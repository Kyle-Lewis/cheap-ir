from pathlib import Path
from datetime import timedelta
import argparse
import logging
import json
import time
import os

import transformers
from transformers import AutoModelForSequenceClassification
from transformers.trainer_pt_utils import get_parameter_names

import torch
from torch.optim import AdamW

from passagerank.model_options import MODEL_OPTIONS
import passagerank.datasets as datasets
from passagerank.pruning.intermediate_pruning import (
    IntermediatesPruner,
    get_prune_schedule
)
from passagerank.trec_run import trec_eval_run


logger = logging.getLogger(__name__)


def train(args):

    if "seen_pairs" in args.model_config:
        model = torch.load(args.checkpoint_dir.joinpath("model.pt"), map_location=args.device)
    
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.checkpoint_dir,
            num_labels=2
        )
        model.to(args.device)

    num_skip = args.seen_pairs
    if "seen_pairs" in args.model_config:
        num_skip = args.model_config["seen_pairs"]

    _train_dataset, train_dataloader = \
        datasets.get_triples_dataloader(
            args.train_triples_file,
            args.model_tag,
            args.batch_size,
            num_skip
        )

    _eval_dataset, eval_dataloader = \
        datasets.get_trec_validation_dataloader(
            args.eval_queries_file,
            args.eval_candidates_file,
            args.model_tag
        )
    
    weight_decay = 1e-2
    # Excluding normalization layers and all biases from weight decay
    # https://github.com/huggingface/transformers/blob/7c6cd0ac28f1b760ccb4d6e4761f13185d05d90b/src/transformers/trainer.py#L800
    decay_parameters = get_parameter_names(model, forbidden_layer_types=[torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    optimizer.load_state_dict(torch.load(args.checkpoint_dir.joinpath("optimizer.pt")))
    scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps,
        num_cycles=args.num_prune_cycles,
    )

    print("last scheduler step:", scheduler.last_epoch)
    if "seen_pairs" in args.model_config:
        scheduler.load_state_dict(torch.load(args.checkpoint_dir.joinpath("scheduler.pt")))

    print("last scheduler step:", scheduler.last_epoch)

    intermediates_pruner = IntermediatesPruner(model, exclude_layers=[0])
    
    model.train()
    start_time = time.time()

    prune_cycle = args.model_config.get("prune_cycle", 0)
    num_pruned = sum(p[1] for p in args.prune_schedule[:prune_cycle])
    start_dim = args.intermediate_hdim
    seen_pairs = args.model_config.get("seen_pairs", args.seen_pairs)
    total_steps = args.num_training_steps
    print("start_step:", 0)
    print("seen_pairs:", seen_pairs)
    print("total_steps:", total_steps)
    print("lr:", scheduler.get_last_lr()[0])
    print("prune_cycle:", prune_cycle)
    print("next prune step:", args.prune_schedule[prune_cycle][0])
    print("num_pruned:", num_pruned)
    print("cur dim a:", start_dim - num_pruned)
    if "seen_pairs" in args.model_config:
        print("true cur dim:", torch.sum(model.base_model.transformer.layer[-1].ffn.lin1.weight_mask[:, 0]))
    
    data = iter(train_dataloader)
    
    for step in range(scheduler.last_epoch, total_steps):
        
        print_status(step, total_steps, scheduler, start_time, num_pruned)
        
        ### Record Step
        if step > 0 and step % args.report_steps == 0:
            print_status(step, total_steps, scheduler, start_time, num_pruned, record=True)

        if (prune_cycle < len(args.prune_schedule)) and step == args.prune_schedule[prune_cycle][0]:
            
            _step, n_prune = args.prune_schedule[prune_cycle]

            make_checkpoint = False
            if prune_cycle == 0:
                pass
            elif step == 185000:
                pass
            elif prune_cycle == 1:  # for an early sanity check
                make_checkpoint = True
            elif prune_cycle < (0.5 * args.num_prune_cycles):
                if prune_cycle % (args.num_prune_cycles // 10) == 0:
                    make_checkpoint = True
            elif prune_cycle < (0.9 * args.num_prune_cycles):
                if prune_cycle % (args.num_prune_cycles // 20) == 0:
                    make_checkpoint = True
            elif prune_cycle >= (0.9 * args.num_prune_cycles):
                make_checkpoint = True

            if make_checkpoint:
                model_output = args.output_dir.joinpath(
                    'seen_{:07}_idim_{:04}'.format(seen_pairs, start_dim - num_pruned))
                
                if not args.dry_run:
                    os.makedirs(model_output, exist_ok=True)
                    torch.save(model, os.path.join(model_output, "model.pt"))
                    torch.save(optimizer.state_dict(), os.path.join(model_output, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(model_output, "scheduler.pt"))

                print_status(step, total_steps, scheduler, start_time, num_pruned, checkpoint=model_output)
                if not args.dry_run:
                    trec_eval_run(model, eval_dataloader, model_output, split=args.eval_split)

            if not args.dry_run:
                intermediates_pruner.prune_intermediate_layers(amount=n_prune)

            num_pruned += n_prune
            prune_cycle += 1
            print("prune_cycle:", prune_cycle)
        
        ### Training
        if not args.dry_run:
            batch = next(data)

            batch = {k: v.to(args.device) for k, v in batch.items()}

            outputs = model(**batch)

            loss = outputs.loss
            loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        seen_pairs += args.batch_size
    
    model_output = args.output_dir.joinpath(
        'seen_{:07}_idim_{:04}'.format(seen_pairs, start_dim - num_pruned))
    
    if not args.dry_run:
        os.makedirs(model_output, exist_ok=True)
        torch.save(model, os.path.join(model_output, "model.pt"))
        torch.save(optimizer.state_dict(), os.path.join(model_output, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(model_output, "scheduler.pt"))
    
    print_status(step, total_steps, scheduler, start_time, num_pruned, checkpoint=model_output)
    if not args.dry_run:
        trec_eval_run(model, eval_dataloader, model_output, split=args.eval_split)


def print_status(this_step, total_steps, scheduler, start_time, n_pruned, record=False, checkpoint=None):
    
    lr = scheduler.get_last_lr()[0]
    elapsed = str(timedelta(seconds=int(time.time() - start_time)))

    if checkpoint is not None:
        status_fmt = "({: 6}/{}) - elapsed: {} - lr: {:.2E} - pruned: {} - checkpoint: {}"
        print(status_fmt.format(this_step, total_steps, elapsed, lr, n_pruned, str(checkpoint)))
    else:
        ending = "\n" if record else "\r"
        status_fmt = "({: 6}/{}) - elapsed: {} - lr: {:.2E} - pruned: {}"
        print(status_fmt.format(this_step, total_steps, elapsed, lr, n_pruned), end=ending)


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    
    # required
    parser.add_argument(
        "--train_triples_file", type=Path, required=True,
        help="Input training triples from MSMARCO, e.g., triples.train.small.tsv"
    )
    parser.add_argument(
        "--checkpoint_dir", type=Path, required=True,
        help="Directory of a previous fine tuned checkpoint to begin from"
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Output directory for further model checkpoints"
    )
    parser.add_argument(
        "--eval_candidates_file", type=Path, required=True,
        help="The candidates file for the TREC eval split to run on checkpoint."
    )
    parser.add_argument(
        "--seen-pairs", type=int, required=True,
        help="The number of training samples to skip as they have been seen." 
    )
    parser.add_argument(
        "--target_sparsity", type=float, required=True,
        help="The fraction of attention heads that should be pruned. " \
             "Higher targets may not be reached given the policy:\n" \
             "    - Never prune from layer 0\n" \
             "    - Never prune from a layer with only 1 head remaining"
    )
    parser.add_argument(
        "--num_prune_cycles", type=int, required=True,
        help="The number of pruning cycles to run to reach target sparsity."
    )
    parser.add_argument(
        "--num_warmup_samples", type=int, required=True,
        help="The number of query passage pairs to warmup on.\n"\
             "Determines the number of warmup steps based on batch size."
    )
    parser.add_argument(
        "--num_training_samples", type=int, required=True,
        help="The number of query passage pairs to train on.\n"\
             "Determines the number of training steps based on batch size."
    )
    
    # optional
    parser.add_argument(
        "--report_every", type=int, default=10000,
        help="Number of samples between which to log stats to console (default 10000)."
    )
    parser.add_argument(
        "-lr", "--learning-rate", type=float, default=3e-6,
        help="Learning rate to reach after warmup (default 3e-6)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without touching the model to see the status outputs that will result."\
             "Used to evaluate whether the resulting pruning schedule looks reasonable."
    )
    
    args = parser.parse_args()

    if not os.path.isabs(args.checkpoint_dir):
        args.checkpoint_dir = Path(__file__).absolute().parent.joinpath(args.checkpoint_dir)

    # resolve relative paths and validate
    if not os.path.isabs(args.train_triples_file):
        args.train_triples_file = \
            Path(__file__).absolute().parent.joinpath(args.train_triples_file)

    if not os.path.isabs(args.eval_candidates_file):
        args.eval_candidates_file = \
            Path(__file__).absolute().parent.joinpath(args.eval_candidates_file)

    if not os.path.isabs(args.checkpoint_dir):
        args.checkpoint_dir = \
            Path(__file__).absolute().parent.joinpath(args.checkpoint_dir)
    
    if not os.path.isabs(args.output_dir):
        args.output_dir = Path(__file__).absolute().parent.joinpath(args.output_dir)
    
    if not (args.train_triples_file.exists() and args.train_triples_file.is_file()):
        raise argparse.ArgumentTypeError(
            "Invalid train triples file:", args.train_triples_file)
    
    if not (args.checkpoint_dir.exists() and args.checkpoint_dir.is_dir()):
        raise argparse.ArgumentTypeError(
            "Invalid checkpoint directory:", args.checkpoint_dir)
    
    if not (args.eval_candidates_file.exists() and args.eval_candidates_file.is_file()):
        raise argparse.ArgumentTypeError(
            "Invalid eval candidates file:", args.eval_candidates_file)
    
    eval_split = args.eval_candidates_file.stem.split('_')[0]
    queries_name = "{}_queries.tsv".format(eval_split)
    args.eval_queries_file = args.eval_candidates_file.parent.joinpath(queries_name)

    if os.path.exists(args.checkpoint_dir.joinpath("state.json")):
        # determine model architecture from continued checkpoint
        with open(args.checkpoint_dir.joinpath("state.json"), "r") as state_file:
            model_config = json.loads(state_file.read())
            if model_config["model_type"] == "roberta":
                args.model_tag = "roberta-base"
            elif model_config["model_type"] == "distilbert":
                args.model_tag = "distilbert-base-uncased"
            else:
                raise ValueError(
                    "Checkpoint is for unsupported model type:",
                    model_config["model_type"])

    else:
        # determine model architecture from initial checkpoint
        with open(args.checkpoint_dir.joinpath("config.json"), "r") as config_file:
            model_config = json.loads(config_file.read())
            if model_config["model_type"] == "roberta":
                args.model_tag = "roberta-base"
            elif model_config["model_type"] == "distilbert":
                args.model_tag = "distilbert-base-uncased"
            else:
                raise ValueError(
                    "Checkpoint is for unsupported model type:",
                    model_config["model_type"])

    args.model_config = model_config

    # determine batch size from model architecture
    args.batch_size = MODEL_OPTIONS[args.model_tag]['batch_size']

    # Number of batches to train on between logging
    args.report_steps = args.report_every // args.batch_size

    # Set schedule parameters such that for a given batch size and pruning level
    # the model sees a fixed number of query : passage pairs.
    args.num_prune_cycles = args.num_prune_cycles
    args.num_warmup_steps = args.num_warmup_samples // args.batch_size
    args.num_training_steps = args.num_training_samples // args.batch_size
    args.cycle_training_steps = args.num_training_steps // args.num_prune_cycles

    num_training_steps_after_warmup = args.num_training_steps - args.num_warmup_steps
    
    if args.model_tag == "roberta-base":
        n_heads = model_config['num_attention_heads']
        n_layers = model_config['num_hidden_layers']
        intermediate_hdim = model_config['intermediate_size']

    elif args.model_tag == "distilbert-base-uncased":
        n_heads = model_config['n_layers']
        n_layers = model_config['n_heads']
        intermediate_hdim = model_config['hidden_dim']
    
    args.intermediate_hdim = intermediate_hdim

    args.prune_schedule = get_prune_schedule(
        num_training_steps_after_warmup,
        args.num_prune_cycles,
        args.target_sparsity,
        intermediate_hdim,
        warmup_steps=args.num_warmup_steps
    )

    # get device as available
    cpu = torch.device('cpu')
    gpu = torch.device('cuda')
    args.device = gpu if torch.cuda.is_available() else cpu

    for arg, value in args.__dict__.items():
        print("{:>20} - {}".format(arg, value))

    # done with args
    ############################################################################
    train(args)
    

if __name__ == "__main__":
    main()