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
from torch.utils.data import DataLoader

from passagerank.model_options import MODEL_OPTIONS
from passagerank.datasets import TriplesDataset, TrecValidationDataset
from passagerank.pruning.head_pruning import HeadPruner
from passagerank.pruning.intermediate_pruning import IntermediatesPruner
from passagerank.pruning.schedule import get_mixed_pruning_schedule
from passagerank.trec_run import trec_eval_run

def train(args):

    cpu = torch.device('cpu')
    gpu = torch.device('cuda')
    device = gpu if torch.cuda.is_available() else cpu

    if args.command == 'initial':
        model = AutoModelForSequenceClassification.from_pretrained(
                args.checkpoint_dir,
                num_labels=2
            )
        model.to(device)

    elif args.command == 'continue':
        model = torch.load(
            args.checkpoint_dir.joinpath("model.pt"),
            map_location=device
        )

    train_dataset = TriplesDataset(
        args.train_triples_file,
        model_tag=args.model_tag,
        batch_size=args.batch_size,
        num_seen=args.seen_pairs,
    )

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=1,
        worker_init_fn=TriplesDataset.worker_init_fn,
        batch_size=None
    )

    trec_val_dataset = TrecValidationDataset(
        args.eval_queries_file,
        args.eval_candidates_file,
        model_tag=args.model_tag
    )

    trec_val_dataloader = DataLoader(
        trec_val_dataset,
        num_workers=1,
        worker_init_fn=TrecValidationDataset.worker_init_fn,
        batch_size=None
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
    if args.command != 'continue':
        optimizer.load_state_dict(torch.load(args.checkpoint_dir.joinpath("optimizer.pt")))

    scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps,
        num_cycles=args.num_prune_cycles,
    )
    

    head_pruner = HeadPruner(model)
    intermediates_pruner = IntermediatesPruner(model, exclude_layers=[0])

    prune_stage = 0
    prune_cycle = 0
    num_pruned_neurons = 0
    num_pruned_heads = 0
    seen_pairs = args.seen_pairs

    if args.command == 'continue':
        
        scheduler.load_state_dict(torch.load(args.checkpoint_dir.joinpath("scheduler.pt")))
        # rewind
        scheduler.step(scheduler.last_epoch-100)

        cur_dim = int(torch.sum(model.base_model.transformer.layer[-1].ffn.lin1.weight_mask[:, 0]).item())
        num_pruned_heads = sum(len(v) for v in model.config.pruned_heads.values())
        num_pruned_neurons = args.intermediate_hidden_dim - cur_dim
        while args.prune_schedule[prune_stage]['step'] < scheduler.last_epoch:
            if args.prune_schedule[prune_stage]['action'] == 'prune':
                prune_cycle += 1
            prune_stage += 1
        
        seen_pairs += args.batch_size * scheduler.last_epoch
    
    if args.model_tag == "roberta-base":
        num_heads_per_layer = args.initial_model_config['num_attention_heads']
        n_layers = args.initial_model_config['num_hidden_layers']
        num_heads = num_heads_per_layer * n_layers

    elif args.model_tag == "distilbert-base-uncased":
        num_heads_per_layer = args.initial_model_config['n_layers']
        n_layers = args.initial_model_config['n_heads']
        num_heads = num_heads_per_layer * n_layers

    num_neurons = args.intermediate_hidden_dim - num_pruned_neurons
    num_heads -= num_pruned_heads

    print("start_step:", scheduler.last_epoch + 1)
    print("seen_pairs:", seen_pairs)
    print("num_training_steps:", args.num_training_steps)
    print("lr:", scheduler.get_last_lr()[0])
    print("prune_stage:", prune_stage)
    print("prune_cycle:", prune_cycle)
    print("next prune step:", args.prune_schedule[prune_stage])
    print("num_heads:", num_heads)
    print("num pruned heads:", num_pruned_heads)
    print("num_neurons:", num_neurons)
    print("num_pruned_neurons:", num_pruned_neurons)
    print("cur dim:", args.intermediate_hidden_dim - num_pruned_neurons)
    if args.command == 'continue':
        print("true cur dim:", torch.sum(model.base_model.transformer.layer[-1].ffn.lin1.weight_mask[:, 0]))

    model.train()
    start_time = time.time()
    
    accumulate_attentions = False
    data = iter(train_dataloader)

    for step in range(scheduler.last_epoch + 1, args.num_training_steps):
        
        print_status(step, args.num_training_steps, scheduler, start_time, num_heads, num_neurons)
        if step > 0 and step % args.report_steps == 0:
            print_status(step, args.num_training_steps, scheduler, start_time, num_heads, num_neurons, record=True)

        step_action = {'action': 'continue'}
        if prune_stage < len(args.prune_schedule) and step == args.prune_schedule[prune_stage]['step']:
            step_action = args.prune_schedule[prune_stage]
            prune_stage += 1

        if step_action['action'] == 'continue':
            pass
        
        elif step_action['action'] == 'accumulate':
            accumulate_attentions = True

        elif step_action['action'] == 'prune':
            
            make_checkpoint = False
            if prune_cycle == 0:
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
            
            model_output = os.path.join(args.output_dir,
                'seen_{:08}_nheads_{:04}_idim_{:04}'.format(
                    seen_pairs, num_heads, num_neurons
            ))

            # When resuming training there is no need to save over the existing
            # checkpoint and run eval again.
            if args.command == 'continue' and model_output == str(args.checkpoint_dir):
                print('Skipping existing checkpoint')
                make_checkpoint = False

            if make_checkpoint:
                if not args.dry_run:
                    os.makedirs(model_output, exist_ok=True)
                    torch.save(model, os.path.join(model_output, "model.pt"))
                    torch.save(optimizer.state_dict(), os.path.join(model_output, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(model_output, "scheduler.pt"))

                print_status(step, args.num_training_steps, scheduler, start_time, num_heads, num_neurons, checkpoint=model_output)
                if not args.dry_run:
                    trec_eval_run(model, trec_val_dataloader, model_output, affix=args.eval_split)
            
            if not args.dry_run:
                intermediates_pruner.prune_intermediate_layers(amount=step_action['neurons_to_prune'])
                pruned_head = head_pruner.prune_head()
                print('pruned:', pruned_head)

            print(step_action)
            num_neurons -= step_action['neurons_to_prune']
            num_heads -= 1
            prune_cycle += 1
            accumulate_attentions = False

        else:
            raise ValueError(step_action)
    
        ### Training
        if not args.dry_run:
            batch = next(data)

            batch = {k: v.to(device) for k, v in batch.items()}

            if accumulate_attentions:
                outputs = model(**batch, output_attentions=True)
                head_pruner.accumulate(outputs)
            else:
                outputs = model(**batch)

            loss = outputs.loss
            loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        seen_pairs += args.batch_size
    
    #########
    model_output = os.path.join(args.output_dir,
        'seen_{:08}_nheads_{:04}_idim_{:04}'.format(
            seen_pairs, num_heads, num_neurons
    ))
    
    if not args.dry_run:
        os.makedirs(model_output, exist_ok=True)
        torch.save(model, os.path.join(model_output, "model.pt"))
        torch.save(optimizer.state_dict(), os.path.join(model_output, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(model_output, "scheduler.pt"))
    
    print_status(step, args.num_training_steps, scheduler, start_time, num_heads, num_neurons, checkpoint=model_output)
    if not args.dry_run:
        trec_eval_run(model, trec_val_dataloader, model_output, affix=args.eval_split)


def print_status(this_step, total_steps, scheduler, start_time, n_heads, n_neurons, record=False, checkpoint=None):
    
    lr = scheduler.get_last_lr()[0]
    elapsed = str(timedelta(seconds=int(time.time() - start_time)))

    if checkpoint is not None:
        status_fmt = "({: 6}/{}) - elapsed: {} - lr: {:.2E} - n_heads: {} - n_neurons: {} - checkpoint: {}"
        print(status_fmt.format(this_step, total_steps, elapsed, lr, n_heads, n_neurons, str(checkpoint)))
    else:
        ending = "\n" if record else "\r"
        status_fmt = "({: 6}/{}) - elapsed: {} - lr: {:.2E} - n_heads: {} - n_neurons: {}"
        print(status_fmt.format(this_step, total_steps, elapsed, lr, n_heads, n_neurons), end=ending)


def main():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    
    subparsers = parser.add_subparsers(dest="command")

    ############################################################################
    # Begin new training / fine tuning run

    from_initial_parser = subparsers.add_parser(
        "initial",
        help="Begin pruning and continue training from an initial training checkpoint"
    )
    
    # required
    from_initial_parser.add_argument(
        "--train_triples_file", type=Path, required=True,
        help="Input training triples from MSMARCO, e.g., triples.train.small.tsv"
    )
    from_initial_parser.add_argument(
        "--checkpoint_dir", type=Path, required=True,
        help="Directory of a previous fine tuned checkpoint to begin from"
    )
    from_initial_parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Output directory for further model checkpoints"
    )
    from_initial_parser.add_argument(
        "--eval_candidates_file", type=Path, required=True,
        help="The candidates file for the TREC eval split to run on checkpoint."
    )
    from_initial_parser.add_argument(
        "--seen-pairs", type=int, required=True,
        help="The number of training samples to skip as they have been seen." 
    )
    from_initial_parser.add_argument(
        "--target_sparsity", type=float, required=True,
        help="The fraction of attention heads that should be pruned. " \
             "Higher targets may not be reached given the policy:\n" \
             "    - Never prune from layer 0\n" \
             "    - Never prune from a layer with only 1 head remaining"
    )
    from_initial_parser.add_argument(
        "--num_prune_cycles", type=int, required=True,
        help="The number of pruning cycles to run to reach target sparsity."
    )
    from_initial_parser.add_argument(
        "--num_warmup_samples", type=int, required=True,
        help="The number of query passage pairs to warmup on.\n"\
             "Determines the number of warmup steps based on batch size."
    )
    from_initial_parser.add_argument(
        "--num_training_samples", type=int, required=True,
        help="The number of query passage pairs to train on.\n"\
             "Determines the number of training steps based on batch size."
    )

    # optional
    from_initial_parser.add_argument(
        "--report_every", type=int, default=10000,
        help="Number of samples between which to log stats to console (default 10000)."
    )
    from_initial_parser.add_argument(
        "-lr", "--learning-rate", type=float, default=3e-6,
        help="Learning rate to reach after warmup (default 3e-6)."
    )
    from_initial_parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without touching the model to see the status outputs that will result."\
             "Used to evaluate whether the resulting pruning schedule looks reasonable."
    )
    
    ############################################################################
    # Continue from existing fine tuning run
    continue_from_parser = subparsers.add_parser(
        "continue",
        help="Continue pruning and training from an existing pruning checkpoint"
    )

    # required
    continue_from_parser.add_argument(
        "--checkpoint_dir", type=Path, required=True,
        help="Directory of a previous fine tuned checkpoint to begin from"
    )

    args = parser.parse_args()

    if args.command == 'continue':
        if not os.path.isabs(args.checkpoint_dir):
            args.checkpoint_dir = \
                Path(__file__).absolute().parent.joinpath(args.checkpoint_dir)
        
        if not (args.checkpoint_dir.exists() and args.checkpoint_dir.is_dir()):
            raise argparse.ArgumentTypeError(
                "Invalid checkpoint directory:", args.checkpoint_dir)
    
        with open(args.checkpoint_dir.parent.joinpath('args.json'), 'r') as argsfile:
            orig_args = json.loads(argsfile.read())
            for k, v in orig_args.items():
                if k == 'command':
                    continue
                elif k == 'checkpoint_dir':
                    args.orig_checkpoint_dir = v
                else:
                    setattr(args, k, v)

    else:
    
        # resolve relative paths and validate
        if not os.path.isabs(args.train_triples_file):
            args.train_triples_file = \
                Path(__file__).absolute().parent.joinpath(args.train_triples_file)

        if not (args.train_triples_file.exists() and args.train_triples_file.is_file()):
            raise argparse.ArgumentTypeError(
                "Invalid train triples file:", args.train_triples_file)
        
        if not os.path.isabs(args.checkpoint_dir):
            args.checkpoint_dir = \
                Path(__file__).absolute().parent.joinpath(args.checkpoint_dir)
        
        if not (args.checkpoint_dir.exists() and args.checkpoint_dir.is_dir()):
            raise argparse.ArgumentTypeError(
                "Invalid checkpoint directory:", args.checkpoint_dir)
        
        if not os.path.isabs(args.output_dir):
            args.output_dir = Path(__file__).absolute().parent.joinpath(args.output_dir)

        if args.output_dir.exists():
            raise argparse.ArgumentTypeError(
                "Output directory already exists:", args.output_dir)
        
        if not os.path.isabs(args.eval_candidates_file):
            args.eval_candidates_file = \
                Path(__file__).absolute().parent.joinpath(args.eval_candidates_file)
        
        if not (args.eval_candidates_file.exists() and args.eval_candidates_file.is_file()):
            raise argparse.ArgumentTypeError(
                "Invalid eval candidates file:", args.eval_candidates_file)

        eval_split = args.eval_candidates_file.stem.split('_')[0]
        queries_name = "{}_queries.tsv".format(eval_split)
        args.eval_queries_file = args.eval_candidates_file.parent.joinpath(queries_name)

        # determine model architecture from initial checkpoint
        with open(args.checkpoint_dir.joinpath("config.json"), "r") as config_file:
            
            initial_model_config = json.loads(config_file.read())
            if initial_model_config["model_type"] == "roberta":
                args.model_tag = "roberta-base"
                args.intermediate_hidden_dim = initial_model_config['intermediate_size']

            elif initial_model_config["model_type"] == "distilbert":
                args.model_tag = "distilbert-base-uncased"
                args.intermediate_hidden_dim = initial_model_config['hidden_dim']
            
            else:
                raise ValueError(
                    "Checkpoint is for unsupported model type:",
                    initial_model_config["model_type"])

        args.initial_model_config = initial_model_config

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

        args.prune_schedule = get_mixed_pruning_schedule(
            args.intermediate_hidden_dim,
            args.target_sparsity,
            args.num_warmup_steps,
            args.num_training_steps,
            args.num_prune_cycles,
        )

        os.makedirs(args.output_dir)
        with open(args.output_dir.joinpath('args.json'), 'w+') as argsfile:
            dct = {}
            for k, v in args.__dict__.items():
                if isinstance(v, Path):
                    dct[k] = str(v)
                else:
                    dct[k] = v
            argsfile.write(json.dumps(dct, indent=4))

    for arg, value in args.__dict__.items():
        print("{:>20} - {}".format(arg, value))

    # done with args
    ############################################################################
    train(args)
    

if __name__ == "__main__":
    main()