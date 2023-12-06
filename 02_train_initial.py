from pathlib import Path
from datetime import timedelta
import argparse
import logging
import json
import time
import os

from transformers import AutoModelForSequenceClassification, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names

import torch
from torch.optim import AdamW

from passagerank.model_options import MODEL_OPTIONS
import passagerank.datasets as datasets
from passagerank.trec_run import trec_eval_run


logger = logging.getLogger(__name__)


def train(args):

    checkpoint_or_tag = args.checkpoint_dir
    if checkpoint_or_tag is None:
        checkpoint_or_tag = args.model_tag

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_or_tag,
        num_labels=2
    )

    model.to(args.device)

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

    # Scaled from Passage Re-Ranking With BERT
    # https://arxiv.org/pdf/1901.04085.pdf
    paper_batch_size = 128
    paper_warmup_steps = 10000
    paper_training_steps = 100000

    warmup_steps = int((paper_batch_size / args.batch_size) * paper_warmup_steps)
    training_steps = int((paper_batch_size / args.batch_size) * paper_training_steps)

    scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=training_steps
    )
    print('scaled warmup steps: {:,}'.format(warmup_steps))
    print('scaled training steps: {:,}'.format(training_steps))

    if args.checkpoint_dir is not None:
        if os.path.exists(args.checkpoint_dir.joinpath("optimizer.pt")):
            optimizer.load_state_dict(torch.load(args.checkpoint_dir.joinpath("optimizer.pt")))
            print("loaded optimizer from state")
        
        if os.path.exists(args.checkpoint_dir.joinpath("scheduler.pt")):
            scheduler.load_state_dict(torch.load(args.checkpoint_dir.joinpath("scheduler.pt")))
            print("loaded scheduler from state")

    model.train()
    start_time = time.time()

    start_step = scheduler.state_dict()['last_epoch']
    seen_pairs = scheduler.state_dict()['last_epoch'] * args.batch_size
    total_steps = args.num_training_steps
    print(
        "start_step:", start_step,
        "seen_pairs:", seen_pairs,
        "total_steps:", total_steps,
        "lr:", scheduler.get_last_lr()[0],
    )

    _train_dataset, train_dataloader = \
        datasets.get_triples_dataloader(
            args.train_triples_file,
            args.model_tag,
            args.batch_size,
            seen_pairs
        )

    data = iter(train_dataloader)
    for step in range(start_step, total_steps):
        
        print_status(step, total_steps, scheduler, start_time)

        ### Record Step
        if step > 0 and step % args.report_steps == 0:
            print_status(step, total_steps, scheduler, start_time, record=True)

        ### Checkpoint step
        if (step > 0 and step != start_step and step % args.checkpoint_steps == 0) or step == 10000:

            model_output = args.output_dir.joinpath('seen_{:07}'.format(seen_pairs))
            
            if not args.dry_run:
                model.save_pretrained(model_output)
                torch.save(optimizer.state_dict(), model_output.joinpath("optimizer.pt"))
                torch.save(scheduler.state_dict(), model_output.joinpath("scheduler.pt"))
            
            print_status(step, total_steps, scheduler, start_time, checkpoint=model_output)

            if not args.dry_run:
                trec_eval_run(model, eval_dataloader, model_output, split=args.eval_split)

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
    
    model_output = args.output_dir.joinpath('seen_{:07}'.format(seen_pairs))

    if not args.dry_run:
        model.save_pretrained(model_output)
        torch.save(optimizer.state_dict(), model_output.joinpath("optimizer.pt"))
        torch.save(scheduler.state_dict(), model_output.joinpath("scheduler.pt"))

    print_status(step, total_steps, scheduler, start_time, checkpoint=model_output)

    if not args.dry_run:
        trec_eval_run(model, eval_dataloader, model_output, split=args.eval_split)


def print_status(this_step, total_steps, scheduler, start_time, record=False, checkpoint=None):
    
    lr = scheduler.get_last_lr()[0]
    elapsed = str(timedelta(seconds=int(time.time() - start_time)))

    if checkpoint is not None:
        status_fmt = "({: 6}/{}) - elapsed: {} - lr: {:.2E} - checkpoint: {}"
        print(status_fmt.format(this_step, total_steps, elapsed, lr, str(checkpoint)))
    else:
        ending = "\n" if record else "\r"
        status_fmt = "({: 6}/{}) - elapsed: {} - lr: {:.2E}"
        print(status_fmt.format(this_step, total_steps, elapsed, lr), end=ending)


def main():

    allowed_models = list(MODEL_OPTIONS.keys())
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    
    tag_or_checkpoint = parser.add_mutually_exclusive_group(required=True)
    tag_or_checkpoint.add_argument(
        "--model_tag", type=str, choices=allowed_models,
        help="HF base checkpoint to start from, one of " + str(allowed_models)
    )
    tag_or_checkpoint.add_argument(
        "--checkpoint_dir", type=Path,
        help="Directory of a previous fine tuned checkpoint"
    )

    # required
    parser.add_argument(
        "--train_triples_file", type=Path, required=True,
        help="Input training triples from MSMARCO, e.g., triples.train.small.tsv"
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Output directory for model checkpoints"
    )
    parser.add_argument(
        "--eval_candidates_file", type=Path, required=True,
        help="The candidates file for the TREC eval split to run on checkpoint."
    )
    parser.add_argument(
        "--num_training_samples", type=int, required=True,
        help="The number of query passage pairs to train on.\n"\
             "Determines the number of training steps based on batch size."
    )
    parser.add_argument(
        "--num_checkpoint_samples", type=int, required=True,
        help="The number of query passage pairs between checkpoints."
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
             "Used to evaluate whether the resulting schedule looks reasonable."
    )
    args = parser.parse_args()

    # resolve relative paths and validate
    if args.checkpoint_dir is not None and not os.path.isabs(args.checkpoint_dir):
        args.checkpoint_dir = Path(__file__).absolute().parent.joinpath(args.checkpoint_dir)

    if not os.path.isabs(args.train_triples_file):
        args.train_triples_file = \
            Path(__file__).absolute().parent.joinpath(args.train_triples_file)

    if not os.path.isabs(args.eval_candidates_file):
        args.eval_candidates_file = \
            Path(__file__).absolute().parent.joinpath(args.eval_candidates_file)

    if not os.path.isabs(args.output_dir):
        args.output_dir = \
            Path(__file__).absolute().parent.joinpath(args.checkpoint_dir)
    
    if not os.path.isabs(args.output_dir):
        args.output_dir = Path(__file__).absolute().parent.joinpath(args.output_dir)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if not (args.train_triples_file.exists() and args.train_triples_file.is_file()):
        raise argparse.ArgumentTypeError(
            "Invalid train triples file:", args.train_triples_file)

    if not (args.eval_candidates_file.exists() and args.eval_candidates_file.is_file()):
        raise argparse.ArgumentTypeError(
            "Invalid eval candidates file:", args.eval_candidates_file)
    
    eval_split = args.eval_candidates_file.stem.split('_')[0]
    queries_name = "{}_queries.tsv".format(eval_split)
    args.eval_queries_file = args.eval_candidates_file.parent.joinpath(queries_name)

    if args.checkpoint_dir is not None:
        if not (args.checkpoint_dir.exists() and args.checkpoint_dir.is_dir()):
            raise argparse.ArgumentTypeError(
                "Invalid checkpoint directory:", args.checkpoint_dir)
        
        # determine model architecture from checkpoint
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

    # determine batch size from model architecture
    args.batch_size = MODEL_OPTIONS[args.model_tag]['batch_size']

    # Number of batches to train on between logging
    args.report_steps = args.report_every // args.batch_size

    # Number of batches to train on between checkpoints
    args.checkpoint_steps = args.num_checkpoint_samples // args.batch_size

    # Set schedule parameters such that for a given batch size
    # the model sees a fixed number of query : passage pairs.
    args.num_training_steps = args.num_training_samples // args.batch_size

    # get device as available
    cpu = torch.device('cpu')
    gpu = torch.device('cuda')
    args.device = gpu if torch.cuda.is_available() else cpu

    for arg, value in args.__dict__.items():
        print("{:>30} - {}".format(arg, value))

    # done with args
    ############################################################################
    train(args)
    

if __name__ == "__main__":
    main()