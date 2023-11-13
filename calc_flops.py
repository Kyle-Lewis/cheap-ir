from pathlib import Path
import argparse
import json

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from transformers import AutoModelForSequenceClassification

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--checkpoints_dir", type=Path, required=True,
        help="Run directory to search for model checkpoints."
    )
    parser.add_argument(
        "--pruned", action="store_true",
        help="If the directory contains pruned checkpoints that should be loaded"\
             "as opposed to the masked checkpoints that preceed them."
    )

    args = parser.parse_args()

    model_checkpoints = list(args.checkpoints_dir.iterdir())
    model_checkpoints = [d for d in model_checkpoints if d.is_dir()]

    cpu = torch.device('cpu')
    gpu = torch.device('cuda')
    device = gpu if torch.cuda.is_available() else cpu

    B = 100
    T = 512
    tokens = torch.randint(0, 3000, (B, T)).to(device)

    for checkpoint in model_checkpoints:

        print(checkpoint)
        if args.pruned:
            model = torch.load(checkpoint.joinpath("pruned_model.pt"), map_location=device)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            model.to(device)

        model.eval()
        with torch.no_grad():
            flops = FlopCountAnalysis(model, tokens)
            print(flops.total())
            print(flop_count_table(flops, max_depth=2, show_param_shapes=False))

        with open(checkpoint.joinpath('flops.json'), 'w+') as outfile:
            outfile.write(json.dumps({"flops": flops.total()}))
            
if __name__ == "__main__":
    main()