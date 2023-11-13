from pathlib import Path
import argparse

import torch

from passagerank.pruning.intermediate_pruning import finalize_model

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--checkpoints_dir", type=Path, required=True,
        help="Run directory to search for model checkpoints."
    )
    args = parser.parse_args()

    model_checkpoints = list(args.checkpoints_dir.iterdir())
    model_checkpoints = [d for d in model_checkpoints if d.is_dir()]

    for checkpoint in model_checkpoints:

        model = torch.load(str(checkpoint) + "/model.pt")
        print()
        print(checkpoint.stem)
        
        print(torch.sum(model.distilbert.transformer.layer[5].ffn.lin1.weight_mask[:,0]).item())
        print(sum(p.numel() for p in model.parameters()) * 4)
        finalize_model(model, exclude_layers=[0])
        print(sum(p.numel() for p in model.parameters()) * 4)
        torch.save(model, str(checkpoint) + "/pruned_model.pt")
            
if __name__ == "__main__":
    main()