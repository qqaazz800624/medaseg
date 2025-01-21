from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path

import torch
import pytorch_lightning

def main(args):
    """
    Load a checkpoint from the source file and save a modified version of it to the destination file.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        None
    """
    ckpt = torch.load(args.src, map_location='cpu')

    output = OrderedDict()

    output["state_dict"] = OrderedDict()
    for key, value in ckpt["model"].items():
        output["state_dict"]["model." + key] = value

    output["epoch"] = 0
    output["global_step"] = 0
    output["pytorch-lightning_version"] = pytorch_lightning.__version__

    torch.save(output, args.dst)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src", "-s", type=Path, required=True)
    parser.add_argument("--dst", "-d", type=Path, required=True)
    args = parser.parse_args()
    main(args)

