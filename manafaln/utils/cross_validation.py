import argparse
import os
import sys
import tempfile
from copy import deepcopy

import torch

from manafaln.utils import load_yaml, save_yaml


def get_next_version(lightning_logs_dir) -> int:
    try:
        listdir_info = os.listdir(lightning_logs_dir)
    except OSError:
        return 0

    existing_versions = []
    for d in listdir_info:
        bn = os.path.basename(d)
        if bn.startswith("version_"):
            dir_ver = bn.split("_")[1].replace("/", "")
            existing_versions.append(int(dir_ver))
    if len(existing_versions) == 0:
        return 0
    return max(existing_versions) + 1


def prepare_config(base_config, fold, total_folds, lightning_version, devices):
    # Replace data list key for different folds
    fold_config = deepcopy(base_config)
    fold_config["data"]["training"]["data_list_key"] = [
        f"fold_{(fold+i)%total_folds}" for i in range(1, total_folds)
    ]
    fold_config["data"]["validation"]["data_list_key"] = f"fold_{fold}"

    # Replace devices if given
    if devices is not None:
        fold_config["trainer"]["settings"]["devices"] = devices

    # Replace logger version for different folds
    fold_config["trainer"]["logger"] = {
        "name": "TensorBoardLogger",
        "args": {
            "version": f"version_{lightning_version}/fold_{fold}",
        },
    }

    # Add CheckpointMetricSaver callback for evaluation
    if "CheckpointMetricSaver" not in set(
        callback["name"] for callback in fold_config["trainer"]["callbacks"]
    ):
        fold_config["trainer"]["callbacks"].append({"name": "CheckpointMetricSaver"})

    # Create temp file for the training config of this fold
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml")
    save_yaml(temp_file.name, fold_config)
    temp_file.flush()
    return temp_file


def train_folds(total_folds, base_config, lightning_version, training_folds, devices):
    for fold in training_folds:
        temp_file = prepare_config(
            base_config, fold, total_folds, lightning_version, devices
        )
        exit_code = os.system(
            f"""python -m manafaln.apps.train \
                -c {temp_file.name} \
                -s 42"""
        )
        if exit_code != 0:
            raise RuntimeError


def evaluate_folds(total_folds, lightning_version):
    folds_metrics = {}
    for fold in range(total_folds):
        fold_metrics = torch.load(
            f"lightning_logs/version_{lightning_version}/fold_{fold}/checkpoints/best_model.ckpt",
            map_location="cpu",
        )["logged_metrics"]
        for fold_metric_name, fold_metric_value in fold_metrics.items():
            folds_metrics[fold_metric_name] = folds_metrics.get(
                fold_metric_name, []
            ) + [fold_metric_value]

    folds_metrics_stats = {}
    for metric_name, metric_value in folds_metrics.items():
        metric_value = torch.tensor(metric_value)
        mean = torch.mean(metric_value).item()
        std = torch.std(metric_value).item()
        folds_metrics_stats[metric_name] = f"{mean:.3f} +/- {std:.3f}"

    save_yaml(
        f"lightning_logs/version_{lightning_version}/cross_validation_result.yaml",
        folds_metrics_stats,
    )


if __name__ == "__main__":
    sys.path.append(".")  # For unpickling custom module when loading ckpt

    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=5, help="The total number of folds.")
    parser.add_argument("-e", action="store_true", help="Evaluate the folds")
    parser.add_argument("-s", action="store_true", help="Skip training.")
    parser.add_argument(
        "-c", type=str, default="./configs/train.yaml", help="Path to training config."
    )
    parser.add_argument(
        "-v",
        type=int,
        default=argparse.SUPPRESS,
        help="The version number in lightning logs, create new version if not specified.",
    )
    parser.add_argument(
        "-t",
        type=int,
        nargs="+",
        default=argparse.SUPPRESS,
        help="The folds to be trained, train all if not specified.",
    )
    parser.add_argument(
        "-d",
        type=int,
        nargs="+",
        default=None,
        help="If given, overwrites the devices in config.",
    )
    args = parser.parse_args()

    total_folds = args.k
    base_config = load_yaml(args.c)
    devices = args.d

    if "v" in args:
        lightning_version = args.v
    else:
        lightning_version = get_next_version("lightning_logs")

    if "t" in args:
        training_folds = args.t
    else:
        training_folds = range(total_folds)

    if not args.s:
        train_folds(
            total_folds, base_config, lightning_version, training_folds, devices
        )

    if args.e:
        evaluate_folds(total_folds, lightning_version)
