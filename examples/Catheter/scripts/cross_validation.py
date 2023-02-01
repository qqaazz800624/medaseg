import argparse
import os
from copy import deepcopy

import torch

from manafaln.utils import load_yaml, save_yaml

def get_next_version(root_dir) -> int:
    try:
        listdir_info = os.listdir(root_dir)
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

def prepare_configs(version, k, base_dir):
    fold_configs_dir = os.path.join(base_dir, f"{k}fold")
    fold_config_path = os.path.join(fold_configs_dir, "fold_{fold}.yaml")
    os.makedirs(fold_configs_dir, exist_ok=True)

    base_config = load_yaml(os.path.join(base_dir, "train.yaml"))

    for fold in range(k):
        # Replace data list key for different folds
        fold_config = deepcopy(base_config)
        fold_config["data"]["training"]["data_list_key"] = [f"fold_{(fold+i)%k}" for i in range(1, k)]
        fold_config["data"]["validation"]["data_list_key"] = f"fold_{fold}"

        # Replace logger version for different folds
        fold_config["trainer"]["logger"] = {
                "name": "TensorBoardLogger",
                "args": {
                    "version": f"version_{version}/fold_{fold}",
                }
        }
        save_yaml(fold_config_path.format(fold=fold), fold_config)

def train_folds(k, base_dir):
    fold_configs_dir = os.path.join(base_dir, f"{k}fold")
    fold_config_path = os.path.join(fold_configs_dir, "fold_{fold}.yaml")
    for fold in range(k):
        os.system(
            f"""python -m manafaln.apps.train \
                -c {fold_config_path.format(fold=fold)} \
                -s 42"""
            )

def evaluate_folds(k, version):
    folds_metrics = {}
    for fold in range(k):
        fold_metrics = torch.load(f"lightning_logs/version_{version}/fold_{fold}/checkpoints/best_model.ckpt", map_location="cpu")["logged_metrics"]
        for fold_metric_name, fold_metric_value in fold_metrics.items():
            folds_metrics[fold_metric_name] = folds_metrics.get(fold_metric_name, []) + [fold_metric_value]

    folds_metrics_stats = {}
    for metric_name, metric_value in folds_metrics.items():
        metric_value = torch.tensor(metric_value)
        mean = torch.mean(metric_value).item()
        std  = torch.std(metric_value).item()
        folds_metrics_stats[metric_name] = f"{mean:.3f} +/- {std:.3f}"

    save_yaml(f"lightning_logs/version_{version}/cross_validation_result.yaml", folds_metrics_stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("-r", "--root", type=str, default="./config/")
    args = parser.parse_args()

    k = args.k
    base_dir = args.root

    version = get_next_version("lightning_logs")

    prepare_configs(version, k, base_dir)
    train_folds(k, base_dir)
    evaluate_folds(k, version)
