from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import yaml

def main(args):
    if not args.input.exists():
        raise FileNotFoundError(f"File not found: {args.input}")

    with open(args.input, "r") as f:
        data = yaml.safe_load(f)

    records = data["val_results"]

    metric_names = set()
    for record in records:
        metric_names.update(record["metrics"].keys())
    metric_names = sorted(list(metric_names))

    rows = []
    columns = ["model", "data"] + metric_names
    for record in records:
        row = [record["model_owner"], record["data_client"]]
        for metric in metric_names:
            row.append(record["metrics"].get(metric, None))
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df.sort_values(by=["model", "data"], inplace=True)

    if len(args.exclude_metrics) > 0:
        df.drop(args.exclude_metrics, axis=1, inplace=True)

    if args.compact:
        df.drop(["data"], axis=1, inplace=True)
        df = df.groupby("model")[df.columns].first()

    df.to_csv(args.output, index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--exclude_metrics", "-e", nargs='+', default=[])
    parser.add_argument("--compact", "-c", action="store_true")

    args = parser.parse_args()
    main(args)

