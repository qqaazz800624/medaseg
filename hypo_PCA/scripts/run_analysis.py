#!/usr/bin/env python

import json
import logging
from argparse import ArgumentParser
from pathlib import Path, PurePath
from pprint import PrettyPrinter
from typing import Dict, List

import nibabel as nib
import numpy as np
from tqdm import tqdm
from ruamel.yaml import YAML

def resolve_datalist(
    dataroot: str,
    datalist: str,
    split_keys: List[str],
    img_key: str,
    seg_key: str
):
    with open(datalist) as f:
        raw_data = json.load(f)

    cases = []
    for split in split_keys:
        if split not in raw_data:
            raise ValueError(f"Split key '{split}' not found in datalist.")
        cases += raw_data[split]

    data = []
    prefix = Path(dataroot)
    for case in cases:
        img_list = [prefix / img_path for img_path in case[img_key]]
        seg = prefix / case[seg_key]

        # 檢查每個影像是否存在
        for img in img_list:
            if not img.exists():
                raise RuntimeError(f"Image file {img} does not exist!")
        if not seg.exists():
            raise RuntimeError(f"Segmentation label file {seg} does not exist!")

        data.append({"image": [str(p) for p in img_list], "label": str(seg)})

    return data



def percentile(hist):
    cs = np.cumsum(hist)
    low  = float(np.argmax(cs > cs[-1] * 0.005))
    high = float(np.argmax(cs > cs[-1] * 0.995))
    return low, high

def analysis(data):
    shapes = []
    spacings = []
    histogram = np.zeros(2049, dtype=np.int64)

    for case in tqdm(data, desc="Collecting information from data"):
        seg = nib.load(case["label"])
        seg = nib.as_closest_canonical(seg)
        seg_shape = seg.shape

        for img_path in case["image"]:
            img = nib.load(img_path)
            img = nib.as_closest_canonical(img)

            if img.shape != seg_shape:
                raise RuntimeError(f"Shape mismatch between image {img_path} and segmentation.")

            x, y, z = img.shape
            xs, ys, zs, _ = np.abs(np.diag(img.affine))
            shapes.append([x / xs, y / ys, z / zs])
            spacings.append([xs, ys, zs])

            img_data = np.asanyarray(img.dataobj).astype(np.int16)
            seg_data = np.asanyarray(seg.dataobj).astype(np.uint8)

            # 遮罩只看 label > 0 的區域（只統計 foreground）
            mask = np.where(seg_data > 0, False, True)
            masked_data = np.ma.array(img_data, mask=mask)

            hist, _ = np.histogram(masked_data.compressed(), bins=2049, range=(-1024, 1024))
            histogram += hist

    # 統計 intensity 範圍與 mean/std
    min_intensity, max_intensity = percentile(histogram)
    min_intensity -= 1024
    max_intensity -= 1024

    intensity = np.arange(-1024, 1025)
    mean = np.average(intensity, weights=histogram)
    var = np.sum(((intensity - mean) ** 2) * histogram) / np.sum(histogram)
    std = np.sqrt(var)

    spacing = np.median(spacings, axis=0)
    shape = (np.median(shapes, axis=0) * spacing).astype(np.int32)

    stats = {
        "intensity": {
            "min": min_intensity,
            "max": max_intensity,
            "mean": mean,
            "std": std
        },
        "spacing": spacing.tolist(),
        "shape": shape.tolist()
    }
    return stats


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to training config file.")
    parser.add_argument("--datalist", "-l", type=str, default=None, help="Path to datalist.")
    parser.add_argument("--dataroot", "-r", type=str, default=None, help="Path prefix for data in datalist.")
    parser.add_argument("--data_split_keys", nargs="+", default=["training", "validation"])
    parser.add_argument("--img_key", type=str, default="image", help="Dictionary key for the image file.")
    parser.add_argument("--seg_key", type=str, default="label", help="Dictionary key for the segmentation label file.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Write analysis output as JSON file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("nnUNetDataAnalyzer")

    # Get correct settings from argument
    if args.dataroot is None and args.datalist is None:
        if args.config is None:
            raise ValueError("Must given a config file or a pair of dataroot and datalist.")

        loader = YAML()
        with open(args.config) as f:
            config = loader.load(f)

        dataroot = config["data"]["settings"]["data_root"]
        datalist = config["data"]["settings"]["data_list"]
    elif args.dataroot is not None and args.datalist is not None:
        if args.config is not None:
            logger.warning(
                f"Both 'dataroot' and 'datalist' are set, ignore the settings in the config file {args.config}"
            )
        dataroot = args.dataroot
        datalist = args.datalist
    else:
        raise ValueError(
            "The 'datalist' and 'dataroot' must be set at the same time, "
            "if you wish to use the training config, please leave these two arguments blank."
        )

    logger.info(f"Start to extract file information:")
    logger.info(f"Target dataroot: {dataroot}")
    logger.info(f"Target datalist: {datalist}")
    logger.info(f"Target split: {args.data_split_keys}")

    # Get the real datalist from dataroot & datalist
    data = resolve_datalist(
        dataroot,
        datalist,
        split_keys=args.data_split_keys,
        img_key=args.img_key,
        seg_key=args.seg_key
    )
    logger.info(f"Collected {len(data)} pairs of image and segmentation labels.")

    # Run analysis
    stats = analysis(data)
    logger.info("Analysis complete!")

    # Print in prettry format
    printer = PrettyPrinter(indent=4)
    printer.pprint(stats)

    # [Optional] Save result to file
    if args.output is not None:
        with open(args.output, "w") as f:
            json.dump(stats, f, indent=4)

