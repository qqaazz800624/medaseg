import os
import json
import logging

from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import nibabel as nib
import typer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


class JSONDataLoader:
    def __init__(self,
                 data_root: str,
                 json_path: str):
        self.data_root = data_root
        self.json_path = json_path

    def _normalize(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.normpath(
            os.path.join(self.data_root, path)
        )

    def _is_valid_case(
        self,
        img: str,
        lbl: str
    ) -> bool:
        return True
        if not os.path.exists(img):
            logging.warning("Image missing: %s", img)
            return False
        if not os.path.exists(lbl):
            logging.warning("Label missing: %s", lbl)
            return False
        return True

    def load(self) -> Dict[str, List[Dict[str, str]]]:
        raw = json.load(open(self.json_path))
        splits: Dict[str, List[Dict[str, str]]] = {}
        for split, cases in raw.items():
            valid_cases: List[Dict[str, str]] = []
            for case in cases:
                img, lbl = case.get('image'), case.get('label')
                if not img or not lbl:
                    continue

                if isinstance(img, list):
                    img_path = [self._normalize(p) for p in img]
                else:
                    img_path = self._normalize(img)

                if isinstance(lbl, list):
                    lbl_path = [self._normalize(p) for p in lbl]
                else:
                    lbl_path = self._normalize(lbl)

                if self._is_valid_case(img_path, lbl_path):
                    valid_cases.append({
                        'image': img_path,
                        'label': lbl_path
                    })
            splits[split] = valid_cases
        return splits


class NiftiReader:
    def read(
        self,
        path: list[str] | str
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        if isinstance(path, str):
            img = nib.load(path)
            data = img.get_fdata(dtype=np.float32)
            zooms = img.header.get_zooms()
            if len(zooms) >= 3:
                spacing = (
                    float(zooms[0]),
                    float(zooms[1]),
                    float(zooms[2])
                )
            else:
                spacing = (1.0, 1.0, 1.0)
        else:
            img = [nib.load(p) for p in path]
            data = np.stack([i.get_fdata(dtype=np.float32) for i in img], axis=0)
            zooms = img[0].header.get_zooms()
            if len(zooms) >= 3:
                spacing = (
                    float(zooms[0]),
                    float(zooms[1]),
                    float(zooms[2])
                )
            else:
                spacing = (1.0, 1.0, 1.0)
        return data, spacing

class Cropper:
    @staticmethod
    def crop_to_bbox(
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray,
               Tuple[int, int, int, int, int, int]]:
        coords = np.argwhere(mask > 0)
        if not coords.size:
            return image, mask, (0, 0, 0, 0, 0, 0)
        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0)
        if image.ndim == mask.ndim + 1:
            slices = [slice(None),
                      slice(z0, z1 + 1),
                      slice(y0, y1 + 1),
                      slice(x0, x1 + 1)]
            cropped_img = image[tuple(slices)]
        else:
            cropped_img = image[
                z0:z1 + 1,
                y0:y1 + 1,
                x0:x1 + 1
            ]
        cropped_mask = mask[
            z0:z1 + 1,
            y0:y1 + 1,
            x0:x1 + 1
        ]
        return cropped_img, cropped_mask, (z0, z1, y0, y1, x0, x1)


class IntensityCollector:
    def __init__(self,
                 samples: int,
                 seed: int = 1234):
        self.samples = samples
        self.rng = np.random.RandomState(seed)

    def _sample(self, pixels: np.ndarray) -> np.ndarray:
        if pixels.size:
            need = self.samples
            avail = pixels.size
            replace = avail < need
            return self.rng.choice(pixels, need, replace=replace)
        return np.array([], dtype=pixels.dtype)

    def _stats(self, pixels: np.ndarray) -> Dict[str, float]:
        if pixels.size:
            p = np.percentile(pixels, [0.5, 50.0, 99.5])
            return {
                'percentile_00_5': float(p[0]),
                'median': float(p[1]),
                'percentile_99_5': float(p[2]),
                'mean': float(pixels.mean()),
                'min': float(pixels.min()),
                'max': float(pixels.max())
            }
        keys = ['mean', 'median', 'min', 'max',
                'percentile_00_5', 'percentile_99_5']
        return dict.fromkeys(keys, float('nan'))

    def collect_per_channel(
        self,
        imgs: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
        if imgs.ndim == 3:
            imgs = imgs[np.newaxis]
        fg = mask > 0
        intensities: List[np.ndarray] = []
        stats: List[Dict[str, float]] = []
        for ch in imgs:
            pix = ch[fg]
            intensities.append(self._sample(pix))
            stats.append(self._stats(pix))
        return intensities, stats


class CaseAnalyzer:
    def __init__(
        self,
        reader: NiftiReader,
        cropper: Cropper,
        collector: IntensityCollector
    ):
        self.reader = reader
        self.cropper = cropper
        self.collector = collector

    def _load(
        self,
        img_path: str,
        lbl_path: str
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
        img, sp = self.reader.read(img_path)
        seg, _ = self.reader.read(lbl_path)
        return img, seg, sp

    def _prepare(
        self,
        img: np.ndarray,
        seg: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if img.ndim == 3:
            img = img[np.newaxis]
        if seg.ndim == 4:
            seg = np.any(seg > 0, axis=0)
        mask = seg > 0
        return img, mask

    def analyze(
        self,
        img_path: str,
        lbl_path: str
    ) -> Optional[Dict]:
        try:
            img, seg, spacing = self._load(img_path, lbl_path)
            img, mask = self._prepare(img, seg)
            cropped_img, cropped_mask, bbox = (
                self.cropper.crop_to_bbox(img, mask)
            )
            intensities, stats = (
                self.collector.collect_per_channel(
                    cropped_img, cropped_mask
                )
            )
            before = np.prod(img.shape[1:])
            after = np.prod(cropped_img.shape[1:])
            rel = float(after / before) if before else float('nan')
            return {
                'shape_after_crop': tuple(cropped_img.shape[1:]),
                'spacing': spacing,
                'intensities_per_channel': intensities,
                'intensity_statistics': stats,
                'relative_size': rel,
                'bounding_box': bbox
            }
        except Exception as e:
            logging.error("Analysis failed %s: %s", img_path, e)
            return None


def _analyze_case_with_samples(
    img_path: str,
    lbl_path: str,
    samples_per_case: int
) -> Optional[Dict]:
    collector = IntensityCollector(samples_per_case)
    analyzer = CaseAnalyzer(
        reader=NiftiReader(),
        cropper=Cropper(),
        collector=collector
    )
    return analyzer.analyze(img_path, lbl_path)

class DatasetAnalyzer:
    def __init__(
        self,
        data_root: str,
        json_path: str,
        splits: List[str],
        strategy: str = 'per_case',
        samples: int = 10000,
        total_voxels: int = int(1e8),
        workers: int = 8
    ):
        self.loader = JSONDataLoader(data_root, json_path)
        self.splits = splits
        self.strategy = strategy
        self.samples = samples
        self.total_voxels = total_voxels
        self.workers = workers

    def _get_cases(self) -> List[Tuple[str, str]]:
        data = self.loader.load()
        cases: List[Tuple[str, str]] = []
        for s in self.splits:
            for c in data.get(s, []):
                cases.append((c['image'], c['label']))
        return cases

    def _compute_samples_per_case(
        self,
        n_cases: int
    ) -> int:
        if self.strategy == 'equal_dataset' and n_cases:
            return self.total_voxels // n_cases
        return self.samples

    def _run_analysis(
        self,
        cases: List[Tuple[str, str]],
        samples_per_case: int
    ) -> List[Dict]:
        if self.workers > 1:
            with ProcessPoolExecutor(
                max_workers=self.workers
            ) as executor:
                futures = [
                    executor.submit(
                        _analyze_case_with_samples,
                        img, lbl, samples_per_case
                    )
                    for img, lbl in cases
                ]
                return [f.result() for f in futures if f.result()]
        return [
            _analyze_case_with_samples(img, lbl, samples_per_case)
            for img, lbl in cases
        ]

    def _aggregate(
        self,
        results: List[Dict]
    ) -> Dict:
        rel = [r['relative_size'] for r in results]
        num_ch = (
            len(results[0]['intensities_per_channel'])
            if results else 0
        )
        stats: Dict[int, Dict[str, float]] = {}
        for ch in range(num_ch):
            pix = np.concatenate([
                r['intensities_per_channel'][ch]
                for r in results
            ])
            if pix.size:
                stats[ch] = {
                    'mean': float(pix.mean()),
                    'median': float(np.median(pix)),
                    'std': float(pix.std()),
                    'min': float(pix.min()),
                    'max': float(pix.max()),
                    'percentile_00_5': float(
                        np.percentile(pix, 0.5)
                    ),
                    'percentile_99_5': float(
                        np.percentile(pix, 99.5)
                    )
                }
            else:
                stats[ch] = dict.fromkeys([
                    'mean', 'median', 'std', 'min',
                    'max', 'percentile_00_5',
                    'percentile_99_5'
                ], 0.0)
        shapes = [r['shape_after_crop'] for r in results]
        spacings = [r['spacing'] for r in results]
        return {
            'spacings': spacings,
            'shapes_after_crop': shapes,
            'foreground_intensity_properties_per_channel': stats,
            'median_relative_size_after_cropping':
                float(np.median(rel)) if rel else float('nan')
        }

    def analyze(self) -> Dict:
        cases = self._get_cases()
        per_case = self._compute_samples_per_case(
            len(cases)
        )
        logging.info(
            "Strategy %s: %d samples per case",
            self.strategy, per_case
        )
        results = self._run_analysis(cases, per_case)
        return self._aggregate(results)

app = typer.Typer(
    help="""
    Analyze a medical image dataset and compute a fingerprint for planning.
    This includes cropping to foreground, extracting intensity statistics,
    and computing relative volume after cropping. Supports multi-process
    execution and two sampling strategies.
    """
)

@app.command()
def main(
    data_root: str = typer.Argument(
        ..., help="Root directory of the dataset containing image/label files"
    ),
    data_list: str = typer.Argument(
        ..., help="Path to JSON file describing image/label case pairs"
    ),
    data_list_keys: List[str] = typer.Option(
        ["training"],
        help="List of dataset splits to analyze (e.g., training, validation, test)"
    ),
    sampling_strategy: str = typer.Option(
        "per_case",
        help="Sampling strategy for foreground voxel selection. "
             "Choose 'per_case' to sample a fixed number of voxels per case, "
             "or 'equal_dataset' to distribute a fixed total number of voxels "
             "across all cases."
    ),
    total_voxels_target: int = typer.Option(
        int(1e8),
        help="Total number of foreground voxels to sample across the dataset. "
             "Only used if sampling_strategy='equal_dataset'."
    ),
    num_processes: int = typer.Option(
        8,
        help="Number of parallel processes to use for analysis."
    ),
    num_samples: int = typer.Option(
        10000,
        help="Number of foreground samples per case. Only used if sampling_strategy='per_case'."
    ),
    output_file: Optional[str] = typer.Option(
        None,
        help="Path to save the output fingerprint JSON file."
    )
):
    analyzer = DatasetAnalyzer(
        data_root,
        data_list,
        data_list_keys,
        sampling_strategy,
        num_samples,
        total_voxels_target,
        num_processes
    )
    fingerprint = analyzer.analyze()
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(fingerprint, f, indent=4)
        logging.info(f"Saved fingerprint to {output_file}")
    typer.secho("Analysis completed", fg=typer.colors.GREEN)

    # Print the fingerprint to the console
    typer.echo(f"Median spacing: {np.median(fingerprint['spacings'], axis=0)}")
    typer.echo(f"Median shape after crop: {np.median(fingerprint['shapes_after_crop'], axis=0)}")
    typer.echo(f"Intensity statistics per channel:")
    for ch in fingerprint['foreground_intensity_properties_per_channel']:
        stats = fingerprint['foreground_intensity_properties_per_channel'][ch]
        typer.echo(f"  Channel {ch}:")
        for key, value in stats.items():
            typer.echo(f"    {key}: {value}")
    typer.echo(f"Median relative size after cropping: {fingerprint['median_relative_size_after_cropping']}")

if __name__ == '__main__':
    typer.run(main)

