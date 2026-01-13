"""
Utility script to convert raw Retina project data into NPZ files compatible
with ``NPZRawDataset`` in ``training.dataset.vos_raw_dataset``.

The expected input directory structure is:
    root_dir/
        train/
            case_0001/
                image.png
                gt_0.png
                gt_1.png
                ...
        val/
            case_0009/
                image.png
                gt_0.png
                ...

For each ``gt_*.png`` mask, this script creates one ``image_XXX.npz`` file that
contains the corresponding ``image.png`` and mask in the ``imgs`` and ``gts``
keys. Files originating from the ``train`` split include ``train: True`` and
``val: False`` flags, while ``val`` split files set ``val: True`` and
``train: False``.

Usage example:
    python data_produce.py \
        --root /data/Retina_Project \
        --train-output /data/Retina_Project/train_npz \
        --val-output /data/Retina_Project/val_npz \
        --train-start 1 --val-start 349
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)


def load_image_color(image_path: Path) -> np.ndarray:
    """Load an image as an RGB uint8 numpy array in CHW layout."""
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        image_np = np.array(img)

    image_np = image_np.astype(np.uint8)
    return np.transpose(image_np, (2, 0, 1))


def load_mask_grayscale(mask_path: Path) -> np.ndarray:
    """Load a mask as a single-channel uint8 numpy array."""
    with Image.open(mask_path) as img:
        if img.mode not in ("1", "L", "I;16", "P"):
            img = img.convert("L")
        mask_np = np.array(img)

    if mask_np.ndim == 3:
        # If conversion still yields multiple channels (e.g., palette), keep first.
        mask_np = mask_np[:, :, 0]

    return mask_np.astype(np.uint8)

    return mask_np.astype(np.uint8)

def validate_pair(image_array: np.ndarray, mask_array: np.ndarray, case_name: str, mask_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure image and mask shapes match by resizing the mask if needed."""
    image_shape = image_array.shape[1:3]
    mask_shape = mask_array.shape[:2]
    if image_shape != mask_shape:
        logging.warning(
            "Shape mismatch for %s/%s: image %s vs mask %s. Resizing mask to match image.",
            case_name,
            mask_name,
            image_shape,
            mask_shape,
        )
        mask_image = Image.fromarray(mask_array)
        mask_image = mask_image.resize((image_shape[1], image_shape[0]), resample=Image.NEAREST)
        mask_array = np.array(mask_image, dtype=mask_array.dtype)
    return image_array, mask_array


def process_split(
    split: str,
    input_dir: Path,
    output_dir: Path,
    start_index: int,
) -> int:
    """Convert one split (train/val) into NPZ files.

    Returns the next available index after processing.
    """
    if not input_dir.exists():
        logging.error("Input directory %s does not exist.", input_dir)
        return start_index

    output_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith("case_")])
    if not case_dirs:
        logging.warning("No case folders found in %s", input_dir)

    counter = start_index
    for case_dir in case_dirs:
        image_path = case_dir / "image.png"
        if not image_path.exists():
            logging.warning("Missing image.png in %s, skipping.", case_dir)
            continue

        try:
            image_array = load_image_color(image_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.error("Failed to load image %s: %s", image_path, exc)
            continue

        gt_files = sorted(case_dir.glob("gt_*.png"))
        if not gt_files:
            logging.warning("No gt_*.png files found in %s", case_dir)
            continue

        for gt_path in gt_files:
            try:
                mask_array = load_mask_grayscale(gt_path)
            except Exception as exc:  # pragma: no cover - defensive logging
                logging.error("Failed to load mask %s: %s", gt_path, exc)
                continue

            image_array, mask_array = validate_pair(image_array, mask_array, case_dir.name, gt_path.name)

            npz_name = f"image_{counter:03d}.npz"
            npz_path = output_dir / npz_name

            np.savez_compressed(
                npz_path,
                imgs=image_array[np.newaxis, ...],
                gts=mask_array[np.newaxis, ...],
                train=split == "train",
                val=split == "val",
            )
            logging.info("Saved %s", npz_path)
            counter += 1

    return counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Retina project PNGs to NPZ format for NPZRawDataset.")
    parser.add_argument("--root", type=Path, default=Path("/data/Retina_Project"), help="Root directory containing train/ and val/ folders.")
    parser.add_argument("--train-output", type=Path, default=None, help="Output directory for training NPZ files.")
    parser.add_argument("--val-output", type=Path, default=None, help="Output directory for validation NPZ files.")
    parser.add_argument("--train-start", type=int, default=1, help="Starting index for training NPZ filenames.")
    parser.add_argument("--val-start", type=int, default=349, help="Starting index for validation NPZ filenames.")
    return parser.parse_args()


def main():
    args = parse_args()

    train_output = args.train_output or args.root / "train_npz"
    val_output = args.val_output or args.root / "val_npz"

    train_dir = args.root / "train"
    val_dir = args.root / "val"

    next_train_index = process_split("train", train_dir, train_output, args.train_start)
    process_split("val", val_dir, val_output, args.val_start)

    logging.info(
        "Processing complete. Next available indices - train: %d, val: %d",
        next_train_index,
        args.val_start + (next_train_index - args.train_start),
    )


if __name__ == "__main__":
    main()
