"""Interactive prediction script using a SAM2 checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_images(image_dir: Path) -> List[Path]:
    paths = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in SUPPORTED_SUFFIXES]
    if not paths:
        raise RuntimeError(f"No images found in {image_dir}")
    return paths


def generate_hex_centers(mask: np.ndarray, spacing: float = 24.0) -> List[Tuple[int, int]]:
    height, width = mask.shape
    centers: List[Tuple[int, int]] = []
    row_spacing = spacing * math.sqrt(3) / 2
    y = 0.0
    row = 0
    while y < height:
        x_offset = 0.0 if row % 2 == 0 else spacing / 2
        x = x_offset
        while x < width:
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= xi < width and 0 <= yi < height and mask[yi, xi]:
                centers.append((xi, yi))
            x += spacing
        y += row_spacing
        row += 1
    return centers


def draw_circles(image: Image.Image, mask: np.ndarray) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    radius = 8
    for x, y in generate_hex_centers(mask):
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            outline=(0, 0, 255),
            width=2,
        )
    return output


def resolve_checkpoint(path: str | None, script_dir: Path) -> Path:
    if path:
        return Path(path)
    candidates = sorted(script_dir.glob("*.pt"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError("No .pt checkpoint found. Please provide --checkpoint.")
    raise RuntimeError(
        "Multiple .pt checkpoints found. Please select one with --checkpoint."
    )


def run_interactive(
    predictor: SAM2ImagePredictor, image_path: Path, max_points: int
) -> bool:
    original = Image.open(image_path).convert("RGB")
    image_np = np.array(original)
    predictor.set_image(image_np)

    click_points: List[Tuple[float, float]] = []
    click_labels: List[int] = []
    history: List[Dict[str, np.ndarray]] = []
    prev_low_res: np.ndarray | None = None

    fig, ax = plt.subplots()
    ax.axis("off")
    image_artist = ax.imshow(original)

    should_continue = {"next": False, "quit": False}

    def on_click(event):
        nonlocal prev_low_res
        if event.inaxes != ax:
            return
        if event.button not in (1, 3):
            return
        label = 1 if event.button == 1 else 0
        click_points.append((event.xdata, event.ydata))
        click_labels.append(label)
        if len(click_points) > max_points:
            click_points.pop(0)
            click_labels.pop(0)
        point_coords = np.array(click_points, dtype=np.float32)
        point_labels = np.array(click_labels, dtype=np.int32)
        masks, _, low_res = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=prev_low_res,
            multimask_output=False,
            return_logits=False,
            normalize_coords=False,
        )
        prev_low_res = low_res
        mask = masks[0]
        history.append(
            {
                "points": point_coords.copy(),
                "labels": point_labels.copy(),
                "mask": mask.copy(),
            }
        )
        overlay = draw_circles(original, mask)
        image_artist.set_data(overlay)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in {"enter", "n"}:
            should_continue["next"] = True
            plt.close(fig)
        if event.key in {"escape", "q"}:
            should_continue["quit"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    return not should_continue["quit"]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="interactive predictor using SAM2 checkpoint")
    parser.add_argument(
        "--checkpoint",
        default=r"C:\work space\prp\predict_260107\best_dice_epoch_0238_first_0p768089_final_0p901605.pt",
        help="Path to the SAM2 .pt checkpoint (defaults to a single .pt in script dir).",
    )
    parser.add_argument(
        "--config",
        default=r"C:\work space\prp\predict_260107\sam2\configs\sam2.1_hiera_tiny512_laser.yaml",
        help="Path to SAM2 config yaml (defaults to sam2.1_hiera_tiny512_laser.yaml).",
    )
    parser.add_argument(
        "--image-dir",
        default=r"C:\work space\prp\predict_260107\demo",
        help="Directory containing images to predict.",
    )
    parser.add_argument("--device", default=None, help="Device to run inference on")
    parser.add_argument(
        "--max-points",
        type=int,
        default=8,
        help="Maximum number of user clicks to keep (oldest clicks are dropped).",
    )
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint(args.checkpoint, script_dir)
    config_path = Path(
        args.config
        if args.config
        else script_dir / "sam2" / "configs" / "sam2.1_hiera_tiny512_laser.yaml"
    )

    sam2_model = build_sam2(config_path.as_posix(), checkpoint_path.as_posix(), device=args.device)
    predictor = SAM2ImagePredictor(sam2_model)

    image_paths = load_images(Path(args.image_dir))
    for image_path in image_paths:
        print(f"Processing {image_path.name} (press 'n' to advance, 'q' to quit)")
        if not run_interactive(predictor, image_path, args.max_points):
            break


if __name__ == "__main__":
    main()
