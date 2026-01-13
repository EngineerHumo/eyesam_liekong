"""Interactive prediction script using an exported ONNX model."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_images(image_dir: Path) -> List[Path]:
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in supported]
    if not paths:
        raise RuntimeError(f"No images found in {image_dir}")
    return paths


def preprocess_image(image: Image.Image, image_size: int) -> np.ndarray:
    resized = image.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    arr = (arr - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


def resize_mask(mask_logits: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(mask_logits)
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy()


class ONNXPredictor:
    def __init__(
        self,
        model_path: Path,
        device: str | None = None,
    ) -> None:
        available = ort.get_available_providers()
        if device is None:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available
                else ["CPUExecutionProvider"]
            )
        elif device.lower() == "cuda":
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available
                else ["CPUExecutionProvider"]
            )
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path.as_posix(), providers=providers)
        inputs = {input_info.name: input_info for input_info in self.session.get_inputs()}
        image_shape = inputs["image"].shape
        self.image_size = int(image_shape[2])

    def predict(
        self,
        image: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        prev_low_res: np.ndarray | None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if prev_low_res is None:
            mask_inputs = np.zeros(
                (
                    image.shape[0],
                    1,
                    self.image_size // 4,
                    self.image_size // 4,
                ),
                dtype=np.float32,
            )
            has_mask = np.zeros((image.shape[0], 1, 1, 1), dtype=np.float32)
        else:
            mask_inputs = prev_low_res.astype(np.float32)
            has_mask = np.ones((image.shape[0], 1, 1, 1), dtype=np.float32)
        outputs = self.session.run(
            None,
            {
                "image": image.astype(np.float32),
                "point_coords": point_coords.astype(np.float32),
                "point_labels": point_labels.astype(np.int64),
                "mask_inputs": mask_inputs,
                "has_mask": has_mask,
            },
        )
        low_res_masks, high_res_masks = outputs
        return low_res_masks, high_res_masks

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


def run_interactive(predictor: ONNXPredictor, image_path: Path) -> bool:
    original = Image.open(image_path).convert("RGB")
    width, height = original.size
    image_array = preprocess_image(original, predictor.image_size)

    click_points: List[Tuple[float, float]] = []
    click_labels: List[int] = []
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
        points = np.array(click_points, dtype=np.float32)
        points[:, 0] = points[:, 0] / width * predictor.image_size
        points[:, 1] = points[:, 1] / height * predictor.image_size
        point_coords = points[None, :, :].astype(np.float32)
        point_labels = np.array(click_labels, dtype=np.int64)[None, :]
        low_res, high_res = predictor.predict(
            image_array, point_coords, point_labels, prev_low_res
        )
        prev_low_res = low_res
        mask_logits = resize_mask(high_res, (height, width))
        mask = mask_logits > 0.0
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
    parser = argparse.ArgumentParser(description="interactive predictor")
    parser.add_argument("--onnx-model", default=r"C:\work space\prp\predict_260107\sam2_interactive.onnx", help="Path to ONNX model")
    parser.add_argument("--image-dir", default=r"C:\work space\prp\predict_260107\demo", help="Directory containing images")
    parser.add_argument("--device", default=None, help="Device to run inference on")
    args = parser.parse_args()

    predictor = ONNXPredictor(
        Path(args.onnx_model), device=args.device
    )

    image_paths = load_images(Path(args.image_dir))
    for image_path in image_paths:
        print(f"Processing {image_path.name} (press 'n' to advance, 'q' to quit)")
        if not run_interactive(predictor, image_path):
            break


if __name__ == "__main__":
    main()
