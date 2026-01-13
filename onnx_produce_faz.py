"""Export SAM2 model to ONNX for prompt-free prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sam2.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
from training.utils.train_utils import register_omegaconf_resolvers


def _ensure_omegaconf_resolvers() -> None:
    if not OmegaConf.has_resolver("times"):
        register_omegaconf_resolvers()


def _load_config(config_path: Path) -> Dict[str, Any]:
    _ensure_omegaconf_resolvers()
    config = OmegaConf.create(yaml.safe_load(config_path.read_text()))
    OmegaConf.resolve(config)
    return config


def _get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if "trainer" in config and "model" in config["trainer"]:
        return config["trainer"]["model"]
    if "model" in config:
        return config["model"]
    raise KeyError("Config must contain trainer.model or model section.")


def _build_model(model_cfg: Dict[str, Any]) -> torch.nn.Module:
    return instantiate(model_cfg, _recursive_=True)


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = state.get("model", state)
    model.load_state_dict(state_dict, strict=True)


class SAM2AutoOnnxWrapper(torch.nn.Module):
    def __init__(self, model: SAM2Base):
        super().__init__()
        self.model = model

    def _empty_sparse_embeddings(self, batch_size: int, device: torch.device) -> torch.Tensor:
        prompt_encoder = self.model.sam_prompt_encoder
        return torch.empty((batch_size, 0, prompt_encoder.embed_dim), device=device)

    def _no_mask_embeddings(self, batch_size: int, device: torch.device) -> torch.Tensor:
        prompt_encoder = self.model.sam_prompt_encoder
        image_embedding_size = prompt_encoder.image_embedding_size
        return prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            batch_size, -1, image_embedding_size[0], image_embedding_size[1]
        ).to(device)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        backbone_out = self.model.forward_image(image)
        _, vision_feats, _, feat_sizes = self.model._prepare_backbone_features(
            backbone_out
        )
        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        top_feat = vision_feats[-1]
        pix_feat = top_feat.permute(1, 2, 0).view(
            top_feat.size(1), top_feat.size(2), *feat_sizes[-1]
        )

        batch_size = image.size(0)
        device = image.device
        sparse_embeddings = self._empty_sparse_embeddings(batch_size, device)
        dense_embeddings = self._no_mask_embeddings(batch_size, device)

        prompt_encoder = self.model.sam_prompt_encoder
        multimask_output = self.model.multimask_output_in_sam
        (
            low_res_multimasks,
            ious,
            _,
            object_score_logits,
        ) = self.model.sam_mask_decoder(
            image_embeddings=pix_feat,
            image_pe=prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        if self.model.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0
            no_obj_score = low_res_multimasks.new_tensor(NO_OBJ_SCORE)
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None], low_res_multimasks, no_obj_score
            )

        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = torch.nn.functional.interpolate(
            low_res_multimasks,
            size=(self.model.image_size, self.model.image_size),
            mode="bilinear",
            align_corners=False,
        )

        if multimask_output:
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(
                low_res_multimasks.size(0), device=low_res_multimasks.device
            )
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
        else:
            low_res_masks = low_res_multimasks
            high_res_masks = high_res_multimasks

        return low_res_masks, high_res_masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SAM2 ONNX model (auto)")
    parser.add_argument(
        "--checkpoint",
        default="/home/wensheng/gjq_workspace/eyesam/exp_log_260110_area/checkpoints/best_dice_epoch_0358_first_0p875572_final_0p875572.pt",
        help="Path to model weights",
    )
    parser.add_argument(
        "--config",
        default="sam2/configs/sam2.1_hiera_tiny512_laser_faz.yaml",
        help="Path to SAM2 config",
    )
    parser.add_argument(
        "--output",
        default="onnx/area_358.onnx",
        help="Output ONNX path",
    )
    parser.add_argument("--device", default="cpu", help="Device for export")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    config_path = Path(args.config)
    ckpt_path = Path(args.checkpoint)
    output_path = Path(args.output)

    config = _load_config(config_path)
    model_cfg = _get_model_config(config)
    model = _build_model(model_cfg)
    _load_checkpoint(model, ckpt_path)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    wrapper = SAM2AutoOnnxWrapper(model).to(device)

    image_size = model.image_size
    dummy_image = torch.randn(1, 3, image_size, image_size, device=device)

    torch.onnx.export(
        wrapper,
        (dummy_image,),
        output_path.as_posix(),
        input_names=["image"],
        output_names=["low_res_masks", "high_res_masks"],
        dynamic_axes={
            "image": {0: "batch"},
            "low_res_masks": {0: "batch"},
            "high_res_masks": {0: "batch"},
        },
        opset_version=args.opset,
    )

    print(f"ONNX model saved to {output_path}")


if __name__ == "__main__":
    main()
