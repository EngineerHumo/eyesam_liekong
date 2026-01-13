# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.utils.misc import mask_to_box


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """
    Select up to `max_cond_frame_num` conditioning frames from `cond_frame_outputs`
    that are temporally closest to the current frame at `frame_idx`. Here, we take
    - a) the closest conditioning frame before `frame_idx` (if any);
    - b) the closest conditioning frame after `frame_idx` (if any);
    - c) any other temporally closest conditioning frames until reaching a total
         of `max_cond_frame_num` conditioning frames.

    Outputs:
    - selected_outputs: selected items (keys & values) from `cond_frame_outputs`.
    - unselected_outputs: items (keys & values) not selected in `cond_frame_outputs`.
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}

        # the closest conditioning frame before `frame_idx` (if any)
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # the closest conditioning frame after `frame_idx` (if any)
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # add other temporally closest conditioning frames until reaching a total
        # of `max_cond_frame_num` conditioning frames.
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {
            t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
        }

    return selected_outputs, unselected_outputs


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """
    Get 1D sine positional embedding as in the original Transformer paper.
    """
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DropPath(nn.Module):
    # adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def sample_box_points(
    masks: torch.Tensor,
    noise: float = 0.1,  # SAM default
    noise_bound: int = 20,  # SAM default
    top_left_label: int = 2,
    bottom_right_label: int = 3,
) -> Tuple[np.array, np.array]:
    """
    Sample a noised version of the top left and bottom right corners of a given `bbox`

    Inputs:
    - masks: [B, 1, H,W] boxes, dtype=torch.Tensor
    - noise: noise as a fraction of box width and height, dtype=float
    - noise_bound: maximum amount of noise (in pure pixesl), dtype=int

    Returns:
    - box_coords: [B, num_pt, 2], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.float
    - box_labels: [B, num_pt], label 2 is reserverd for top left and 3 for bottom right corners, dtype=torch.int32
    """
    device = masks.device
    box_coords = mask_to_box(masks)
    B, _, H, W = masks.shape
    box_labels = torch.tensor(
        [top_left_label, bottom_right_label], dtype=torch.int, device=device
    ).repeat(B)
    if noise > 0.0:
        if not isinstance(noise_bound, torch.Tensor):
            noise_bound = torch.tensor(noise_bound, device=device)
        bbox_w = box_coords[..., 2] - box_coords[..., 0]
        bbox_h = box_coords[..., 3] - box_coords[..., 1]
        max_dx = torch.min(bbox_w * noise, noise_bound)
        max_dy = torch.min(bbox_h * noise, noise_bound)
        box_noise = 2 * torch.rand(B, 1, 4, device=device) - 1
        box_noise = box_noise * torch.stack((max_dx, max_dy, max_dx, max_dy), dim=-1)

        box_coords = box_coords + box_noise
        img_bounds = (
            torch.tensor([W, H, W, H], device=device) - 1
        )  # uncentered pixel coords
        box_coords.clamp_(torch.zeros_like(img_bounds), img_bounds)  # In place clamping

    box_coords = box_coords.reshape(-1, 2, 2)  # always 2 points
    box_labels = box_labels.reshape(-1, 2)
    return box_coords, box_labels


def sample_random_points_from_errors(gt_masks, pred_masks, num_pt=1):
    """
    Sample `num_pt` random points (along with their labels) independently from the error regions.

    Inputs:
    - gt_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool
    - pred_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool or None
    - num_pt: int, number of points to sample independently for each of the B error maps

    Outputs:
    - points: [B, num_pt, 2], dtype=torch.float, contains (x, y) coordinates of each sampled point
    - labels: [B, num_pt], dtype=torch.int32, where 1 means positive clicks and 0 means
      negative clicks
    """
    if pred_masks is None:  # if pred_masks is not provided, treat it as empty
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape
    assert num_pt >= 0

    B, _, H_im, W_im = gt_masks.shape
    device = gt_masks.device

    # false positive region, a new point sampled in this region should have
    # negative label to correct the FP error
    fp_masks = ~gt_masks & pred_masks
    # false negative region, a new point sampled in this region should have
    # positive label to correct the FN error
    fn_masks = gt_masks & ~pred_masks
    # whether the prediction completely match the ground-truth on each mask
    all_correct = torch.all((gt_masks == pred_masks).flatten(2), dim=2)
    all_correct = all_correct[..., None, None]

    # channel 0 is FP map, while channel 1 is FN map
    pts_noise = torch.rand(B, num_pt, H_im, W_im, 2, device=device)
    # sample a negative new click from FP region or a positive new click
    # from FN region, depend on where the maximum falls,
    # and in case the predictions are all correct (no FP or FN), we just
    # sample a negative click from the background region
    pts_noise[..., 0] *= fp_masks | (all_correct & ~gt_masks)
    pts_noise[..., 1] *= fn_masks
    pts_idx = pts_noise.flatten(2).argmax(dim=2)
    labels = (pts_idx % 2).to(torch.int32)
    pts_idx = pts_idx // 2
    pts_x = pts_idx % W_im
    pts_y = pts_idx // W_im
    points = torch.stack([pts_x, pts_y], dim=2).to(torch.float)
    return points, labels


def sample_one_point_from_error_center(gt_masks, pred_masks, padding=True):
    """
    Sample 1 random point (along with its label) from the center of each error region,
    that is, the point with the largest distance to the boundary of each error region.
    This is the RITM sampling method from https://github.com/saic-vul/ritm_interactive_segmentation/blob/master/isegm/inference/clicker.py

    Inputs:
    - gt_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool
    - pred_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool or None
    - padding: if True, pad with boundary of 1 px for distance transform

    Outputs:
    - points: [B, 1, 2], dtype=torch.float, contains (x, y) coordinates of each sampled point
    - labels: [B, 1], dtype=torch.int32, where 1 means positive clicks and 0 means negative clicks
    """
    import cv2

    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape

    B, _, _, W_im = gt_masks.shape
    device = gt_masks.device

    # false positive region, a new point sampled in this region should have
    # negative label to correct the FP error
    fp_masks = ~gt_masks & pred_masks
    # false negative region, a new point sampled in this region should have
    # positive label to correct the FN error
    fn_masks = gt_masks & ~pred_masks

    fp_masks = fp_masks.cpu().numpy()
    fn_masks = fn_masks.cpu().numpy()
    points = torch.zeros(B, 1, 2, dtype=torch.float)
    labels = torch.ones(B, 1, dtype=torch.int32)
    for b in range(B):
        fn_mask = fn_masks[b, 0]
        fp_mask = fp_masks[b, 0]
        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), "constant")
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), "constant")
        # compute the distance of each point in FN/FP region to its boundary
        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)
        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        # take the point in FN/FP region with the largest distance to its boundary
        fn_mask_dt_flat = fn_mask_dt.reshape(-1)
        fp_mask_dt_flat = fp_mask_dt.reshape(-1)
        fn_argmax = np.argmax(fn_mask_dt_flat)
        fp_argmax = np.argmax(fp_mask_dt_flat)
        is_positive = fn_mask_dt_flat[fn_argmax] > fp_mask_dt_flat[fp_argmax]
        pt_idx = fn_argmax if is_positive else fp_argmax
        points[b, 0, 0] = pt_idx % W_im  # x
        points[b, 0, 1] = pt_idx // W_im  # y
        labels[b, 0] = int(is_positive)

    points = points.to(device)
    labels = labels.to(device)
    return points, labels


def sample_center_biased_points_from_mask(
    gt_masks, p_center=0.7, use_area_norm=True
):
    if gt_masks.dtype != torch.bool:
        gt_masks = gt_masks > 0
    assert gt_masks.dim() == 4 and gt_masks.size(1) == 1
    B, _, H_im, W_im = gt_masks.shape
    device = gt_masks.device
    points = torch.zeros(B, 1, 2, dtype=torch.float32, device=device)
    labels = torch.ones(B, 1, dtype=torch.int32, device=device)

    import cv2

    for b in range(B):
        mask = gt_masks[b, 0]
        if mask.sum() == 0:
            points[b, 0, 0] = torch.randint(0, W_im, (1,), device=device)
            points[b, 0, 1] = torch.randint(0, H_im, (1,), device=device)
            labels[b, 0] = 0
            continue
        if torch.rand(1, device=device).item() < p_center:
            mask_np = mask.cpu().numpy().astype(np.uint8)
            dist = cv2.distanceTransform(mask_np, cv2.DIST_L2, 0).astype(np.float64)
            dist_flat = dist.reshape(-1)
            if use_area_norm and dist_flat.sum() > 0:
                probs = dist_flat / dist_flat.sum()
                idx = np.random.choice(dist_flat.size, p=probs)
            else:
                ys, xs = np.where(mask_np > 0)
                sel = np.random.randint(len(xs))
                idx = ys[sel] * W_im + xs[sel]
            points[b, 0, 0] = idx % W_im
            points[b, 0, 1] = idx // W_im
        else:
            ys, xs = torch.nonzero(mask, as_tuple=True)
            sel = torch.randint(0, ys.numel(), (1,), device=device)
            points[b, 0, 0] = xs[sel]
            points[b, 0, 1] = ys[sel]
    return points, labels


def sample_largest_error_region_point(
    gt_masks,
    pred_masks,
    p_largest=0.8,
    largest_region_prefer="largest",
):
    if pred_masks is None:
        return sample_random_points_from_errors(gt_masks, pred_masks)
    if gt_masks.dtype != torch.bool:
        gt_masks = gt_masks > 0
    if pred_masks.dtype != torch.bool:
        pred_masks = pred_masks > 0
    assert gt_masks.dim() == 4 and gt_masks.size(1) == 1
    B, _, H_im, W_im = gt_masks.shape
    device = gt_masks.device
    points = torch.zeros(B, 1, 2, dtype=torch.float32, device=device)
    labels = torch.zeros(B, 1, dtype=torch.int32, device=device)

    import cv2

    def _sample_random_point_from_other_errors(
        fp_mask_np: np.ndarray,
        fn_mask_np: np.ndarray,
        chosen_mask: Optional[np.ndarray],
    ):
        if chosen_mask is None:
            other_fp = fp_mask_np
            other_fn = fn_mask_np
        else:
            other_fp = fp_mask_np & ~chosen_mask
            other_fn = fn_mask_np & ~chosen_mask
        other_mask = other_fp | other_fn
        ys, xs = np.where(other_mask > 0)
        if ys.size == 0:
            return None
        sel = np.random.randint(ys.size)
        y = ys[sel]
        x = xs[sel]
        label = 0 if other_fp[y, x] > 0 else 1
        return x, y, label

    for b in range(B):
        fp_mask = (~gt_masks[b, 0] & pred_masks[b, 0]).cpu().numpy().astype(np.uint8)
        fn_mask = (gt_masks[b, 0] & ~pred_masks[b, 0]).cpu().numpy().astype(np.uint8)

        def largest_component(mask_np):
            if mask_np.sum() == 0:
                return None, 0
            _, comp_labels = cv2.connectedComponents(mask_np, connectivity=8)
            areas = np.bincount(comp_labels.reshape(-1))
            areas[0] = 0
            comp_id = areas.argmax()
            comp_mask = (comp_labels == comp_id).astype(np.uint8)
            return comp_mask, areas[comp_id]

        fn_comp, fn_area = largest_component(fn_mask)
        fp_comp, fp_area = largest_component(fp_mask)

        chosen_mask = None
        chosen_label = 1
        if largest_region_prefer == "fn":
            chosen_mask = fn_comp if fn_comp is not None else fp_comp
            chosen_label = 1 if fn_comp is not None else 0
        elif largest_region_prefer == "fp":
            chosen_mask = fp_comp if fp_comp is not None else fn_comp
            chosen_label = 0 if fp_comp is not None else 1
        else:
            if fn_area >= fp_area:
                chosen_mask = fn_comp if fn_comp is not None else fp_comp
                chosen_label = 1 if fn_comp is not None else 0
            else:
                chosen_mask = fp_comp if fp_comp is not None else fn_comp
                chosen_label = 0 if fp_comp is not None else 1

        if chosen_mask is None or chosen_mask.sum() == 0:
            rand_points, rand_labels = sample_random_points_from_errors(
                gt_masks[b : b + 1], pred_masks[b : b + 1]
            )
            points[b] = rand_points[0]
            labels[b] = rand_labels[0]
            continue

        if torch.rand(1, device=device).item() >= p_largest:
            other_sample = _sample_random_point_from_other_errors(
                fp_mask, fn_mask, chosen_mask
            )
            if other_sample is None:
                rand_points, rand_labels = sample_random_points_from_errors(
                    gt_masks[b : b + 1], pred_masks[b : b + 1]
                )
                points[b] = rand_points[0]
                labels[b] = rand_labels[0]
            else:
                x, y, label = other_sample
                points[b, 0, 0] = x
                points[b, 0, 1] = y
                labels[b, 0] = label
            continue

        dist = cv2.distanceTransform(chosen_mask, cv2.DIST_L2, 0)
        max_dist = float(dist.max())
        idx = int(dist.reshape(-1).argmax())
        center_x = idx % W_im
        center_y = idx // W_im
        if max_dist <= 0:
            points[b, 0, 0] = center_x
            points[b, 0, 1] = center_y
            labels[b, 0] = chosen_label
            continue
        radius = max_dist
        x0 = max(int(center_x - radius), 0)
        x1 = min(int(center_x + radius) + 1, W_im)
        y0 = max(int(center_y - radius), 0)
        y1 = min(int(center_y + radius) + 1, H_im)
        ys, xs = np.ogrid[y0:y1, x0:x1]
        circle_mask = (xs - center_x) ** 2 + (ys - center_y) ** 2 <= radius**2
        region_mask = chosen_mask[y0:y1, x0:x1] > 0
        valid_mask = circle_mask & region_mask
        valid_ys, valid_xs = np.where(valid_mask)
        if valid_ys.size == 0:
            points[b, 0, 0] = center_x
            points[b, 0, 1] = center_y
            labels[b, 0] = chosen_label
            continue
        sel = np.random.randint(valid_ys.size)
        points[b, 0, 0] = valid_xs[sel] + x0
        points[b, 0, 1] = valid_ys[sel] + y0
        labels[b, 0] = chosen_label

    return points, labels


def get_next_point(
    gt_masks,
    pred_masks,
    method,
    p_center=0.7,
    p_largest_region=0.8,
    largest_region_prefer="largest",
    use_area_norm=True,
):
    if method == "uniform":
        return sample_random_points_from_errors(gt_masks, pred_masks)
    elif method == "center":
        return sample_one_point_from_error_center(gt_masks, pred_masks)
    elif method == "center_biased_uniform":
        return sample_center_biased_points_from_mask(
            gt_masks, p_center=p_center, use_area_norm=use_area_norm
        )
    elif method == "largest_error_center":
        return sample_largest_error_region_point(
            gt_masks,
            pred_masks,
            p_largest=p_largest_region,
            largest_region_prefer=largest_region_prefer,
        )
    else:
        raise ValueError(f"unknown sampling method {method}")
