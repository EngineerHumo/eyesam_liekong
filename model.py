"""Standalone SAM2 model builder and predictor helpers."""

from __future__ import annotations

import inspect
from typing import Dict, Any

import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf
from training.utils.train_utils import register_omegaconf_resolvers


def _ensure_omegaconf_resolvers() -> None:
    if not OmegaConf.has_resolver("times"):
        register_omegaconf_resolvers()

_CONFIG_YAML = "# @package _global_\n\nscratch:\n  resolution: 512\n  train_video_batch_size: 2 # increase batch size based on your computing\n  num_train_workers: 15\n  num_frames: 1\n  max_num_objects: 3\n  base_lr: 5.0e-5\n  vision_lr: 3.0e-05\n  phases_per_epoch: 1\n  num_epochs: 300\n\ndataset:\n  # PATHS to Dataset\n  folder: /home/wensheng/gjq_workspace/eyesam/data/Retina_Project  # Root path containing train/ and val/ splits\n  multiplier: 1\n\n# Video transforms\nvos:\n  train_transforms:\n    - _target_: training.dataset.transforms.ComposeAPI\n      transforms:\n        - _target_: training.dataset.transforms.RandomHorizontalFlip\n          consistent_transform: True\n        - _target_: training.dataset.transforms.RandomAffine\n          degrees: 25\n          shear: 20\n          image_interpolation: bilinear\n          consistent_transform: True\n        - _target_: training.dataset.transforms.RandomResizeAPI\n          sizes: ${scratch.resolution}\n          square: true\n          consistent_transform: True\n        - _target_: training.dataset.transforms.ColorJitter\n          consistent_transform: True\n          brightness: 0.1\n          contrast: 0.03\n          saturation: 0.03\n          hue: null\n        - _target_: training.dataset.transforms.RandomGrayscale\n          p: 0.05\n          consistent_transform: True\n        - _target_: training.dataset.transforms.ColorJitter\n          consistent_transform: False\n          brightness: 0.1\n          contrast: 0.05\n          saturation: 0.05\n          hue: null\n        - _target_: training.dataset.transforms.ToTensorAPI\n        - _target_: training.dataset.transforms.NormalizeAPI\n          mean: [0.485, 0.456, 0.406]\n          std: [0.229, 0.224, 0.225]\n  val_transforms:\n    - _target_: training.dataset.transforms.ComposeAPI\n      transforms:\n        - _target_: training.dataset.transforms.RandomResizeAPI\n          sizes: ${scratch.resolution}\n          square: true\n          consistent_transform: True\n        - _target_: training.dataset.transforms.ToTensorAPI\n        - _target_: training.dataset.transforms.NormalizeAPI\n          mean: [0.485, 0.456, 0.406]\n          std: [0.229, 0.224, 0.225]\n\n\ntrainer:\n  _target_: training.trainer.Trainer\n  mode: train\n  max_epochs: ${times:${scratch.num_epochs},${scratch.phases_per_epoch}}\n  accelerator: cuda\n  seed_value: 123\n\n  model:\n    _target_: training.model.sam2.SAM2Train\n    image_encoder:\n      _target_: sam2.modeling.backbones.image_encoder.ImageEncoder\n      scalp: 1\n      trunk:\n        _target_: sam2.modeling.backbones.hieradet.Hiera\n        embed_dim: 96\n        num_heads: 1\n        stages: [1, 2, 7, 2]\n        global_att_blocks: [5, 7, 9]\n        window_pos_embed_bkg_spatial_size: [7, 7]\n      neck:\n        _target_: sam2.modeling.backbones.image_encoder.FpnNeck\n        position_encoding:\n          _target_: sam2.modeling.position_encoding.PositionEmbeddingSine\n          num_pos_feats: 256\n          normalize: true\n          scale: null\n          temperature: 10000\n        d_model: 256\n        backbone_channel_list: [768, 384, 192, 96]\n        fpn_top_down_levels: [2, 3]  # output level 0 and 1 directly use the backbone features\n        fpn_interp_model: nearest\n\n    memory_attention:\n      _target_: sam2.modeling.memory_attention.MemoryAttention\n      d_model: 256\n      pos_enc_at_input: true\n      layer:\n        _target_: sam2.modeling.memory_attention.MemoryAttentionLayer\n        activation: relu\n        dim_feedforward: 2048\n        dropout: 0.1\n        pos_enc_at_attn: false\n        self_attention:\n          _target_: sam2.modeling.sam.transformer.RoPEAttention\n          rope_theta: 10000.0\n          feat_sizes: [32, 32]\n          embedding_dim: 256\n          num_heads: 1\n          downsample_rate: 1\n          dropout: 0.1\n        d_model: 256\n        pos_enc_at_cross_attn_keys: true\n        pos_enc_at_cross_attn_queries: false\n        cross_attention:\n          _target_: sam2.modeling.sam.transformer.RoPEAttention\n          rope_theta: 10000.0\n          feat_sizes: [32, 32]\n          rope_k_repeat: True\n          embedding_dim: 256\n          num_heads: 1\n          downsample_rate: 1\n          dropout: 0.1\n          kv_in_dim: 64\n      num_layers: 4\n\n    memory_encoder:\n        _target_: sam2.modeling.memory_encoder.MemoryEncoder\n        out_dim: 64\n        position_encoding:\n          _target_: sam2.modeling.position_encoding.PositionEmbeddingSine\n          num_pos_feats: 64\n          normalize: true\n          scale: null\n          temperature: 10000\n        mask_downsampler:\n          _target_: sam2.modeling.memory_encoder.MaskDownSampler\n          kernel_size: 3\n          stride: 2\n          padding: 1\n        fuser:\n          _target_: sam2.modeling.memory_encoder.Fuser\n          layer:\n            _target_: sam2.modeling.memory_encoder.CXBlock\n            dim: 256\n            kernel_size: 7\n            padding: 3\n            layer_scale_init_value: 1.0e-6\n            use_dwconv: True  # depth-wise convs\n          num_layers: 2\n\n    num_maskmem: 7\n    image_size: ${scratch.resolution}\n    # apply scaled sigmoid on mask logits for memory encoder, and directly feed input mask as output mask\n    # SAM decoder\n    sigmoid_scale_for_mem_enc: 20.0\n    sigmoid_bias_for_mem_enc: -10.0\n    use_mask_input_as_output_without_sam: true\n    # Memory\n    directly_add_no_mem_embed: true\n    no_obj_embed_spatial: true\n    # use high-resolution feature map in the SAM mask decoder\n    use_high_res_features_in_sam: true\n    # output 3 masks on the first click on initial conditioning frames\n    multimask_output_in_sam: true\n    # SAM heads\n    iou_prediction_use_sigmoid: True\n    # cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder\n    use_obj_ptrs_in_encoder: true\n    add_tpos_enc_to_obj_ptrs: true\n    proj_tpos_enc_in_obj_ptrs: true\n    use_signed_tpos_enc_to_obj_ptrs: true\n    only_obj_ptrs_in_the_past_for_eval: true\n    # object occlusion prediction\n    pred_obj_scores: true\n    pred_obj_scores_mlp: true\n    fixed_no_obj_ptr: true\n    # multimask tracking settings\n    multimask_output_for_tracking: true\n    use_multimask_token_for_obj_ptr: true\n    multimask_min_pt_num: 0\n    multimask_max_pt_num: 1\n    use_mlp_for_obj_ptr_proj: true\n    # Compilation flag\n    # compile_image_encoder: False\n\n    ####### Training specific params #######\n    # box/point input and corrections\n    prob_to_use_pt_input_for_train: 1.0\n    prob_to_use_pt_input_for_eval: 1.0\n    prob_to_use_box_input_for_train: 0.0\n    prob_to_use_box_input_for_eval: 0.0\n    prob_to_sample_from_gt_for_train: 0.1  # with a small prob, sampling correction points from GT mask instead of prediction errors\n    num_frames_to_correct_for_train: 1  # single-frame data, only correct the first frame\n    num_frames_to_correct_for_eval: 1  # only iteratively sample on first frame\n    rand_frames_to_correct_for_train: False  # single-frame data, no random correction frames\n    add_all_frames_to_correct_as_cond: True  # when a frame receives a correction click, it becomes a conditioning frame (even if it's not initially a conditioning frame)\n    # maximum 1 initial conditioning frame\n    num_init_cond_frames_for_train: 1\n    rand_init_cond_frames_for_train: False  # single-frame data, no random conditioning frames\n    num_correction_pt_per_frame: 7\n    use_act_ckpt_iterative_pt_sampling: false\n    \n\n    \n    num_init_cond_frames_for_eval: 1  # only mask on the first frame\n    forward_backbone_per_frame_for_eval: True\n    \n\n  data:\n    train:\n      _target_: training.dataset.sam2_datasets.TorchTrainMixedDataset\n      phases_per_epoch: ${scratch.phases_per_epoch}\n      batch_sizes:\n        - ${scratch.train_video_batch_size}\n      datasets:\n        - _target_: training.dataset.utils.RepeatFactorWrapper\n          dataset:\n            _target_: training.dataset.utils.ConcatDataset\n            datasets:\n            # CT Lesion npz dataset\n            - _target_: training.dataset.vos_dataset.VOSDataset\n              transforms: ${vos.train_transforms}\n              training: true\n              video_dataset:\n                _target_: training.dataset.vos_raw_dataset.NPZRawDataset\n                folder: /home/wensheng/gjq_workspace/eyesam/data/Retina_Project/train_npz # must be absolute path\n              sampler:\n                _target_: training.dataset.vos_sampler.RandomUniformSampler\n                num_frames: ${scratch.num_frames}\n                max_num_objects: ${scratch.max_num_objects}\n              multiplier: 1\n            \n\n      shuffle: True\n      num_workers: ${scratch.num_train_workers}\n      pin_memory: True\n      drop_last: True\n      collate_fn:\n        _target_: training.utils.data_utils.collate_fn\n        _partial_: true\n        dict_key: laser\n    val:\n      _target_: training.dataset.sam2_datasets.TorchTrainMixedDataset\n      phases_per_epoch: 1\n      batch_sizes:\n        - ${scratch.train_video_batch_size}\n      datasets:\n        - _target_: training.dataset.utils.RepeatFactorWrapper\n          dataset:\n            _target_: training.dataset.utils.ConcatDataset\n            datasets:\n            - _target_: training.dataset.vos_dataset.VOSDataset\n              transforms: ${vos.val_transforms}\n              training: false\n              video_dataset:\n                _target_: training.dataset.vos_raw_dataset.NPZRawDataset\n                folder: /home/wensheng/gjq_workspace/eyesam/data/Retina_Project/val_npz\n              sampler:\n                _target_: training.dataset.vos_sampler.RandomUniformSampler\n                num_frames: ${scratch.num_frames}\n                max_num_objects: ${scratch.max_num_objects}\n              multiplier: 1\n      shuffle: False\n      num_workers: ${scratch.num_train_workers}\n      pin_memory: True\n      drop_last: False\n      collate_fn:\n        _target_: training.utils.data_utils.collate_fn\n        _partial_: true\n        dict_key: laser\n\n  optim:\n    amp:\n      enabled: True\n      amp_dtype: bfloat16\n\n    optimizer:\n      _target_: torch.optim.AdamW\n\n    gradient_clip:\n      _target_: training.optimizer.GradientClipper\n      max_norm: 0.1\n      norm_type: 2\n\n    param_group_modifiers:\n      - _target_: training.optimizer.layer_decay_param_modifier\n        _partial_: True\n        layer_decay_value: 0.9\n        apply_to: 'image_encoder.trunk'\n        overrides:\n          - pattern: '*pos_embed*'\n            value: 1.0\n\n    options:\n      lr:\n        - scheduler:\n            _target_: fvcore.common.param_scheduler.CosineParamScheduler\n            start_value: ${scratch.base_lr}\n            end_value: ${divide:${scratch.base_lr},10}\n        - scheduler:\n            _target_: fvcore.common.param_scheduler.CosineParamScheduler\n            start_value: ${scratch.vision_lr}\n            end_value: ${divide:${scratch.vision_lr},10}\n          param_names:\n            - 'image_encoder.*'\n      weight_decay:\n        - scheduler:\n            _target_: fvcore.common.param_scheduler.ConstantParamScheduler\n            value: 0.1\n        - scheduler:\n            _target_: fvcore.common.param_scheduler.ConstantParamScheduler\n            value: 0.0\n          param_names:\n            - '*bias*'\n          module_cls_names: ['torch.nn.LayerNorm']\n\n  loss:\n    laser:\n      _target_: training.loss_fns.MultiStepMultiMasksAndIous\n      weight_dict:\n        loss_mask: 20\n        loss_dice: 1\n        loss_iou: 1\n        loss_class: 1\n      supervise_all_iou: true\n      iou_use_l1_loss: true\n      pred_obj_scores: true\n      focal_gamma_obj_score: 0.0\n      focal_alpha_obj_score: -1.0\n\n  distributed:\n    backend: nccl #  gloo or nccl\n    find_unused_parameters: True\n\n  logging:\n    tensorboard_writer:\n      _target_: training.utils.logger.make_tensorboard_logger\n      log_dir:  ${launcher.experiment_log_dir}/tensorboard\n      flush_secs: 120\n      should_log: True\n    visdom_writer:\n      _target_: training.utils.logger.make_visdom_logger\n      env: sam2_train\n      server: http://localhost\n      port: 8097\n      raise_exceptions: False\n      should_log: True\n    log_dir: ${launcher.experiment_log_dir}/logs\n    log_freq: 10\n    log_visual_frequency: 100\n    visdom_image_mean: [0.485, 0.456, 0.406]\n    visdom_image_std: [0.229, 0.224, 0.225]\n\n  # initialize from a SAM 2 checkpoint\n  checkpoint:\n    save_dir: ${launcher.experiment_log_dir}/checkpoints\n    save_freq: 10 # 0 only last checkpoint is saved.\n    model_weight_initializer:\n      _partial_: True\n      _target_: training.utils.checkpoint_utils.load_state_dict_into_model\n      strict: True\n      ignore_unexpected_keys: null\n      ignore_missing_keys: null\n\n      state_dict:\n        _target_: training.utils.checkpoint_utils.load_checkpoint_and_apply_kernels\n        checkpoint_path: checkpoints/MedSAM2_latest.pt # PATH to SAM 2.1 checkpoint\n        ckpt_state_dict_keys: ['model']\n\nlauncher:\n  num_nodes: 1\n  gpus_per_node: 2\n  experiment_log_dir: exp_log # Path to log directory, defaults to ./sam2_logs/${config_name}\n\n# SLURM args if running on a cluster\nsubmitit:\n  partition: gpu_bwanggroup\n  account: null\n  qos: null\n  cpus_per_task: 10\n  use_cluster: false\n  timeout_hour: 24\n  name: null\n  port_range: [10000, 65000]\n"


def get_best_available_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _prepare_model_config(
    predictor_target: str | None,
    apply_postprocessing: bool,
    overrides: Dict[str, Any] | None = None,
):
    _ensure_omegaconf_resolvers()
    cfg = OmegaConf.create(yaml.safe_load(_CONFIG_YAML))
    OmegaConf.resolve(cfg)
    model_cfg = cfg["trainer"]["model"]
    if predictor_target is not None:
        model_cfg["_target_"] = predictor_target
    if not apply_postprocessing:
        model_cfg["fill_hole_area"] = 0
    if overrides:
        for key, value in overrides.items():
            model_cfg[key] = value
    return model_cfg


def _filter_model_kwargs(model_cfg, *allowed_types) -> Dict[str, Any]:
    allowed = set()
    for allowed_type in allowed_types:
        allowed.update(inspect.signature(allowed_type.__init__).parameters)
    allowed.discard("self")
    return {k: v for k, v in model_cfg.items() if k in allowed}


def _load_checkpoint(model, ckpt_path: str | None) -> None:
    if ckpt_path is None:
        return
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = state.get("model", state)
    model.load_state_dict(state_dict, strict=True)


def build_sam2_base(
    ckpt_path: str | None = None,
    device: str | None = None,
    mode: str = "eval",
    apply_postprocessing: bool = False,
    overrides: Dict[str, Any] | None = None,
):
    from sam2.modeling.sam2_base import SAM2Base
    device = device or get_best_available_device()
    model_cfg = _prepare_model_config(None, apply_postprocessing, overrides)
    model_kwargs = _filter_model_kwargs(model_cfg, SAM2Base)
    model = instantiate(model_kwargs, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor_npz(
    ckpt_path: str | None = None,
    device: str | None = None,
    mode: str = "eval",
    apply_postprocessing: bool = False,
    overrides: Dict[str, Any] | None = None,
):
    from sam2.modeling.sam2_base import SAM2Base
    from sam2.sam2_video_predictor_npz import SAM2VideoPredictorNPZ
    device = device or get_best_available_device()
    model_cfg = _prepare_model_config(
        "sam2.sam2_video_predictor_npz.SAM2VideoPredictorNPZ",
        apply_postprocessing,
        overrides,
    )
    model_kwargs = _filter_model_kwargs(model_cfg, SAM2Base, SAM2VideoPredictorNPZ)
    model = instantiate(model_kwargs, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    ckpt_path: str | None = None,
    device: str | None = None,
    mode: str = "eval",
    apply_postprocessing: bool = False,
    overrides: Dict[str, Any] | None = None,
):
    from sam2.modeling.sam2_base import SAM2Base
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    device = device or get_best_available_device()
    model_cfg = _prepare_model_config(
        "sam2.sam2_video_predictor.SAM2VideoPredictor",
        apply_postprocessing,
        overrides,
    )
    model_kwargs = _filter_model_kwargs(model_cfg, SAM2Base, SAM2VideoPredictor)
    model = instantiate(model_kwargs, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_image_predictor(
    ckpt_path: str | None = None,
    device: str | None = None,
    mode: str = "eval",
    apply_postprocessing: bool = False,
    overrides: Dict[str, Any] | None = None,
    **predictor_kwargs,
):
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam_model = build_sam2_base(
        ckpt_path=ckpt_path,
        device=device,
        mode=mode,
        apply_postprocessing=apply_postprocessing,
        overrides=overrides,
    )
    return SAM2ImagePredictor(sam_model, **predictor_kwargs)
