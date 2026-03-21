"""High-level API for CrossScore image quality assessment."""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from omegaconf import OmegaConf

from crossscore._download import get_checkpoint_path
from crossscore.utils.io.images import ImageNetMeanStd
from crossscore.dataloading.dataset.simple_reference import SimpleReference
from crossscore.dataloading.transformation.crop import CropperFactory


def _build_config(
    metric_type: str = "ssim",
    metric_min: int = 0,
    metric_max: int = 1,
    batch_size: int = 8,
    num_workers: int = 4,
    resize_short_side: int = 518,
    devices: Optional[list] = None,
    out_dir: Optional[str] = None,
) -> OmegaConf:
    """Build an OmegaConf config object for prediction."""
    use_gpu = torch.cuda.is_available() and devices != "cpu"
    if devices is None:
        devices = [0] if use_gpu else "auto"
    elif devices == "cpu":
        devices = "auto"
        use_gpu = False

    config_dir = Path(__file__).parent / "config"
    # Load base configs
    base_cfg = OmegaConf.load(config_dir / "default_predict.yaml")
    model_cfg = OmegaConf.load(config_dir / "model" / "model.yaml")
    data_cfg = OmegaConf.load(config_dir / "data" / "SimpleReference.yaml")

    # Merge model config into base
    base_cfg.model = model_cfg
    base_cfg.data = data_cfg

    # Apply overrides
    base_cfg.model.predict.metric.type = metric_type
    base_cfg.model.predict.metric.min = metric_min
    base_cfg.model.predict.metric.max = metric_max
    base_cfg.data.loader.validation.batch_size = batch_size
    base_cfg.data.loader.validation.num_workers = num_workers
    base_cfg.trainer.devices = devices
    base_cfg.trainer.precision = "16-mixed" if use_gpu else "32-true"
    base_cfg.trainer.accelerator = "gpu" if use_gpu else "cpu"

    if out_dir is not None:
        base_cfg.logger.predict.out_dir = out_dir

    return base_cfg


def score(
    query_dir: str,
    reference_dir: str,
    ckpt_path: Optional[str] = None,
    metric_type: str = "ssim",
    batch_size: int = 8,
    num_workers: int = 4,
    resize_short_side: int = 518,
    devices: Optional[list] = None,
    out_dir: Optional[str] = None,
    write_outputs: bool = True,
) -> dict:
    """Score query images against reference images using CrossScore.

    Args:
        query_dir: Directory containing query images (e.g., NVS rendered images).
        reference_dir: Directory containing reference images (e.g., real captured images).
        ckpt_path: Path to model checkpoint. Auto-downloads if not provided.
        metric_type: Metric type to predict. One of "ssim", "mae", "mse".
        batch_size: Batch size for inference.
        num_workers: Number of data loading workers.
        resize_short_side: Resize images so short side equals this value. Set to -1 to disable.
        devices: List of GPU device indices. Defaults to [0] if CUDA available.
        out_dir: Output directory for score maps. Defaults to a timestamped directory
            under the checkpoint's parent.
        write_outputs: Whether to write score maps and visualizations to disk.

    Returns:
        Dictionary with:
            - "score_maps": List of predicted score map tensors
            - "out_dir": Output directory path (if write_outputs=True)

    Example:
        >>> import crossscore
        >>> results = crossscore.score(
        ...     query_dir="path/to/query/images",
        ...     reference_dir="path/to/reference/images",
        ... )
    """
    import lightning
    from datetime import datetime
    from crossscore.task.core import CrossScoreLightningModule

    # Get checkpoint
    if ckpt_path is None:
        ckpt_path = get_checkpoint_path()

    # Build config
    metric_min = -1 if metric_type == "ssim" else 0
    # For SSIM, CrossScore predicts in [0, 1] by default (the common sub-range)
    if metric_type == "ssim":
        metric_min = 0

    cfg = _build_config(
        metric_type=metric_type,
        metric_min=metric_min,
        batch_size=batch_size,
        num_workers=num_workers,
        resize_short_side=resize_short_side,
        devices=devices,
        out_dir=out_dir,
    )

    # Set checkpoint path
    cfg.trainer.ckpt_path_to_load = ckpt_path

    # Determine output directory
    if cfg.logger.predict.out_dir is None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S.%f")
        log_dir = Path(ckpt_path).parents[1] if Path(ckpt_path).parent.name == "ckpt" else Path(".")
        cfg.logger.predict.out_dir = str(log_dir / "predict" / now)

    if not write_outputs:
        cfg.logger.predict.write.flag.batch = False
        cfg.logger.predict.write.config.vis_img_every_n_steps = -1

    # Set up data
    lightning.seed_everything(cfg.lightning.seed, workers=True)

    img_norm_stat = ImageNetMeanStd()
    transforms = {
        "img": T.Normalize(mean=img_norm_stat.mean, std=img_norm_stat.std),
    }

    if resize_short_side > 0:
        transforms["resize"] = T.Resize(
            resize_short_side,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )

    dataset = SimpleReference(
        query_dir=query_dir,
        reference_dir=reference_dir,
        transforms=transforms,
        neighbour_config=cfg.data.neighbour_config,
        return_item_paths=True,
        zero_reference=cfg.data.dataset.zero_reference,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.loader.validation.batch_size,
        shuffle=False,
        num_workers=cfg.data.loader.validation.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    # Build model and trainer
    model = CrossScoreLightningModule(cfg)

    NUM_GPUS = len(cfg.trainer.devices) if isinstance(cfg.trainer.devices, list) else 0
    if NUM_GPUS > 1:
        from lightning.pytorch.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=False, static_graph=True)
        use_distributed_sampler = True
    else:
        strategy = "auto"
        use_distributed_sampler = False

    trainer = lightning.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        strategy=strategy,
        use_distributed_sampler=use_distributed_sampler,
        logger=False,
    )

    # Run prediction
    with torch.no_grad():
        predictions = trainer.predict(
            model,
            dataloader,
            ckpt_path=ckpt_path,
        )

    # Collect results
    score_maps = []
    if predictions:
        for batch_output in predictions:
            if "score_map_ref_cross" in batch_output:
                score_maps.append(batch_output["score_map_ref_cross"].cpu())

    results = {"score_maps": score_maps}
    if write_outputs:
        results["out_dir"] = cfg.logger.predict.out_dir

    return results
