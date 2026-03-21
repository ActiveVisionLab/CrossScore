"""High-level API for CrossScore image quality assessment."""

from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from omegaconf import OmegaConf
from tqdm import tqdm

from crossscore._download import get_checkpoint_path
from crossscore.utils.io.images import ImageNetMeanStd
from crossscore.dataloading.dataset.simple_reference import SimpleReference


def _write_score_maps(score_maps, query_paths, out_dir, metric_type, metric_min, metric_max):
    """Write score maps to disk as colorized PNGs."""
    from PIL import Image
    from crossscore.utils.misc.image import gray2rgb

    vrange_vis = [metric_min, metric_max]
    out_dir = Path(out_dir) / "score_maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for batch_maps, batch_paths in zip(score_maps, query_paths):
        for score_map, qpath in zip(batch_maps, batch_paths):
            fname = Path(qpath).stem + ".png"
            rgb = gray2rgb(score_map.cpu().numpy(), vrange_vis)
            Image.fromarray(rgb).save(out_dir / fname)
            idx += 1
    return str(out_dir)


def score(
    query_dir: str,
    reference_dir: str,
    ckpt_path: Optional[str] = None,
    metric_type: str = "ssim",
    batch_size: int = 8,
    num_workers: int = 4,
    resize_short_side: int = 518,
    device: Optional[str] = None,
    out_dir: Optional[str] = None,
    write_score_maps: bool = True,
) -> dict:
    """Score query images against reference images using CrossScore.

    Args:
        query_dir: Directory containing query images (e.g., NVS rendered images).
        reference_dir: Directory containing reference images (e.g., real captured images).
        ckpt_path: Path to model checkpoint. Auto-downloads if not provided.
        metric_type: Metric type to predict. One of "ssim", "mae", "mse".
        batch_size: Batch size for inference.
        num_workers: Number of data loading workers.
        resize_short_side: Resize images so short side equals this value. -1 to disable.
        device: Device string ("cuda", "cuda:0", "cpu"). Auto-detected if None.
        out_dir: Output directory for score maps. Defaults to "./crossscore_output".
        write_score_maps: Whether to write colorized score map PNGs to disk.

    Returns:
        Dictionary with:
            - "score_maps": List of score map tensors, each (B, H, W)
            - "scores": List of per-image mean scores (float)
            - "out_dir": Output directory path (if write_score_maps=True)

    Example:
        >>> import crossscore
        >>> results = crossscore.score(
        ...     query_dir="path/to/query/images",
        ...     reference_dir="path/to/reference/images",
        ... )
        >>> print(results["scores"])  # per-image mean scores
    """
    from crossscore.task.core import load_model

    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            print(
                "Note: CUDA not available, running on CPU. "
                "For GPU acceleration, install with:\n"
                "  conda env create -f environment_gpu.yaml"
            )

    # Get checkpoint
    if ckpt_path is None:
        ckpt_path = get_checkpoint_path()

    # Load model
    model = load_model(ckpt_path, device=device)

    # Set up data transforms
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

    # Build dataset and dataloader
    neighbour_config = {"strategy": "random", "cross": 5, "deterministic": False}
    dataset = SimpleReference(
        query_dir=query_dir,
        reference_dir=reference_dir,
        transforms=transforms,
        neighbour_config=neighbour_config,
        return_item_paths=True,
        zero_reference=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=False,
    )

    # Run inference
    all_score_maps = []
    all_scores = []
    all_query_paths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="CrossScore"):
            query_img = batch["query/img"].to(device)
            ref_imgs = batch.get("reference/cross/imgs")
            if ref_imgs is not None:
                ref_imgs = ref_imgs.to(device)

            outputs = model(
                query_img=query_img,
                ref_cross_imgs=ref_imgs,
                norm_img=False,
            )

            score_map = outputs["score_map_ref_cross"]  # (B, H, W)
            all_score_maps.append(score_map.cpu())

            # Per-image mean score
            for i in range(score_map.shape[0]):
                all_scores.append(score_map[i].mean().item())

            # Track query paths for output naming
            if "item_paths" in batch and "query/img" in batch["item_paths"]:
                all_query_paths.append(batch["item_paths"]["query/img"])

    # Build results
    metric_min = -1 if metric_type == "ssim" else 0
    if metric_type == "ssim":
        metric_min = 0  # CrossScore predicts SSIM in [0, 1] by default

    results = {
        "score_maps": all_score_maps,
        "scores": all_scores,
    }

    # Write outputs
    if write_score_maps and all_score_maps:
        if out_dir is None:
            out_dir = "./crossscore_output"
        written_dir = _write_score_maps(
            all_score_maps, all_query_paths, out_dir,
            metric_type, metric_min, metric_max=1,
        )
        results["out_dir"] = written_dir

    return results
