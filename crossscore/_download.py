"""Utilities for downloading CrossScore model checkpoints."""

import os
import urllib.request
import shutil
from pathlib import Path

# Download directly from GitHub (served via Git LFS)
CHECKPOINT_URL = (
    "https://github.com/ActiveVisionLab/CrossScore/raw/main/ckpt/CrossScore-v1.0.0.ckpt"
)
CHECKPOINT_FILENAME = "CrossScore-v1.0.0.ckpt"


def get_cache_dir() -> Path:
    """Return the cache directory for CrossScore model checkpoints."""
    cache_dir = Path(os.environ.get("CROSSSCORE_CACHE_DIR", Path.home() / ".cache" / "crossscore"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_checkpoint_path() -> str:
    """Get path to the CrossScore checkpoint, downloading it if necessary.

    Downloads from GitHub (Git LFS) on first use and caches locally at
    ~/.cache/crossscore/. Set environment variables to customize:
        CROSSSCORE_CKPT_PATH - use a specific local checkpoint file
        CROSSSCORE_CACHE_DIR - custom cache directory

    Returns:
        Path to the checkpoint file.
    """
    # Allow user to override with a custom path
    custom_path = os.environ.get("CROSSSCORE_CKPT_PATH")
    if custom_path:
        if not Path(custom_path).exists():
            raise FileNotFoundError(f"Checkpoint not found at CROSSSCORE_CKPT_PATH={custom_path}")
        return custom_path

    cache_dir = get_cache_dir()
    ckpt_path = cache_dir / CHECKPOINT_FILENAME

    if ckpt_path.exists():
        return str(ckpt_path)

    print(f"Downloading CrossScore checkpoint (~129MB)...")
    print(f"  From: {CHECKPOINT_URL}")
    print(f"  To:   {ckpt_path}")
    print("  (Set CROSSSCORE_CKPT_PATH to skip download and use a local file)")

    tmp_path = str(ckpt_path) + ".tmp"
    try:
        urllib.request.urlretrieve(CHECKPOINT_URL, tmp_path, _download_progress)
        os.rename(tmp_path, str(ckpt_path))
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    print(f"\n  Download complete.")
    return str(ckpt_path)


def _download_progress(block_count, block_size, total_size):
    """Progress callback for urlretrieve."""
    downloaded = block_count * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  {mb_done:.1f}/{mb_total:.1f} MB ({pct}%)", end="", flush=True)
