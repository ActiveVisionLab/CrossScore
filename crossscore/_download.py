"""Utilities for downloading CrossScore model checkpoints."""

import os
from pathlib import Path

CHECKPOINT_URL = (
    "https://huggingface.co/ActiveVisionLab/CrossScore/resolve/main/CrossScore-v1.0.0.ckpt"
)
CHECKPOINT_FILENAME = "CrossScore-v1.0.0.ckpt"


def get_cache_dir() -> Path:
    """Return the cache directory for CrossScore model checkpoints."""
    cache_dir = Path(os.environ.get("CROSSSCORE_CACHE_DIR", Path.home() / ".cache" / "crossscore"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_checkpoint_path() -> str:
    """Get path to the CrossScore checkpoint, downloading it if necessary.

    Downloads from HuggingFace Hub on first use and caches locally.
    Set CROSSSCORE_CACHE_DIR environment variable to customize cache location.
    Set CROSSSCORE_CKPT_PATH to use a specific local checkpoint file.

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

    print(f"Downloading CrossScore checkpoint to {ckpt_path}...")
    print(f"  Source: {CHECKPOINT_URL}")
    print("  (Set CROSSSCORE_CKPT_PATH to use a local checkpoint instead)")

    try:
        from huggingface_hub import hf_hub_download

        downloaded_path = hf_hub_download(
            repo_id="ActiveVisionLab/CrossScore",
            filename=CHECKPOINT_FILENAME,
            local_dir=str(cache_dir),
        )
        return downloaded_path
    except ImportError:
        # Fallback to urllib if huggingface_hub not installed
        import urllib.request
        import shutil

        tmp_path = str(ckpt_path) + ".tmp"
        try:
            with urllib.request.urlopen(CHECKPOINT_URL) as response, open(tmp_path, "wb") as out:
                shutil.copyfileobj(response, out)
            os.rename(tmp_path, str(ckpt_path))
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    print(f"  Download complete: {ckpt_path}")
    return str(ckpt_path)
