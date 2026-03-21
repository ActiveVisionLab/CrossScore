"""Utilities for downloading CrossScore model checkpoints."""

import os
from pathlib import Path

HF_REPO_ID = "ActiveVisionLab/CrossScore"
CHECKPOINT_FILENAME = "CrossScore-v1.0.0.ckpt"


def get_checkpoint_path() -> str:
    """Get path to the CrossScore checkpoint, downloading it if necessary.

    Downloads from HuggingFace Hub on first use and caches locally.
    Set environment variables to customize:
        CROSSSCORE_CKPT_PATH - use a specific local checkpoint file

    Returns:
        Path to the checkpoint file.
    """
    # Allow user to override with a custom path
    custom_path = os.environ.get("CROSSSCORE_CKPT_PATH")
    if custom_path:
        if not Path(custom_path).exists():
            raise FileNotFoundError(f"Checkpoint not found at CROSSSCORE_CKPT_PATH={custom_path}")
        return custom_path

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=CHECKPOINT_FILENAME,
    )
    return path
