"""CrossScore: Towards Multi-View Image Evaluation and Scoring.

A pip-installable package for neural image quality assessment using
cross-reference scoring with DINOv2 backbone.

Example:
    >>> import crossscore
    >>> results = crossscore.score(
    ...     query_dir="path/to/query/images",
    ...     reference_dir="path/to/reference/images",
    ... )
"""

__version__ = "1.0.0"


def score(*args, **kwargs):
    """Score query images against reference images using CrossScore.

    See crossscore.api.score for full documentation.
    """
    from crossscore.api import score as _score

    return _score(*args, **kwargs)


def get_checkpoint_path():
    """Get path to the CrossScore checkpoint, downloading if necessary.

    See crossscore._download.get_checkpoint_path for full documentation.
    """
    from crossscore._download import get_checkpoint_path as _get

    return _get()


__all__ = ["score", "get_checkpoint_path"]
