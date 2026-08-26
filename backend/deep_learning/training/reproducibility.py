"""Shared reproducibility controls for TFT training processes."""

from __future__ import annotations


def configure_tft_reproducibility() -> None:
    """Use one deterministic CPU execution stream for TFT training.

    Lightning's ``deterministic=True`` selects deterministic algorithms, but
    it does not force CPU thread pools to use one reduction order.  Keeping
    this in a small shared module ensures the final trainer and the separate
    Optuna process use the same reproducibility contract.
    """
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # The process may already have initialized the inter-op pool.  The
        # single intra-op stream still removes the material variation here.
        pass
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
