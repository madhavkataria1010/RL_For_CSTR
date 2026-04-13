from __future__ import annotations


def get_best_device(prefer_accelerator: bool = True) -> str:
    """Return the preferred runtime device as a string."""
    if not prefer_accelerator:
        return "cpu"

    try:  # pragma: no cover - optional dependency
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ModuleNotFoundError:
        pass
    return "cpu"

