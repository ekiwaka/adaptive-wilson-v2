from __future__ import annotations
import random
import numpy as np
import torch


def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
    torch_num_threads: int | None = None,
) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: Integer random seed.
        deterministic: When true, request deterministic PyTorch algorithms where
            supported. This can reduce performance but improves reproducibility.
        torch_num_threads: Optional CPU thread count for PyTorch.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch_num_threads is not None:
        try:
            torch.set_num_threads(int(torch_num_threads))
        except Exception:
            pass

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def set_global_seed(
    seed: int,
    *,
    deterministic: bool = True,
    torch_num_threads: int | None = None,
) -> None:
    """Backward-compatible alias."""
    seed_everything(
        seed,
        deterministic=deterministic,
        torch_num_threads=torch_num_threads,
    )


def build_generator(device: torch.device, seed: int) -> torch.Generator:
    """Create a seeded torch generator on the requested device."""

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def build_cpu_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator