"""Named, order-independent random streams: every stream depends only on (seed, path)."""
import hashlib

import numpy as np


def _key(part: str | int) -> int:
    if isinstance(part, int):
        if part < 0:
            raise ValueError(f"seed path ints must be >= 0, got {part}")
        return part
    return int.from_bytes(hashlib.blake2b(part.encode(), digest_size=8).digest(), "big")


def rng(seed: int, *path: str | int) -> np.random.Generator:
    """Generator for (seed, path); never uses Python's per-process hash()."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=tuple(_key(p) for p in path))))
