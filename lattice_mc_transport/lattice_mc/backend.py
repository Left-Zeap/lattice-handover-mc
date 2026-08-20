from __future__ import annotations

def get_backend(name: str = "auto"):
    """
    Return (xp, resolved_name).

    xp is numpy or cupy and exposes a NumPy-like API.
    """
    name = name.lower()
    if name not in {"auto", "cpu", "gpu"}:
        raise ValueError("backend must be auto/cpu/gpu")

    if name in {"auto", "gpu"}:
        try:
            import cupy as cp
            # Force a tiny runtime check; import alone can succeed with unusable CUDA.
            _ = cp.zeros(1)
            return cp, "gpu"
        except Exception:
            if name == "gpu":
                raise RuntimeError(
                    "GPU backend requested but CuPy/CUDA is unavailable. "
                    "Install a matching CuPy build, e.g. cupy-cuda12x."
                )

    import numpy as np
    return np, "cpu"


class _CupyRngAdapter:
    """
    Give cupy.random.RandomState the numpy.random.Generator API surface
    used by the simulation code (normal / uniform / random).
    """

    def __init__(self, rs):
        self._rs = rs

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self._rs.normal(loc, scale, size)

    def uniform(self, low=0.0, high=1.0, size=None):
        return self._rs.uniform(low, high, size)

    def random(self, size=None):
        return self._rs.random_sample(size)


def make_rng(xp, seed: int):
    """
    Return a random generator compatible with the given array backend.

    CuPy's new-style Generator (cp.random.default_rng) does not implement
    normal()/uniform(), and cupy.random.RandomState lacks random(), so on
    GPU we wrap RandomState in a small adapter exposing the exact API the
    simulation code uses.
    """
    try:
        import cupy as cp
        if xp is cp:
            return _CupyRngAdapter(cp.random.RandomState(seed))
    except ImportError:
        pass
    return xp.random.default_rng(seed)


def to_cpu(x):
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    return x
