"""ctypes bridge to the single Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIB = Path(
    os.environ.get(
        "MOJO_SCIKIT_SURVIVAL_LIB",
        ROOT / "dist/libmojo-scikit-survival.so",
    )
)
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mss_concordance": ([I, I, I, I, I, I, I, F], F),
    "mss_product_limit": ([I, I, I, I, I, I, I, F, I], I),
    "mss_nelson_aalen": ([I, I, I, I, I], I),
    "mss_brier_score": ([I, I, I, I, I, I, I, I, I], None),
    "mss_dynamic_auc": ([I, I, I, I, I, I, I, I, I, I, F], None),
    "mss_cox_evaluate": ([I] * 13, F),
    "mss_cox_baseline": ([I, I, I, I, I, I], I),
    "mss_linear_predict": ([I, I, I, I, I], None),
}

_library: ctypes.CDLL | None = None


def build(force: bool = False) -> Path:
    source = ROOT / "src/survival.mojo"
    if not force and LIB.exists() and LIB.stat().st_mtime >= source.stat().st_mtime:
        return LIB
    if os.environ.get("MOJO_SCIKIT_SURVIVAL_LIB"):
        raise RuntimeError(f"configured shared library is missing or stale: {LIB}")
    subprocess.run(
        ["bash", str(ROOT / "build/build.sh")],
        cwd=ROOT,
        check=True,
        timeout=1800,
    )
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(str(build()))
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def f64(values, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(values)
    if np.iscomplexobj(source):
        raise ValueError("complex values cannot be converted to float64")
    if copy:
        return np.array(source, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(source, dtype=np.float64)


def addr(values: np.ndarray) -> int:
    if not isinstance(values, np.ndarray) or not values.flags.c_contiguous:
        raise TypeError("native buffers must be C-contiguous NumPy arrays")
    address = int(values.ctypes.data)
    if address == 0:
        raise ValueError("native buffers must have a non-null address")
    return address
