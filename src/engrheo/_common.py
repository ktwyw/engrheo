"""Small helpers shared by the modules."""

from __future__ import annotations

import numpy as np


def as_1d(x, name: str = "x", min_n: int = 1) -> np.ndarray:
    """Convert to a 1-D float array and check it (finite values, at least ``min_n`` of them)."""
    a = np.asarray(x, dtype=float).ravel()
    if a.size < min_n:
        raise ValueError(f"{name} needs at least {min_n} values, got {a.size}.")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return a


def log_slope(x, y) -> np.ndarray:
    """Local slope d ln y / d ln x at every point (second-order differences on the log scale; works
    for unevenly spaced data). Used by the rheometer corrections, which need n' = d ln(torque)/d ln(rate)."""
    lx, ly = np.log(as_1d(x, "x", 3)), np.log(as_1d(y, "y", 3))
    if np.any(np.diff(lx) <= 0):
        raise ValueError("x must be strictly increasing and positive.")
    return np.gradient(ly, lx)


def table(headers, rows, floatfmt: str = ".4g") -> str:
    """Plain-text table (no dependencies)."""
    def fmt(v):
        return format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v)
    cells = [[fmt(v) for v in r] for r in rows]
    widths = [max(len(str(h)), *(len(r[i]) for r in cells)) if cells else len(str(h)) for i, h in enumerate(headers)]
    line = "  ".join(str(h).rjust(w) for h, w in zip(headers, widths))
    return "\n".join([line, "-" * len(line)] + ["  ".join(c.rjust(w) for c, w in zip(r, widths)) for r in cells])
