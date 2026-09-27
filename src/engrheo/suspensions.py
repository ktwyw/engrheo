"""Viscosity of suspensions and emulsions as a function of the solid (or dispersed-phase) volume fraction.

>>> from engrheo import suspensions as su
>>> round(su.krieger_dougherty(0.3, phi_max=0.64), 4)
2.7512
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from ._common import as_1d


def einstein(phi):
    """Dilute hard spheres: eta_r = 1 + 2.5 phi (valid for phi below about 0.02)."""
    return 1 + 2.5 * np.asarray(phi, float)


def batchelor(phi):
    """Second order (Brownian hard spheres, low shear): eta_r = 1 + 2.5 phi + 6.2 phi^2."""
    p = np.asarray(phi, float)
    return 1 + 2.5 * p + 6.2 * p**2


def krieger_dougherty(phi, phi_max: float, intrinsic: float = 2.5):
    """Krieger-Dougherty: eta_r = (1 - phi/phi_max)^(-[eta] phi_max); diverges at the maximum packing fraction."""
    p = np.asarray(phi, float)
    if np.any(p >= phi_max):
        raise ValueError("phi must be below phi_max.")
    out = (1 - p / phi_max) ** (-intrinsic * phi_max)
    return float(out) if np.ndim(out) == 0 else out


def quemada(phi, phi_max: float):
    """Quemada (Maron-Pierce): eta_r = (1 - phi/phi_max)^(-2)."""
    p = np.asarray(phi, float)
    if np.any(p >= phi_max):
        raise ValueError("phi must be below phi_max.")
    return (1 - p / phi_max) ** -2


def fit_krieger_dougherty(phi, eta_r, fix_intrinsic: float | None = 2.5) -> dict:
    """Fit phi_max (and [eta] unless fixed) to relative-viscosity data, on log residuals."""
    p, e = as_1d(phi, "phi", 2), as_1d(eta_r, "eta_r", 2)
    if fix_intrinsic is not None:
        f = lambda x: np.log((1 - p / x[0]) ** (-fix_intrinsic * x[0])) - np.log(e)  # noqa: E731
        res = optimize.least_squares(f, [min(0.99, p.max() + 0.1)], bounds=([p.max() + 1e-6], [1.0]))
        return {"phi_max": float(res.x[0]), "intrinsic": fix_intrinsic, "rms": float(np.sqrt(np.mean(res.fun**2)))}
    f = lambda x: np.log((1 - p / x[0]) ** (-x[1] * x[0])) - np.log(e)  # noqa: E731
    res = optimize.least_squares(f, [min(0.99, p.max() + 0.1), 2.5], bounds=([p.max() + 1e-6, 0.5], [1.0, 10.0]))
    return {"phi_max": float(res.x[0]), "intrinsic": float(res.x[1]), "rms": float(np.sqrt(np.mean(res.fun**2)))}
