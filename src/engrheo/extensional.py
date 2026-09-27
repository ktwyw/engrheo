"""Extensional rheology: Trouton ratio and capillary-thinning (CaBER) analysis.

In a capillary-thinning test a liquid bridge between two plates thins under surface tension sigma; its
mid-filament diameter D(t) is recorded. Newtonian liquids thin linearly, D = 0.1418 (sigma/eta)(t_c - t)
(Papageorgiou 1995); dilute polymer solutions thin exponentially in the elasto-capillary regime,
D/D0 = (G D0 / 4 sigma)^(1/3) exp(-t / (3 lambda_E)) (Entov and Hinch 1997).
The apparent extensional viscosity is eta_E = (2X - 1) sigma / (-dD/dt) with X = 0.7127 for Newtonian thinning.

>>> from engrheo import extensional as ex
>>> ex.trouton_ratio(3.0, 1.0)
3.0
"""

from __future__ import annotations

import numpy as np

from ._common import as_1d

X_NEWTONIAN = 0.7127        # Eggers / Papageorgiou similarity solution
C_NEWTONIAN = 0.1418        # D = C sigma/eta (t_c - t)


def trouton_ratio(eta_E, eta):
    """Extensional viscosity divided by shear viscosity (3 for Newtonian liquids in uniaxial extension)."""
    out = np.asarray(eta_E, float) / np.asarray(eta, float)
    return float(out) if np.ndim(out) == 0 else out


def newtonian_thinning(t, D0: float, sigma: float, eta: float):
    """Mid-filament diameter of a thinning Newtonian filament (Papageorgiou): D = max(D0 - C sigma/eta t, 0)."""
    return np.maximum(D0 - C_NEWTONIAN * sigma / eta * np.asarray(t, float), 0.0)


def apparent_extensional_viscosity(t, D, sigma: float, X: float = X_NEWTONIAN):
    """eta_E,app = (2X - 1) sigma / (-dD/dt) from a measured thinning curve (derivative by second-order
    differences). With X = 0.7127 a Newtonian liquid gives exactly 3 times its shear viscosity."""
    t, D = as_1d(t, "t", 3), as_1d(D, "D", 3)
    dDdt = np.gradient(D, t)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(dDdt < 0, (2 * X - 1) * sigma / -dDdt, np.nan)      # NaN once the filament has broken


def fit_elastocapillary(t, D) -> dict:
    """Extensional relaxation time from the exponential (elasto-capillary) part of a thinning curve:
    ln D = const - t / (3 lambda_E); give only the data of that regime."""
    t, D = as_1d(t, "t", 3), as_1d(D, "D", 3)
    slope, intercept = np.polyfit(t, np.log(D), 1)
    return {"lambda_E": float(-1 / (3 * slope)), "D_fit0": float(np.exp(intercept))}
