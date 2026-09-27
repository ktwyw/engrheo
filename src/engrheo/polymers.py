"""Polymer melt rheology from molecular parameters: zero-shear viscosity scaling with molecular weight,
entanglement molecular weight from the plateau modulus, and the melt flow index.

>>> from engrheo import polymers as po
>>> round(po.entanglement_molar_mass(2.0e5, 1000.0, 453.15), 3)       # kg/mol
18.838
"""

from __future__ import annotations

import numpy as np

from .tts import R_GAS

# ISO 1133 / ASTM D1238 standard die and piston
MFI_DIE_RADIUS = 2.095e-3 / 2
MFI_DIE_LENGTH = 8.000e-3
MFI_PISTON_RADIUS = 9.55e-3 / 2


def zero_shear_viscosity(M, M_c: float, eta_c: float, exponent: float = 3.4):
    """eta0(M): proportional to M below the critical molecular weight M_c (Rouse regime), to M^exponent above
    (entangled, ~3.4); continuous at M_c where eta0 = eta_c."""
    m = np.asarray(M, float)
    out = np.where(m < M_c, eta_c * m / M_c, eta_c * (m / M_c) ** exponent)
    return float(out) if np.ndim(out) == 0 else out


def entanglement_molar_mass(G_N0: float, density: float, T: float) -> float:
    """Entanglement molar mass M_e = rho R T / G_N0 (kg/mol) from the plateau modulus (Pa), density (kg/m3)
    and temperature (K). (Some texts use 4/5 rho R T / G_N0.)"""
    return float(density * R_GAS * T / G_N0)


def mfi_conditions(mfi_g_per_10min: float, load_kg: float, melt_density: float) -> dict:
    """Apparent wall shear stress, apparent shear rate and apparent viscosity in a standard melt-flow-index test.
    Stress: piston pressure m g / (pi Rp^2) times R/(2L) of the die; rate: 4Q/(pi R^3) with Q = MFI / (rho 600 s).
    (No Bagley or Rabinowitsch corrections: the MFI is a single-point index, not a viscosity.)"""
    tau = load_kg * 9.80665 / (np.pi * MFI_PISTON_RADIUS**2) * MFI_DIE_RADIUS / (2 * MFI_DIE_LENGTH)
    Q = mfi_g_per_10min * 1e-3 / melt_density / 600.0
    rate = 4 * Q / (np.pi * MFI_DIE_RADIUS**3)
    return {"tau_w": float(tau), "rate_apparent": float(rate), "eta_apparent": float(tau / rate)}
