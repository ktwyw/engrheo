"""Large-amplitude oscillatory shear (LAOS): Fourier-transform rheology and the Chebyshev decomposition
of Ewoldt, Hosoi and McKinley (J. Rheol. 52, 2008).

For a strain gamma0 sin(w t), the periodic stress contains odd harmonics:
    stress = gamma0 * sum_n [G'_n sin(n w t) + G''_n cos(n w t)],  n = 1, 3, 5, ...
Elastic Chebyshev coefficients e_n = G'_n (-1)^((n-1)/2); viscous v_n = G''_n / w. Material measures:
    G'_M = sum n G'_n  (minimum-strain modulus, tangent at zero strain)
    G'_L = sum G'_n (-1)^((n-1)/2)  (large-strain modulus, secant at maximum strain)
    S = (G'_L - G'_M) / G'_L  (> 0: intra-cycle strain stiffening)
    T = (eta'_L - eta'_M) / eta'_L  (> 0: intra-cycle shear thickening)

>>> import numpy as np
>>> from engrheo import laos
>>> t = np.linspace(0, 2 * np.pi, 512, endpoint=False)
>>> r = laos.chebyshev(t, 0.5 * np.sin(t), 100 * (0.5 * np.sin(t) + (0.5 * np.sin(t))**3), omega=1.0)
>>> round(r["G_L"], 6), round(r["G_M"], 6)                  # spring stress 100 (g + g^3): secant 125, tangent 100
(125.0, 100.0)
"""

from __future__ import annotations

import numpy as np

from ._common import as_1d


def harmonics(t, strain, stress, omega: float, n_max: int = 9) -> dict:
    """Odd-harmonic moduli G'_n, G''_n (n = 1, 3, ..., n_max) by least squares over the given time series
    (whole cycles of a steady oscillation), plus the relative third harmonic I3/I1."""
    t, g, s = as_1d(t, "t", 8), as_1d(strain, "strain", 8), as_1d(stress, "stress", 8)
    base = np.column_stack([np.sin(omega * t), np.cos(omega * t)])
    gamma0 = float(np.hypot(*np.linalg.lstsq(base, g, rcond=None)[0]))
    ns = np.arange(1, n_max + 1, 2)
    X = np.column_stack([f(n * omega * t) for n in ns for f in (np.sin, np.cos)] + [np.ones_like(t)])
    c = np.linalg.lstsq(X, s, rcond=None)[0]
    Gp, Gpp = c[0:-1:2] / gamma0, c[1:-1:2] / gamma0
    intensity = np.hypot(Gp, Gpp)
    ratio = float(intensity[1] / intensity[0]) if ns.size > 1 else 0.0
    return {"n": ns, "Gp": Gp, "Gpp": Gpp, "strain_amplitude": gamma0, "I3_I1": ratio}


def chebyshev(t, strain, stress, omega: float, n_max: int = 9) -> dict:
    """Chebyshev coefficients and LAOS material measures (G'_M, G'_L, eta'_M, eta'_L, S, T). The strain is
    assumed to be gamma0 sin(omega t) with zero phase (as produced by strain-controlled rheometers)."""
    h = harmonics(t, strain, stress, omega, n_max)
    n, Gp, Gpp = h["n"], h["Gp"], h["Gpp"]
    sign = (-1.0) ** ((n - 1) // 2)
    e, v = Gp * sign, Gpp / omega
    G_M, G_L = float(np.sum(n * Gp)), float(np.sum(e))
    eta_M, eta_L = float(np.sum(n * v * sign)), float(np.sum(v))
    return {"n": n, "e": e, "v": v, "G_M": G_M, "G_L": G_L, "eta_M": eta_M, "eta_L": eta_L,
            "S": (G_L - G_M) / G_L, "T": (eta_L - eta_M) / eta_L, "I3_I1": h["I3_I1"],
            "strain_amplitude": h["strain_amplitude"]}
