"""Oscillatory (dynamic) rheology: complex modulus and viscosity, loss tangent, crossover, the
linear-viscoelastic limit of amplitude sweeps, the Winter-Chambon gel point, and G', G'' from raw
stress and strain waveforms.

>>> from engrheo import oscillatory
>>> d = oscillatory.complex_quantities(omega=10.0, Gp=300.0, Gpp=400.0)
>>> d["G_star"], round(d["tan_delta"], 4), d["eta_star"]
(500.0, 1.3333, 50.0)
"""

from __future__ import annotations

import numpy as np

from ._common import as_1d


def complex_quantities(omega, Gp, Gpp) -> dict:
    """|G*| = sqrt(G'^2 + G''^2), tan(delta) = G''/G', phase angle delta (degrees), complex viscosity
    |eta*| = |G*|/omega, dynamic viscosity eta' = G''/omega and eta'' = G'/omega."""
    w, gp, gpp = np.asarray(omega, float), np.asarray(Gp, float), np.asarray(Gpp, float)
    G_star = np.hypot(gp, gpp)
    out = {"G_star": G_star, "tan_delta": gpp / gp, "delta_deg": np.degrees(np.arctan2(gpp, gp)),
           "eta_star": G_star / w, "eta_prime": gpp / w, "eta_double_prime": gp / w}
    return {k: (float(v) if np.ndim(v) == 0 else v) for k, v in out.items()}


def crossover(omega, Gp, Gpp) -> tuple[float, float] | None:
    """Frequency and modulus where G' = G'' (log-log interpolation between the bracketing points); the
    inverse of the crossover frequency is a characteristic relaxation time. None if the curves do not cross."""
    w, gp, gpp = as_1d(omega, "omega", 2), as_1d(Gp, "Gp", 2), as_1d(Gpp, "Gpp", 2)
    d = np.log(gp) - np.log(gpp)
    exact = np.nonzero(d == 0)[0]                   # a sampled point exactly at the crossover
    idx = np.nonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    if exact.size and (idx.size == 0 or exact[0] <= idx[0]):
        return float(w[exact[0]]), float(gp[exact[0]])
    if idx.size == 0:
        return None
    i = idx[0]
    f = d[i] / (d[i] - d[i + 1])
    lw = np.log(w[i]) + f * (np.log(w[i + 1]) - np.log(w[i]))
    lg = np.log(gp[i]) + f * (np.log(gp[i + 1]) - np.log(gp[i]))
    return float(np.exp(lw)), float(np.exp(lg))


def lve_limit(strain, Gp, tolerance: float = 0.05, n_plateau: int = 3) -> float | None:
    """End of the linear-viscoelastic region in an amplitude sweep: the strain at which G' has fallen by
    ``tolerance`` (default 5 %) below its low-strain plateau (mean of the first ``n_plateau`` points);
    interpolated on log scales. None if G' never falls that far."""
    s, gp = as_1d(strain, "strain", n_plateau + 1), as_1d(Gp, "Gp", n_plateau + 1)
    plateau = gp[:n_plateau].mean()
    target = (1 - tolerance) * plateau
    below = np.nonzero(gp < target)[0]
    if below.size == 0:
        return None
    i = below[0]
    f = (np.log(gp[i - 1]) - np.log(target)) / (np.log(gp[i - 1]) - np.log(gp[i]))
    return float(np.exp(np.log(s[i - 1]) + f * (np.log(s[i]) - np.log(s[i - 1]))))


def gel_point(times, tan_delta) -> float:
    """Winter-Chambon gel point from time sweeps at several frequencies: at the gel point tan(delta) is the
    same at every frequency. ``tan_delta`` has shape (len(times), n_frequencies). Each frequency's log tan(delta)
    is interpolated linearly in time, and the gel time is where their spread (standard deviation) is smallest.
    Near the gel point the curves cross, so the spread is V-shaped; a direct bounded minimisation locates the
    crossing exactly where the curves are locally straight (a parabola through the minimum would be biased)."""
    from scipy import optimize

    t = as_1d(times, "times", 3)
    ltd = np.log(np.asarray(tan_delta, float))
    if ltd.shape[0] != t.size or ltd.ndim != 2 or ltd.shape[1] < 2:
        raise ValueError("tan_delta must have shape (len(times), n_frequencies >= 2).")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be increasing.")

    def spread(x):
        return float(np.std([np.interp(x, t, ltd[:, j]) for j in range(ltd.shape[1])]))

    i = int(np.argmin(np.std(ltd, axis=1)))
    lo, hi = t[max(i - 1, 0)], t[min(i + 1, t.size - 1)]
    return float(optimize.minimize_scalar(spread, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12}).x)


def moduli_from_waveforms(t, strain, stress, omega: float) -> dict:
    """G' and G'' from sampled strain and stress signals of an oscillation at frequency omega (rad/s),
    using their first Fourier harmonic (least-squares fit of sine and cosine over whole cycles):
    G' = (stress amplitude in phase with strain) / strain amplitude, G'' = (in phase with the rate) / strain amplitude.
    Higher harmonics (nonlinear response, notebook 11) are ignored here."""
    t, e, s = as_1d(t, "t", 8), as_1d(strain, "strain", 8), as_1d(stress, "stress", 8)
    X = np.column_stack([np.sin(omega * t), np.cos(omega * t), np.ones_like(t)])
    ce, cs = np.linalg.lstsq(X, e, rcond=None)[0], np.linalg.lstsq(X, s, rcond=None)[0]
    amp_e, ph_e = np.hypot(ce[0], ce[1]), np.arctan2(ce[1], ce[0])
    amp_s, ph_s = np.hypot(cs[0], cs[1]), np.arctan2(cs[1], cs[0])
    delta = np.angle(np.exp(1j * (ph_s - ph_e)))
    G_star = amp_s / amp_e
    return {"Gp": float(G_star * np.cos(delta)), "Gpp": float(G_star * np.sin(delta)), "strain_amplitude": float(amp_e),
            "stress_amplitude": float(amp_s), "delta_deg": float(np.degrees(delta))}


def cox_merz_ratio(rate, eta, omega, eta_star) -> np.ndarray:
    """Cox-Merz rule check: steady viscosity eta(rate) divided by complex viscosity |eta*|(omega) at
    omega = rate (log-log interpolation of |eta*|). Ratios near 1 mean the rule holds."""
    r, e = as_1d(rate, "rate", 1), as_1d(eta, "eta", 1)
    w, es = as_1d(omega, "omega", 2), as_1d(eta_star, "eta_star", 2)
    inside = (r >= w.min()) & (r <= w.max())
    out = np.full(r.shape, np.nan)
    out[inside] = e[inside] / np.exp(np.interp(np.log(r[inside]), np.log(w), np.log(es)))
    return out
