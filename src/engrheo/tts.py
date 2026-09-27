"""Temperature dependence and time-temperature superposition (TTS).

For thermorheologically simple materials a change of temperature only rescales time: curves measured at
temperature T, plotted against a_T * omega, superpose onto a master curve at the reference temperature.
Temperatures are in kelvin (WLF uses only differences, so degC gives the same C1, C2).

>>> from engrheo import tts
>>> round(float(tts.wlf(373.15, C1=8.86, C2=101.6, T_ref=373.15)), 12) + 0.0   # log10 aT = 0 at T_ref
0.0
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from ._common import as_1d

R_GAS = 8.314462618  # J/(mol K)


def wlf(T, C1: float, C2: float, T_ref: float):
    """Williams-Landel-Ferry equation: log10 aT = -C1 (T - T_ref) / (C2 + T - T_ref)."""
    dT = np.asarray(T, float) - T_ref
    return -C1 * dT / (C2 + dT)


def arrhenius(T, Ea: float, T_ref: float):
    """Arrhenius shift: ln aT = (Ea/R) (1/T - 1/T_ref); returned as log10 aT. Ea in J/mol, T in K."""
    return Ea / R_GAS * (1 / np.asarray(T, float) - 1 / T_ref) / np.log(10)


def fit_wlf(T, log10_aT, T_ref: float) -> dict:
    """Least-squares WLF constants C1, C2 from measured shift factors (log10 aT)."""
    t, la = as_1d(T, "T", 3), as_1d(log10_aT, "log10_aT", 3)
    res = optimize.least_squares(lambda p: wlf(t, p[0], p[1], T_ref) - la, [10.0, 100.0],
                                 bounds=([0, 1e-6], [np.inf, np.inf]), xtol=1e-14, ftol=1e-14)
    return {"C1": float(res.x[0]), "C2": float(res.x[1]), "rms": float(np.sqrt(np.mean(res.fun**2)))}


def fit_arrhenius(T, log10_aT, T_ref: float) -> dict:
    """Activation energy (J/mol) from shift factors: straight line of ln aT against 1/T - 1/T_ref."""
    t, la = as_1d(T, "T", 2), as_1d(log10_aT, "log10_aT", 2)
    x = 1 / t - 1 / T_ref
    slope = float(np.sum(x * la * np.log(10)) / np.sum(x * x))       # line through the origin (aT = 1 at T_ref)
    return {"Ea": slope * R_GAS, "rms": float(np.sqrt(np.mean((la * np.log(10) - slope * x) ** 2)) / np.log(10))}


def shift_factor(x_ref, y_ref, x, y, bounds=(-8.0, 8.0)) -> float:
    """Horizontal shift log10(a) that makes the curve (x * a, y) overlap the reference curve (x_ref, y_ref)
    best - least squares of ln y against log-log interpolation of the reference over their overlap. ``y`` and
    ``y_ref`` may have several columns (e.g. G' and G'') that are shifted together."""
    xr, x = as_1d(x_ref, "x_ref", 2), as_1d(x, "x", 2)
    yr, y = np.asarray(y_ref, float).reshape(xr.size, -1), np.asarray(y, float).reshape(x.size, -1)
    lxr, lx = np.log10(xr), np.log10(x)
    order = np.argsort(lxr)

    def mismatch(la):
        s = lx + la
        inside = (s >= lxr.min()) & (s <= lxr.max())
        if inside.sum() < 2:
            return 1e6 * (1 + abs(la))                 # no overlap: push the search back
        err = [np.log(y[inside, j]) - np.interp(s[inside], lxr[order], np.log(yr[order, j])) for j in range(y.shape[1])]
        return float(np.mean(np.concatenate(err) ** 2))

    grid = np.linspace(bounds[0], bounds[1], 321)       # coarse scan (the mismatch can have local minima) ...
    vals = [mismatch(v) for v in grid]
    k = int(np.argmin(vals))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
    return float(optimize.minimize_scalar(mismatch, bounds=(lo, hi), method="bounded", options={"xatol": 1e-10}).x)


def master_curve(curves: dict, T_ref: float) -> dict:
    """Shift curves measured at several temperatures onto a master curve at T_ref.

    ``curves`` maps temperature -> (omega, y), y with one or several columns (e.g. G' and G''). Curves are
    shifted outwards from T_ref, each onto its already-shifted neighbour, so every pair overlaps.
    Returns {"log10_aT": {T: value}, "omega_reduced": {T: a_T omega}, "y": {T: y}}."""
    temps = sorted(curves)
    if T_ref not in curves:
        raise ValueError("T_ref must be one of the measured temperatures.")
    i0 = temps.index(T_ref)
    la = {T_ref: 0.0}
    for direction in (1, -1):
        prev = T_ref
        i = i0 + direction
        while 0 <= i < len(temps):
            T = temps[i]
            w_prev, y_prev = curves[prev]
            w, y = curves[T]
            la[T] = la[prev] + shift_factor(w_prev, y_prev, w, y)   # shift onto the neighbour, then chain
            prev = T
            i += direction
    return {"log10_aT": {T: la[T] for T in temps},
            "omega_reduced": {T: np.asarray(curves[T][0], float) * 10 ** la[T] for T in temps},
            "y": {T: np.asarray(curves[T][1], float) for T in temps}}
