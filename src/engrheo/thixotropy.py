"""Thixotropy with a structural-kinetics model (Moore / Houska type).

A structure parameter lam (1 = fully built up, 0 = fully broken down) evolves as
    d lam/dt = k_build (1 - lam) - k_break * rate * lam
and the stress depends on the current structure:
    tau = lam * tau_y + (eta_inf + d_eta * lam) * rate
At constant rate lam relaxes exponentially to lam_eq = k_build / (k_build + k_break rate) with rate constant
k_build + k_break rate - so the stress depends on the shear history, not only on the present rate.

>>> from engrheo import thixotropy as th
>>> p = th.Params(tau_y=10.0, eta_inf=0.05, d_eta=0.5, k_build=0.01, k_break=0.05)
>>> round(th.equilibrium_structure(p, 0.0), 6), round(th.equilibrium_structure(p, 1.0), 6)
(1.0, 0.166667)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._common import as_1d


@dataclass
class Params:
    tau_y: float        # yield stress of the fully built structure (Pa)
    eta_inf: float      # viscosity of the fully broken structure (Pa s)
    d_eta: float        # extra viscosity contributed by the structure (Pa s)
    k_build: float      # build-up (recovery) rate constant (1/s)
    k_break: float      # breakdown constant (dimensionless: per unit strain)


def equilibrium_structure(p: Params, rate):
    """Steady-state structure at a constant shear rate."""
    r = np.asarray(rate, float)
    out = p.k_build / (p.k_build + p.k_break * r)
    return float(out) if np.ndim(out) == 0 else out


def stress(p: Params, lam, rate):
    """Shear stress for structure lam at shear rate rate."""
    return np.asarray(lam, float) * p.tau_y + (p.eta_inf + p.d_eta * np.asarray(lam, float)) * np.asarray(rate, float)


def equilibrium_flow_curve(p: Params, rate):
    """Stress on the equilibrium flow curve (each rate held long enough for the structure to settle)."""
    return stress(p, equilibrium_structure(p, rate), rate)


def simulate(p: Params, t, rate, lam0: float = 1.0) -> dict:
    """Structure and stress for a shear-rate history given at the times ``t``, treated as piecewise constant
    (rate[i] applies from t[i] to t[i+1]) and integrated exactly interval by interval - ideal for step tests.
    For smoothly varying rates (ramps) use a fine time grid, or ``hysteresis_loop``, which integrates the
    continuous rate."""
    t, r = as_1d(t, "t", 2), as_1d(rate, "rate", 2)
    if t.size != r.size or np.any(np.diff(t) <= 0):
        raise ValueError("t must increase and have the same length as rate.")
    lam = np.empty_like(t)
    lam[0] = lam0
    for i in range(t.size - 1):
        k = p.k_build + p.k_break * r[i]
        eq = p.k_build / k
        lam[i + 1] = eq + (lam[i] - eq) * np.exp(-k * (t[i + 1] - t[i]))
    return {"t": t, "rate": r, "structure": lam, "stress": stress(p, lam, r)}


def hysteresis_loop(p: Params, rate_max: float, t_ramp: float, lam0: float = 1.0, n: int = 400) -> dict:
    """Up-and-down ramp of the shear rate (0 -> rate_max -> 0, each in t_ramp seconds): stresses on the up and
    down curves and the enclosed loop area (Pa/s), a common thixotropy index. The structure equation is
    integrated with the continuously varying rate (stiff solver), so the loop is not distorted by time steps."""
    from scipy import integrate

    def rate(t):
        return rate_max * t / t_ramp if t <= t_ramp else rate_max * (2 * t_ramp - t) / t_ramp

    t = np.linspace(0, 2 * t_ramp, 2 * n + 1)
    def rhs(s_, y):
        return [p.k_build * (1 - y[0]) - p.k_break * rate(s_) * y[0]]

    sol = integrate.solve_ivp(rhs, (0, 2 * t_ramp), [lam0], t_eval=t, method="Radau", rtol=1e-10, atol=1e-12,
                              max_step=t_ramp / 50)
    r = np.array([rate(x) for x in t])
    tau = stress(p, sol.y[0], r)
    up, down = slice(0, n + 1), slice(n, 2 * n + 1)
    r_up, s_up = r[up], tau[up]
    r_dn, s_dn = r[down][::-1], tau[down][::-1]
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return {"rate_up": r_up, "stress_up": s_up, "rate_down": r_dn, "stress_down": s_dn,
            "area": float(trap(s_up, r_up) - trap(s_dn, r_dn))}
