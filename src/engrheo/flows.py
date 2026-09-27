"""Pipe flow of non-Newtonian fluids: velocity profiles, flow rate, pressure drop, the Metzner-Reed
Reynolds number and friction factors. SI units; friction factors are Fanning factors (f = tau_w / (rho V^2 / 2)),
one quarter of the Darcy factor.

>>> from engrheo import flows
>>> r = flows.pipe_power_law(K=0.5, n=0.6, R=0.025, Q=1e-3)
>>> round(r["dp_per_L"], 1)                      # Pa/m
615.0
"""

from __future__ import annotations

import numpy as np
from scipy import optimize


# ------------------------------------------------------------------ laminar flow: power law
def pipe_power_law(K: float, n: float, R: float, *, dp_per_L: float | None = None, Q: float | None = None) -> dict:
    """Laminar pipe flow of a power-law fluid. Give the pressure gradient ``dp_per_L`` (Pa/m) or the flow
    rate ``Q`` (m3/s). Returns wall stress, flow rate, pressure gradient, mean and centre-line velocity.
    Q = pi R^3 n/(3n + 1) (tau_w/K)^(1/n), tau_w = dp_per_L R / 2."""
    if (dp_per_L is None) == (Q is None):
        raise ValueError("Give either dp_per_L or Q.")
    if Q is not None:
        tau_w = K * ((3 * n + 1) / n * Q / (np.pi * R**3)) ** n
        dp_per_L = 2 * tau_w / R
    tau_w = dp_per_L * R / 2
    Q = np.pi * R**3 * n / (3 * n + 1) * (tau_w / K) ** (1 / n)
    V = Q / (np.pi * R**2)
    return {"tau_w": float(tau_w), "Q": float(Q), "dp_per_L": float(dp_per_L), "V": float(V),
            "u_max": float(V * (3 * n + 1) / (n + 1))}


def velocity_power_law(r, K: float, n: float, R: float, dp_per_L: float):
    """Velocity profile u(r) = n/(n+1) (dp_per_L / (2K))^(1/n) (R^(1+1/n) - r^(1+1/n))."""
    r = np.asarray(r, float)
    return n / (n + 1) * (dp_per_L / (2 * K)) ** (1 / n) * (R ** (1 + 1 / n) - r ** (1 + 1 / n))


# ------------------------------------------------------------------ laminar flow: yield-stress fluids
def flow_rate_hb(tau_w: float, tau_y: float, K: float, n: float, R: float) -> float:
    """Flow rate of a Herschel-Bulkley fluid for wall stress tau_w (Bingham: n = 1, K = plastic viscosity).
    From Q/(pi R^3) = tau_w^-3 * integral of tau^2 rate(tau) over tau_y..tau_w; zero if tau_w <= tau_y."""
    if tau_w <= tau_y:
        return 0.0
    u, m = tau_w - tau_y, 1 / n
    integral = (u ** (3 + m) / (3 + m) + 2 * tau_y * u ** (2 + m) / (2 + m) + tau_y**2 * u ** (1 + m) / (1 + m))
    return float(np.pi * R**3 / tau_w**3 * integral / K**m)


def pipe_hb(tau_y: float, K: float, n: float, R: float, *, dp_per_L: float | None = None,
            Q: float | None = None) -> dict:
    """Laminar pipe flow of a Herschel-Bulkley (or Bingham, n = 1) fluid. Give ``dp_per_L`` or ``Q``.
    Returns wall stress, flow rate, pressure gradient, mean velocity and the radius of the unsheared plug.
    A pressure gradient below 2 tau_y / R cannot start the flow."""
    if (dp_per_L is None) == (Q is None):
        raise ValueError("Give either dp_per_L or Q.")
    if Q is not None:
        if Q <= 0:
            raise ValueError("Q must be positive.")
        hi = max(2 * tau_y, 1.0)
        while flow_rate_hb(hi, tau_y, K, n, R) < Q:
            hi *= 2
        tau_w = optimize.brentq(lambda t: flow_rate_hb(t, tau_y, K, n, R) - Q, tau_y, hi, xtol=1e-14, rtol=1e-14)
        dp_per_L = 2 * tau_w / R
    tau_w = dp_per_L * R / 2
    Q = flow_rate_hb(tau_w, tau_y, K, n, R)
    return {"tau_w": float(tau_w), "Q": float(Q), "dp_per_L": float(dp_per_L), "V": float(Q / (np.pi * R**2)),
            "plug_radius": float(min(R, R * tau_y / tau_w)) if tau_w > 0 else R}


def velocity_hb(r, tau_y: float, K: float, n: float, R: float, dp_per_L: float):
    """Velocity profile of a Herschel-Bulkley fluid: sheared outside the plug radius R tau_y/tau_w,
    rigid (constant velocity) inside it."""
    r = np.asarray(r, float)
    tau_w = dp_per_L * R / 2
    if tau_w <= tau_y:
        return np.zeros_like(r)
    m = 1 + 1 / n
    rp = R * tau_y / tau_w

    def u(s):
        return R / (tau_w * m * K ** (1 / n)) * ((tau_w - tau_y) ** m - np.maximum(tau_w * s / R - tau_y, 0) ** m)
    return np.where(r <= rp, u(rp), u(r))


# ------------------------------------------------------------------ Reynolds numbers and friction
def reynolds_metzner_reed(rho: float, V: float, D: float, K: float, n: float) -> float:
    """Metzner-Reed Reynolds number for power-law fluids, defined so that laminar flow gives f = 16/Re:
    Re = rho V^(2-n) D^n / (8^(n-1) K ((3n+1)/(4n))^n). Reduces to rho V D / mu for n = 1."""
    return float(rho * V ** (2 - n) * D**n / (8 ** (n - 1) * K * ((3 * n + 1) / (4 * n)) ** n))


def critical_reynolds_power_law(n: float) -> float:
    """End of laminar flow for power-law fluids (Ryan and Johnson 1959): 6464 n (2 + n)^((2+n)/(1+n)) / (1 + 3n)^2.
    About 2100 for n = 1; rises to a maximum near n = 0.4."""
    return float(6464 * n * (2 + n) ** ((2 + n) / (1 + n)) / (1 + 3 * n) ** 2)


def fanning_power_law(Re: float, n: float) -> float:
    """Fanning friction factor in smooth pipes: 16/Re in laminar flow (Re below the Ryan-Johnson limit),
    otherwise the Dodge-Metzner correlation 1/sqrt(f) = 4/n^0.75 log10(Re f^(1 - n/2)) - 0.4/n^1.2."""
    if Re <= critical_reynolds_power_law(n):
        return 16.0 / Re

    def g(x):  # x = 1/sqrt(f)
        f = 1 / x**2
        return x - (4 / n**0.75 * np.log10(Re * f ** (1 - n / 2)) - 0.4 / n**1.2)
    return float(1 / optimize.brentq(g, 0.5, 100.0) ** 2)


def pressure_drop_power_law(Q: float, D: float, L: float, rho: float, K: float, n: float) -> dict:
    """Pressure drop for a power-law fluid in a smooth pipe (laminar or turbulent)."""
    V = Q / (np.pi * D**2 / 4)
    Re = reynolds_metzner_reed(rho, V, D, K, n)
    f = fanning_power_law(Re, n)
    regime = "laminar" if Re <= critical_reynolds_power_law(n) else "turbulent"
    return {"dp": float(4 * f * L / D * rho * V**2 / 2), "V": float(V), "Re_MR": Re, "f_fanning": f,
            "regime": regime, "Re_critical": critical_reynolds_power_law(n)}


def hedstrom(rho: float, D: float, tau_y: float, mu_p: float) -> float:
    """Hedstrom number He = rho tau_y D^2 / mu_p^2 for Bingham plastics."""
    return float(rho * tau_y * D**2 / mu_p**2)


def fanning_bingham_laminar(Re: float, He: float) -> float:
    """Laminar Fanning friction factor of a Bingham plastic (Buckingham-Reiner), Re = rho V D / mu_p:
    f = 16/Re (1 + He/(6 Re) - He^4 / (3 f^3 Re^7)), solved for f."""
    g = lambda f: f - 16 / Re * (1 + He / (6 * Re) - He**4 / (3 * f**3 * Re**7))  # noqa: E731
    lo = 16 / Re
    hi = lo * (1 + He / (6 * Re)) * 1.0001 + 1e-12
    return float(optimize.brentq(g, lo * (1 + 1e-12), hi) if He > 0 else lo)


# ------------------------------------------------------------------ processing: slits, dies, mixing
def slit_power_law(K: float, n: float, W: float, H: float, *, dp_per_L: float | None = None,
                   Q: float | None = None) -> dict:
    """Laminar flow of a power-law fluid in a wide slit (width W >> gap H), e.g. a sheet or film die:
    Q = 2 W (dp_per_L / K)^(1/n) (H/2)^(2+1/n) n/(2n+1); wall stress dp_per_L H/2."""
    if (dp_per_L is None) == (Q is None):
        raise ValueError("Give either dp_per_L or Q.")
    h = H / 2
    if Q is not None:
        dp_per_L = K * (Q * (2 * n + 1) / (2 * W * n * h ** (2 + 1 / n))) ** n
    Q = 2 * W * (dp_per_L / K) ** (1 / n) * h ** (2 + 1 / n) * n / (2 * n + 1)
    return {"Q": float(Q), "dp_per_L": float(dp_per_L), "tau_w": float(dp_per_L * h),
            "rate_w": float((dp_per_L * h / K) ** (1 / n))}


def tanner_swell(N1: float, tau_w: float) -> float:
    """Tanner's estimate of extrudate swell from a capillary die:
    D_extrudate / D_die = 0.1 + (1 + (N1/(2 tau_w))^2 / 2)^(1/6), with N1 and tau_w at the wall shear rate."""
    return float(0.1 + (1 + 0.5 * (N1 / (2 * tau_w)) ** 2) ** (1 / 6))


def mixing_power_laminar(N: float, D: float, K: float, n: float, ks: float, Kp: float) -> dict:
    """Laminar mixing power of a power-law fluid by the Metzner-Otto method: the average shear rate in the
    vessel is ks N (ks is impeller-specific, about 10-13 for turbines), the apparent viscosity K (ks N)^(n-1),
    and P = Kp eta_app N^2 D^3 with Kp = Po Re (the impeller's laminar power constant)."""
    rate = ks * N
    eta = K * rate ** (n - 1)
    return {"rate_avg": float(rate), "eta_app": float(eta), "power": float(Kp * eta * N**2 * D**3)}
