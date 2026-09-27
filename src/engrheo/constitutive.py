"""Nonlinear viscoelastic constitutive models (differential form), solved as ODEs in the standard
rheometric flows: start-up and steady shear, uniaxial extension, and large-amplitude oscillatory shear.

Each mode has a polymer viscosity eta_p (Pa s) and relaxation time lam (s); models:
  "ucm"       upper-convected Maxwell (Oldroyd-B when a solvent viscosity eta_s is added)
  "giesekus"  quadratic stress term with mobility alpha (0 <= alpha <= 0.5): shear thinning, N2 < 0
  "ptt"       linear Phan-Thien-Tanner with parameter eps (xi = 0): shear thinning, bounded extension
Several modes (e.g. from a relaxation spectrum, eta_p,i = g_i tau_i) are summed.

Simple shear v_x = rate(t) y: the polymer stress components T = (Txx, Tyy, Txy) of each mode obey
  lam dTxx/dt = -f Txx + 2 lam rate Txy - (alpha lam/eta_p)(Txx^2 + Txy^2)
  lam dTxy/dt = -f Txy + lam rate Tyy + eta_p rate - (alpha lam/eta_p) Txy (Txx + Tyy)
  lam dTyy/dt = -f Tyy - (alpha lam/eta_p)(Txy^2 + Tyy^2)
with f = 1 (UCM, Giesekus) or f = 1 + (eps lam/eta_p)(Txx + Tyy + Tzz) (PTT; Tzz = 0 in shear).

>>> import numpy as np
>>> from engrheo import constitutive as cm
>>> r = cm.startup_shear([cm.Mode(eta_p=100.0, lam=1.0)], rate=1.0, t=np.array([0.0, 50.0]))
>>> round(float(r["stress"][-1]), 6), round(float(r["N1"][-1]), 6)       # UCM steady state: eta rate, 2 eta lam rate^2
(100.0, 200.0)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate


@dataclass
class Mode:
    eta_p: float
    lam: float
    model: str = "ucm"
    alpha: float = 0.0      # Giesekus mobility
    eps: float = 0.0        # PTT extensibility parameter

    def __post_init__(self):
        if self.model not in ("ucm", "giesekus", "ptt"):
            raise ValueError("model must be 'ucm', 'giesekus' or 'ptt'.")
        if self.eta_p <= 0 or self.lam <= 0:
            raise ValueError("eta_p and lam must be positive.")
        if not 0 <= self.alpha <= 0.5:
            raise ValueError("Giesekus alpha must lie between 0 and 0.5.")


def modes_from_spectrum(g, tau, model: str = "ucm", **params) -> list[Mode]:
    """Modes from a discrete relaxation spectrum (eta_p,i = g_i tau_i), all with the same nonlinear model."""
    return [Mode(float(gi * ti), float(ti), model, **params) for gi, ti in zip(g, tau) if gi > 0]


def _shear_rhs(modes, rate_fn):
    def rhs(t, y):
        out = np.empty_like(y)
        rate = rate_fn(t)
        for k, m in enumerate(modes):
            xx, yy, xy = y[3 * k : 3 * k + 3]
            f = 1.0 + (m.eps * m.lam / m.eta_p) * (xx + yy) if m.model == "ptt" else 1.0
            a = m.alpha * m.lam / m.eta_p if m.model == "giesekus" else 0.0
            out[3 * k] = (-f * xx + 2 * m.lam * rate * xy - a * (xx * xx + xy * xy)) / m.lam
            out[3 * k + 1] = (-f * yy - a * (xy * xy + yy * yy)) / m.lam
            out[3 * k + 2] = (-f * xy + m.lam * rate * yy + m.eta_p * rate - a * xy * (xx + yy)) / m.lam
        return out
    return rhs


def _solve(rhs, n_state, t, max_step=np.inf):
    t = np.asarray(t, float)
    sol = integrate.solve_ivp(rhs, (t[0], t[-1]), np.zeros(n_state), t_eval=t, method="LSODA",
                              rtol=1e-10, atol=1e-12, max_step=max_step)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y


def startup_shear(modes, rate: float, t, eta_s: float = 0.0) -> dict:
    """Start-up of steady shear at constant ``rate`` from rest: shear stress, first and second normal-stress
    differences N1 = Txx - Tyy, N2 = Tyy - Tzz, and the transient viscosity eta+ = stress / rate."""
    modes = list(modes)
    y = _solve(_shear_rhs(modes, lambda _t: rate), 3 * len(modes), t)
    xx, yy, xy = (y[i::3].sum(axis=0) for i in range(3))
    stress = xy + eta_s * rate
    return {"t": np.asarray(t, float), "stress": stress, "N1": xx - yy, "N2": yy, "eta_plus": stress / rate}


def steady_shear(modes, rates, eta_s: float = 0.0, t_factor: float = 60.0) -> dict:
    """Steady shear viscosity and normal-stress differences, as the long-time limit of start-up flow
    (integrated to ``t_factor`` times the longest relaxation time)."""
    modes = list(modes)
    t_end = t_factor * max(m.lam for m in modes)
    out = {"rate": np.asarray(rates, float), "eta": [], "N1": [], "N2": []}
    for r in np.atleast_1d(rates):
        res = startup_shear(modes, float(r), np.array([0.0, t_end]), eta_s)
        out["eta"].append(res["eta_plus"][-1])
        out["N1"].append(res["N1"][-1])
        out["N2"].append(res["N2"][-1])
    return {k: np.asarray(v, float) for k, v in out.items()}


def startup_extension(modes, rate: float, t, eta_s: float = 0.0) -> dict:
    """Start-up of uniaxial extension at constant Hencky strain rate: tensile stress difference
    Tzz - Trr and the transient extensional viscosity eta_E+ = (Tzz - Trr)/rate (+ 3 eta_s)."""
    modes = list(modes)

    def rhs(_t, y):
        out = np.empty_like(y)
        for k, m in enumerate(modes):
            zz, rr = y[2 * k : 2 * k + 2]
            f = 1.0 + (m.eps * m.lam / m.eta_p) * (zz + 2 * rr) if m.model == "ptt" else 1.0
            a = m.alpha * m.lam / m.eta_p if m.model == "giesekus" else 0.0
            out[2 * k] = (-f * zz + 2 * m.lam * rate * zz + 2 * m.eta_p * rate - a * zz * zz) / m.lam
            out[2 * k + 1] = (-f * rr - m.lam * rate * rr - m.eta_p * rate - a * rr * rr) / m.lam
        return out

    y = _solve(rhs, 2 * len(modes), t)
    diff = y[0::2].sum(axis=0) - y[1::2].sum(axis=0) + 3 * eta_s * rate
    return {"t": np.asarray(t, float), "tensile_stress": diff, "eta_E_plus": diff / rate}


def laos_shear(modes, strain_amplitude: float, omega: float, n_cycles: int = 10, points_per_cycle: int = 256,
               eta_s: float = 0.0) -> dict:
    """Oscillatory shear with strain strain_amplitude * sin(omega t): returns time, strain and shear stress
    of the last cycle (after the start-up transient has decayed; use enough cycles for the longest mode)."""
    modes = list(modes)
    period = 2 * np.pi / omega
    t = np.linspace(0, n_cycles * period, n_cycles * points_per_cycle + 1)
    rate_fn = lambda s: strain_amplitude * omega * np.cos(omega * s)  # noqa: E731
    y = _solve(_shear_rhs(modes, rate_fn), 3 * len(modes), t, max_step=period / 50)
    last = slice(-points_per_cycle - 1, None)
    tl = t[last]
    stress = y[2::3].sum(axis=0)[last] + eta_s * rate_fn(tl)
    return {"t": tl - tl[0], "strain": strain_amplitude * np.sin(omega * tl), "stress": stress}
