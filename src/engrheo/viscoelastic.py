"""Linear viscoelasticity: mechanical models, discrete relaxation spectra, Boltzmann superposition,
interconversion between relaxation and creep, and spectrum fitting.

A discrete (generalised Maxwell) spectrum is a set of moduli g_i (Pa) and relaxation times tau_i (s),
plus an optional equilibrium modulus Ge (Pa) for solids:
    G(t) = Ge + sum g_i exp(-t / tau_i)
    G'(w) = Ge + sum g_i (w tau_i)^2 / (1 + (w tau_i)^2),   G''(w) = sum g_i w tau_i / (1 + (w tau_i)^2)

>>> import numpy as np
>>> from engrheo import viscoelastic as ve
>>> Gp, Gpp = ve.moduli([1000.0], [0.1], omega=10.0)          # a Maxwell element at w tau = 1
>>> float(Gp), float(Gpp)
(500.0, 500.0)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from ._common import as_1d


# ------------------------------------------------------------------ discrete spectra
def _spectrum(g, tau):
    g, tau = np.atleast_1d(np.asarray(g, float)), np.atleast_1d(np.asarray(tau, float))
    if g.shape != tau.shape:
        raise ValueError("g and tau must have the same length.")
    if np.any(tau <= 0):
        raise ValueError("Relaxation times must be positive.")
    return g, tau


def relaxation_modulus(g, tau, t, Ge: float = 0.0):
    """G(t) = Ge + sum g_i exp(-t/tau_i)."""
    g, tau = _spectrum(g, tau)
    t = np.asarray(t, float)
    return Ge + np.sum(g * np.exp(-t[..., None] / tau), axis=-1)


def moduli(g, tau, omega, Ge: float = 0.0):
    """Storage and loss moduli (G', G'') of a discrete spectrum at angular frequency omega (rad/s)."""
    g, tau = _spectrum(g, tau)
    wt = np.asarray(omega, float)[..., None] * tau
    return Ge + np.sum(g * wt**2 / (1 + wt**2), axis=-1), np.sum(g * wt / (1 + wt**2), axis=-1)


def zero_shear_viscosity(g, tau) -> float:
    """eta0 = sum g_i tau_i (liquids)."""
    g, tau = _spectrum(g, tau)
    return float(np.sum(g * tau))


def steady_state_compliance(g, tau) -> float:
    """Je0 = sum g_i tau_i^2 / (sum g_i tau_i)^2 - the recoverable (elastic) part of creep at long times."""
    g, tau = _spectrum(g, tau)
    return float(np.sum(g * tau**2) / np.sum(g * tau) ** 2)


# ------------------------------------------------------------------ mechanical models
def maxwell(G: float, eta: float) -> dict:
    """Spring G in series with dashpot eta (a viscoelastic liquid). Relaxation time tau = eta/G.
    Returns closed-form functions: relaxation G(t), creep J(t) = 1/G + t/eta, and moduli(omega)."""
    tau = eta / G
    return {"tau": tau,
            "relaxation": lambda t: G * np.exp(-np.asarray(t, float) / tau),
            "creep": lambda t: 1 / G + np.asarray(t, float) / eta,
            "moduli": lambda w: moduli([G], [tau], w)}


def kelvin_voigt(G: float, eta: float) -> dict:
    """Spring G parallel to dashpot eta (a viscoelastic solid): creep J(t) = (1 - exp(-t G/eta))/G,
    G' = G, G'' = eta omega (it cannot relax a step strain)."""
    return {"tau": eta / G,
            "creep": lambda t: (1 - np.exp(-np.asarray(t, float) * G / eta)) / G,
            "moduli": lambda w: (G * np.ones_like(np.asarray(w, float)), eta * np.asarray(w, float))}


def standard_linear_solid(Ge: float, G1: float, tau1: float) -> dict:
    """Zener model: equilibrium spring Ge parallel to a Maxwell element (G1, tau1). Relaxes from Ge + G1 to Ge;
    creep J(t) = 1/Ge - G1/(Ge (Ge + G1)) exp(-t/tau_c) with retardation time tau_c = tau1 (Ge + G1)/Ge."""
    tau_c = tau1 * (Ge + G1) / Ge
    return {"tau": tau1, "tau_retardation": tau_c,
            "relaxation": lambda t: relaxation_modulus([G1], [tau1], t, Ge),
            "creep": lambda t: 1 / Ge - G1 / (Ge * (Ge + G1)) * np.exp(-np.asarray(t, float) / tau_c),
            "moduli": lambda w: moduli([G1], [tau1], w, Ge)}


def burgers_creep(t, G1: float, eta1: float, G2: float, eta2: float):
    """Burgers model (Maxwell in series with Kelvin-Voigt), the classic creep model of doughs, asphalt and
    polymers: J(t) = 1/G1 + t/eta1 + (1 - exp(-t G2/eta2))/G2."""
    t = np.asarray(t, float)
    return 1 / G1 + t / eta1 + (1 - np.exp(-t * G2 / eta2)) / G2


# ------------------------------------------------------------------ superposition and interconversion
def boltzmann_stress(G_of_t, t, strain):
    """Stress from an arbitrary strain history by Boltzmann superposition,
    sigma(t_k) = sum over steps of G(t_k - t_j) * (strain increment at t_j), with the increments of the given
    (sampled) strain history taken at mid-intervals. ``G_of_t`` is a function; ``t`` must start at the
    moment the strain starts to change (strain[0] is applied as an initial step)."""
    t, e = as_1d(t, "t", 2), as_1d(strain, "strain", 2)
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be increasing.")
    de = np.r_[e[0], np.diff(e)]
    tm = np.r_[t[0], 0.5 * (t[1:] + t[:-1])]                 # increments act at mid-interval
    out = np.empty_like(t)
    for k in range(t.size):
        out[k] = np.sum(G_of_t(np.maximum(t[k] - tm[: k + 1], 0.0)) * de[: k + 1])
    return out


def _cumulative_integral(G_of_t, x_max: float, x_min: float):
    """I(x) = integral_0^x G(u) du for a generic relaxation function, tabulated on a fine log grid."""
    u = np.r_[0.0, np.logspace(np.log10(x_min) - 2, np.log10(x_max), 20001)]
    Gu = G_of_t(u)
    cum = np.r_[0.0, np.cumsum(0.5 * (Gu[1:] + Gu[:-1]) * np.diff(u))]
    return lambda x: np.interp(x, u, cum)


def creep_from_relaxation(G_of_t, t, G_integral=None):
    """Creep compliance J(t) from the relaxation modulus by solving the interconversion equation
    J(0) G(t) + integral_0^t G(t - s) J'(s) ds = 1 (response to a unit step stress) on the time grid t (t[0] = 0).

    Product integration: J is linear within each interval, so each interval contributes its increment of J
    times the *exact mean* of G over the interval, obtained from the cumulative integral
    I(x) = integral_0^x G(u) du. Give ``G_integral`` (a function I(x)) when it is known in closed form - see
    ``creep_from_spectrum`` - otherwise it is computed numerically. Resolving the mean exactly matters for
    real spectra, whose fast modes make G spike at short times."""
    t = as_1d(t, "t", 2)
    if t[0] != 0 or np.any(np.diff(t) <= 0):
        raise ValueError("t must start at 0 and increase.")
    cum = G_integral if G_integral is not None else _cumulative_integral(G_of_t, t[-1], np.min(np.diff(t)))
    J = np.empty_like(t)
    J[0] = 1.0 / G_of_t(0.0)
    for k in range(1, t.size):
        a, b = t[:k], t[1 : k + 1]
        Gbar = (cum(t[k] - a) - cum(t[k] - b)) / (b - a)           # exact mean of G(t_k - s) over each interval
        acc = J[0] * G_of_t(t[k]) + np.sum(Gbar[:-1] * np.diff(J[:k]))
        J[k] = J[k - 1] + (1.0 - acc) / Gbar[-1]
    return J


def creep_from_spectrum(g, tau, t, Ge: float = 0.0):
    """Creep compliance of a discrete relaxation spectrum (uses the exact integral of G)."""
    g, tau = _spectrum(g, tau)

    def integral(x):                  # exact integral of G from 0 to x
        x = np.asarray(x, float)
        return Ge * x + np.sum(g * tau * (1 - np.exp(-x[..., None] / tau)), axis=-1)

    return creep_from_relaxation(lambda s: relaxation_modulus(g, tau, s, Ge), t, integral)


# ------------------------------------------------------------------ spectrum fitting
@dataclass
class SpectrumFit:
    g: np.ndarray
    tau: np.ndarray
    Ge: float
    rel_rms: float          # root-mean-square relative misfit of G' and G''
    regularization: float

    def moduli(self, omega):
        return moduli(self.g, self.tau, omega, self.Ge)

    def relaxation(self, t):
        return relaxation_modulus(self.g, self.tau, t, self.Ge)

    @property
    def eta0(self) -> float:
        return zero_shear_viscosity(self.g, self.tau)


def fit_spectrum(omega, Gp, Gpp, n_modes: int | None = None, tau=None, regularization: float = 0.0,
                 solid: bool = False) -> SpectrumFit:
    """Fit a discrete relaxation spectrum to G'(omega), G''(omega) data.

    Relaxation times are fixed on a log grid spanning the data (``n_modes``, default 5 per decade of
    frequency, one decade beyond each end) or given as ``tau``; the moduli g_i >= 0 then follow from a linear
    non-negative least-squares problem on relative residuals. ``regularization`` > 0 adds Tikhonov
    smoothing of ln-spaced neighbouring g_i - the spectrum is an ill-posed inverse problem, and without it
    many very different spectra fit the same data equally well. ``solid=True`` adds an equilibrium modulus."""
    w, gp, gpp = as_1d(omega, "omega", 3), as_1d(Gp, "Gp", 3), as_1d(Gpp, "Gpp", 3)
    if not (w.size == gp.size == gpp.size):
        raise ValueError("omega, Gp and Gpp must have the same length.")
    if tau is None:
        lo, hi = np.log10(1 / w.max()) - 1, np.log10(1 / w.min()) + 1
        n_modes = n_modes or int(np.ceil(5 * (hi - lo)))
        tau = np.logspace(lo, hi, n_modes)
    tau = np.asarray(tau, float)
    # unknowns y_i = g_i / G_ref (dimensionless, order one); rows are relative residuals of G' and G''
    G_ref = float(np.median(np.hypot(gp, gpp)))
    wt = w[:, None] * tau
    A = G_ref * np.vstack([wt**2 / (1 + wt**2) / gp[:, None], wt / (1 + wt**2) / gpp[:, None]])
    b = np.ones(2 * w.size)
    if solid:
        A = np.hstack([A, G_ref * np.r_[1 / gp, np.zeros(w.size)][:, None]])
    if regularization > 0 and tau.size > 2:
        # Tikhonov: penalise the curvature of the (dimensionless) spectrum over the log-spaced times
        D = np.diff(np.eye(tau.size), 2, axis=0) * regularization * np.sqrt(2 * w.size)
        if solid:
            D = np.hstack([D, np.zeros((D.shape[0], 1))])
        A, b = np.vstack([A, D]), np.r_[b, np.zeros(D.shape[0])]
    y, _ = optimize.nnls(A, b, maxiter=100 * A.shape[1])
    x = y * G_ref
    g, Ge = (x[:-1], float(x[-1])) if solid else (x, 0.0)
    fp, fpp = moduli(g, tau, w, Ge)
    rel = np.r_[fp / gp - 1, fpp / gpp - 1]
    return SpectrumFit(g, tau, Ge, float(np.sqrt(np.mean(rel**2))), float(regularization))
