"""Steady-shear (generalised Newtonian) models: shear stress as a function of shear rate.

All functions take the shear rate (1/s, positive) and return the shear stress (Pa); ``viscosity``
divides by the rate. Parameters are in SI units (Pa, Pa·s, Pa·s^n, s).

>>> from engrheo import models
>>> round(float(models.stress("herschel_bulkley", 10.0, tau_y=5.0, K=2.0, n=0.5)), 4)
11.3246
>>> round(float(models.viscosity("carreau", 1e-6, eta0=10.0, eta_inf=0.0, lam=1.0, n=0.5)), 6)
10.0
"""

from __future__ import annotations

import numpy as np


def newtonian(rate, mu):
    """tau = mu * rate."""
    return mu * rate


def power_law(rate, K, n):
    """Ostwald-de Waele: tau = K rate^n (n < 1 shear-thinning, n > 1 shear-thickening)."""
    return K * rate**n


def cross(rate, eta0, eta_inf, lam, m):
    """Cross: eta = eta_inf + (eta0 - eta_inf) / (1 + (lam rate)^m)."""
    return (eta_inf + (eta0 - eta_inf) / (1 + (lam * rate) ** m)) * rate


def carreau(rate, eta0, eta_inf, lam, n):
    """Carreau: eta = eta_inf + (eta0 - eta_inf) [1 + (lam rate)^2]^((n - 1)/2)."""
    return (eta_inf + (eta0 - eta_inf) * (1 + (lam * rate) ** 2) ** ((n - 1) / 2)) * rate


def carreau_yasuda(rate, eta0, eta_inf, lam, a, n):
    """Carreau-Yasuda: eta = eta_inf + (eta0 - eta_inf) [1 + (lam rate)^a]^((n - 1)/a)."""
    return (eta_inf + (eta0 - eta_inf) * (1 + (lam * rate) ** a) ** ((n - 1) / a)) * rate


def sisko(rate, eta_inf, K, n):
    """Sisko: eta = eta_inf + K rate^(n - 1) (power law plus a high-rate Newtonian limit)."""
    return (eta_inf + K * rate ** (n - 1)) * rate


def bingham(rate, tau_y, mu_p):
    """Bingham plastic (flowing region): tau = tau_y + mu_p rate."""
    return tau_y + mu_p * rate


def herschel_bulkley(rate, tau_y, K, n):
    """Herschel-Bulkley (flowing region): tau = tau_y + K rate^n."""
    return tau_y + K * rate**n


def casson(rate, tau_y, eta_c):
    """Casson (flowing region): sqrt(tau) = sqrt(tau_y) + sqrt(eta_c rate)."""
    return (np.sqrt(tau_y) + np.sqrt(eta_c * rate)) ** 2


# name -> (function, parameter names, lower bounds, upper bounds)
MODELS = {
    "newtonian": (newtonian, ("mu",), (0,), (np.inf,)),
    "power_law": (power_law, ("K", "n"), (0, 0.01), (np.inf, 3.0)),
    "cross": (cross, ("eta0", "eta_inf", "lam", "m"), (0, 0, 0, 0.05), (np.inf, np.inf, np.inf, 3.0)),
    "carreau": (carreau, ("eta0", "eta_inf", "lam", "n"), (0, 0, 0, 0.01), (np.inf, np.inf, np.inf, 1.5)),
    "carreau_yasuda": (carreau_yasuda, ("eta0", "eta_inf", "lam", "a", "n"), (0, 0, 0, 0.1, 0.01),
                       (np.inf, np.inf, np.inf, 10.0, 1.5)),
    "sisko": (sisko, ("eta_inf", "K", "n"), (0, 0, 0.01), (np.inf, np.inf, 1.5)),
    "bingham": (bingham, ("tau_y", "mu_p"), (0, 0), (np.inf, np.inf)),
    "herschel_bulkley": (herschel_bulkley, ("tau_y", "K", "n"), (0, 0, 0.01), (np.inf, np.inf, 3.0)),
    "casson": (casson, ("tau_y", "eta_c"), (0, 0), (np.inf, np.inf)),
}
YIELD_STRESS_MODELS = ("bingham", "herschel_bulkley", "casson")


def _model(name: str):
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}. Available: {', '.join(MODELS)}")
    return MODELS[name]


def parameter_names(name: str) -> tuple[str, ...]:
    """Names of a model's parameters, in order."""
    return _model(name)[1]


def stress(name: str, rate, *args, **kwargs):
    """Shear stress of model ``name`` at shear rate(s) ``rate`` (parameters by position or name)."""
    fn, names, *_ = _model(name)
    if kwargs:
        if args:
            raise TypeError("Give the parameters either by position or by name.")
        missing = set(names) - set(kwargs)
        if missing:
            raise TypeError(f"Missing parameter(s) for {name}: {', '.join(sorted(missing))}")
        args = tuple(kwargs[k] for k in names)
    if len(args) != len(names):
        raise TypeError(f"{name} needs {len(names)} parameters {names}, got {len(args)}.")
    r = np.asarray(rate, float)
    if np.any(r <= 0):
        raise ValueError("Shear rates must be positive.")
    return fn(r, *args)


def viscosity(name: str, rate, *args, **kwargs):
    """Apparent viscosity tau / rate of model ``name`` (Pa·s)."""
    return stress(name, rate, *args, **kwargs) / np.asarray(rate, float)


def rate_from_stress(name: str, tau, *args):
    """Inverse of ``stress``: the shear rate at which model ``name`` carries shear stress ``tau``
    (zero below the yield stress of yield-stress models). Closed form where available, otherwise a
    root search (the stress of every model increases with rate)."""
    from scipy import optimize

    t = np.atleast_1d(np.asarray(tau, float))
    fn, names, *_ = _model(name)
    if len(args) != len(names):
        raise TypeError(f"{name} needs {len(names)} parameters {names}, got {len(args)}.")
    if name == "newtonian":
        out = t / args[0]
    elif name == "power_law":
        out = (np.maximum(t, 0) / args[0]) ** (1 / args[1])
    elif name == "bingham":
        out = np.maximum(t - args[0], 0) / args[1]
    elif name == "herschel_bulkley":
        out = (np.maximum(t - args[0], 0) / args[1]) ** (1 / args[2])
    elif name == "casson":
        out = np.maximum(np.sqrt(np.maximum(t, 0)) - np.sqrt(args[0]), 0) ** 2 / args[1]
    else:
        def one(ti):
            if ti <= 0:
                return 0.0
            hi = 1.0
            while fn(hi, *args) < ti:
                hi *= 10
            lo = np.finfo(float).tiny     # just above zero: some models (e.g. Sisko) are singular at rate 0
            return optimize.brentq(lambda g: fn(g, *args) - ti, lo, hi, xtol=1e-300, rtol=1e-13)
        out = np.array([one(ti) for ti in t])
    return out if np.ndim(tau) else float(out[0])
