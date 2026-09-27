"""Fitting steady-shear models to flow curves, with parameter uncertainty and model comparison.

Flow curves span decades of shear rate and stress, and rheometer errors are roughly *relative*, so
residuals are taken on the log scale, r = ln(tau_model / tau_measured): every point counts equally,
whatever its magnitude. Standard errors come from the Jacobian at the optimum.

>>> import numpy as np
>>> from engrheo import fitting, models
>>> rate = np.logspace(-1, 3, 20)
>>> fit = fitting.fit_flow_curve(rate, models.stress("power_law", rate, K=3.0, n=0.4), "power_law")
>>> [round(v, 6) for v in fit.params.values()]
[3.0, 0.4]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

from ._common import as_1d, table
from .models import MODELS, YIELD_STRESS_MODELS


@dataclass
class FlowFit:
    model: str
    params: dict
    se: dict
    cov: np.ndarray
    rate: np.ndarray
    stress: np.ndarray
    log_residuals: np.ndarray
    sigma: float            # standard deviation of the log residuals ~ typical relative scatter
    aic: float
    at_bound: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.rate.size

    def predict_stress(self, rate):
        return MODELS[self.model][0](np.asarray(rate, float), *self.params.values())

    def predict_viscosity(self, rate):
        r = np.asarray(rate, float)
        return self.predict_stress(r) / r

    def conf_int(self, conf: float = 0.95) -> dict:
        """Wald confidence intervals (parameter +/- t * SE)."""
        t = stats.t.ppf(0.5 + conf / 2, max(self.n - len(self.params), 1))
        return {k: (v - t * self.se[k], v + t * self.se[k]) for k, v in self.params.items()}

    def __str__(self) -> str:
        ci = self.conf_int()
        rows = [(k, v, self.se[k], ci[k][0], ci[k][1]) for k, v in self.params.items()]
        out = [f"{self.model} fit to {self.n} points: relative scatter {100 * self.sigma:.2f} %, AIC {self.aic:.2f}",
               table(["parameter", "value", "std. error", "95% low", "95% high"], rows, ".5g")]
        if self.at_bound:
            out.append(f"note: {', '.join(self.at_bound)} at a bound - standard errors not meaningful")
        return "\n".join(out)


def _initial_guess(model: str, rate: np.ndarray, tau: np.ndarray) -> list:
    eta = tau / rate
    lr, lt = np.log(rate), np.log(tau)
    n_pl, lnK = np.polyfit(lr, lt, 1)
    K, n = float(np.exp(lnK)), float(np.clip(n_pl, 0.05, 2.0))
    if model == "newtonian":
        return [float(np.median(eta))]
    if model == "power_law":
        return [K, n]
    if model == "sisko":
        return [0.1 * float(eta.min()), K, n]
    if model in ("cross", "carreau", "carreau_yasuda"):
        eta0, eta_inf = float(eta.max()), 0.01 * float(eta.min())
        half = np.nonzero(eta < 0.5 * eta0)[0]
        lam = 1.0 / float(rate[half[0]]) if half.size else 1.0 / float(rate.max())
        slope = np.polyfit(lr[-max(3, rate.size // 3):], np.log(eta[-max(3, rate.size // 3):]), 1)[0]
        n_hi = float(np.clip(1 + slope, 0.05, 0.95))
        return {"cross": [eta0, eta_inf, lam, float(np.clip(1 - n_hi, 0.2, 1.5))],
                "carreau": [eta0, eta_inf, lam, n_hi],
                "carreau_yasuda": [eta0, eta_inf, lam, 2.0, n_hi]}[model]
    if model == "bingham":
        mu_p, ty = np.polyfit(rate, tau, 1)
        return [max(float(ty), 0.1 * float(tau.min())), max(float(mu_p), 1e-12)]
    if model == "herschel_bulkley":
        ty = 0.8 * float(tau.min())
        n2, lnK2 = np.polyfit(lr, np.log(np.maximum(tau - ty, 1e-12 * tau.max())), 1)
        return [ty, float(np.exp(lnK2)), float(np.clip(n2, 0.05, 2.0))]
    if model == "casson":
        b, a = np.polyfit(np.sqrt(rate), np.sqrt(tau), 1)
        return [max(float(a), 1e-6) ** 2, max(float(b), 1e-6) ** 2]
    raise KeyError(model)


def fit_flow_curve(rate, stress=None, model: str = "power_law", viscosity=None, p0=None) -> FlowFit:
    """Fit a steady-shear model (see ``models.MODELS``) to a flow curve given as shear stress or as
    viscosity versus shear rate. Starting values are estimated automatically unless ``p0`` is given."""
    if model not in MODELS:
        raise KeyError(f"Unknown model {model!r}. Available: {', '.join(MODELS)}")
    r = as_1d(rate, "rate", 3)
    if (stress is None) == (viscosity is None):
        raise ValueError("Give either stress or viscosity.")
    tau = as_1d(stress, "stress", 3) if stress is not None else as_1d(viscosity, "viscosity", 3) * r
    if r.size != tau.size:
        raise ValueError("rate and stress/viscosity must have the same length.")
    if np.any(r <= 0) or np.any(tau <= 0):
        raise ValueError("Shear rates and stresses must be positive.")
    fn, names, lo, hi = MODELS[model]
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if r.size <= len(names):
        raise ValueError(f"{model} has {len(names)} parameters; need more than {len(names)} data points.")
    start = np.asarray(p0 if p0 is not None else _initial_guess(model, r, tau), float)
    start = np.clip(start, lo + 1e-12 * np.maximum(1, np.abs(lo)), np.where(np.isfinite(hi), hi - 1e-9, np.inf))

    def resid(p):
        with np.errstate(all="ignore"):
            m = fn(r, *p)
        return np.log(np.where(m > 0, m, 1e-300) / tau)

    sol = optimize.least_squares(resid, start, bounds=(lo, hi), x_scale="jac", ftol=1e-14, xtol=1e-14,
                                 gtol=1e-14, max_nfev=20000)
    p, res = sol.x, sol.fun
    dof = r.size - p.size
    s2 = float(res @ res) / dof
    # "at a bound": within a millionth of the parameter's scale (taken from its starting value) of a bound
    scale = np.maximum(np.abs(start), 1e-300)
    at_bound = [nm for nm, v, a, b, sc in zip(names, p, lo, hi, scale)
                if abs(v - a) < 1e-6 * sc or abs(b - v) < 1e-6 * sc]
    try:
        cov = s2 * np.linalg.inv(sol.jac.T @ sol.jac)
    except np.linalg.LinAlgError:
        cov = np.full((p.size, p.size), np.nan)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    # (floored: an exact fit to noise-free data has a zero residual sum)
    aic = r.size * np.log(max(float(res @ res) / r.size, np.finfo(float).tiny)) + 2 * p.size
    return FlowFit(model, dict(zip(names, map(float, p))), dict(zip(names, map(float, se))), cov, r, tau, res,
                   float(np.sqrt(s2)), float(aic), at_bound)


def compare_models(rate, stress=None, models=("power_law", "carreau", "cross", "sisko"), viscosity=None):
    """Fit several models and rank them by AIC (lower is better; differences below ~2 are not
    meaningful). Returns (fits sorted by AIC, text table)."""
    fits = []
    for m in models:
        try:
            fits.append(fit_flow_curve(rate, stress, m, viscosity=viscosity))
        except (ValueError, RuntimeError) as exc:
            fits.append(None)
            print(f"note: {m} could not be fitted ({exc})")
    fits = sorted((f for f in fits if f is not None), key=lambda f: f.aic)
    best = fits[0].aic
    rows = [(f.model, len(f.params), 100 * f.sigma, f.aic, f.aic - best,
             "yes" if f.model in YIELD_STRESS_MODELS else "no") for f in fits]
    return fits, table(["model", "parameters", "scatter %", "AIC", "delta AIC", "yield stress"], rows, ".4g")
