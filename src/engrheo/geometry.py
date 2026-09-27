"""Rheometer geometries: from what the instrument measures (torque, rotation speed, pressure drop, flow
rate) to shear stress and shear rate - including the corrections that are needed when the fluid is not
Newtonian or slips at the wall. SI units throughout (N·m, rad/s, m, Pa, m3/s).

The corrections use the local slope n' = d ln(signal) / d ln(rate) of the measured data.

>>> from engrheo import geometry
>>> tau, rate = geometry.cone_plate(torque=1e-3, omega=1.0, R=0.025, theta=0.0349)
>>> round(float(tau), 3), round(float(rate), 2)
(30.558, 28.65)
"""

from __future__ import annotations

import numpy as np

from ._common import as_1d, log_slope


def cone_plate(torque, omega, R: float, theta: float):
    """Cone and plate (small cone angle ``theta`` in rad): the shear rate is uniform, so
    tau = 3M/(2 pi R^3) and rate = omega/theta hold for any fluid. Returns (stress, rate)."""
    M, w = np.asarray(torque, float), np.asarray(omega, float)
    return 3 * M / (2 * np.pi * R**3), w / theta


def parallel_plate(torque, omega, R: float, h: float, correct: bool = True):
    """Parallel plates of radius R and gap h. The shear rate varies from 0 at the centre to
    rate_R = omega R / h at the rim. Apparent (Newtonian) rim stress 2M/(pi R^3); with ``correct`` the
    Weissenberg-Rabinowitsch-type correction tau_R = M/(2 pi R^3) (3 + d ln M / d ln rate_R).
    Returns (rim stress, rim rate, n') - n' is 1 when not corrected."""
    M, w = as_1d(torque, "torque"), as_1d(omega, "omega")
    rate_R = w * R / h
    if not correct:
        return 2 * M / (np.pi * R**3), rate_R, np.ones_like(M)
    n = log_slope(rate_R, M)
    return M / (2 * np.pi * R**3) * (3 + n), rate_R, n


def couette(torque, omega, Ri: float, Ro: float, L: float, correct: bool = True):
    """Concentric cylinders (bob radius Ri, cup radius Ro, immersed length L), stress and rate at the bob.
    tau_bob = M/(2 pi Ri^2 L). Newtonian rate 2 omega Ro^2/(Ro^2 - Ri^2); with ``correct`` the power-law
    (Krieger) correction rate_bob = 2 omega / (n' (1 - (Ri/Ro)^(2/n'))), n' = d ln M / d ln omega.
    Returns (stress, rate, n')."""
    M, w = as_1d(torque, "torque"), as_1d(omega, "omega")
    tau = M / (2 * np.pi * Ri**2 * L)
    if not correct:
        return tau, 2 * w * Ro**2 / (Ro**2 - Ri**2), np.ones_like(M)
    n = log_slope(w, M)
    return tau, 2 * w / (n * (1 - (Ri / Ro) ** (2 / n))), n


def capillary(pressure_drop, flow_rate, R: float, L: float, correct: bool = True):
    """Capillary (tube) rheometer: wall stress tau_w = dP R / (2L), apparent rate 4Q/(pi R^3). With
    ``correct`` the Weissenberg-Rabinowitsch correction rate_w = apparent (3n' + 1)/(4n'),
    n' = d ln tau_w / d ln apparent. The pressure drop must already be corrected for entrance effects
    (see ``bagley``). Returns (wall stress, wall rate, n')."""
    dp, Q = as_1d(pressure_drop, "pressure_drop"), as_1d(flow_rate, "flow_rate")
    tau_w = dp * R / (2 * L)
    app = 4 * Q / (np.pi * R**3)
    if not correct:
        return tau_w, app, np.ones_like(dp)
    n = log_slope(app, tau_w)
    return tau_w, app * (3 * n + 1) / (4 * n), n


def bagley(L_over_R, pressure_drop) -> dict:
    """Bagley correction for entrance and exit pressure losses: at one apparent shear rate, measure dP
    with dies of different L/R. Then dP = 2 tau_w (L/R) + dP_ends; a straight-line fit gives the true
    wall stress (half the slope), the end loss dP_ends and the Bagley end correction e = dP_ends/(2 tau_w)."""
    x, y = as_1d(L_over_R, "L_over_R", 2), as_1d(pressure_drop, "pressure_drop", 2)
    slope, intercept = np.polyfit(x, y, 1)
    return {"tau_w": float(slope / 2), "dp_ends": float(intercept), "e": float(intercept / slope)}


def mooney(size, apparent_rate, kind: str = "capillary") -> dict:
    """Mooney wall-slip analysis: at one wall stress, measure the apparent shear rate with different
    geometry sizes. With slip velocity u_s at each wall:
    capillaries of radius R: apparent = true + 4 u_s / R;  parallel plates with gap h: apparent = true + 2 u_s / h.
    A straight line against 1/size gives the slip-free rate (intercept) and u_s (slope / 4 or / 2)."""
    if kind not in ("capillary", "plates"):
        raise ValueError("kind must be 'capillary' or 'plates'.")
    d, a = as_1d(size, "size", 2), as_1d(apparent_rate, "apparent_rate", 2)
    slope, intercept = np.polyfit(1 / d, a, 1)
    return {"true_rate": float(intercept), "slip_velocity": float(slope / (4 if kind == "capillary" else 2))}


def couette_speed(model: str, params, torque, Ri: float, Ro: float, L: float):
    """Rotation speed (rad/s) of an ideal Couette cell for a fluid obeying ``model`` (see engrheo.models)
    at the given torque: omega = integral over the gap of rate(tau(r)) / r dr, tau(r) = M / (2 pi r^2 L).
    Handles yield-stress fluids whose outer layers do not flow (partially sheared gap). Useful to
    simulate instruments and to test the conversion formulas."""
    from scipy import integrate

    from .models import rate_from_stress

    def one(M):
        f = lambda r: rate_from_stress(model, M / (2 * np.pi * r**2 * L), *params) / r  # noqa: E731
        return integrate.quad(f, Ri, Ro, epsabs=0, epsrel=1e-12, limit=200)[0]
    M = np.atleast_1d(np.asarray(torque, float))
    out = np.array([one(m) for m in M])
    return out if np.ndim(torque) else float(out[0])


# ------------------------------------------------------------------ Fann 35 (oilfield) viscometer
FANN_RATE_PER_RPM = 1.7023      # 1/s per rpm (Newtonian rate at the bob; R1 bob 1.7245 cm, R2 rotor 1.8415 cm)
FANN_PA_PER_DEGREE = 0.5113     # Pa per dial degree: 1.0678 lbf/100 ft2 x 0.47880 Pa (standard B1 bob, F1 spring)


def fann35(rpm, dial):
    """Convert Fann 35 (API RP 13B) viscometer readings to (shear stress in Pa, shear rate in 1/s) with the
    standard factors. The rate is the *Newtonian* rate at the bob, as in field practice."""
    return FANN_PA_PER_DEGREE * np.asarray(dial, float), FANN_RATE_PER_RPM * np.asarray(rpm, float)


def api_bingham(theta600: float, theta300: float) -> dict:
    """API field formulas from the 600 and 300 rpm dial readings: plastic viscosity PV = theta600 - theta300
    (mPa·s) and yield point YP = theta300 - PV (lbf/100 ft2), also returned in SI (Pa·s, Pa)."""
    pv = theta600 - theta300
    yp = theta300 - pv
    return {"PV_mPas": float(pv), "YP_lbf100ft2": float(yp), "mu_p": pv / 1000.0, "tau_y": yp * 0.4788}


def api_power_law(theta600: float, theta300: float) -> dict:
    """API power-law parameters from two readings: n = 3.32 log10(theta600/theta300),
    K = 0.511 theta300 / 511^n (Pa·s^n)."""
    n = 3.32 * np.log10(theta600 / theta300)
    return {"n": float(n), "K": float(FANN_PA_PER_DEGREE * theta300 / 511.0**n)}
