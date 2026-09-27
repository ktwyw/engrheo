"""Validation of engrheo against exact solutions and independent calculations.

Run:  python docs/validate.py      (writes docs/VALIDATION.md)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import integrate, optimize

from engrheo import fitting, flows, geometry, models

RESULTS: list[tuple] = []
rng = np.random.default_rng(2026)


def check(section, name, reference, value, expected, rtol=0.0, atol=0.0):
    ok = bool(np.isclose(value, expected, rtol=rtol, atol=atol))
    RESULTS.append((section, name, reference, float(value), float(expected), ok))
    if not ok:
        print(f"FAIL: {section} | {name}: computed {value!r}, expected {expected!r}")


# ------------------------------------------------------------------ 1. models
S = "Steady-shear models"
rate = np.logspace(-3, 4, 50)
check(S, "Herschel-Bulkley with tau_y = 0 equals power law (max rel. diff)", "exact",
      np.max(np.abs(models.stress("herschel_bulkley", rate, 0.0, 2.0, 0.6) / models.stress("power_law", rate, 2.0, 0.6) - 1)), 0, atol=1e-14)
check(S, "Herschel-Bulkley with n = 1 equals Bingham", "exact",
      np.max(np.abs(models.stress("herschel_bulkley", rate, 3.0, 0.2, 1.0) / models.stress("bingham", rate, 3.0, 0.2) - 1)), 0, atol=1e-14)
check(S, "Carreau-Yasuda with a = 2 equals Carreau", "exact",
      np.max(np.abs(models.stress("carreau_yasuda", rate, 5.0, 0.01, 2.0, 2.0, 0.4) / models.stress("carreau", rate, 5.0, 0.01, 2.0, 0.4) - 1)), 0, atol=1e-14)
check(S, "Carreau viscosity at very low rate -> eta0", "limit", models.viscosity("carreau", 1e-9, 5.0, 0.0, 2.0, 0.4), 5.0, rtol=1e-12)
hi = np.array([1e6, 1e7])
eta_hi = models.viscosity("carreau", hi, 5.0, 0.0, 2.0, 0.4)
check(S, "Carreau high-rate slope d ln(eta)/d ln(rate) -> n - 1", "limit", np.log(eta_hi[1] / eta_hi[0]) / np.log(10), -0.6, atol=1e-9)
check(S, "Casson: sqrt(stress) linear in sqrt(rate) (spread of the slope)", "definition",
      np.ptp(np.diff(np.sqrt(models.stress("casson", rate, 4.0, 0.05))) / np.diff(np.sqrt(rate))), 0, atol=1e-12)

# ------------------------------------------------------------------ 2. fitting
S = "Flow-curve fitting"
truth = {"newtonian": (0.8,), "power_law": (3.0, 0.4), "cross": (12.0, 0.02, 3.0, 0.8), "carreau": (8.0, 0.01, 2.0, 0.35),
         "carreau_yasuda": (8.0, 0.01, 2.0, 1.2, 0.35), "sisko": (0.02, 5.0, 0.3), "bingham": (4.0, 0.05),
         "herschel_bulkley": (6.0, 1.5, 0.5), "casson": (3.0, 0.02)}
r_fit = np.logspace(-2, 3, 30)
for name, p in truth.items():
    fit = fitting.fit_flow_curve(r_fit, models.stress(name, r_fit, *p), name)
    check(S, f"{name}: parameters recovered from exact data (max rel. error, automatic start)", "exact",
          np.max(np.abs(np.array(list(fit.params.values())) / np.array(p) - 1)), 0, atol=1e-6)
noisy = models.stress("power_law", r_fit, 3.0, 0.4) * np.exp(rng.normal(0, 0.05, r_fit.size))
pl = fitting.fit_flow_curve(r_fit, noisy, "power_law")
b, a = np.polyfit(np.log(r_fit), np.log(noisy), 1)
check(S, "power law fitted in log space = straight line in log-log coordinates: n", "numpy.polyfit", pl.params["n"], b, rtol=1e-8)
check(S, "... K", "numpy.polyfit", pl.params["K"], np.exp(a), rtol=1e-8)
car_noisy = models.stress("carreau", r_fit, *truth["carreau"]) * np.exp(rng.normal(0, 0.03, r_fit.size))
car = fitting.fit_flow_curve(r_fit, car_noisy, "carreau")
logmodel = lambda x, e0, ei, lam, n: np.log(models.carreau(x, e0, ei, lam, n))  # noqa: E731
popt, pcov = optimize.curve_fit(logmodel, r_fit, np.log(car_noisy), p0=list(car.params.values()), maxfev=20000)
check(S, "Carreau (noisy data): parameters vs scipy curve_fit (max rel. diff)", "scipy curve_fit",
      np.max(np.abs(np.array(list(car.params.values())) / popt - 1)), 0, atol=1e-5)
check(S, "Carreau (noisy data): standard error of n vs curve_fit", "scipy curve_fit", car.se["n"], np.sqrt(pcov[3, 3]), rtol=1e-3)
hits = 0
for _ in range(2000):
    y = models.stress("power_law", r_fit, 3.0, 0.4) * np.exp(rng.normal(0, 0.05, r_fit.size))
    lo, hi_ = fitting.fit_flow_curve(r_fit, y, "power_law").conf_int()["n"]
    hits += lo <= 0.4 <= hi_
check(S, "95 % confidence interval for n: simulated coverage (2000 data sets)", "nominal 0.95", hits / 2000, 0.95, atol=0.015)

# ------------------------------------------------------------------ 3. rheometer geometries
S = "Rheometer geometries and corrections"
K, n = 2.0, 0.45
R, h = 0.025, 1e-3
omega = np.logspace(-2, 2, 25)
rate_R = omega * R / h
M_plate = 2 * np.pi * K * rate_R**n * R**3 / (n + 3)                   # exact torque for a power-law fluid
tau_R, _, n_est = geometry.parallel_plate(M_plate, omega, R, h)
check(S, "parallel plates, power-law fluid: corrected rim stress (max rel. error)", "exact solution",
      np.max(np.abs(tau_R / (K * rate_R**n) - 1)), 0, atol=1e-10)
tau_app, _, _ = geometry.parallel_plate(M_plate, omega, R, h, correct=False)
check(S, "... uncorrected (Newtonian) formula overestimates the stress by 4/(3+n) - 1", "exact",
      np.mean(tau_app / (K * rate_R**n)) - 1, 4 / (3 + n) - 1, atol=1e-10)
car_p = (5.0, 0.0, 1.0, 0.4)
M_car = np.array([2 * np.pi * integrate.quad(lambda r, g=g: models.carreau(g * r / R, *car_p) * r**2, 0, R, epsabs=0, epsrel=1e-12)[0]
                  for g in rate_R])
tau_c, _, _ = geometry.parallel_plate(M_car, omega, R, h)
check(S, "parallel plates, Carreau fluid (torque by integration): corrected stress (max rel. error)", "numerical integration",
      np.max(np.abs(tau_c / models.carreau(rate_R, *car_p) - 1)), 0, atol=5e-3)
Ri, Ro, Lc = 0.0125, 0.0136, 0.0375
M_c = np.logspace(-5, -2, 25)
tau_b = M_c / (2 * np.pi * Ri**2 * Lc)
om_c = n / 2 * (M_c / (2 * np.pi * Lc * K)) ** (1 / n) * (Ri ** (-2 / n) - Ro ** (-2 / n))    # exact power-law Couette
_, rate_b, _ = geometry.couette(M_c, om_c, Ri, Ro, Lc)
check(S, "Couette, power-law fluid: Krieger-corrected shear rate at the bob (max rel. error)", "exact solution",
      np.max(np.abs(rate_b / (tau_b / K) ** (1 / n) - 1)), 0, atol=1e-8)
Rc, Lcap = 0.5e-3, 0.02
tau_w = np.logspace(3, 5, 25)
Q_pl = np.pi * Rc**3 * n / (3 * n + 1) * (tau_w / K) ** (1 / n)
_, rate_w, _ = geometry.capillary(2 * Lcap * tau_w / Rc, Q_pl, Rc, Lcap)
check(S, "capillary, power-law fluid: Weissenberg-Rabinowitsch wall rate (max rel. error)", "exact solution",
      np.max(np.abs(rate_w / (tau_w / K) ** (1 / n) - 1)), 0, atol=1e-10)


def rate_of(t):
    return optimize.brentq(lambda g: models.carreau(g, *car_p) - t, 0.0, 1e12) if t > 0 else 0.0


def capillary_error(points_per_decade):
    """Max error of the corrected wall rate for a Carreau fluid; flow rates by numerical integration."""
    tw = np.logspace(0, 3, 3 * points_per_decade + 1)
    qs = np.array([(np.pi * Rc**3 / t**3 * integrate.quad(lambda s: s**2 * rate_of(s), 0, t, epsrel=1e-11)[0], rate_of(t)) for t in tw])
    _, rw, _ = geometry.capillary(2 * Lcap * tw / Rc, qs[:, 0], Rc, Lcap)
    return np.max(np.abs(rw / qs[:, 1] - 1))


# the correction needs the local slope n' from the data: its accuracy depends on how densely the curve is measured
check(S, "capillary, Carreau fluid, 8 points/decade: corrected wall rate (max rel. error < 1 %)", "numerical integration",
      capillary_error(8), 0, atol=0.01)
check(S, "... error ratio when doubling the density from 16 to 32 points/decade (second order: ~4)", "convergence theory",
      capillary_error(16) / capillary_error(32), 4.0, atol=0.5)
LR = np.array([5.0, 10, 20, 40])
bag = geometry.bagley(LR, 2 * 1.8e4 * LR + 3.1e5)
check(S, "Bagley plot: true wall stress", "exact (synthetic)", bag["tau_w"], 1.8e4, rtol=1e-12)
check(S, "Bagley plot: end correction e = dP_ends / (2 tau_w)", "exact (synthetic)", bag["e"], 3.1e5 / 3.6e4, rtol=1e-12)
radii = np.array([0.25e-3, 0.5e-3, 1.0e-3])
moo = geometry.mooney(radii, 120.0 + 4 * 2e-3 / radii)
check(S, "Mooney plot: slip velocity", "exact (synthetic)", moo["slip_velocity"], 2e-3, rtol=1e-10)
check(S, "Mooney plot: slip-free shear rate", "exact (synthetic)", moo["true_rate"], 120.0, rtol=1e-10)
radii_h = np.array([0.5e-3, 1.0e-3, 2.0e-3])
moo_p = geometry.mooney(radii_h, 40.0 + 2 * 1e-4 / radii_h, kind="plates")
check(S, "Mooney plot for parallel plates (apparent = true + 2 u_s / h): slip velocity", "exact (synthetic)",
      moo_p["slip_velocity"], 1e-4, rtol=1e-10)
Rb, Rr, Lf = 0.017245, 0.018415, 0.038                     # Fann 35 bob and rotor
check(S, "Fann 35: shear-rate factor from the bob and rotor radii (1/s per rpm)", "API RP 13B value 1.7023",
      2 * (2 * np.pi / 60) * Rr**2 / (Rr**2 - Rb**2), geometry.FANN_RATE_PER_RPM, rtol=1e-4)
Mtest = np.logspace(-5, -3, 7)
check(S, "Couette simulation, Newtonian fluid (max rel. error)", "exact", np.max(np.abs(
      geometry.couette_speed("newtonian", (0.02,), Mtest, Rb, Rr, Lf) / (Mtest / (4 * np.pi * Lf * 0.02) * (Rb**-2 - Rr**-2)) - 1)), 0, atol=1e-10)
check(S, "Couette simulation, power-law fluid (max rel. error)", "exact", np.max(np.abs(
      geometry.couette_speed("power_law", (K, n), Mtest, Rb, Rr, Lf)
      / (n / 2 * (Mtest / (2 * np.pi * Lf * K)) ** (1 / n) * (Rb ** (-2 / n) - Rr ** (-2 / n))) - 1)), 0, atol=1e-9)
ty_b, mu_b = 5.0, 0.03


def reiner_riwlin(M):
    """Bingham fluid in a Couette gap; partially sheared when the stress at the rotor is below tau_y."""
    rc = min(Rr, np.sqrt(M / (2 * np.pi * Lf * ty_b)))
    if rc <= Rb:
        return 0.0
    return M / (4 * np.pi * Lf * mu_b) * (Rb**-2 - rc**-2) - ty_b / mu_b * np.log(rc / Rb)


M_b = np.array([2 * np.pi * Rb**2 * Lf * ty_b * f for f in (1.02, 1.08, 1.2, 1.6, 3.0)])     # includes partially sheared gaps
check(S, "Couette simulation, Bingham fluid incl. partially sheared gap (max rel. error)", "Reiner-Riwlin equation",
      np.max(np.abs(geometry.couette_speed("bingham", (ty_b, mu_b), M_b, Rb, Rr, Lf) / np.array([reiner_riwlin(m) for m in M_b]) - 1)),
      0, atol=1e-8)
th600, th300 = 2 ** 0.6 * 30.0, 30.0                          # a power-law fluid read with the Newtonian rate conversion
check(S, "API power-law index n = 3.32 log10(theta600/theta300) for n = 0.6", "exact value", geometry.api_power_law(th600, th300)["n"], 0.6,
      rtol=1e-3)
tau_cp, rate_cp = geometry.cone_plate(1e-3, 2.0, R, 0.0349)
M_back = 2 * np.pi * integrate.quad(lambda r: tau_cp * r**2, 0, R)[0]
check(S, "cone-plate: uniform stress integrates back to the torque", "numerical integration", M_back, 1e-3, rtol=1e-12)

# ------------------------------------------------------------------ 4. pipe flow
S = "Pipe flow"
mu, Rp, G = 0.9, 0.02, 2000.0
check(S, "power law with n = 1: Hagen-Poiseuille flow rate", "exact", flows.pipe_power_law(mu, 1.0, Rp, dp_per_L=G)["Q"],
      np.pi * Rp**4 * G / (8 * mu), rtol=1e-12)
pw = flows.pipe_power_law(0.5, 0.6, Rp, dp_per_L=G)
Qint = integrate.quad(lambda r: 2 * np.pi * r * flows.velocity_power_law(r, 0.5, 0.6, Rp, G), 0, Rp, epsabs=0, epsrel=1e-12)[0]
check(S, "power law: flow rate = integral of the velocity profile", "numerical integration", pw["Q"], Qint, rtol=1e-10)
check(S, "power law: centre-line velocity = profile at r = 0", "exact", pw["u_max"], flows.velocity_power_law(0.0, 0.5, 0.6, Rp, G), rtol=1e-12)
hb = flows.pipe_hb(12.0, 0.5, 0.6, Rp, dp_per_L=G)
Qhb = integrate.quad(lambda r: 2 * np.pi * r * flows.velocity_hb(r, 12.0, 0.5, 0.6, Rp, G), 0, Rp, epsabs=0, epsrel=1e-12,
                     points=[hb["plug_radius"]])[0]
check(S, "Herschel-Bulkley: closed-form flow rate = integral of the velocity profile", "numerical integration", hb["Q"], Qhb, rtol=1e-9)
check(S, "Herschel-Bulkley with tau_y = 0 equals power law", "exact", flows.pipe_hb(0.0, 0.5, 0.6, Rp, dp_per_L=G)["Q"], pw["Q"], rtol=1e-12)
tw_b = G * Rp / 2
phi = 8.0 / tw_b
check(S, "Bingham (n = 1): Buckingham-Reiner equation", "exact", flows.pipe_hb(8.0, 0.1, 1.0, Rp, dp_per_L=G)["Q"],
      np.pi * Rp**4 * G / (8 * 0.1) * (1 - 4 / 3 * phi + phi**4 / 3), rtol=1e-12)
check(S, "Herschel-Bulkley: pressure gradient for a given flow rate (round trip)", "inverse", flows.pipe_hb(12.0, 0.5, 0.6, Rp, Q=hb["Q"])["dp_per_L"],
      G, rtol=1e-10)
rho, D = 1000.0, 2 * Rp
V = pw["V"]
Re = flows.reynolds_metzner_reed(rho, V, D, 0.5, 0.6)
check(S, "Metzner-Reed: laminar f = 16/Re reproduces the exact power-law pressure gradient", "exact",
      4 * (16 / Re) / D * rho * V**2 / 2, G, rtol=1e-12)
check(S, "Metzner-Reed Reynolds number with n = 1 = rho V D / mu", "exact", flows.reynolds_metzner_reed(rho, 1.3, D, 0.002, 1.0),
      rho * 1.3 * D / 0.002, rtol=1e-12)
check(S, "Ryan-Johnson critical Reynolds number at n = 1", "classical value ~2100", flows.critical_reynolds_power_law(1.0), 2100, rtol=0.001)
for Re_t in (1e4, 1e5, 1e6):
    fd = optimize.brentq(lambda f, Re_t=Re_t: 1 / np.sqrt(f) + 2 * np.log10(2.51 / (Re_t * np.sqrt(f))), 1e-4, 0.2)    # Colebrook, smooth (Darcy)
    check(S, f"Dodge-Metzner with n = 1 vs Colebrook smooth pipe, Re = {Re_t:.0e} (Fanning)", "Colebrook (Darcy/4)",
          flows.fanning_power_law(Re_t, 1.0), fd / 4, rtol=0.01)
V_b = 0.4
bing = flows.pipe_hb(8.0, 0.1, 1.0, Rp, Q=V_b * np.pi * Rp**2)
Re_b, He = rho * V_b * D / 0.1, flows.hedstrom(rho, D, 8.0, 0.1)
check(S, "Bingham: Buckingham-Reiner friction factor vs exact laminar solution", "exact", flows.fanning_bingham_laminar(Re_b, He),
      bing["tau_w"] / (rho * V_b**2 / 2), rtol=1e-9)

# ------------------------------------------------------------------ 5. linear viscoelasticity
from engrheo import oscillatory, tts  # noqa: E402
from engrheo import viscoelastic as ve  # noqa: E402

S = "Linear viscoelasticity"
g4, t4 = np.array([5e4, 2e4, 6e3, 1e3]), np.array([1e-3, 1e-2, 0.1, 1.0])
for w_ in (0.3, 3.0, 30.0):
    Gp_num = w_ * integrate.quad(lambda s: ve.relaxation_modulus(g4, t4, s), 0, 60, weight="sin", wvar=w_, limit=400)[0]
    Gpp_num = w_ * integrate.quad(lambda s: ve.relaxation_modulus(g4, t4, s), 0, 60, weight="cos", wvar=w_, limit=400)[0]
    Gp_, Gpp_ = ve.moduli(g4, t4, w_)
    check(S, f"G', G'' = Fourier integrals of G(t) at omega = {w_} (max rel. error)", "numerical integration",
          max(abs(Gp_ / Gp_num - 1), abs(Gpp_ / Gpp_num - 1)), 0, atol=1e-7)
check(S, "zero-shear viscosity = integral of G(t)", "numerical integration", ve.zero_shear_viscosity(g4, t4),
      integrate.quad(lambda s: ve.relaxation_modulus(g4, t4, s), 0, 60, limit=400)[0], rtol=1e-7)
eta0_4 = ve.zero_shear_viscosity(g4, t4)
check(S, "steady-state compliance = integral of t G(t) / eta0^2", "numerical integration", ve.steady_state_compliance(g4, t4),
      integrate.quad(lambda s: s * ve.relaxation_modulus(g4, t4, s), 0, 60, limit=400)[0] / eta0_4**2, rtol=1e-7)
mx = ve.maxwell(1e4, 500.0)
check(S, "Maxwell element at omega tau = 1: G' = G'' = G/2", "exact", ve.moduli([1e4], [mx["tau"]], 1 / mx["tau"])[0], 5e3, rtol=1e-14)
tt = np.r_[0.0, np.logspace(-6, 1, 3000)]
check(S, "creep from relaxation (interconversion), Maxwell: max rel. error", "exact J(t) = 1/G + t/eta",
      np.max(np.abs(ve.creep_from_relaxation(mx["relaxation"], tt) / mx["creep"](tt) - 1)), 0, atol=1e-5)
sls = ve.standard_linear_solid(2e3, 8e3, 0.05)
check(S, "creep from relaxation, standard linear solid: max rel. error", "exact (retardation time)",
      np.max(np.abs(ve.creep_from_relaxation(sls["relaxation"], tt) / sls["creep"](tt) - 1)), 0, atol=2e-3)
g7 = np.array([2.7e6, 1.08e6, 6.3e5, 5.4e5, 4.95e5, 3.6e5, 1.35e5])        # a realistic, wide melt spectrum
t7 = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0, 20.0])
tt7 = np.r_[0.0, np.logspace(-6, 3.5, 1000)]
long7 = ve.steady_state_compliance(g7, t7) + tt7[-1] / ve.zero_shear_viscosity(g7, t7)
check(S, "creep of a wide 7-mode spectrum (9 decades of relaxation times) at long times", "exact Je0 + t/eta0",
      ve.creep_from_spectrum(g7, t7, tt7)[-1], long7, rtol=1e-5)
check(S, "... same with a generic G(t) function (numerical cumulative integral)", "exact Je0 + t/eta0",
      ve.creep_from_relaxation(lambda s: ve.relaxation_modulus(g7, t7, s), tt7)[-1], long7, rtol=1e-5)
ts = np.linspace(0, 0.5, 4001)
rate_su = 2.0
check(S, "Boltzmann superposition, start-up at constant rate (Maxwell): max rel. error", "exact eta rate (1 - exp(-t/tau))",
      np.max(np.abs(ve.boltzmann_stress(mx["relaxation"], ts, rate_su * ts)[1:] / (500.0 * rate_su * (1 - np.exp(-ts[1:] / mx["tau"]))) - 1)),
      0, atol=1e-4)
w_sp = np.logspace(-2, 4, 31)
Gp4, Gpp4 = ve.moduli(g4, t4, w_sp)
fit_ex = ve.fit_spectrum(w_sp, Gp4, Gpp4, tau=t4)
check(S, "spectrum fit on the true relaxation times: recovered moduli g_i (max rel. error)", "exact", np.max(np.abs(fit_ex.g / g4 - 1)), 0, atol=1e-8)
noisy_p, noisy_pp = Gp4 * np.exp(rng.normal(0, 0.02, w_sp.size)), Gpp4 * np.exp(rng.normal(0, 0.02, w_sp.size))
fit_n = ve.fit_spectrum(w_sp, noisy_p, noisy_pp)                  # free grid, 5 modes per decade
check(S, "spectrum fit to 2 % noisy data (free grid): rms relative misfit reaches the noise level", "noise level 0.02",
      fit_n.rel_rms, 0.02, atol=0.005)
check(S, "... zero-shear viscosity from the fitted spectrum", "true eta0", fit_n.eta0, eta0_4, rtol=0.02)
fit_r = ve.fit_spectrum(w_sp, noisy_p, noisy_pp, regularization=0.01)
check(S, "... with Tikhonov regularisation: zero-shear viscosity unchanged", "true eta0", fit_r.eta0, eta0_4, rtol=0.02)

# ------------------------------------------------------------------ 6. oscillatory rheology
S = "Oscillatory rheology"
w_dense = np.logspace(0, 3, 301)
cx = oscillatory.crossover(w_dense, *ve.moduli([1e4], [0.01], w_dense))
check(S, "Maxwell crossover frequency 1/tau", "exact", cx[0], 100.0, rtol=1e-4)
check(S, "Maxwell crossover modulus G/2", "exact", cx[1], 5e3, rtol=1e-4)
tw = np.linspace(0, 4 * 2 * np.pi / 5.0, 800, endpoint=False)
wave = oscillatory.moduli_from_waveforms(tw, 0.01 * np.sin(5.0 * tw), 0.01 * (320 * np.sin(5.0 * tw) + 150 * np.cos(5.0 * tw)), 5.0)
check(S, "G' from sampled strain and stress waveforms", "exact (320 Pa)", wave["Gp"], 320.0, rtol=1e-10)
check(S, "G'' from sampled strain and stress waveforms", "exact (150 Pa)", wave["Gpp"], 150.0, rtol=1e-10)
gam = np.logspace(-4, 0, 400)
check(S, "LVE limit: 5 % drop of G' for G' = G0 / (1 + (strain/0.02)^2)", "exact 0.02 sqrt(1/0.95 - 1)",
      oscillatory.lve_limit(gam, 1000 / (1 + (gam / 0.02) ** 2)), 0.02 * np.sqrt(1 / 0.95 - 1), rtol=1e-3)
tg = np.linspace(0, 100, 26)
wg = np.array([1.0, 3, 10, 30])
check(S, "Winter-Chambon gel point from crossing tan(delta) curves", "exact (43.7 min, between samples)",
      oscillatory.gel_point(tg, np.exp(0.02 * (tg[:, None] - 43.7) * np.log(wg) + 0.1)), 43.7, rtol=1e-6)

# ------------------------------------------------------------------ 7. time-temperature superposition
S = "Time-temperature superposition"
T_ref, C1, C2 = 443.15, 8.86, 101.6
temps = [423.15, 433.15, 443.15, 453.15, 473.15, 493.15]
w_m = np.logspace(-1, 2, 16)
curves = {}
for T in temps:
    aT = 10 ** tts.wlf(T, C1, C2, T_ref)
    curves[T] = (w_m, np.column_stack(ve.moduli(g4 * 1e1, t4 * 100, aT * w_m)))
mc = tts.master_curve(curves, T_ref)
check(S, "master curve, 5 points/decade: recovered shift factors log10 aT (max abs. error)", "exact WLF shifts",
      max(abs(mc["log10_aT"][T] - tts.wlf(T, C1, C2, T_ref)) for T in temps), 0, atol=0.005)
w_m10 = np.logspace(-1, 2, 31)
mc10 = tts.master_curve({T: (w_m10, np.column_stack(ve.moduli(g4 * 1e1, t4 * 100, 10 ** tts.wlf(T, C1, C2, T_ref) * w_m10)))
                         for T in temps}, T_ref)
check(S, "... 10 points/decade: error falls with data density", "exact WLF shifts",
      max(abs(mc10["log10_aT"][T] - tts.wlf(T, C1, C2, T_ref)) for T in temps), 0, atol=0.001)
fw = tts.fit_wlf(temps, [mc["log10_aT"][T] for T in temps], T_ref)
check(S, "WLF fit to the recovered shift factors (5 points/decade): C1", "8.86", fw["C1"], C1, rtol=0.005)
check(S, "... C2", "101.6 K", fw["C2"], C2, rtol=0.005)
Ea = 45e3
Ta = np.array([293.15, 313.15, 333.15, 353.15])
fa = tts.fit_arrhenius(Ta, tts.arrhenius(Ta, Ea, 313.15), 313.15)
check(S, "Arrhenius activation energy from exact shift factors", "45 kJ/mol", fa["Ea"], Ea, rtol=1e-12)

# ------------------------------------------------------------------ 8. nonlinear viscoelasticity
from engrheo import constitutive as cm  # noqa: E402
from engrheo import extensional as ex  # noqa: E402
from engrheo import laos, polymers, thixotropy  # noqa: E402
from engrheo import suspensions as su  # noqa: E402

S = "Nonlinear viscoelastic models"
eta_p, lam = 100.0, 0.5
ucm = [cm.Mode(eta_p, lam)]
t_su = np.linspace(0, 3, 61)
r_su = cm.startup_shear(ucm, 2.0, t_su)
check(S, "UCM start-up shear stress (max rel. error)", "exact eta rate (1 - exp(-t/lam))",
      np.max(np.abs(r_su["stress"][1:] / (eta_p * 2.0 * (1 - np.exp(-t_su[1:] / lam))) - 1)), 0, atol=1e-7)
N1_exact = 2 * eta_p * lam * 4.0 * (1 - np.exp(-t_su / lam) * (1 + t_su / lam))
check(S, "UCM start-up first normal-stress difference (max rel. error)", "exact", np.max(np.abs(r_su["N1"][5:] / N1_exact[5:] - 1)), 0, atol=1e-6)
for Wi in (0.1, 0.3, 0.45):
    ext = cm.startup_extension(ucm, Wi / lam, np.array([0.0, 400 * lam]))
    check(S, f"UCM steady extensional viscosity at Wi = {Wi} (diverges at 0.5)", "exact 3 eta/((1 - 2Wi)(1 + Wi))",
          ext["eta_E_plus"][-1], 3 * eta_p / ((1 - 2 * Wi) * (1 + Wi)), rtol=1e-6)
a_g = 0.3
for Wi in (0.5, 5.0):
    chi = np.sqrt((np.sqrt(1 + 16 * a_g * (1 - a_g) * Wi**2) - 1) / (8 * a_g * (1 - a_g) * Wi**2))
    f_ = (1 - chi) / (1 + (1 - 2 * a_g) * chi)
    check(S, f"Giesekus steady shear viscosity, alpha = 0.3, Wi = {Wi}", "Bird, Armstrong & Hassager (analytic)",
          cm.steady_shear([cm.Mode(eta_p, lam, "giesekus", alpha=a_g)], [Wi / lam])["eta"][0], eta_p * (1 - f_) ** 2 / (1 + (1 - 2 * a_g) * f_),
          rtol=1e-7)
check(S, "Giesekus with alpha = 0 equals UCM (start-up N1)", "exact", np.max(np.abs(
      cm.startup_shear([cm.Mode(eta_p, lam, "giesekus", alpha=0.0)], 2.0, t_su)["N1"] - r_su["N1"])), 0, atol=1e-6)
eps_p, Wi_p = 0.1, 3.0
f_ptt = optimize.brentq(lambda f: f**3 - f**2 - 2 * eps_p * Wi_p**2, 1, 100)
check(S, "PTT (linear, xi = 0) steady shear viscosity: f^3 - f^2 = 2 eps Wi^2, eta = eta_p/f", "exact cubic",
      cm.steady_shear([cm.Mode(eta_p, lam, "ptt", eps=eps_p)], [Wi_p / lam])["eta"][0], eta_p / f_ptt, rtol=1e-7)
osc = cm.laos_shear([cm.Mode(eta_p, lam, "giesekus", alpha=0.3)], 1e-3, 4.0, n_cycles=20)
mo = oscillatory.moduli_from_waveforms(osc["t"], osc["strain"], osc["stress"], 4.0)
Gp_l, Gpp_l = ve.moduli([eta_p / lam], [lam], 4.0)
check(S, "Giesekus at small strain amplitude: G' = Maxwell G' (linear limit)", "linear viscoelasticity", mo["Gp"], Gp_l, rtol=1e-4)
check(S, "... G''", "linear viscoelasticity", mo["Gpp"], Gpp_l, rtol=1e-4)

# ------------------------------------------------------------------ 9. LAOS
S = "Large-amplitude oscillatory shear"
tl_ = np.linspace(0, 2 * np.pi, 1024, endpoint=False)
g0, Gs, cs = 0.8, 200.0, 1.5
che = laos.chebyshev(tl_, g0 * np.sin(tl_), Gs * (g0 * np.sin(tl_) + cs * (g0 * np.sin(tl_)) ** 3), omega=1.0)
check(S, "stiffening spring: large-strain modulus G'_L = secant stress(g0)/g0", "exact", che["G_L"], Gs * (1 + cs * g0**2), rtol=1e-12)
check(S, "stiffening spring: minimum-strain modulus G'_M = tangent at zero strain", "exact", che["G_M"], Gs, rtol=1e-12)
check(S, "stiffening spring: third Chebyshev coefficient e3", "exact G c g0^2/4", che["e"][1], Gs * cs * g0**2 / 4, rtol=1e-12)
hs = laos.harmonics(tl_, 0.1 * np.sin(tl_), 0.1 * (50 * np.sin(tl_) + 20 * np.cos(tl_) - 4 * np.sin(3 * tl_) + 1.5 * np.cos(3 * tl_)), 1.0)
check(S, "harmonic moduli from a synthetic stress: G'_3", "exact (-4 Pa)", hs["Gp"][1], -4.0, rtol=1e-10)
check(S, "... G''_3", "exact (1.5 Pa)", hs["Gpp"][1], 1.5, rtol=1e-10)
gm = [cm.Mode(eta_p, lam, "giesekus", alpha=0.3)]
I3 = [laos.harmonics(**{k: v for k, v in cm.laos_shear(gm, a, 4.0, n_cycles=20).items() if k != "t"},
                     t=cm.laos_shear(gm, a, 4.0, n_cycles=20)["t"], omega=4.0)["I3_I1"] for a in (0.05, 0.1)]
check(S, "Giesekus: I3/I1 grows as strain amplitude^2 at moderate amplitude (ratio for doubled amplitude)", "theory: 4",
      I3[1] / I3[0], 4.0, rtol=0.05)

# ------------------------------------------------------------------ 10. thixotropy, suspensions, extension, polymers, processing
S = "Thixotropy, suspensions, extensional flow, polymers, processing"
pth = thixotropy.Params(tau_y=10.0, eta_inf=0.05, d_eta=0.5, k_build=0.02, k_break=0.05)
tq = np.linspace(0, 200, 401)
rate_h = np.where(tq < 100, 5.0, 0.2)
sim = thixotropy.simulate(pth, tq, rate_h, lam0=1.0)
ode = integrate.solve_ivp(lambda s, y: [pth.k_build * (1 - y[0]) - pth.k_break * (5.0 if s < 100 else 0.2) * y[0]], (0, 200), [1.0],
                          t_eval=tq, rtol=1e-11, atol=1e-13, max_step=0.5)
check(S, "thixotropy: exact interval solution vs independent ODE integration (max abs. diff)", "scipy solve_ivp",
      np.max(np.abs(sim["structure"] - ode.y[0])), 0, atol=1e-6)
fast = thixotropy.Params(10.0, 0.05, 0.5, 1e3, 2.5e3)
loop = thixotropy.hysteresis_loop(fast, 50.0, 60.0)
check(S, "hysteresis loop area -> 0 for instantaneous structural kinetics (relative to loop scale)", "limit",
      abs(loop["area"]) / integrate.trapezoid(loop["stress_up"], loop["rate_up"]), 0, atol=1e-3)
pp = np.array([1e-4, 2e-4])
check(S, "Krieger-Dougherty initial slope -> Einstein 2.5", "exact limit",
      np.diff(su.krieger_dougherty(pp, 0.64))[0] / 1e-4, 2.5, rtol=1e-3)
check(S, "Krieger-Dougherty with [eta] phi_max = 2 equals Quemada", "exact", su.krieger_dougherty(0.4, 0.8, intrinsic=2.5),
      su.quemada(0.4, 0.8), rtol=1e-12)
phis = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
check(S, "Krieger-Dougherty fit recovers phi_max", "exact (0.63)", su.fit_krieger_dougherty(phis, su.krieger_dougherty(phis, 0.63))["phi_max"],
      0.63, rtol=1e-6)
tc_, sig_, eta_n = np.linspace(0, 0.25, 51), 0.03, 1.2               # break-up at 0.28 s
etaE = ex.apparent_extensional_viscosity(tc_, ex.newtonian_thinning(tc_, 1e-3, sig_, eta_n), sig_)
check(S, "capillary thinning of a Newtonian liquid: Trouton ratio of the apparent extensional viscosity", "exact 3",
      np.mean(etaE) / eta_n, 3.0, rtol=1e-3)
te = np.linspace(0.05, 0.5, 30)
check(S, "elasto-capillary thinning: extensional relaxation time", "exact 0.04 s",
      ex.fit_elastocapillary(te, 2e-4 * np.exp(-te / (3 * 0.04)))["lambda_E"], 0.04, rtol=1e-10)
check(S, "UCM transient extension at small rate: Trouton ratio -> 3", "linear limit",
      cm.startup_extension(ucm, 1e-4 / lam, np.array([0.0, 50 * lam]))["eta_E_plus"][-1] / eta_p, 3.0, rtol=1e-3)
check(S, "zero-shear viscosity continuous at M_c", "definition", polymers.zero_shear_viscosity(3.0e4 * (1 - 1e-12), 3.0e4, 1e3), 1e3, rtol=1e-9)
mu_m, rho_m = 900.0, 750.0                    # a Newtonian 'melt' in the MFI die
dp_m = 2.16 * 9.80665 / (np.pi * polymers.MFI_PISTON_RADIUS**2)
Q_m = np.pi * polymers.MFI_DIE_RADIUS**4 * dp_m / (8 * mu_m * polymers.MFI_DIE_LENGTH)
mfi = Q_m * rho_m * 600 * 1e3
check(S, "melt flow index test: apparent viscosity of a Newtonian melt", "Hagen-Poiseuille in the ISO 1133 die",
      polymers.mfi_conditions(mfi, 2.16, rho_m)["eta_apparent"], mu_m, rtol=1e-12)
check(S, "slit flow, n = 1: Q = W H^3 (dP/L) / (12 mu)", "exact", flows.slit_power_law(0.5, 1.0, 0.3, 2e-3, dp_per_L=5e4)["Q"],
      0.3 * (2e-3) ** 3 * 5e4 / (12 * 0.5), rtol=1e-12)
sl = flows.slit_power_law(2.0, 0.4, 0.3, 2e-3, dp_per_L=5e4)
check(S, "slit flow, power law: flow rate round trip (Q -> pressure gradient)", "inverse",
      flows.slit_power_law(2.0, 0.4, 0.3, 2e-3, Q=sl["Q"])["dp_per_L"], 5e4, rtol=1e-10)
check(S, "Tanner swell of an inelastic fluid (N1 = 0)", "exact 1.1", flows.tanner_swell(0.0, 1e4), 1.1, rtol=1e-12)

# ------------------------------------------------------------------ report
passed = sum(r[5] for r in RESULTS)
lines = ["# Validation report", "", "Generated by `python docs/validate.py`; rerun in CI on every push.", "",
         f"**{passed} / {len(RESULTS)} checks pass.**", ""]
section = None
for sec, name, ref, val, exp, ok in RESULTS:
    if sec != section:
        lines += ["", f"## {sec}", "", "| Check | Reference | engrheo | Expected | Result |", "|---|---|---|---|:-:|"]
        section = sec
    lines.append(f"| {name} | {ref} | {val:.10g} | {exp:.10g} | {'pass' if ok else '**FAIL**'} |")
Path(__file__).with_name("VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{passed}/{len(RESULTS)} checks pass")
