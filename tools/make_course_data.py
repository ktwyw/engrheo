"""Write the course data files into src/engrheo/data (reproducible: fixed random seeds).

Every file is synthetic, generated from a known model with realistic noise and instrument behaviour, so
that analyses in the notebooks can be checked against the truth. Run from the repository root:
    python tools/make_course_data.py
"""
from pathlib import Path

import numpy as np
from scipy import integrate, optimize

from engrheo import geometry, models

OUT = Path(__file__).resolve().parents[1] / "src" / "engrheo" / "data"
rng = np.random.default_rng(2026)


def write(name, header_lines, columns, names, fmt):
    lines = [f"# {h}" for h in header_lines] + [",".join(names)]
    for row in zip(*columns):
        lines.append(",".join(f % v for f, v in zip(fmt, row)))
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", name)


# 1. xanthan gum solution, cone-plate (true model: Carreau eta0 35 Pa s, eta_inf 0.002 Pa s, lambda 12 s, n 0.25)
car = (35.0, 0.002, 12.0, 0.25)
rate = np.logspace(-2, 3, 31)
tau = models.stress("carreau", rate, *car) * np.exp(rng.normal(0, 0.02, rate.size))
R, theta = 0.025, np.radians(1.0)
torque = tau * 2 * np.pi * R**3 / 3
write("xanthan_flow_curve.csv",
      ["0.5 wt% xanthan gum in water, 25 degC. Cone-plate: R = 25 mm, cone angle 1.0 deg. Steady-state flow curve.",
       "Columns: shear rate (1/s), shear stress (Pa), viscosity (Pa s), torque (N m)."],
      [rate, tau, tau / rate, torque], ["shear_rate_1_s", "shear_stress_Pa", "viscosity_Pa_s", "torque_Nm"],
      ["%.5g", "%.5g", "%.5g", "%.5g"])

# 2. polymer melt, capillary rheometer with four dies (true: Carreau eta0 3000 Pa s, lambda 0.3 s, n 0.3)
melt = (3000.0, 0.0, 0.3, 0.3)
Rd = 0.5e-3


def q_of_tau(tw):
    rate_of = lambda t: models.rate_from_stress("carreau", t, *melt)  # noqa: E731
    return np.pi * Rd**3 / tw**3 * integrate.quad(lambda s: s**2 * rate_of(s), 0, tw, epsrel=1e-10)[0]


apparent = np.logspace(1, 3.5, 11)
rows = []
for a in apparent:
    Q = a * np.pi * Rd**3 / 4
    tw = optimize.brentq(lambda t, Q=Q: q_of_tau(t) - Q, 1.0, 1e7, rtol=1e-12)
    e = 2.0 + 1.5 * np.log10(a)                               # Bagley end correction grows with rate
    for LR in (5.0, 10.0, 20.0, 32.0):
        dp = 2 * tw * (LR + e) * np.exp(rng.normal(0, 0.01))
        rows.append((LR * Rd * 1e3, Rd * 1e3, Q, dp))
rows = np.array(rows)
write("polymer_melt_capillary.csv",
      ["Polymer melt at 200 degC, capillary rheometer. Four dies of radius 0.5 mm and different lengths;",
       "at each piston speed (flow rate) the TOTAL pressure drop is recorded (entrance losses included).",
       "Columns: die length (mm), die radius (mm), volumetric flow rate (m3/s), pressure drop (Pa)."],
      list(rows.T), ["die_length_mm", "die_radius_mm", "flow_rate_m3_s", "pressure_drop_Pa"], ["%.4g", "%.4g", "%.6g", "%.6g"])

# 3. toothpaste between smooth parallel plates at three gaps: wall slip (true: HB tau_y 120 Pa, K 25, n 0.45;
#    slip velocity u_s = 4e-6 * tau at each plate)
hb = (120.0, 25.0, 0.45)
stresses = np.round(np.logspace(np.log10(60), np.log10(600), 10), 1)
rows = []
for h in (0.5e-3, 1.0e-3, 2.0e-3):
    for t in stresses:
        app = models.rate_from_stress("herschel_bulkley", t, *hb) + 2 * 4e-6 * t / h
        rows.append((h * 1e3, t, app * np.exp(rng.normal(0, 0.02))))
rows = np.array(rows)
write("toothpaste_plates_slip.csv",
      ["Toothpaste, 25 degC, smooth parallel plates (R = 20 mm) at three gaps, stress-controlled steady shear.",
       "Columns: gap (mm), rim shear stress (Pa), apparent rim shear rate omega R / h (1/s)."],
      list(rows.T), ["gap_mm", "shear_stress_Pa", "apparent_shear_rate_1_s"], ["%.3g", "%.5g", "%.5g"])

# 4. water-based drilling mud, Fann 35 readings (true: HB tau_y 4.8 Pa, K 0.35 Pa s^n, n 0.62), simulated by
#    integrating the fluid behaviour across the real gap, read to the nearest half degree
mud = (4.8, 0.35, 0.62)
Rb, Rr, Lb = 0.017245, 0.018415, 0.038
rpm = np.array([600, 300, 200, 100, 6, 3])
dial = []
for r in rpm:
    omega = r * 2 * np.pi / 60
    lo = 2 * np.pi * Rb**2 * Lb * mud[0] * 1.0000001
    M = optimize.brentq(lambda m, w=omega: geometry.couette_speed("herschel_bulkley", mud, m, Rb, Rr, Lb) - w,
                        lo, 1.0, rtol=1e-12)
    dial.append(np.round(2 * M / (2 * np.pi * Rb**2 * Lb) / geometry.FANN_PA_PER_DEGREE) / 2)
write("drilling_mud_fann.csv",
      ["Water-based drilling fluid, 49 degC (120 degF), Fann 35 six-speed viscometer (R1 rotor-B1 bob-F1 spring),",
       "dial readings as recorded by the technician (to the nearest half degree). Columns: rotor speed (rpm), dial reading (deg)."],
      [rpm, np.array(dial)], ["rpm", "dial_reading_deg"], ["%d", "%.1f"])

# 5. molten milk chocolate (true: Casson tau_y 12 Pa, eta_c 2.1 Pa s), ICA-type test 2-50 1/s
rate_c = np.round(np.logspace(np.log10(2), np.log10(50), 10), 2)
tau_c = models.stress("casson", rate_c, 12.0, 2.1) * np.exp(rng.normal(0, 0.015, rate_c.size))
write("chocolate_casson.csv",
      ["Molten milk chocolate, 40 degC, concentric cylinders, increasing shear rate 2-50 1/s after pre-shear.",
       "Columns: shear rate (1/s), shear stress (Pa)."],
      [rate_c, tau_c], ["shear_rate_1_s", "shear_stress_Pa"], ["%.4g", "%.5g"])

# 6. fresh concrete (true: Bingham tau_y 600 Pa, plastic viscosity 45 Pa s)
rate_k = np.array([1.0, 2, 3, 5, 7, 10, 14, 20])
tau_k = models.stress("bingham", rate_k, 600.0, 45.0) * np.exp(rng.normal(0, 0.04, rate_k.size))
write("concrete_flow_curve.csv",
      ["Fresh self-compacting concrete, 20 degC, wide-gap concrete rheometer (converted to stress and rate).",
       "Columns: shear rate (1/s), shear stress (Pa)."],
      [rate_k, tau_k], ["shear_rate_1_s", "shear_stress_Pa"], ["%.4g", "%.5g"])


# ================================================================= viscoelasticity
from engrheo import tts  # noqa: E402
from engrheo import viscoelastic as ve  # noqa: E402

# 7. engine oil viscosity vs temperature (true: Vogel eta = 6e-5 exp(1100 / (T - 160)) Pa s, T in K; 1 % scatter)
T_oil = np.arange(0, 121, 10) + 273.15
eta_oil = 6e-5 * np.exp(1100 / (T_oil - 160)) * np.exp(rng.normal(0, 0.01, T_oil.size))
write("engine_oil_viscosity_temperature.csv",
      ["Multigrade engine oil, dynamic viscosity at low shear rate between 0 and 120 degC.",
       "Columns: temperature (degC), viscosity (Pa s)."],
      [T_oil - 273.15, eta_oil], ["temperature_C", "viscosity_Pa_s"], ["%.1f", "%.5g"])

# 8. polymer melt frequency sweeps at six temperatures (true: 7-mode spectrum at 170 degC, WLF C1 8.86, C2 101.6 K)
g_ps = np.array([2.7e6, 1.08e6, 6.3e5, 5.4e5, 4.95e5, 3.6e5, 1.35e5])
tau_ps = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0, 20.0])
T_ref_ps = 443.15
w_ps = np.logspace(-1, 2, 16)
rows = []
for T_C in (150, 160, 170, 180, 200, 220):
    aT = 10 ** tts.wlf(T_C + 273.15, 8.86, 101.6, T_ref_ps)
    Gp, Gpp = ve.moduli(g_ps, tau_ps, aT * w_ps)
    for w_, a, b in zip(w_ps, Gp * np.exp(rng.normal(0, 0.02, w_ps.size)), Gpp * np.exp(rng.normal(0, 0.02, w_ps.size))):
        rows.append((T_C, w_, a, b))
rows = np.array(rows)
write("polystyrene_frequency_sweeps.csv",
      ["Polystyrene-like melt, small-amplitude oscillatory shear (frequency sweeps) at six temperatures,",
       "parallel plates 25 mm, strain within the linear range. Columns: temperature (degC), angular frequency (rad/s), G' (Pa), G'' (Pa)."],
      list(rows.T), ["temperature_C", "omega_rad_s", "G_storage_Pa", "G_loss_Pa"], ["%.0f", "%.5g", "%.5g", "%.5g"])

# 9. bread dough creep (50 Pa for 120 s) and recovery (180 s) (true: Burgers G1 5e3, eta1 1e6, G2 2e3 Pa, eta2 5e4 Pa s)
bur = (5e3, 1e6, 2e3, 5e4)
t_c = np.r_[np.linspace(0, 120, 121), np.linspace(121, 300, 180)]
J = lambda t: np.where(t >= 0, ve.burgers_creep(np.maximum(t, 0), *bur), 0.0)  # noqa: E731
strain = 50.0 * (J(t_c) - np.where(t_c > 120, J(t_c - 120), 0.0))
strain = strain + rng.normal(0, 1e-4, t_c.size)
write("dough_creep_recovery.csv",
      ["Bread dough (wheat flour, water, salt), 25 degC, parallel plates. Creep at a constant stress of 50 Pa",
       "for 120 s, then recovery at zero stress. Columns: time (s), shear stress (Pa), shear strain (-)."],
      [t_c, np.where(t_c <= 120, 50.0, 0.0), strain], ["time_s", "shear_stress_Pa", "strain"], ["%.1f", "%.1f", "%.6f"])

# 10. gelling food system: time sweeps at four frequencies (true gel time 23.4 min; tan(delta) at the gel point tan(0.6 pi/2))
t_g = np.linspace(0, 60, 61)
omegas = np.array([1.0, 3.162, 10.0, 31.62])
rows = []
for t_ in t_g:
    ltd = np.log(np.tan(0.6 * np.pi / 2)) - 0.035 * (t_ - 23.4) * np.log(omegas)
    Gs = 2.0 * np.exp(0.12 * (t_ - 23.4)) * omegas**0.6 + 0.05
    delta = np.arctan(np.exp(ltd))
    for w_, gs, d in zip(omegas, Gs, delta):
        rows.append((t_, w_, gs * np.cos(d) * np.exp(rng.normal(0, 0.02)), gs * np.sin(d) * np.exp(rng.normal(0, 0.02))))
rows = np.array(rows)
write("food_gel_time_sweep.csv",
      ["Gelling food system (e.g. a protein gel), 25 degC, repeated small-amplitude frequency sweeps during gelation.",
       "Columns: time (min), angular frequency (rad/s), G' (Pa), G'' (Pa)."],
      list(rows.T), ["time_min", "omega_rad_s", "G_storage_Pa", "G_loss_Pa"], ["%.1f", "%.4g", "%.5g", "%.5g"])

# 11. cosmetic cream amplitude sweep at 1 Hz (true linear limit of G' at 5 %: strain 0.05 (1/0.95 - 1)^(1/1.6))
gam = np.logspace(-4, 0, 33)
Gp_c = 2000 / (1 + (gam / 0.05) ** 1.6)
Gpp_c = 400 * (1 + 3 * (gam / 0.08)) / (1 + (gam / 0.12) ** 1.8)
write("cream_amplitude_sweep.csv",
      ["Cosmetic cream (oil-in-water emulsion), 25 degC, amplitude sweep at 1 Hz (6.283 rad/s), sandblasted plates.",
       "Columns: strain amplitude (-), G' (Pa), G'' (Pa)."],
      [gam, Gp_c * np.exp(rng.normal(0, 0.01, gam.size)), Gpp_c * np.exp(rng.normal(0, 0.01, gam.size))],
      ["strain_amplitude", "G_storage_Pa", "G_loss_Pa"], ["%.5g", "%.5g", "%.5g"])

# 12. polymer solution: steady flow curve and frequency sweep (Cox-Merz holds by construction)
g_s, tau_s = np.array([40.0, 25.0, 12.0, 4.0]), np.array([0.001, 0.01, 0.1, 1.0])
w_s = np.logspace(-1, 2.5, 15)
Gp_s, Gpp_s = ve.moduli(g_s, tau_s, w_s)
eta_star = np.hypot(Gp_s, Gpp_s) / w_s * np.exp(rng.normal(0, 0.02, w_s.size))
r_s = np.logspace(-0.5, 3, 12)
Gp_r, Gpp_r = ve.moduli(g_s, tau_s, r_s)
eta_steady = np.hypot(Gp_r, Gpp_r) / r_s * np.exp(rng.normal(0, 0.02, r_s.size))
kinds = ["oscillatory"] * w_s.size + ["steady"] * r_s.size
lines = ["# Aqueous polymer solution (2 wt% polyethylene oxide), 25 degC, cone-plate. Two tests in one file:",
         "# 'oscillatory' rows: angular frequency (rad/s) and complex viscosity |eta*| (Pa s);",
         "# 'steady' rows: shear rate (1/s) and steady shear viscosity (Pa s).",
         "test,rate_or_frequency,viscosity_Pa_s"]
lines += [f"{k},{x:.5g},{v:.5g}" for k, x, v in zip(kinds, np.r_[w_s, r_s], np.r_[eta_star, eta_steady])]
(OUT / "polymer_solution_cox_merz.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote polymer_solution_cox_merz.csv")


# ================================================================= nonlinear rheology, materials, processing
from engrheo import constitutive as cm  # noqa: E402
from engrheo import thixotropy as th  # noqa: E402

# 13. paint: up-and-down shear-rate ramp (true: structural kinetics tau_y 3 Pa, eta_inf 0.08, d_eta 0.4 Pa s,
#     k_build 0.02 1/s, k_break 0.08; 0 -> 200 1/s in 60 s and back; 1 % scatter)
p_paint = th.Params(3.0, 0.08, 0.4, 0.02, 0.08)
loop = th.hysteresis_loop(p_paint, 200.0, 60.0, n=60)
rates = np.r_[loop["rate_up"], loop["rate_down"][::-1][1:]]
stresses = np.r_[loop["stress_up"], loop["stress_down"][::-1][1:]] * np.exp(rng.normal(0, 0.01, rates.size))
direction = ["up"] * loop["rate_up"].size + ["down"] * (loop["rate_down"].size - 1)
t_loop = np.linspace(0, 120, rates.size)
lines = ["# Water-based wall paint, 23 degC, concentric cylinders. Shear rate ramped linearly from 0 to 200 1/s in 60 s",
         "# and back to 0 in 60 s, after 30 min rest. Columns: time (s), ramp direction, shear rate (1/s), shear stress (Pa).",
         "time_s,direction,shear_rate_1_s,shear_stress_Pa"]
lines += [f"{a:.2f},{d},{r:.4g},{s:.5g}" for a, d, r, s in zip(t_loop, direction, rates, stresses)]
(OUT / "paint_hysteresis_loop.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote paint_hysteresis_loop.csv")

# 14. cement paste: step test (true: tau_y 25 Pa, eta_inf 0.1, d_eta 1.5 Pa s, k_build 0.01 1/s, k_break 0.02; 1 % scatter)
p_cem = th.Params(25.0, 0.1, 1.5, 0.01, 0.02)
t_st = np.linspace(0, 1200, 1201)
rate_st = np.select([t_st < 200, t_st < 400, t_st < 600, t_st < 900], [10.0, 100.0, 10.0, 0.0], 1.0)
sim = th.simulate(p_cem, t_st, rate_st, lam0=th.equilibrium_structure(p_cem, 10.0))
write("cement_paste_step_test.csv",
      ["Cement paste (water/cement 0.45), 20 degC, vane geometry. Steps in shear rate 10 -> 100 -> 10 1/s (200 s each),",
       "then 300 s at rest (rate 0) and a probe at 1 1/s. Columns: time (s), shear rate (1/s), shear stress (Pa)."],
      [t_st, rate_st, sim["stress"] * np.exp(rng.normal(0, 0.01, t_st.size))], ["time_s", "shear_rate_1_s", "shear_stress_Pa"],
      ["%.0f", "%.4g", "%.5g"])

# 15. polymer solution: start-up of shear at four rates (true: 3-mode Giesekus, alpha 0.25; spectrum known from SAOS)
g_ps3, tau_ps3 = np.array([60.0, 25.0, 8.0]), np.array([0.02, 0.2, 2.0])
modes_true = cm.modes_from_spectrum(g_ps3, tau_ps3, "giesekus", alpha=0.25)
rows = []
for rate in (0.1, 1.0, 10.0, 100.0):
    t_u = np.r_[np.logspace(-3, 1.5, 40)]
    r = cm.startup_shear(modes_true, rate, np.r_[0.0, t_u])
    for tt_, s_, n1 in zip(t_u, r["stress"][1:], r["N1"][1:]):
        rows.append((rate, tt_, s_ * np.exp(rng.normal(0, 0.01)), n1 * np.exp(rng.normal(0, 0.03))))
rows = np.array(rows)
write("polymer_solution_startup.csv",
      ["Semi-dilute polymer solution (polyacrylamide in glycerol-water), 25 degC, cone-plate with normal-force transducer.",
       "Start-up of steady shear at four rates. Linear spectrum from a frequency sweep: g = 60, 25, 8 Pa at tau = 0.02, 0.2, 2 s.",
       "Columns: shear rate (1/s), time (s), shear stress (Pa), first normal-stress difference N1 (Pa)."],
      list(rows.T), ["shear_rate_1_s", "time_s", "shear_stress_Pa", "N1_Pa"], ["%.4g", "%.5g", "%.5g", "%.5g"])

# 16. stirred yoghurt: LAOS waveforms at five amplitudes, 1 Hz (phenomenological weak gel: strain-stiffening
#     elastic part that yields, plus a thinning viscous part; 1 % noise)
w_y = 2 * np.pi
t_y = np.linspace(0, 1, 256, endpoint=False)
rows = []
for g0 in (0.005, 0.02, 0.08, 0.3, 1.0):
    g = g0 * np.sin(w_y * t_y)
    rate_y = g0 * w_y * np.cos(w_y * t_y)
    elastic = 300 * g * (1 + (g / 0.1) ** 2) / (1 + (np.abs(g) / 0.15) ** 3.2)
    viscous = 20 * rate_y / (1 + (np.abs(rate_y) / 0.5) ** 0.6)
    s = elastic + viscous
    s = s + rng.normal(0, 0.01 * np.abs(s).max(), s.size)
    rows += [(g0, a, b, c) for a, b, c in zip(t_y, g, s)]
rows = np.array(rows)
write("yoghurt_laos.csv",
      ["Stirred yoghurt, 10 degC, serrated plates, oscillatory shear at 1 Hz (6.283 rad/s), five strain amplitudes;",
       "one steady-state cycle each. Columns: strain amplitude (-), time (s), strain (-), shear stress (Pa)."],
      list(rows.T), ["strain_amplitude", "time_s", "strain", "shear_stress_Pa"], ["%.4g", "%.5f", "%.6g", "%.5g"])

# 17. polyethylene grades: zero-shear viscosity vs weight-average molar mass at 190 degC
#     (true: M_c = 4 kg/mol with eta_c = 0.8 Pa s; exponent 3.4 above, 1 below; 5 % scatter)
M = np.array([0.8, 1.5, 2.5, 3.2, 5.0, 8.0, 15, 30, 60, 100, 200, 400]) * 1e3        # g/mol
eta0 = np.where(M < 4e3, 0.8 * M / 4e3, 0.8 * (M / 4e3) ** 3.4) * np.exp(rng.normal(0, 0.05, M.size))
write("polyethylene_eta0_vs_mw.csv",
      ["Linear polyethylene grades, zero-shear viscosity at 190 degC (from creep or low-frequency oscillation).",
       "Columns: weight-average molar mass (g/mol), zero-shear viscosity (Pa s)."],
      [M, eta0], ["Mw_g_mol", "eta0_Pa_s"], ["%.4g", "%.4g"])

# 18. glass beads in silicone oil: relative viscosity vs volume fraction (true: Krieger-Dougherty phi_max 0.61, [eta] 2.5; 3 %)
phi_b = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
eta_r = (1 - phi_b / 0.61) ** (-2.5 * 0.61) * np.exp(rng.normal(0, 0.03, phi_b.size))
write("glass_beads_suspension.csv",
      ["Glass beads (40-60 um) in a Newtonian silicone oil, 25 degC, low-shear relative viscosity (suspension / oil).",
       "Columns: solid volume fraction (-), relative viscosity (-)."],
      [phi_b, eta_r], ["volume_fraction", "relative_viscosity"], ["%.3f", "%.4g"])

# 19. cornstarch in water (phi ~ 0.44): shear thinning, then discontinuous shear thickening above ~ 5 1/s
rate_cs = np.logspace(-1, 1.3, 16)
eta_cs = 2.0 * rate_cs ** -0.3 * (1 + (rate_cs / 5.0) ** 4)
write("cornstarch_flow_curve.csv",
      ["Cornstarch in water (about 44 vol%), 22 degC, stress-controlled, sandblasted plates. Steady flow curve.",
       "Columns: shear rate (1/s), viscosity (Pa s)."],
      [rate_cs, eta_cs * np.exp(rng.normal(0, 0.03, rate_cs.size))], ["shear_rate_1_s", "viscosity_Pa_s"], ["%.4g", "%.4g"])

# 20. dilute polymer solution: capillary thinning (true: elasto-capillary exponential, lambda_E 12 ms, then a
#     linear finite-extensibility end; surface tension 0.062 N/m; diameter noise 1 um)
t_cb = np.linspace(0, 0.25, 126)
D_cb = np.where(t_cb < 0.20, 0.35e-3 * np.exp(-t_cb / (3 * 0.012)), 0.0)
t_lin = t_cb >= 0.20
D_cb[t_lin] = np.maximum(0.35e-3 * np.exp(-0.20 / 0.036) * (1 - (t_cb[t_lin] - 0.20) / 0.02), 0)
D_cb = np.maximum(D_cb + rng.normal(0, 1e-6, t_cb.size), 0)
write("polymer_solution_caber.csv",
      ["Dilute polyethylene oxide solution (0.1 wt%, inkjet-type fluid), 22 degC, capillary break-up (CaBER), plates 6 mm,",
       "surface tension 0.062 N/m. Columns: time after plate separation (s), mid-filament diameter (m)."],
      [t_cb, D_cb], ["time_s", "diameter_m"], ["%.4f", "%.4g"])


# ================================================================= a messy multi-test export (notebook 17)
# Ketchup, 25 degC: true Herschel-Bulkley tau_y 15 Pa, K 4.5 Pa s^n, n 0.35 (equilibrium flow);
# amplitude sweep G' 250 Pa plateau, G'' 60 Pa; weak-gel frequency sweep; thixotropic loop (down curve 12 % lower).
def fmt(v):
    return f"{v:.5g}".replace(".", ",")                     # decimal comma, as exported on a German-language PC


lines = ["Rheometer export;;;;", "Sample;Ketchup (tomato ketchup, retail);;;", "Operator;JM;;;", "Date;14.03.2026;;;",
         "Geometry;PP25/S (sandblasted plate, 25 mm), gap 1,000 mm;;;", ""]
# amplitude sweep at 1 Hz
gam = np.logspace(-4, 0, 21)
Gp_k = 250 / (1 + (gam / 0.03) ** 1.4)
Gpp_k = 60 * (1 + 2.5 * (gam / 0.05)) / (1 + (gam / 0.08) ** 1.6)
lines += ["Test;Amplitude sweep;;;", "Temperature;25,0 °C;;;", "Frequency;1 Hz;;;", "Point;Strain;Storage modulus;Loss modulus;Status",
          ";%;Pa;Pa;"]
for i, (g, a, b) in enumerate(zip(gam, Gp_k * np.exp(rng.normal(0, 0.015, gam.size)), Gpp_k * np.exp(rng.normal(0, 0.015, gam.size))), 1):
    lines.append(f"{i};{fmt(100 * g)};{fmt(a)};{fmt(b)};ok")
# frequency sweep at 0.5 % strain
w_k = np.logspace(-1, 2, 16)
lines += ["", "Test;Frequency sweep;;;", "Temperature;25,0 °C;;;", "Strain;0,5 %;;;",
          "Point;Angular frequency;Storage modulus;Loss modulus;Status", ";rad/s;Pa;Pa;"]
for i, (w_, a, b) in enumerate(zip(w_k, 230 * w_k**0.12 * np.exp(rng.normal(0, 0.015, w_k.size)),
                                   55 * w_k**0.15 * np.exp(rng.normal(0, 0.015, w_k.size))), 1):
    lines.append(f"{i};{fmt(w_)};{fmt(a)};{fmt(b)};ok")
# flow curves up and down: viscosity in mPa s; low torque at the lowest rates; one failed point
rate_k = np.logspace(-2, 2.5, 19)
tau_up = (15.0 + 4.5 * rate_k**0.35) * np.exp(rng.normal(0, 0.02, rate_k.size))
for block, factor, order in (("Flow curve (up)", 1.0, rate_k), ("Flow curve (down)", 0.88, rate_k[::-1])):
    lines += ["", f"Test;{block};;;", "Temperature;25,0 °C;;;",
              "Point;Shear rate;Shear stress;Viscosity;Status", ";1/s;Pa;mPa·s;"]
    for i, r in enumerate(order, 1):
        tau = np.interp(np.log(r), np.log(rate_k), np.log(tau_up))
        tau = np.exp(tau) * (factor if r < 50 else 1.0 - (1 - factor) * (316 - r) / 266)
        status = "ok"
        if block.endswith("(up)") and r < 0.03:
            tau *= 1.6                                       # too little time at a very low rate: stress still decaying
            status = "Steady state not reached"
        text_tau, text_eta = fmt(tau), fmt(tau / r * 1000)
        if block.endswith("(down)") and i == 7:
            text_tau, text_eta, status = "---", "---", "Measurement failed"
        lines.append(f"{i};{fmt(r)};{text_tau};{text_eta};{status}")
(OUT / "ketchup_rheometer_export.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote ketchup_rheometer_export.csv")
