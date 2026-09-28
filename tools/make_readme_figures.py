"""Generate the README figures from the library (docs/figures/*.png). Run: python tools/make_readme_figures.py"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from engrheo import constitutive, flows, models, thixotropy, viscoelastic  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(2, 3, figsize=(13, 7.2))

# 1. flow curves of the classical models
rate = np.logspace(-2, 3, 200)
curves = {"Newtonian": models.viscosity("newtonian", rate, mu=1.0), "power law (n = 0.5)": models.viscosity("power_law", rate, K=3.0, n=0.5),
          "Carreau": models.viscosity("carreau", rate, eta0=20.0, eta_inf=0.01, lam=2.0, n=0.4),
          "Herschel-Bulkley": models.viscosity("herschel_bulkley", rate, tau_y=5.0, K=2.0, n=0.6)}
for name, eta in curves.items():
    ax[0, 0].loglog(rate, eta, label=name)
ax[0, 0].set(xlabel="shear rate (1/s)", ylabel="viscosity (Pa s)", title="Flow curves of the classical models")
ax[0, 0].legend(frameon=False, fontsize=8)

# 2. pipe velocity profiles: Newtonian, power law, Herschel-Bulkley with its plug
R = 0.05
r = np.linspace(-R, R, 400)
for name, v in (("Newtonian (n = 1)", flows.velocity_power_law(np.abs(r), K=1.0, n=1.0, R=R, dp_per_L=200.0)),
                ("power law, n = 0.4", flows.velocity_power_law(np.abs(r), K=1.0, n=0.4, R=R, dp_per_L=200.0)),
                ("Herschel-Bulkley (plug)", flows.velocity_hb(np.abs(r), tau_y=2.0, K=1.0, n=0.6, R=R, dp_per_L=200.0))):
    ax[0, 1].plot(v / v.max(), r / R, label=name)
ax[0, 1].set(xlabel="velocity / maximum", ylabel="r / R", title="Laminar pipe flow profiles")
ax[0, 1].legend(frameon=False, fontsize=8, loc="center left")

# 3. G' and G'' of a discrete Maxwell spectrum
omega = np.logspace(-3, 3, 300)
g, tau = [2e4, 1e4, 5e3, 2e3], [10.0, 1.0, 0.1, 0.01]
Gp, Gpp = viscoelastic.moduli(g, tau, omega)
ax[0, 2].loglog(omega, Gp, label="G' (storage)")
ax[0, 2].loglog(omega, Gpp, label="G'' (loss)")
ax[0, 2].set(xlabel="angular frequency (rad/s)", ylabel="modulus (Pa)", title="Frequency sweep of a 4-mode Maxwell spectrum")
ax[0, 2].legend(frameon=False)

# 4. Giesekus start-up: stress overshoot
modes = constitutive.modes_from_spectrum([1000.0], [1.0], "giesekus", alpha=0.3)
t = np.linspace(0, 6, 400)
for gd in (0.5, 2.0, 8.0):
    su = constitutive.startup_shear(modes, gd, t)
    ax[1, 0].plot(t, su["eta_plus"] / 1000.0, label=f"shear rate {gd:g} 1/s")
ax[1, 0].set(xlabel="time (s)", ylabel="eta+ / eta0", title="Giesekus start-up: overshoot grows with Wi")
ax[1, 0].legend(frameon=False, fontsize=8)

# 5. LAOS Lissajous curves
for amp, col in ((0.2, "C0"), (2.0, "C1"), (6.0, "C3")):
    la = constitutive.laos_shear(modes, amp, 1.0, n_cycles=25, points_per_cycle=256)
    last = la["t"] >= la["t"][-1] - 2 * np.pi / 1.0 - 1e-9              # the last full cycle, selected by time
    ax[1, 1].plot(la["strain"][last] / amp, la["stress"][last] / np.max(np.abs(la["stress"][last])), color=col, label=f"strain amplitude {amp:g}")
ax[1, 1].set(xlabel="strain / amplitude", ylabel="stress / maximum", title="LAOS Lissajous curves (Giesekus)")
ax[1, 1].legend(frameon=False, fontsize=8)

# 6. thixotropic hysteresis loop
p = thixotropy.Params(tau_y=5.0, eta_inf=0.5, d_eta=20.0, k_build=0.05, k_break=0.02)
for t_ramp, col in ((10.0, "C0"), (100.0, "C1")):
    loop = thixotropy.hysteresis_loop(p, rate_max=50.0, t_ramp=t_ramp)
    ax[1, 2].plot(loop["rate_up"], loop["stress_up"], color=col, label=f"ramp {t_ramp:g} s each way")
    ax[1, 2].plot(loop["rate_down"], loop["stress_down"], color=col, ls="--")
ax[1, 2].set(xlabel="shear rate (1/s)", ylabel="stress (Pa)", title="Thixotropic hysteresis loops (solid up, dashed down)")
ax[1, 2].legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig(OUT / "gallery.png", dpi=130)
print("wrote", OUT / "gallery.png")
