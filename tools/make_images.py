"""Generate the README images (docs/images/*.png and *.gif) from the library. Run: python tools/make_images.py"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from engrheo import constitutive, flows, models, thixotropy, viscoelastic  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=110)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ static figures
fig, ax = plt.subplots(figsize=(5.6, 4))
rate = np.logspace(-2, 3, 200)
for name, eta in (("Newtonian", models.viscosity("newtonian", rate, mu=1.0)), ("power law, n = 0.5", models.viscosity("power_law", rate, K=3.0, n=0.5)),
                  ("Carreau", models.viscosity("carreau", rate, eta0=20.0, eta_inf=0.01, lam=2.0, n=0.4)),
                  ("Herschel-Bulkley", models.viscosity("herschel_bulkley", rate, tau_y=5.0, K=2.0, n=0.6))):
    ax.loglog(rate, eta, label=name)
ax.set(xlabel="shear rate (1/s)", ylabel="viscosity (Pa s)", title="Flow curves of the classical models")
ax.legend(frameon=False, fontsize=8)
save(fig, "flow_curves.png")

fig, ax = plt.subplots(figsize=(5.6, 4))
R = 0.05
r = np.linspace(-R, R, 400)
for name, v in (("Newtonian", flows.velocity_power_law(np.abs(r), K=1.0, n=1.0, R=R, dp_per_L=200.0)),
                ("power law, n = 0.4", flows.velocity_power_law(np.abs(r), K=1.0, n=0.4, R=R, dp_per_L=200.0)),
                ("Herschel-Bulkley (plug)", flows.velocity_hb(np.abs(r), tau_y=2.0, K=1.0, n=0.6, R=R, dp_per_L=200.0))):
    ax.plot(v / v.max(), r / R, label=name)
ax.set(xlabel="velocity / maximum", ylabel="r / R", title="Laminar pipe-flow profiles")
ax.legend(frameon=False, fontsize=8, loc="center left")
save(fig, "pipe_profiles.png")

fig, ax = plt.subplots(figsize=(5.6, 4))
omega = np.logspace(-3, 3, 300)
g, tau = [2e4, 1e4, 5e3, 2e3], [10.0, 1.0, 0.1, 0.01]
Gp, Gpp = viscoelastic.moduli(g, tau, omega)
ax.loglog(omega, Gp, label="G' (storage)")
ax.loglog(omega, Gpp, label="G'' (loss)")
ax.set(xlabel="angular frequency (rad/s)", ylabel="modulus (Pa)", title="Frequency sweep of a 4-mode Maxwell spectrum")
ax.legend(frameon=False)
save(fig, "frequency_sweep.png")

modes = constitutive.modes_from_spectrum([1000.0], [1.0], "giesekus", alpha=0.3)
fig, ax = plt.subplots(figsize=(5.6, 4))
t = np.linspace(0, 6, 400)
for gd in (0.5, 2.0, 8.0):
    su = constitutive.startup_shear(modes, gd, t)
    ax.plot(t, su["eta_plus"] / 1000.0, label=f"shear rate {gd:g} 1/s (Wi = {gd:g})")
ax.set(xlabel="time (s)", ylabel="eta+ / eta0", title="Giesekus start-up of shear: stress overshoot")
ax.legend(frameon=False, fontsize=8)
save(fig, "startup_overshoot.png")

fig, ax = plt.subplots(figsize=(5.6, 4))
for amp, col in ((0.2, "C0"), (2.0, "C1"), (6.0, "C3")):
    la = constitutive.laos_shear(modes, amp, 1.0, n_cycles=25, points_per_cycle=256)
    last = la["t"] >= la["t"][-1] - 2 * np.pi - 1e-9
    ax.plot(la["strain"][last] / amp, la["stress"][last] / np.max(np.abs(la["stress"][last])), color=col, label=f"strain amplitude {amp:g}")
ax.set(xlabel="strain / amplitude", ylabel="stress / maximum", title="LAOS Lissajous curves (Giesekus)")
ax.legend(frameon=False, fontsize=8)
save(fig, "lissajous.png")

fig, ax = plt.subplots(figsize=(5.6, 4))
p = thixotropy.Params(tau_y=5.0, eta_inf=0.5, d_eta=20.0, k_build=0.05, k_break=0.02)
for t_ramp, col in ((10.0, "C0"), (100.0, "C1")):
    loop = thixotropy.hysteresis_loop(p, rate_max=50.0, t_ramp=t_ramp)
    ax.plot(loop["rate_up"], loop["stress_up"], color=col, label=f"ramp {t_ramp:g} s each way")
    ax.plot(loop["rate_down"], loop["stress_down"], color=col, ls="--")
ax.set(xlabel="shear rate (1/s)", ylabel="stress (Pa)", title="Thixotropic hysteresis loops (solid up, dashed down)")
ax.legend(frameon=False, fontsize=8)
save(fig, "thixotropy_loop.png")

# ------------------------------------------------------------------ hero animation: LAOS as the amplitude grows
amps = np.geomspace(0.05, 8.0, 45)
curves = []
for amp in amps:
    la = constitutive.laos_shear(modes, amp, 1.0, n_cycles=25, points_per_cycle=200)
    last = la["t"] >= la["t"][-1] - 2 * np.pi - 1e-9
    curves.append((la["strain"][last], la["stress"][last], amp))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.6))
line1, = a1.plot([], [], "C3", lw=2)
line2, = a2.plot([], [], "C0", lw=2)
a1.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), xlabel="strain / amplitude", ylabel="stress / maximum")
a2.set(xlim=(0, 2 * np.pi), ylim=(-1.1, 1.1), xlabel="omega t (rad)", ylabel="stress / maximum")
title = fig.suptitle("")
fig.tight_layout()


def update(i):
    strain, stress, amp = curves[i]
    line1.set_data(strain / amp, stress / np.max(np.abs(stress)))
    line2.set_data(np.linspace(0, 2 * np.pi, stress.size), stress / np.max(np.abs(stress)))
    title.set_text(f"Large-amplitude oscillatory shear (Giesekus): strain amplitude {amp:.2f} - " + ("linear" if amp < 0.5 else "nonlinear" if amp < 3 else "strongly nonlinear"))
    return line1, line2, title


anim = FuncAnimation(fig, update, frames=len(curves), blit=False)
anim.save(OUT / "laos.gif", writer=PillowWriter(fps=8), dpi=80)
plt.close(fig)
print("wrote laos.gif")
