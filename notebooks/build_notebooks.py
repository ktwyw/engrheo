"""Generate the course notebooks (single source of truth) and execute them.

    python notebooks/build_notebooks.py            # write and execute all notebooks
    python notebooks/build_notebooks.py 03 15      # only notebooks whose names start with 03 or 15
    python notebooks/build_notebooks.py --no-run   # write without executing
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
NOTEBOOKS: dict[str, list] = {}
EXTRAS: dict[str, dict] = {}

COLAB = '''try:                      # on Google Colab (or anywhere engrheo is missing): install it from GitHub
    import engrheo
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/ktwyw/engrheo"], check=True)'''

STYLE = '''plt.rcParams.update({"figure.figsize": (7, 4), "figure.dpi": 90, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

import inspect
from IPython.display import Code

def show_source(obj):
    """Display the source code of a library function or class."""
    return Code(inspect.getsource(obj), language="python")

def heading(text):
    print(f"\\n{text}\\n" + "=" * len(text))'''


def md(text):
    return new_markdown_cell(text.strip("\n"))


def code(text):
    return new_code_cell(text.strip("\n"))


def setup_cell(extra_imports=()):
    lines = ["import matplotlib.pyplot as plt", "import numpy as np", "import pandas as pd"]
    for imp in extra_imports:
        if imp not in lines:
            lines.append(imp)
    return code(COLAB + "\n" + "\n".join(lines) + "\n" + STYLE)


def read_data(name):
    """Code snippet reading a bundled CSV file (comment lines start with #)."""
    return f'pd.read_csv(datasets.path("{name}"), comment="#")'


# =====================================================================================================
NOTEBOOKS["00_python_for_rheology_data"] = [
    md(r"""
# 00 · Python for rheology data

**Goal:** the Python tools used throughout the course - reading rheometer exports, units, log-scale
plots, local slopes and a first model fit - on one realistic data set.

**The data.** A flow curve of a 0.5 % xanthan gum solution (a thickener used in food, cosmetics and
drilling fluids) measured with a cone-plate rheometer. The file is a plain CSV export with a short
description at the top - the format every rheometer can produce.
"""),
    setup_cell(["from engrheo import datasets, fitting, geometry"]),
    md("## Look at the file, then read it"),
    code(r"""path = datasets.path("xanthan_flow_curve.csv")
with open(path, encoding="utf-8") as fh:
    print("".join(fh.readlines()[:6]))
df = pd.read_csv(path, comment="#")
print(df.shape); print(df.head(3))"""),
    md(r"""
The comment lines hold what the numbers mean: geometry, temperature, units. Always read them - a flow
curve without its temperature and geometry is only half a result.

## Check the units and the internal consistency
Viscosity should equal stress divided by shear rate, and the stress should follow from the torque with the
cone-plate formula $\tau = 3M/(2\pi R^3)$:
"""),
    code(r"""rate, tau, eta, M = df.shear_rate_1_s.to_numpy(), df.shear_stress_Pa.to_numpy(), df.viscosity_Pa_s.to_numpy(), df.torque_Nm.to_numpy()
print("max |eta - tau/rate| / eta:", np.max(np.abs(eta - tau / rate) / eta))
tau_from_torque, rate_check = geometry.cone_plate(M, rate * np.radians(1.0), R=0.025, theta=np.radians(1.0))
print("max |tau from torque - tau| / tau:", np.max(np.abs(tau_from_torque - tau) / tau))"""),
    md(r"""
Both agree to the rounding of the file. Checks like these catch unit mistakes (mPa·s against Pa·s, rpm
against rad/s) before they propagate into an analysis.

## Why rheologists plot on log scales
"""),
    code(r"""fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
a1.plot(rate, eta, "o", ms=4); a1.set(xlabel="shear rate (1/s)", ylabel="viscosity (Pa s)", title="Linear axes")
a2.loglog(rate, eta, "o", ms=4); a2.set(xlabel="shear rate (1/s)", ylabel="viscosity (Pa s)", title="Log-log axes")
plt.show()
print(f"the viscosity falls from {eta.max():.3g} to {eta.min():.3g} Pa s: a factor of {eta.max()/eta.min():.0f}")"""),
    md(r"""
On linear axes everything below about 50 1/s is squeezed into one corner; on log axes the whole curve is
visible: a **Newtonian plateau** at low rates, then **shear thinning**. Flow curves span many decades of
shear rate, so log scales are the norm.

## The local slope: how strongly does it shear-thin?
In the shear-thinning region the curve is nearly straight on log axes, $\eta \propto \dot\gamma^{\,n-1}$. The local
slope $d\ln\eta / d\ln\dot\gamma$ shows where that holds:
"""),
    code(r"""slope = np.gradient(np.log(eta), np.log(rate))
plt.semilogx(rate, slope, "o-", ms=4)
plt.axhline(0, color="k", lw=0.8)
plt.xlabel("shear rate (1/s)"); plt.ylabel("d ln(eta) / d ln(rate)"); plt.title("Local slope of the flow curve")
plt.show()
high = rate > 10
print(f"mean slope above 10 1/s: {slope[high].mean():.3f}  ->  power-law index n = {1 + slope[high].mean():.3f}")"""),
    md(r"""
The slope is near 0 on the plateau and settles to a nearly constant negative value at high rates - the
power-law region. Local slopes of noisy data scatter; averaging over a region steadies them.

## A first fit
A power law $\tau = K\dot\gamma^n$ is a straight line in log-log coordinates, so it can be fitted by linear regression of
$\ln\tau$ on $\ln\dot\gamma$ - which is exactly what `fitting.fit_flow_curve` does for this model:
"""),
    code(r"""b, a = np.polyfit(np.log(rate[high]), np.log(tau[high]), 1)
print(f"straight line in log-log coordinates: n = {b:.4f}, K = {np.exp(a):.4f} Pa s^n")
fit = fitting.fit_flow_curve(rate[high], tau[high], "power_law")
print(fit)"""),
    md(r"""
Same numbers - and the library adds the uncertainty of each parameter. Notebook 03 fits models that cover
the whole curve.

## Interpolating on the right scale
What is the viscosity at 50 1/s, between two measured points? Interpolate where the curve is straight -
on the log scale:
"""),
    code(r"""target = 50.0
linear = np.interp(target, rate, eta)
logscale = np.exp(np.interp(np.log(target), np.log(rate), np.log(eta)))
print(f"linear interpolation {linear:.4f} Pa s, log-log interpolation {logscale:.4f} Pa s")"""),
    md(r"""
The two differ: linear interpolation draws a straight line on linear axes, which for a power-law curve
lies above the true curve between the points. The difference grows with the spacing of the data.

## Exercises
1. The rheometer's minimum reliable torque is 1 µN·m. What is the lowest shear stress this cone (R = 25 mm)
   can measure? Draw the corresponding "low-torque limit" line $\eta_{min} = \tau_{min}/\dot\gamma$ on the viscosity plot.
   Are all the data safely above it?
2. Read the file again with `np.loadtxt` instead of pandas (hint: `delimiter`, `comments`, `skiprows`).
   When is each tool more convenient?
"""),
]

NOTEBOOKS["01_what_rheology_measures"] = [
    md(r"""
# 01 · What rheology measures

**Question.** A ketchup bottle, a drilling mud and a polymer melt all have "a viscosity" on their data
sheet. Why is one number not enough - and which number matters for a given process?

Rheology describes how materials flow and deform. This notebook introduces the central quantities: shear
stress, shear rate and viscosity; the shear rates of real processes; and two dimensionless numbers that
decide whether a material behaves like a liquid or a solid.
"""),
    setup_cell(["from engrheo import models"]),
    md(r"""
## Simple shear
A layer of fluid of thickness $h$ between a fixed plate and a plate moving at speed $V$ is sheared at the
**shear rate** $\dot\gamma = V/h$ (units 1/s). The force $F$ per plate area $A$ is the **shear stress** $\tau = F/A$ (Pa).
Their ratio is the **viscosity** $\eta = \tau/\dot\gamma$ (Pa·s). For a Newtonian fluid (water, oils, honey) $\eta$ is a
constant; for most engineering fluids it depends on the shear rate.
"""),
    code(r"""fluids = {"water": 1.0e-3, "olive oil": 0.08, "glycerol": 1.4, "honey": 10.0}   # Pa s, about 20 degC
V, h, A = 0.1, 1e-3, 0.01                     # 0.1 m/s, 1 mm film, 10 cm x 10 cm plate
rate = V / h
for name, mu in fluids.items():
    print(f"{name:<10}: viscosity {mu:8.4g} Pa s -> stress {mu*rate:9.3g} Pa, force on the plate {mu*rate*A:8.3g} N")"""),
    md(r"""
## Shear rates of real processes
Each process shears a fluid at a characteristic rate. Rough estimates (velocity divided by a length):
"""),
    code(r"""processes = {
    "particle settling (100 µm, 1 µm/s)": 1e-6 / 100e-6,
    "draining from a wall": 1e-2 / 1e-3,
    "pouring from a bottle": 1.0 / 5e-3,
    "pipe flow (8V/D, 1 m/s, 5 cm)": 8 * 1.0 / 0.05,
    "stirring (Metzner-Otto, 11 x 2 rev/s)": 11 * 2.0,
    "brushing paint (1 m/s, 0.1 mm)": 1.0 / 1e-4,
    "spraying or high-speed coating": 1e5,
}
names, rates = list(processes), np.array(list(processes.values()))
plt.figure(figsize=(8, 3.8)); plt.barh(names, rates); plt.xscale("log")
plt.xlabel("typical shear rate (1/s)"); plt.show()
eta_x = models.viscosity("carreau", rates, 35.0, 0.002, 12.0, 0.25)   # the xanthan solution of notebook 00
for n_, r_, e_ in zip(names, rates, eta_x):
    print(f"{n_:<40} {r_:10.3g} 1/s -> xanthan viscosity {e_:9.3g} Pa s")"""),
    md(r"""
The rates span about seven decades, and over that range the xanthan solution's viscosity changes by a
factor of about ten thousand: thick at rest (it keeps particles suspended), thin when pumped or sprayed. A single
"viscosity" is meaningless for such a fluid unless the shear rate is stated - **measure at the shear rates
of your process**.

## Liquid or solid? The Deborah number
Many materials remember their shape for a characteristic **relaxation time** $\lambda$. The Deborah number
compares it with the time scale of the observation, $De = \lambda / t_{obs}$: for $De \ll 1$ the material flows like a
liquid, for $De \gg 1$ it responds like a solid.
"""),
    code(r"""cases = [("silicone putty bouncing (contact 1 ms)", 1.0, 1e-3),
         ("silicone putty left on a table (1 h)", 1.0, 3600.0),
         ("polymer melt filling a mould (1 s)", 0.5, 1.0),
         ("glacier ice flowing over a century", 1e8, 3e9),
         ("water in a pipe (1 s)", 1e-12, 1.0)]
for name, lam, t in cases:
    De = lam / t
    print(f"{name:<40} De = {De:9.2g} -> {'solid-like' if De > 1 else 'liquid-like' if De < 0.1 else 'viscoelastic'}")"""),
    md(r"""
The same putty is a bouncing solid or a slowly spreading liquid depending on how fast it is deformed.
In a steady flow, the analogous **Weissenberg number** $Wi = \lambda\dot\gamma$ measures how strongly the flow
stretches the material's microstructure; elastic effects (notebooks 07-11) become important when $Wi \gtrsim 1$.

## Exercises
1. Water and a power-law fluid with $n$ = 0.5 flow at a mean velocity of 2 m/s in a 5 cm pipe. The wall shear
   rate of a Newtonian fluid is $8V/D$; for a power-law fluid it is $(3n + 1)/(4n) \cdot 8V/D$ (notebook 02). By
   how much does shear thinning increase the wall shear rate?
2. A polymer melt with relaxation time 0.5 s is extruded through a 1 mm die at 0.5 m/s. Estimate the shear
   rate, the Weissenberg number and the Deborah number (residence time in a 20 mm die). Would you expect
   elastic effects such as die swell?
"""),
]

NOTEBOOKS["02_rheometers_and_pitfalls"] = [
    md(r"""
# 02 · Rheometers and their pitfalls

A rheometer measures a **torque and a rotation speed** (rotational instruments) or a **pressure drop and a
flow rate** (capillaries). Converting them to shear stress and shear rate needs a geometry formula - and
the simple formulas assume a Newtonian fluid, no slip at the walls, and no end effects. This notebook
shows when those assumptions fail, by how much, and how to correct for it.
"""),
    setup_cell(["from engrheo import datasets, fitting, geometry, models"]),
    md(r"""
## Cone-plate: the reference geometry
A small cone angle makes the shear rate the same everywhere in the gap, so $\tau = 3M/(2\pi R^3)$ and
$\dot\gamma = \Omega/\theta$ hold for any fluid. Parallel plates are easier to load, but the shear rate grows from
zero at the centre to its maximum at the rim, and the simple (Newtonian) stress formula becomes wrong for
shear-thinning fluids:
"""),
    code(r"""R, h = 0.025, 1e-3
omega = np.logspace(-2, 2, 25)
rate_R = omega * R / h
K, n = 2.0, 0.45                                    # a shear-thinning fluid
M = 2 * np.pi * K * rate_R**n * R**3 / (n + 3)      # exact torque for a power-law fluid between plates
tau_newt, _, _ = geometry.parallel_plate(M, omega, R, h, correct=False)
tau_corr, _, n_est = geometry.parallel_plate(M, omega, R, h)
print(f"Newtonian formula overestimates the rim stress by {np.mean(tau_newt/(K*rate_R**n)) - 1:.1%}")
print(f"corrected (Weissenberg-Rabinowitsch type) formula: max error {np.max(np.abs(tau_corr/(K*rate_R**n) - 1)):.1e}")"""),
    md(r"""
## Couette cells and yield-stress fluids
In a concentric-cylinder (Couette) cell the stress falls as $1/r^2$ across the gap. For a yield-stress fluid
at low speeds, the stress near the outer cylinder can fall *below* the yield stress: only part of the gap
flows, and the usual conversion formulas no longer apply.
"""),
    code(r"""Rb, Rr, L = 0.017245, 0.018415, 0.038                # Fann 35 bob and rotor (drilling-fluid viscometer)
from scipy import optimize
ty, mup = 5.0, 0.03                                 # a Bingham fluid (a drilling-mud-like yield stress)
M_yield = 2 * np.pi * Rb**2 * L * ty                 # below this torque nothing flows
for rpm in (3, 6, 100, 600):
    omega_ = rpm * 2 * np.pi / 60
    M_ = optimize.brentq(lambda m: geometry.couette_speed("bingham", (ty, mup), m, Rb, Rr, L) - omega_, M_yield * 1.0000001, 1.0)
    r_c = min(Rr, np.sqrt(M_ / (2 * np.pi * L * ty)))   # radius where the stress falls to the yield stress
    print(f"{rpm:4d} rpm: torque {M_*1e3:6.3f} mN m, sheared fraction of the gap {(r_c - Rb)/(Rr - Rb):6.1%}")"""),
    md(r"""
At the low speeds used to estimate yield stresses - the 3 and 6 rpm readings of drilling-fluid viscometers -
only part of the gap is sheared, so the standard conversion (which assumes the whole gap flows) does not
apply to exactly the readings used for the yield stress. This is one reason why field "yield points"
differ from true yield stresses (notebook 04).

## Capillary rheometers: two corrections
A capillary rheometer pushes a melt through a die and records the pressure drop. Two corrections turn the
raw data into a flow curve:

1. **Bagley:** part of the pressure is lost entering the die. With several die lengths at the same flow rate,
   $\Delta P = 2\tau_w (L/R) + \Delta P_{ends}$ - a straight line in $L/R$ whose slope gives the true wall stress.
2. **Weissenberg-Rabinowitsch:** the apparent shear rate $4Q/(\pi R^3)$ assumes a parabolic (Newtonian) profile;
   for shear-thinning fluids the true wall rate is higher by $(3n' + 1)/(4n')$.
"""),
    code(r"""melt = pd.read_csv(datasets.path("polymer_melt_capillary.csv"), comment="#")
melt["L_over_R"] = melt.die_length_mm / melt.die_radius_mm
Rd = melt.die_radius_mm.iloc[0] * 1e-3
tau_w, e_bag, app = [], [], []
fig, ax = plt.subplots()
for Q, grp in melt.groupby("flow_rate_m3_s"):
    b = geometry.bagley(grp.L_over_R, grp.pressure_drop_Pa)
    tau_w.append(b["tau_w"]); e_bag.append(b["e"]); app.append(4 * Q / (np.pi * Rd**3))
    ax.plot(grp.L_over_R, grp.pressure_drop_Pa / 1e6, "o-", ms=4)
ax.set(xlabel="die L/R", ylabel="pressure drop (MPa)", title="Bagley plot: one line per flow rate"); plt.show()
tau_w, app = np.array(tau_w), np.array(app)
print(f"Bagley end correction e from {min(e_bag):.1f} to {max(e_bag):.1f} die radii (grows with the rate)")
_, rate_w, n_prime = geometry.capillary(2 * tau_w, app * np.pi * Rd**3 / 4, R=Rd, L=Rd)   # tau_w already corrected
print(f"local power-law index n' from {n_prime.min():.2f} to {n_prime.max():.2f}")"""),
    code(r"""single = melt[melt.L_over_R == 10]
tau_app_raw = single.pressure_drop_Pa * Rd / (2 * 10 * Rd)          # one die, no corrections
true_rate = np.logspace(0.5, 4, 100)
plt.loglog(4 * single.flow_rate_m3_s / (np.pi * Rd**3), tau_app_raw / (4 * single.flow_rate_m3_s / (np.pi * Rd**3)),
           "s", label="raw: one die (L/R = 10), no corrections")
plt.loglog(app, tau_w / app, "^", label="Bagley-corrected, apparent rate")
plt.loglog(rate_w, tau_w / rate_w, "o", label="Bagley + Rabinowitsch")
plt.loglog(true_rate, models.viscosity("carreau", true_rate, 3000.0, 0.0, 0.3, 0.3), "k-", lw=1, label="true viscosity")
plt.xlabel("shear rate (1/s)"); plt.ylabel("viscosity (Pa s)"); plt.legend(fontsize=8); plt.show()
raw_eta = (tau_app_raw / (4 * single.flow_rate_m3_s / (np.pi * Rd**3))).to_numpy()
raw_rate = (4 * single.flow_rate_m3_s / (np.pi * Rd**3)).to_numpy()
print(f"raw single-die viscosity / true viscosity at the same nominal rate: "
      f"{np.min(raw_eta / models.viscosity('carreau', raw_rate, 3000.0, 0.0, 0.3, 0.3)):.2f} to "
      f"{np.max(raw_eta / models.viscosity('carreau', raw_rate, 3000.0, 0.0, 0.3, 0.3)):.2f}")"""),
    md(r"""
Uncorrected, a single die overestimates the viscosity because the entrance pressure loss is counted as
if it were friction in the die; the Bagley correction removes that, and the Rabinowitsch correction then
moves each point to its true shear rate. (The file is synthetic, generated from a known melt, which is how
the "true" curve can be drawn.)

## Wall slip
Pastes, suspensions and gels often slip at smooth walls: a thin, particle-depleted layer lubricates the
surface. The hallmark: **the measured flow curve depends on the gap**.
"""),
    code(r"""paste = pd.read_csv(datasets.path("toothpaste_plates_slip.csv"), comment="#")
for gap, grp in paste.groupby("gap_mm"):
    plt.loglog(grp.apparent_shear_rate_1_s, grp.shear_stress_Pa, "o-", ms=4, label=f"gap {gap} mm")
plt.xlabel("apparent shear rate (1/s)"); plt.ylabel("shear stress (Pa)"); plt.legend(); plt.title("Toothpaste, smooth plates")
plt.show()
rows = []
for stress, grp in paste.groupby("shear_stress_Pa"):
    m = geometry.mooney(grp.gap_mm * 1e-3, grp.apparent_shear_rate_1_s, kind="plates")
    true_slip = 4e-6 * stress * 1e3                   # the slip law used to generate the file (mm/s)
    share = 2 * true_slip * 1e-3 / 0.5e-3 / grp.apparent_shear_rate_1_s.iloc[0]
    rows.append((stress, m["true_rate"], m["slip_velocity"] * 1e3, true_slip, share))
res = pd.DataFrame(rows, columns=["stress_Pa", "slip_free_rate_1_s", "slip_velocity_mm_s", "true_slip_mm_s",
                                  "slip_share_at_0.5mm"])
print(res.round(3).to_string(index=False))"""),
    md(r"""
The Mooney analysis separates the true (slip-free) shear rate from the slip velocity at each stress. Below
the yield stress the true rate is essentially zero - the paste does not flow; it only slides on the plates.
Without the correction, that slip looks like flow, and the yield stress is missed.

The comparison with the true slip velocities (known here because the file is synthetic) shows the method's
limit: it is accurate where slip makes up a large share of the apparent rate (low stresses), but at high
stresses slip contributes only 1-2 % of the measured rate - no more than the 2 % measurement scatter - and
the estimates become meaningless, even negative. A Mooney analysis needs slip to be a substantial part of
the signal. In practice slip is prevented with roughened or serrated plates, and checked by testing at two
gaps.

## Instrument limits
Every measurement has a window: too little torque and the signal drowns in noise and friction; too
much speed and inertia or secondary flows (Taylor vortices in Couette cells) add to the torque.
"""),
    code(r"""M_min = 1e-7                                         # N m, a typical minimum for a good rotational rheometer
for name, (Rg, thg) in {"cone 50 mm / 1 deg": (0.025, np.radians(1)), "cone 25 mm / 1 deg": (0.0125, np.radians(1))}.items():
    tau_min, _ = geometry.cone_plate(M_min, 1.0, Rg, thg)
    print(f"{name}: lowest stress {tau_min*1e3:.2f} mPa -> lowest viscosity at 1 1/s {tau_min*1e3:.2f} mPa s, at 0.01 1/s {tau_min/0.01*1e3:.0f} mPa s")"""),
    md(r"""
A larger geometry lowers the minimum measurable stress by the cube of the radius ratio. Low-viscosity
fluids at low shear rates are therefore measured with large cones or double-gap cylinders; data below the
low-torque limit should be discarded, not fitted.
"""),
]


# =====================================================================================================
# Teaching layer
def _x(objectives, prereq, time, inside, exercises=(), implement=""):
    return dict(objectives=objectives, prereq=prereq, time=time, inside=inside, exercises=list(exercises),
                implement=implement)


EXTRAS["00_python_for_rheology_data"] = _x(
    ["read a rheometer CSV export with pandas and check its units", "plot flow curves on log scales",
     "compute local slopes and a power-law fit", "interpolate on the right scale"],
    "basic Python (variables, lists, functions); no rheology needed", "45 min",
    [md(r"""
## Inside the algorithm: the local slope on an uneven grid
`np.gradient` estimates derivatives with second-order differences that allow unequal spacing - here in
$\ln\dot\gamma$. For interior points it weights the two neighbouring slopes by the opposite spacing:
"""), code(r"""x, y = np.log(rate), np.log(eta)
h1, h2 = x[1:-1] - x[:-2], x[2:] - x[1:-1]
interior = (h1**2 * y[2:] + (h2**2 - h1**2) * y[1:-1] - h2**2 * y[:-2]) / (h1 * h2 * (h1 + h2))
print("max difference from np.gradient:", np.max(np.abs(interior - np.gradient(y, x)[1:-1])))""")],
    implement="**Implement it yourself:** write `loglog_interp(x0, x, y)` that interpolates linearly in log-log "
              "coordinates, and check it against the notebook's result at 50 1/s.")

EXTRAS["01_what_rheology_measures"] = _x(
    ["define shear stress, shear rate and viscosity", "estimate the shear rates of real processes",
     "explain why one viscosity number is not enough for most fluids", "use the Deborah and Weissenberg numbers"],
    "notebook 00; basic mechanics (force, stress)", "45 min",
    [md(r"""
## Inside the algorithm: the Newtonian torque on a cone and on plates
The torque is the stress integrated over the plate, $M = \int_0^R \tau(r)\,2\pi r^2\,dr$. For a cone the stress is
uniform; for plates it grows with $r$ because the shear rate does:
"""), code(r"""from scipy import integrate
mu, R_, h_, th_, om_ = 1.0, 0.025, 1e-3, np.radians(1.0), 2.0
M_cone = integrate.quad(lambda r: mu * om_ / th_ * 2 * np.pi * r**2, 0, R_)[0]
M_plates = integrate.quad(lambda r: mu * om_ * r / h_ * 2 * np.pi * r**2, 0, R_)[0]
print(f"cone:   M = {M_cone:.6e} N m, formula 2 pi R^3 mu omega/(3 theta) = {2*np.pi*R_**3*mu*om_/(3*th_):.6e}")
print(f"plates: M = {M_plates:.6e} N m, formula pi R^4 mu omega/(2 h)      = {np.pi*R_**4*mu*om_/(2*h_):.6e}")""")],
    implement="**Implement it yourself:** write `wall_shear_rate(V, D, n=1.0)` for pipe flow and use it to make a "
              "table of wall shear rates for velocities 0.1-3 m/s in a 5 cm pipe, for n = 1, 0.6 and 0.3.")

EXTRAS["02_rheometers_and_pitfalls"] = _x(
    ["convert torque and speed (or pressure and flow rate) into stress and shear rate",
     "apply the Weissenberg-Rabinowitsch, Bagley and Mooney corrections",
     "recognise partially sheared gaps, wall slip and instrument limits"],
    "notebooks 00-01", "75 min",
    [md(r"""
## Inside the algorithm: the Weissenberg-Rabinowitsch correction
From the Bagley-corrected wall stresses and apparent rates, the local slope $n' = d\ln\tau_w/d\ln\dot\gamma_a$ gives the
true wall rate $\dot\gamma_w = \dot\gamma_a(3n' + 1)/(4n')$:
"""), code(r"""order = np.argsort(app)
n_hand = np.gradient(np.log(tau_w[order]), np.log(app[order]))
rate_hand = app[order] * (3 * n_hand + 1) / (4 * n_hand)
print("max difference from geometry.capillary:", np.max(np.abs(rate_hand / rate_w[order] - 1)))""")],
    ["Fit a Carreau model to the melt data three ways - raw single-die data, Bagley-corrected data at the apparent "
     "rate, and fully corrected data - and compare $\\eta_0$ and $n$ with the true values (3000 Pa·s, 0.3).",
     "Taylor vortices appear in a Couette cell when $Ta = \\Omega^2 R_i (R_o - R_i)^3 / \\nu^2$ exceeds about 1700. "
     "At what rotation speed does this happen for water in a cell with $R_i$ = 12.5 mm, $R_o$ = 13.6 mm, and "
     "what shear rate does that correspond to?"],
    "**Implement it yourself:** do the Mooney analysis for one stress level of the toothpaste data by hand "
    "(straight-line fit of apparent rate against 1/gap) and compare with `geometry.mooney`.")


# =====================================================================================================
NOTEBOOKS["03_shear_thinning_fluids"] = [
    md(r"""
# 03 · Shear-thinning fluids

**Problem.** A beverage maker thickens a fruit drink with 0.5 % xanthan gum so that pulp stays suspended on
the shelf, yet the drink must pour and pump easily. Which model describes the flow curve, what are its
parameters (with uncertainties), and what viscosity does the drink have at rest and when poured?
"""),
    setup_cell(["from scipy import optimize", "from engrheo import datasets, fitting, models"]),
    code(r"""df = pd.read_csv(datasets.path("xanthan_flow_curve.csv"), comment="#")
rate, tau = df.shear_rate_1_s.to_numpy(), df.shear_stress_Pa.to_numpy()
fits, table_text = fitting.compare_models(rate, tau, ("power_law", "cross", "carreau", "carreau_yasuda", "sisko"))
print(table_text)"""),
    md(r"""
AIC balances goodness of fit against the number of parameters; differences below about 2 are not
meaningful. The power law and the Sisko model have no low-rate plateau and are far behind. Carreau fits
within the measurement noise (about 2 %); Carreau-Yasuda's extra parameter brings no improvement
($\Delta$AIC about 1), and the Cross model, although it has a plateau, describes this transition clearly worse. Choose
the simplest model that fits within the noise - here Carreau.
"""),
    code(r"""best = next(f for f in fits if f.model == "carreau")
print(best)
rr = np.logspace(-2.5, 3.5, 200)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
a1.loglog(rate, tau / rate, "o", ms=4, label="data")
for f in fits:
    a1.loglog(rr, f.predict_viscosity(rr), lw=1, label=f.model)
a1.set(xlabel="shear rate (1/s)", ylabel="viscosity (Pa s)"); a1.legend(fontsize=7)
a2.semilogx(rate, 100 * best.log_residuals, "o", ms=4); a2.axhline(0, color="k", lw=0.8)
a2.set(xlabel="shear rate (1/s)", ylabel="residual (%)", title="Carreau: relative residuals")
plt.show()"""),
    md(r"""
The residuals scatter randomly around zero with no trend - the model describes the data within the
measurement noise. The parameters have a physical meaning: $\eta_0$ is the viscosity at rest, $1/\lambda$ the
shear rate where shear thinning sets in, and $n$ the power-law index at high rates. Note the infinite-shear
viscosity $\eta_\infty$: its confidence interval includes zero, because the data never reach the upper plateau - the
data cannot determine it, and it should not be quoted.

## Why fit on the log scale?
Ordinary least squares on the stress itself minimises *absolute* errors; the high-rate points, whose
stresses are largest, then dominate the fit:
"""),
    code(r"""p_lin, _ = optimize.curve_fit(models.carreau, rate, tau, p0=list(best.params.values()),
                              bounds=([0, 0, 0, 0.01], [np.inf, np.inf, np.inf, 1.5]), maxfev=20000)
rel_lin = models.carreau(rate, *p_lin) / tau - 1
rel_log = best.predict_stress(rate) / tau - 1
for name, rel in (("absolute residuals (curve_fit on stress)", rel_lin), ("relative residuals (log scale)", rel_log)):
    print(f"{name:<42}: largest error {np.max(np.abs(rel)):6.1%}, "
          f"below 0.1 1/s {np.max(np.abs(rel[rate < 0.1])):6.1%}")
print(f"eta0: log fit {best.params['eta0']:.2f} Pa s, absolute fit {p_lin[0]:.2f} Pa s")"""),
    md(r"""
Rheometer errors are roughly proportional to the signal, and flow curves span decades: relative
(logarithmic) residuals give every decade its fair weight.

## The danger of extrapolating a power law
"""),
    code(r"""hi = rate >= 10
pl = fitting.fit_flow_curve(rate[hi], tau[hi], "power_law")
for g in (0.01, 1.0, 100.0):
    print(f"at {g:6g} 1/s: power law from 10-1000 1/s predicts {pl.predict_viscosity(g):9.3g} Pa s, "
          f"Carreau {best.predict_viscosity(g):8.3g} Pa s")"""),
    md(r"""
Inside its fitting range the power law is excellent; outside it, the error grows without limit, because a
power law has no plateau - its viscosity tends to infinity at zero shear rate. For anything happening at
rest (sedimentation, levelling, sagging) use a model with a zero-shear viscosity, and measure low enough
shear rates to see the plateau.
"""),
]

NOTEBOOKS["04_yield_stress_fluids"] = [
    md(r"""
# 04 · Yield-stress fluids

Drilling muds carry rock cuttings out of a well and must hold them in suspension when circulation stops.
Chocolate must coat a biscuit without running off. Fresh concrete must stand in a pile yet flow when pumped.
All of them are **yield-stress fluids**: below a critical stress they behave like soft solids, above it they
flow. This notebook fits yield-stress models to three industrial data sets and shows why "the" yield
stress is a slippery quantity.
"""),
    setup_cell(["from engrheo import datasets, fitting, geometry, models"]),
    md(r"""
## A drilling mud: field formulas against proper fits
Drilling-fluid laboratories record six dial readings of a Fann 35 viscometer and use simple API formulas:
plastic viscosity PV = $\theta_{600} - \theta_{300}$ and yield point YP = $\theta_{300}$ - PV (the Bingham line through the two
highest readings).
"""),
    code(r"""mud = pd.read_csv(datasets.path("drilling_mud_fann.csv"), comment="#")
print(mud.to_string(index=False))
th = dict(zip(mud.rpm, mud.dial_reading_deg))
api = geometry.api_bingham(th[600], th[300])
print(f"\nAPI: PV = {api['PV_mPas']:.1f} mPa s, YP = {api['YP_lbf100ft2']:.1f} lbf/100 ft2 = {api['tau_y']:.2f} Pa")
tau_m, rate_m = geometry.fann35(mud.rpm.to_numpy(), mud.dial_reading_deg.to_numpy())
order = np.argsort(rate_m)
rate_m, tau_m = rate_m[order], tau_m[order]
for name in ("bingham", "herschel_bulkley", "power_law"):
    f = fitting.fit_flow_curve(rate_m, tau_m, name)
    print(f"{name:<17}: " + ", ".join(f"{k} = {v:.3g}" for k, v in f.params.items()) + f"   (scatter {100*f.sigma:.1f} %)")
print("true fluid (used to generate the readings): Herschel-Bulkley tau_y = 4.8 Pa, K = 0.35 Pa s^n, n = 0.62")"""),
    code(r"""rr = np.logspace(0, 3.1, 200)
hb = fitting.fit_flow_curve(rate_m, tau_m, "herschel_bulkley")
plt.plot(rate_m, tau_m, "ko", label="Fann readings")
plt.plot(rr, models.stress("bingham", rr, api["tau_y"], api["mu_p"]), label="API Bingham line (600/300 rpm)")
plt.plot(rr, hb.predict_stress(rr), label="Herschel-Bulkley fit (all six readings)")
plt.xscale("log"); plt.xlabel("shear rate (1/s)"); plt.ylabel("shear stress (Pa)"); plt.legend(); plt.show()"""),
    md(r"""
The API yield point is the intercept of a straight line through the two *highest* readings - an
extrapolation over two decades of shear rate to zero. For a shear-thinning mud it overestimates the stress
at low rates, which is exactly where cuttings are suspended. The Herschel-Bulkley fit uses all six readings
and lands much closer to the true yield stress. (The low-speed readings have their own problem: at 3 and
6 rpm only part of the gap is sheared - notebook 02 - so their Newtonian rate conversion is approximate.)

## Chocolate: the yield stress depends on the model
The confectionery industry has long described molten chocolate with the Casson model.
"""),
    code(r"""choc = pd.read_csv(datasets.path("chocolate_casson.csv"), comment="#")
rc, tc = choc.shear_rate_1_s.to_numpy(), choc.shear_stress_Pa.to_numpy()
fits, text = fitting.compare_models(rc, tc, ("casson", "bingham", "herschel_bulkley"))
print(text)
for f in fits:
    print(f"{f.model:<17}: yield stress {f.params['tau_y']:6.2f} Pa")
print("true fluid: Casson tau_y = 12 Pa, eta_c = 2.1 Pa s")"""),
    md(r"""
Casson and Herschel-Bulkley both fit the measured range (2-50 1/s) within the noise; Bingham fits
noticeably worse. Yet even the two good models disagree about the yield stress by almost half, and the
three together span a factor of two. None of these values is *measured*: each is an extrapolation of a
model to zero shear rate, below the lowest data point. A yield
stress is only meaningful together with the model and the shear-rate range used - which is why standards
(for chocolate, the ICA method) fix both. Direct methods - stress ramps, creep tests, oscillatory amplitude
sweeps (notebook 08) - measure the onset of flow instead of extrapolating.

## Fresh concrete
"""),
    code(r"""conc = pd.read_csv(datasets.path("concrete_flow_curve.csv"), comment="#")
bing = fitting.fit_flow_curve(conc.shear_rate_1_s, conc.shear_stress_Pa, "bingham")
print(bing)
print("true fluid: Bingham tau_y = 600 Pa, plastic viscosity 45 Pa s")"""),
    md(r"""
For concrete the Bingham model is the standard, and its two parameters map onto the two things site
engineers care about: the yield stress governs whether concrete holds its shape (slump), the plastic
viscosity how hard it is to pump (notebook 15).
"""),
]

NOTEBOOKS["15_pipe_flow_of_non_newtonian_fluids"] = [
    md(r"""
# 15 · Pipe flow of non-Newtonian fluids

Pumping drilling mud down a well, concrete up a building, slurries along a pipeline, sauces through a
factory: every one needs the pressure drop for a given flow rate. For non-Newtonian fluids the Newtonian
formulas fail - the velocity profile changes shape, a yield-stress fluid moves partly as a solid plug, and
the Reynolds number and friction factor need new definitions.
"""),
    setup_cell(["from scipy import integrate", "from engrheo import datasets, fitting, flows, geometry"]),
    md("## Velocity profiles at the same flow rate"),
    code(r"""R, Q = 0.025, 1.5e-3                              # 5 cm pipe, 1.5 L/s
r = np.linspace(0, R, 400)
newt = flows.pipe_power_law(0.1, 1.0, R, Q=Q)
pl = flows.pipe_power_law(0.5, 0.4, R, Q=Q)
hb = flows.pipe_hb(8.0, 0.3, 0.5, R, Q=Q)
V = newt["V"]
plt.plot(flows.velocity_power_law(r, 0.1, 1.0, R, newt["dp_per_L"]) / V, r / R, label="Newtonian")
plt.plot(flows.velocity_power_law(r, 0.5, 0.4, R, pl["dp_per_L"]) / V, r / R, label="power law, n = 0.4")
plt.plot(flows.velocity_hb(r, 8.0, 0.3, 0.5, R, hb["dp_per_L"]) / V, r / R, label="Herschel-Bulkley")
plt.axhline(hb["plug_radius"] / R, color="C2", ls=":", lw=1)
plt.xlabel("u / mean velocity"); plt.ylabel("r / R"); plt.legend(); plt.show()
print(f"centre-line / mean velocity: Newtonian {newt['u_max']/V:.2f}, power law {pl['u_max']/V:.2f}; "
      f"plug occupies r/R < {hb['plug_radius']/R:.2f}")"""),
    md(r"""
Shear thinning flattens the profile: the fluid near the wall, where the shear rate is highest, becomes
thin and lubricates the core. A yield-stress fluid goes further - where the stress is below the yield
stress (near the axis) it moves as a rigid **plug**.

## Pumping concrete
Concrete is pumped along steel pipes to the top of high-rise buildings. How much pressure does the pump
need? The Bingham fit of notebook 04:
"""),
    code(r"""conc = pd.read_csv(datasets.path("concrete_flow_curve.csv"), comment="#")
b = fitting.fit_flow_curve(conc.shear_rate_1_s, conc.shear_stress_Pa, "bingham").params
Dp, Lp, Qc = 0.125, 100.0, 30 / 3600               # 125 mm line, 100 m, 30 m3/h
sol = flows.pipe_hb(b["tau_y"], b["mu_p"], 1.0, Dp / 2, Q=Qc)
dp = sol["dp_per_L"] * Lp
rho = 2350.0
Re = rho * sol["V"] * Dp / b["mu_p"]
print(f"mean velocity {sol['V']:.2f} m/s, Reynolds number {Re:.1f} (laminar)")
print(f"pressure drop over {Lp:.0f} m: {dp/1e5:.1f} bar; plug fills r/R < {sol['plug_radius']/(Dp/2):.2f}")
print(f"of which needed just to overcome the yield stress (4 tau_y L / D): {4*b['tau_y']*Lp/Dp/1e5:.1f} bar")"""),
    md(r"""
The flow is deeply laminar. At this flow rate the wall stress is several times the yield stress, so the
plug is small, and about a quarter of the pressure is spent overcoming the yield stress - the rest on the
plastic viscosity. Real concrete pressures are usually lower than this estimate: a thin,
cement-rich lubricating layer forms at the wall, and the concrete slides on it. Industrial practice therefore
uses tribometers and pumping tests; the calculation here is the no-slip upper bound.

## Drilling mud in the drill string: laminar or turbulent?
Mud is circulated down the drill pipe at high rates. Fitting a power law to the high-speed Fann readings
(the range relevant inside the pipe) gives the Metzner-Reed Reynolds number and the flow regime:
"""),
    code(r"""mud = pd.read_csv(datasets.path("drilling_mud_fann.csv"), comment="#")
tau_m, rate_m = geometry.fann35(mud.rpm.to_numpy(), mud.dial_reading_deg.to_numpy())
fast = rate_m >= 170                                  # 100-600 rpm
order = np.argsort(rate_m[fast])
pl_mud = fitting.fit_flow_curve(rate_m[fast][order], tau_m[fast][order], "power_law").params
D_pipe, rho_mud = 0.1086, 1200.0                      # 4.276 in inner diameter; 10 lb/gal mud
for q_gpm in (150, 300, 500):
    Qm = q_gpm * 6.309e-5                             # US gal/min -> m3/s
    res = flows.pressure_drop_power_law(Qm, D_pipe, 1000.0, rho_mud, pl_mud["K"], pl_mud["n"])
    print(f"{q_gpm} gal/min: V = {res['V']:.2f} m/s, Re_MR = {res['Re_MR']:8.0f} (critical {res['Re_critical']:.0f}) "
          f"-> {res['regime']:<9}, pressure loss {res['dp']/1e5:6.1f} bar per 1000 m")"""),
    md(r"""
At low rates the flow is laminar; at typical drilling rates it becomes turbulent and the pressure loss rises
steeply. The critical Reynolds number depends on $n$ (Ryan and Johnson): shear-thinning fluids stay laminar
somewhat longer than Newtonian ones.
"""),
    code(r"""nn = np.linspace(0.2, 1.0, 50)
plt.plot(nn, [flows.critical_reynolds_power_law(v) for v in nn])
plt.xlabel("power-law index n"); plt.ylabel("critical Metzner-Reed Reynolds number"); plt.show()"""),
    md(r"""
## Restarting a gelled line
When circulation stops, a yield-stress fluid sets in the pipe. To restart flow, the pump must at least
overcome the yield stress along the whole line, $\Delta P_{start} = 4\tau_y L/D$ - independent of the flow rate:
"""),
    code(r"""hb_mud = fitting.fit_flow_curve(np.sort(rate_m), tau_m[np.argsort(rate_m)], "herschel_bulkley").params
for L_line in (1000.0, 3000.0):
    print(f"{L_line:.0f} m of 4.276 in pipe: at least {4*hb_mud['tau_y']*L_line/D_pipe/1e5:.1f} bar to restart "
          f"(yield stress {hb_mud['tau_y']:.1f} Pa)")"""),
    md(r"""
Many muds and waxy crude oils gel further while at rest (thixotropy, notebook 06), so their yield stress -
and the restart pressure - grows with the shut-in time; restart pressures are a key design question for
pipelines.
"""),
]


EXTRAS["03_shear_thinning_fluids"] = _x(
    ["describe shear thinning with the power-law, Cross, Carreau(-Yasuda) and Sisko models",
     "fit flow curves with uncertainties and compare models with AIC",
     "explain why fits should use relative (log) residuals", "avoid extrapolating models beyond the data"],
    "notebooks 00-01", "60 min",
    [md(r"""
## Inside the algorithm: a Carreau fit by hand
Minimise the sum of squared log residuals with `scipy.optimize.least_squares`, then estimate standard errors
from the Jacobian, $\text{cov} = s^2 (J^TJ)^{-1}$:
"""), code(r"""def resid(p):
    return np.log(models.carreau(rate, *p) / tau)
sol = optimize.least_squares(resid, [30, 0.01, 10, 0.3], bounds=([0, 0, 0, 0.01], [np.inf, np.inf, np.inf, 1.5]), x_scale="jac")
s2 = np.sum(sol.fun**2) / (rate.size - 4)
se = np.sqrt(np.diag(s2 * np.linalg.inv(sol.jac.T @ sol.jac)))
print("by hand:", np.round(sol.x, 5), np.round(se, 5))
print("library:", np.round(list(best.params.values()), 5), np.round(list(best.se.values()), 5))""")],
    ["Fit a power law to the data between 10 and 1000 1/s only. By what factor does it overestimate the viscosity "
     "at 0.01 1/s? How does that factor change if you fit from 1 to 1000 1/s instead?",
     "Pulp particles (diameter 0.3 mm, 50 kg/m³ denser than the drink) settle under gravity. Estimate the Stokes "
     "settling velocity $v = \\Delta\\rho\\, g\\, d^2/(18\\eta)$ in water and in the xanthan drink (using the viscosity "
     "at rest). Check that the shear rate $v/d$ around the particle really lies on the viscosity plateau."],
    "**Implement it yourself:** compute AIC $= n\\ln(\\text{SSR}/n) + 2k$ from the log residuals of the Carreau fit "
    "and compare with `best.aic`; then do the same for the power-law fit.")

EXTRAS["04_yield_stress_fluids"] = _x(
    ["fit Bingham, Herschel-Bulkley and Casson models", "compare API field formulas with proper fits",
     "explain why a yield stress depends on the model and the measuring range",
     "relate yield-stress parameters to industrial requirements"],
    "notebooks 02-03", "60-75 min",
    [md(r"""
## Inside the algorithm: the classic Casson straight line
Casson plotted $\sqrt\tau$ against $\sqrt{\dot\gamma}$: a straight line with intercept $\sqrt{\tau_y}$ and slope $\sqrt{\eta_c}$.
Ordinary regression on that plot weights the points differently from the library's log-residual fit:
"""), code(r"""slope, intercept = np.polyfit(np.sqrt(rc), np.sqrt(tc), 1)
print(f"Casson straight line: tau_y = {intercept**2:.3f} Pa, eta_c = {slope**2:.3f} Pa s")
cf = fitting.fit_flow_curve(rc, tc, "casson")
print(f"log-residual fit:     tau_y = {cf.params['tau_y']:.3f} Pa, eta_c = {cf.params['eta_c']:.3f} Pa s")""")],
    ["Drilling engineers estimate the true yield stress with the low-shear yield point LSYP = $2\\theta_3 - \\theta_6$ "
     "(lbf/100 ft², × 0.4788 for Pa). Compute it for the mud and compare with the API yield point, the "
     "Herschel-Bulkley $\\tau_y$ and the true value (4.8 Pa).",
     "Fit the Herschel-Bulkley model to the chocolate data using only the rates above 10 1/s. How much does the "
     "yield stress change? What does that say about extrapolated yield stresses?"],
    "**Implement it yourself:** fit the Bingham model to the concrete data by an ordinary straight-line fit of "
    "stress against rate (`np.polyfit`) and compare with the library's log-residual fit. Which points does each "
    "weight most?")

EXTRAS["15_pipe_flow_of_non_newtonian_fluids"] = _x(
    ["compute velocity profiles, flow rates and pressure drops for power-law and yield-stress fluids",
     "use the Metzner-Reed Reynolds number to decide between laminar and turbulent flow",
     "size pumping for concrete and drilling mud", "estimate restart pressures for gelled lines"],
    "notebooks 03-04; basic fluid mechanics (Hagen-Poiseuille, friction factors)", "60-75 min",
    [md(r"""
## Inside the algorithm: flow rate from the velocity profile
The flow rate is the velocity integrated over the cross-section, $Q = \int_0^R u(r)\,2\pi r\,dr$ - here for the
Herschel-Bulkley fluid, with the plug radius as a breakpoint:
"""), code(r"""Q_int = integrate.quad(lambda s: 2*np.pi*s*flows.velocity_hb(s, 8.0, 0.3, 0.5, R, hb["dp_per_L"]), 0, R,
                       points=[hb["plug_radius"]], epsabs=0, epsrel=1e-12)[0]
print(f"integrated profile {Q_int:.10e} m3/s, closed form {hb['Q']:.10e} m3/s (target {Q:.1e})")""")],
    ["How does the concrete pump pressure change if the line diameter is 100, 125 or 150 mm at 30 m³/h? Which part "
     "of the pressure (yield stress or viscous) depends most on the diameter?",
     "For the drilling mud, double the flow rate from 250 to 500 gal/min. By what factor does the pressure loss grow, "
     "and how does that compare with the laminar power-law scaling $\\Delta P \\propto Q^n$?"],
    "**Implement it yourself:** write the Metzner-Reed Reynolds number from its definition and check, for the "
    "power-law fluid above, that $f = 16/Re_{MR}$ reproduces the exact laminar pressure gradient from `flows.pipe_power_law`.")


# =====================================================================================================
# Viscoelasticity
NOTEBOOKS["05_temperature_and_tts"] = [
    md(r"""
# 05 · Temperature dependence and time-temperature superposition

Viscosity falls steeply with temperature - an engine oil thins several-fold between a cold start and
operating temperature, and a polymer melt's viscosity can change by a factor of ten over 20 K. This
notebook describes that dependence and uses it: for many materials a temperature change merely rescales
time, so measurements at several temperatures combine into a **master curve** covering far more decades
than any single test.
"""),
    setup_cell(["from scipy import optimize", "from engrheo import datasets, tts"]),
    md(r"""
## An engine oil: Arrhenius or not?
A thermally activated flow process gives the Arrhenius law $\eta = A\,e^{E_a/RT}$: a straight line of $\ln\eta$
against $1/T$. Over a wide temperature range liquids often curve away from it; the Vogel(-Fulcher-Tammann)
equation $\eta = A\,e^{B/(T - T_0)}$ captures that.
"""),
    code(r"""oil = pd.read_csv(datasets.path("engine_oil_viscosity_temperature.csv"), comment="#")
T = oil.temperature_C.to_numpy() + 273.15
eta = oil.viscosity_Pa_s.to_numpy()
b, a = np.polyfit(1 / T, np.log(eta), 1)                         # Arrhenius: ln eta = a + (Ea/R)/T
vog, _ = optimize.curve_fit(lambda T_, lnA, B, T0: lnA + B / (T_ - T0), T, np.log(eta), p0=[-10, 1000, 150])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
a1.semilogy(1000 / T, eta, "o", label="data")
a1.semilogy(1000 / T, np.exp(a + b / T), label="Arrhenius (straight line)")
a1.semilogy(1000 / T, np.exp(vog[0] + vog[1] / (T - vog[2])), "--", label="Vogel")
a1.set(xlabel="1000 / T (1/K)", ylabel="viscosity (Pa s)"); a1.legend()
a2.plot(T - 273.15, 100 * (np.exp(a + b / T) / eta - 1), "o-", label="Arrhenius")
a2.plot(T - 273.15, 100 * (np.exp(vog[0] + vog[1] / (T - vog[2])) / eta - 1), "s-", label="Vogel")
a2.axhline(0, color="k", lw=0.8); a2.set(xlabel="temperature (degC)", ylabel="relative error (%)"); a2.legend()
plt.show()
print(f"Arrhenius: Ea = {b * tts.R_GAS / 1000:.1f} kJ/mol over the whole range")
print(f"Vogel: B = {vog[1]:.0f} K, T0 = {vog[2]:.0f} K   (true: 1100 K, 160 K)")
local_Ea = np.gradient(np.log(eta), 1 / T) * tts.R_GAS / 1000
print(f"local apparent activation energy: {local_Ea[0]:.0f} kJ/mol at {T[0]-273.15:.0f} degC, {local_Ea[-1]:.0f} kJ/mol at {T[-1]-273.15:.0f} degC")"""),
    md(r"""
The Arrhenius line misses systematically - too low at the ends, too high in the middle - because the
apparent activation energy itself falls with temperature. An Arrhenius law is a good *local* description
(over a few tens of kelvin) but not a global one for such liquids.

## A polymer melt: frequency sweeps at six temperatures
"""),
    code(r"""melt = pd.read_csv(datasets.path("polystyrene_frequency_sweeps.csv"), comment="#")
fig, ax = plt.subplots()
for T_C, grp in melt.groupby("temperature_C"):
    ax.loglog(grp.omega_rad_s, grp.G_storage_Pa, "o-", ms=3, label=f"{T_C:.0f} degC")
ax.set(xlabel="angular frequency (rad/s)", ylabel="G' (Pa)", title="Storage modulus at six temperatures"); ax.legend(fontsize=7)
plt.show()"""),
    md(r"""
Each curve covers only three decades of frequency, and the curves look like shifted copies of one another.
Shift them horizontally onto the 170 degC curve:
"""),
    code(r"""curves = {T_C + 273.15: (g.omega_rad_s.to_numpy(), g[["G_storage_Pa", "G_loss_Pa"]].to_numpy())
          for T_C, g in melt.groupby("temperature_C")}
T_ref = 170 + 273.15
mc = tts.master_curve(curves, T_ref)
fig, ax = plt.subplots()
for T_ in sorted(curves):
    ax.loglog(mc["omega_reduced"][T_], mc["y"][T_][:, 0], "o", ms=3, color="C0")
    ax.loglog(mc["omega_reduced"][T_], mc["y"][T_][:, 1], "s", ms=3, color="C1")
ax.set(xlabel="reduced frequency a_T omega (rad/s)", ylabel="modulus (Pa)", title="Master curve at 170 degC (G' circles, G'' squares)")
plt.show()
temps = np.array(sorted(curves))
laT = np.array([mc["log10_aT"][T_] for T_ in temps])
w = tts.fit_wlf(temps, laT, T_ref)
print("log10 aT:", ", ".join(f"{T_ - 273.15:.0f} degC: {v:+.3f}" for T_, v in zip(temps, laT)))
print(f"WLF fit: C1 = {w['C1']:.2f}, C2 = {w['C2']:.1f} K   (true 8.86, 101.6 K)")
wr = np.concatenate([mc["omega_reduced"][T_] for T_ in temps])
print(f"the master curve spans {np.log10(wr.max() / wr.min()):.1f} decades of frequency; each measurement spanned 3")"""),
    md(r"""
The shifted curves superpose onto one master curve spanning several more decades than any single
measurement - from the terminal zone (where the melt flows) to the rubbery plateau. The shift factors
follow the WLF equation, the standard description for polymers between the glass transition and about
100 K above it. (For precise work a small vertical shift $\rho T/\rho_{ref}T_{ref}$ is also applied; it is ignored here.)
The zero-shear viscosity shifts the same way: $\eta_0(T) = a_T\,\eta_0(T_{ref})$.

Superposition is an assumption. It fails for materials whose structure changes with temperature (phase
separation, crystallisation, some blends) - if the shifted curves do not overlap in shape, TTS must not be
forced.
"""),
]

NOTEBOOKS["07_linear_viscoelasticity"] = [
    md(r"""
# 07 · Linear viscoelasticity

Bread dough stretches and partly springs back; a polymer part under constant load slowly creeps; a
rubber band relaxes. Such materials combine elastic storage with viscous flow. In the linear range
(small deformations) their behaviour is fully described by one material function - the relaxation
modulus $G(t)$ or, equivalently, the creep compliance $J(t)$ - and responses to any loading history follow
by superposition.
"""),
    setup_cell(["from scipy import optimize", "from engrheo import datasets", "from engrheo import viscoelastic as ve"]),
    md(r"""
## Springs and dashpots
A spring stores energy (stress proportional to strain), a dashpot dissipates it (stress proportional to strain
rate). Combining them gives the classical models:
"""),
    code(r"""t = np.linspace(0, 5, 400)
mx, kv = ve.maxwell(1000.0, 1000.0), ve.kelvin_voigt(1000.0, 1000.0)
sls = ve.standard_linear_solid(300.0, 700.0, 1.0)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
a1.plot(t, mx["relaxation"](t), label="Maxwell (liquid)"); a1.plot(t, sls["relaxation"](t), label="standard linear solid")
a1.set(xlabel="time (s)", ylabel="G(t) (Pa)", title="Relaxation after a step strain"); a1.legend()
a2.plot(t, mx["creep"](t) * 1e3, label="Maxwell"); a2.plot(t, kv["creep"](t) * 1e3, label="Kelvin-Voigt (solid)")
a2.plot(t, sls["creep"](t) * 1e3, label="standard linear solid")
a2.set(xlabel="time (s)", ylabel="J(t) (1/kPa)", title="Creep under a step stress"); a2.legend()
plt.show()"""),
    md(r"""
The Maxwell model relaxes completely and creeps without limit - a liquid. The Kelvin-Voigt model creeps to
a finite strain but cannot relax a step strain - a solid. The standard linear solid relaxes to a plateau.

## Bread dough: creep and recovery
A constant stress of 50 Pa was applied for 120 s and then removed:
"""),
    code(r"""dough = pd.read_csv(datasets.path("dough_creep_recovery.csv"), comment="#")
tt, strain = dough.time_s.to_numpy(), dough.strain.to_numpy()
creep = tt <= 120
names = ["G1", "eta1", "G2", "eta2"]

def history(t, G1, eta1, G2, eta2):
    # Boltzmann: stress +50 Pa at t = 0 and -50 Pa at t = 120 s
    J = lambda s: np.where(s >= 0, ve.burgers_creep(np.maximum(s, 0), G1, eta1, G2, eta2), 0.0)
    return 50.0 * (J(t) - np.where(t > 120, J(t - 120), 0.0))

p_creep, cov_c = optimize.curve_fit(ve.burgers_creep, tt[creep], strain[creep] / 50, p0=[4e3, 5e5, 1e3, 1e4], bounds=(0, np.inf))
print("fit to the creep phase only:")
for n_, v, s_ in zip(names, p_creep, np.sqrt(np.diag(cov_c))):
    print(f"   {n_:<5} = {v:9.4g}  +/- {s_/v:4.0%}")
print("true values: G1 = 5000 Pa, eta1 = 1e6 Pa s, G2 = 2000 Pa, eta2 = 5e4 Pa s")
plt.plot(tt, strain, ".", ms=3, label="measured")
plt.plot(tt, history(tt, *p_creep), label="fit to the creep phase, recovery predicted")
plt.axvline(120, color="k", ls=":", lw=1); plt.xlabel("time (s)"); plt.ylabel("strain"); plt.legend(); plt.show()
print(f"predicted permanent strain {50*120/p_creep[1]:.4f}; measured strain at 300 s: {strain[-1]:.4f}")"""),
    md(r"""
The fit follows the creep curve closely - yet its parameters are uncertain by tens of percent, and the
predicted recovery is badly wrong: the model expects about twice as much permanent strain as the dough
shows. Over 120 s, slow delayed elasticity and genuine viscous flow produce nearly the same creep curve,
so the creep phase alone cannot tell them apart. The **recovery** phase can: delayed elasticity comes back,
viscous flow does not. Fitting the whole history with Boltzmann superposition (removing the stress is the
same as adding an opposite stress at 120 s) uses that information:
"""),
    code(r"""p_all, cov_a = optimize.curve_fit(history, tt, strain, p0=[4e3, 5e5, 1e3, 1e4], bounds=(0, np.inf))
print("fit to creep and recovery together:")
for n_, v, s_ in zip(names, p_all, np.sqrt(np.diag(cov_a))):
    print(f"   {n_:<5} = {v:9.4g}  +/- {s_/v:5.1%}")
print(f"retardation time eta2/G2 = {p_all[3]/p_all[2]:.1f} s; permanent strain {50*120/p_all[1]:.4f} (measured {strain[-1]:.4f})")
plt.plot(tt, strain, ".", ms=3, label="measured"); plt.plot(tt, history(tt, *p_all), label="fit to the full history")
plt.axvline(120, color="k", ls=":", lw=1); plt.xlabel("time (s)"); plt.ylabel("strain"); plt.legend(); plt.show()"""),
    md(r"""
Now all four parameters are determined precisely and agree with the true values. The lesson is about
experiment design: **always record the recovery** after a creep test. In baking terms, the elastic parts
describe how dough springs back after sheeting, the viscous part how it flows and spreads - and only the
recovery separates the two.

## From relaxation to creep
$G(t)$ and $J(t)$ carry the same information. For a material described by a relaxation spectrum (the polymer
melt of notebook 05), creep follows by solving the interconversion equation numerically:
"""),
    code(r"""g_ps = np.array([2.7e6, 1.08e6, 6.3e5, 5.4e5, 4.95e5, 3.6e5, 1.35e5])
tau_ps = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0, 20.0])
G_t = lambda s: ve.relaxation_modulus(g_ps, tau_ps, s)
tc = np.r_[0.0, np.logspace(-6, 3, 2500)]
Jc = ve.creep_from_relaxation(G_t, tc)
eta0, Je0 = ve.zero_shear_viscosity(g_ps, tau_ps), ve.steady_state_compliance(g_ps, tau_ps)
plt.loglog(tc[1:], Jc[1:], label="J(t) from G(t)")
plt.loglog(tc[1:], Je0 + tc[1:] / eta0, "--", label="long-time limit Je0 + t / eta0")
plt.loglog(tc[1:], 1 / G_t(tc[1:]), ":", label="1 / G(t)")
plt.xlabel("time (s)"); plt.ylabel("compliance (1/Pa)"); plt.legend(); plt.show()
print(f"J(t) G(t) <= 1 everywhere: {np.all(Jc[1:] * G_t(tc[1:]) <= 1 + 1e-9)};  eta0 = {eta0:.3g} Pa s, Je0 = {Je0:.3g} 1/Pa")"""),
    md(r"""
At long times the melt creeps like a liquid with the zero-shear viscosity plus a recoverable elastic part,
the steady-state compliance $J_e^0$ - the quantity behind die swell and recoil. $J(t)$ is not simply $1/G(t)$:
the two are linked by a convolution, and $J(t)\,G(t) \le 1$ always.
"""),
]

NOTEBOOKS["08_oscillatory_rheology"] = [
    md(r"""
# 08 · Oscillatory rheology

Small-amplitude oscillatory shear is the workhorse of rheology: an oscillating strain probes the material
without destroying its structure, and the stress response splits into an elastic part in phase with the
strain (storage modulus $G'$) and a viscous part in phase with the strain rate (loss modulus $G''$). This
notebook shows how the moduli are obtained, and applies frequency, amplitude and time sweeps to a polymer
melt, a cosmetic cream and a gelling food.
"""),
    setup_cell(["from engrheo import datasets, oscillatory, tts", "from engrheo import viscoelastic as ve"]),
    md("## From waveforms to moduli"),
    code(r"""w = 2 * np.pi * 1.0                                        # 1 Hz
t = np.linspace(0, 3, 600, endpoint=False)
strain = 0.01 * np.sin(w * t)
stress = 0.01 * (800 * np.sin(w * t) + 350 * np.cos(w * t)) + np.random.default_rng(8).normal(0, 0.1, t.size)
r = oscillatory.moduli_from_waveforms(t, strain, stress, w)
plt.plot(t, strain / strain.max(), label="strain (scaled)"); plt.plot(t, stress / np.abs(stress).max(), label="stress (scaled)")
plt.xlim(0, 2); plt.xlabel("time (s)"); plt.legend(); plt.show()
print(f"G' = {r['Gp']:.1f} Pa, G'' = {r['Gpp']:.1f} Pa, phase angle {r['delta_deg']:.1f} deg   (true 800, 350 Pa)")"""),
    md(r"""
The stress leads the strain by the phase angle $\delta$ ($0°$ for an elastic solid, $90°$ for a viscous liquid);
$G' = |G^*|\cos\delta$ and $G'' = |G^*|\sin\delta$.

## Frequency sweep: a polymer melt
"""),
    code(r"""melt = pd.read_csv(datasets.path("polystyrene_frequency_sweeps.csv"), comment="#")
curves = {T_C + 273.15: (g.omega_rad_s.to_numpy(), g[["G_storage_Pa", "G_loss_Pa"]].to_numpy()) for T_C, g in melt.groupby("temperature_C")}
mc = tts.master_curve(curves, 443.15)
wr = np.concatenate([mc["omega_reduced"][T_] for T_ in sorted(curves)])
Y = np.vstack([mc["y"][T_] for T_ in sorted(curves)])
order = np.argsort(wr)
wr, Gp, Gpp = wr[order], Y[order, 0], Y[order, 1]
cx = oscillatory.crossover(wr, Gp, Gpp)
plt.loglog(wr, Gp, "o", ms=3, label="G'"); plt.loglog(wr, Gpp, "s", ms=3, label="G''")
plt.loglog(*cx, "k*", ms=12, label="crossover")
plt.xlabel("a_T omega (rad/s)"); plt.ylabel("modulus (Pa)"); plt.legend(); plt.show()
low = wr < wr.min() * 10
print(f"crossover at {cx[0]:.3g} rad/s -> characteristic relaxation time 1/omega_c = {1/cx[0]:.3g} s")
print(f"terminal slopes (lowest decade): d ln G'/d ln w = {np.polyfit(np.log(wr[low]), np.log(Gp[low]), 1)[0]:.2f} (theory 2), "
      f"d ln G''/d ln w = {np.polyfit(np.log(wr[low]), np.log(Gpp[low]), 1)[0]:.2f} (theory 1)")"""),
    md(r"""
At low frequencies (long times) the melt flows: $G'' > G'$, approaching the terminal slopes 2 and 1. Above
the crossover, elasticity dominates. A higher molecular weight moves the crossover to lower frequencies.

## Amplitude sweep: where does the linear range end?
Before any frequency sweep, an amplitude sweep checks how large a strain the structure tolerates:
"""),
    code(r"""cream = pd.read_csv(datasets.path("cream_amplitude_sweep.csv"), comment="#")
s, gp_c, gpp_c = cream.strain_amplitude.to_numpy(), cream.G_storage_Pa.to_numpy(), cream.G_loss_Pa.to_numpy()
lim = oscillatory.lve_limit(s, gp_c)
flow = oscillatory.crossover(s, gp_c, gpp_c)
plt.loglog(s * 100, gp_c, "o-", ms=3, label="G'"); plt.loglog(s * 100, gpp_c, "s-", ms=3, label="G''")
plt.axvline(lim * 100, color="k", ls=":", label="end of linear range (G' -5 %)")
plt.axvline(flow[0] * 100, color="C3", ls="--", label="flow point (G' = G'')")
plt.xlabel("strain amplitude (%)"); plt.ylabel("modulus (Pa)"); plt.legend(fontsize=8); plt.show()
print(f"linear range up to {lim*100:.2f} % strain (true {100*0.05*(1/0.95-1)**(1/1.6):.2f} %); flow point at {flow[0]*100:.1f} % strain")"""),
    md(r"""
Below the limit, moduli are independent of amplitude and characterise the undisturbed structure; frequency
sweeps must stay there. The estimate itself is approximate: a 5 % criterion applied to data with about 1 %
scatter and eight points per decade can only locate the limit roughly - here somewhat early - so stay well
below it in practice. Beyond the **flow point** ($G' = G''$) the cream's structure breaks down and it flows -
the oscillatory counterpart of a yield stress. The rise of $G''$ just before is typical of soft solids.

## Time sweep: when does a food gel set?
At the gel point a percolating network first spans the sample. Winter and Chambon showed that there
$\tan\delta$ is independent of frequency, so time sweeps at several frequencies cross at one instant:
"""),
    code(r"""gel = pd.read_csv(datasets.path("food_gel_time_sweep.csv"), comment="#")
piv = gel.assign(tan_delta=gel.G_loss_Pa / gel.G_storage_Pa).pivot(index="time_min", columns="omega_rad_s", values="tan_delta")
for w_ in piv.columns:
    plt.semilogy(piv.index, piv[w_], label=f"{w_:g} rad/s")
tg = oscillatory.gel_point(piv.index.to_numpy(), piv.to_numpy())
plt.axvline(tg, color="k", ls=":"); plt.xlabel("time (min)"); plt.ylabel("tan(delta)"); plt.legend(); plt.show()
single = gel[gel.omega_rad_s == 10.0]
cross_10 = oscillatory.crossover(single.time_min, single.G_loss_Pa, single.G_storage_Pa)
print(f"Winter-Chambon gel time {tg:.2f} min (true 23.4); simple G' = G'' crossover at 10 rad/s: {cross_10[0]:.1f} min")"""),
    md(r"""
The frequently used shortcut - the time at which $G'$ overtakes $G''$ at a single frequency - depends on the
chosen frequency and does not coincide with the gel point in general; the frequency-independent crossing
of $\tan\delta$ does.

## The Cox-Merz rule
For many polymer solutions and melts, the steady-shear viscosity at shear rate $\dot\gamma$ equals the magnitude
of the complex viscosity at $\omega = \dot\gamma$ - a practical shortcut when steady shear is difficult:
"""),
    code(r"""sol = pd.read_csv(datasets.path("polymer_solution_cox_merz.csv"), comment="#")
osc, stdy = sol[sol.test == "oscillatory"], sol[sol.test == "steady"]
ratio = oscillatory.cox_merz_ratio(stdy.rate_or_frequency, stdy.viscosity_Pa_s, osc.rate_or_frequency, osc.viscosity_Pa_s)
plt.loglog(osc.rate_or_frequency, osc.viscosity_Pa_s, "o", label="|eta*| (oscillatory)")
plt.loglog(stdy.rate_or_frequency, stdy.viscosity_Pa_s, "s", label="eta (steady shear)")
plt.xlabel("omega or shear rate"); plt.ylabel("viscosity (Pa s)"); plt.legend(); plt.show()
print(f"steady / complex viscosity over the common range: {np.nanmin(ratio):.2f} to {np.nanmax(ratio):.2f}")"""),
    md(r"""
The rule holds well for this polymer solution. It generally fails for materials with a structure that
steady shear destroys but small oscillations do not - gels, emulsions like the cream above, suspensions,
yield-stress fluids - whose complex viscosity lies far above the steady viscosity.
"""),
]

NOTEBOOKS["09_relaxation_spectra"] = [
    md(r"""
# 09 · Relaxation spectra

A discrete relaxation spectrum - moduli $g_i$ at relaxation times $\tau_i$ - condenses a master curve into a
compact model from which every linear viscoelastic function follows: $G(t)$, $J(t)$, $\eta_0$, $J_e^0$. It is
also the input to the nonlinear constitutive models of notebook 10. But finding the spectrum from data is
an **ill-posed inverse problem**: many very different spectra fit the same data. This notebook fits
spectra, shows the ill-posedness, and identifies what can be trusted.
"""),
    setup_cell(["from engrheo import datasets, tts", "from engrheo import viscoelastic as ve"]),
    code(r"""melt = pd.read_csv(datasets.path("polystyrene_frequency_sweeps.csv"), comment="#")
curves = {T_C + 273.15: (g.omega_rad_s.to_numpy(), g[["G_storage_Pa", "G_loss_Pa"]].to_numpy()) for T_C, g in melt.groupby("temperature_C")}
mc = tts.master_curve(curves, 443.15)
w = np.concatenate([mc["omega_reduced"][T_] for T_ in sorted(curves)])
Y = np.vstack([mc["y"][T_] for T_ in sorted(curves)])
o = np.argsort(w); w, Gp, Gpp = w[o], Y[o, 0], Y[o, 1]
fit = ve.fit_spectrum(w, Gp, Gpp)
g_true = np.array([2.7e6, 1.08e6, 6.3e5, 5.4e5, 4.95e5, 3.6e5, 1.35e5]); tau_true = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0, 20.0])
fp, fpp = fit.moduli(w)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
a1.loglog(w, Gp, "o", ms=3, label="G' data"); a1.loglog(w, Gpp, "s", ms=3, label="G'' data")
a1.loglog(w, fp, "k-", lw=1); a1.loglog(w, fpp, "k--", lw=1, label="spectrum fit")
a1.set(xlabel="a_T omega (rad/s)", ylabel="modulus (Pa)"); a1.legend(fontsize=8)
used = fit.g > 0
a2.loglog(fit.tau[used], fit.g[used], "o-", label="fitted spectrum")
a2.loglog(tau_true, g_true, "k*", ms=10, label="true modes")
a2.set(xlabel="relaxation time (s)", ylabel="g_i (Pa)"); a2.legend()
plt.show()
print(f"{fit.tau.size} modes on the grid, {used.sum()} non-zero; rms misfit {100*fit.rel_rms:.1f} %")
print(f"eta0: fit {fit.eta0:.4g} Pa s, true {ve.zero_shear_viscosity(g_true, tau_true):.4g} Pa s")
print(f"Je0:  fit {ve.steady_state_compliance(fit.g, fit.tau):.3g} 1/Pa, true {ve.steady_state_compliance(g_true, tau_true):.3g} 1/Pa")"""),
    md(r"""
The spectrum reproduces the data within their scatter, and the zero-shear viscosity and steady-state
compliance agree with the true values. The individual modes, however, look nothing like the true seven
modes - and that is not a failure of the fit.

## Many spectra, one data set
"""),
    code(r"""fits = {"no regularisation": ve.fit_spectrum(w, Gp, Gpp),
        "regularisation 0.01": ve.fit_spectrum(w, Gp, Gpp, regularization=0.01),
        "regularisation 0.1": ve.fit_spectrum(w, Gp, Gpp, regularization=0.1),
        "2 modes per decade": ve.fit_spectrum(w, Gp, Gpp, n_modes=int(2 * 7))}
fig, ax = plt.subplots()
for name, f in fits.items():
    ax.loglog(f.tau[f.g > 0], f.g[f.g > 0], "o-", ms=3, label=name)
ax.loglog(tau_true, g_true, "k*", ms=10, label="true")
ax.set(xlabel="relaxation time (s)", ylabel="g_i (Pa)"); ax.legend(fontsize=8); plt.show()
for name, f in fits.items():
    print(f"{name:<20}: misfit {100*f.rel_rms:4.1f} %, eta0 {f.eta0:.4g} Pa s, Je0 {ve.steady_state_compliance(f.g, f.tau):.3g} 1/Pa")"""),
    md(r"""
Very different spectra reproduce the same data almost equally well: the data determine the spectrum only
in an averaged, smoothed sense. Regularisation chooses the smoothest among the acceptable spectra at the
cost of a slightly larger misfit. What is robust are the **integral properties** - the zero-shear viscosity
(the first moment) and the steady-state compliance - and the moduli themselves within the measured
frequency range. Never interpret individual modes as molecular relaxation processes, and never trust a
spectrum outside the frequency range of the data.

## Using the spectrum
Once fitted, the spectrum predicts any linear response - here the relaxation modulus and the creep
compliance, compared with those of the true spectrum:
"""),
    code(r"""t = np.logspace(-3, 2, 200)
plt.loglog(t, fit.relaxation(t), label="G(t) from fitted spectrum")
plt.loglog(t, ve.relaxation_modulus(g_true, tau_true, t), "k--", lw=1, label="G(t) true")
plt.xlabel("time (s)"); plt.ylabel("G(t) (Pa)"); plt.legend(); plt.show()
inside = (t > 1 / w.max()) & (t < 1 / w.min())
print(f"max relative error of G(t) for 1/omega_max < t < 1/omega_min: "
      f"{np.max(np.abs(fit.relaxation(t[inside]) / ve.relaxation_modulus(g_true, tau_true, t[inside]) - 1)):.1%}")"""),
    md(r"""
Within the time window corresponding to the measured frequencies, the prediction agrees with the truth to
within about 6 %, even though the modes themselves differ: the spectrum is a good *model* of the
material, not a unique description of it.
"""),
]


EXTRAS["05_temperature_and_tts"] = _x(
    ["describe the temperature dependence of viscosity with the Arrhenius and Vogel equations",
     "build a master curve by time-temperature superposition", "fit WLF constants and use shift factors",
     "recognise when superposition must not be used"],
    "notebooks 00 and 03; logarithms", "60 min",
    [md(r"""
## Inside the algorithm: one shift factor by hand
Shift the 180 degC curve onto the 170 degC curve: try a range of shifts $\log_{10} a$, and for each compare
$\ln G'$ and $\ln G''$ of the shifted curve with the reference curve (interpolated on log scales) where they overlap:
"""), code(r"""w_ref, y_ref = curves[443.15]; w_180, y_180 = curves[453.15]
trial = np.linspace(-1.5, 0.5, 2001)
def mismatch(la):
    s = np.log10(w_180) + la
    m = (s >= np.log10(w_ref).min()) & (s <= np.log10(w_ref).max())
    return np.mean([(np.log(y_180[m, j]) - np.interp(s[m], np.log10(w_ref), np.log(y_ref[:, j])))**2 for j in (0, 1)])
best = trial[np.argmin([mismatch(v) for v in trial])]
print(f"by hand: log10 aT(180 degC) = {best:.4f};  library: {mc['log10_aT'][453.15]:.4f}")""")],
    ["Fit the Arrhenius law to the engine-oil data between 60 and 120 degC only, and use it to predict the "
     "viscosity at 0 degC. By how much is the cold-start viscosity misjudged?",
     "With the fitted WLF constants, by what factor does the melt's longest relaxation time change between "
     "170 degC and 220 degC, and between 170 degC and 150 degC? What does that mean for processing?"],
    "**Implement it yourself:** write `wlf_shift(T, C1, C2, T_ref)` and a function that converts WLF constants "
    "from one reference temperature to another ($C_2' = C_2 + T_{ref}' - T_{ref}$, $C_1' = C_1C_2/C_2'$); check that "
    "both descriptions give identical shift factors.")

EXTRAS["07_linear_viscoelasticity"] = _x(
    ["describe creep and relaxation with spring-dashpot models", "fit a creep test and predict recovery by "
     "Boltzmann superposition", "separate elastic, delayed and permanent deformation",
     "convert a relaxation modulus into a creep compliance"],
    "notebooks 01 and 03; exponential functions", "60-75 min",
    [md(r"""
## Inside the algorithm: Boltzmann superposition by hand
For a strain applied in steps, the stress is the sum of the relaxation responses of all steps so far,
$\sigma(t) = \sum_k \Delta\gamma_k\,G(t - t_k)$. A strain of 0.1 applied at t = 0 and removed at t = 2 s:
"""), code(r"""G_mx = mx["relaxation"]
t_ = np.linspace(0, 5, 11)
by_hand = 0.1 * G_mx(t_) - 0.1 * np.where(t_ >= 2, G_mx(np.maximum(t_ - 2, 0)), 0.0)
strain_hist = np.where(t_ < 2, 0.1, 0.0)
library = ve.boltzmann_stress(G_mx, np.r_[0, 1e-9, 2 - 1e-9, 2 + 1e-9, 5], [0.1, 0.1, 0.1, 0.0, 0.0])
print("stress by hand at t = 0..5 s:", np.round(by_hand, 2))
print("stress just after removal (library, t = 2 s):", round(float(library[3]), 2), " by hand:", round(float(0.1*G_mx(2) - 0.1*G_mx(0)), 2))""")],
    ["A cross-linked rubber must not creep indefinitely. Which model - Maxwell or standard linear solid - can "
     "describe it? Compare their creep compliances after 10 relaxation times.",
     "What fraction of the dough's strain at 120 s has been recovered 180 s after unloading? Compare the "
     "measurement, the Burgers prediction and the limit for very long times."],
    "**Implement it yourself:** fit a single Maxwell element to relaxation data by a straight line of $\\ln G$ "
    "against $t$ (generate the data from `ve.maxwell(2000, 500)` with 2 % noise). How accurate are $G$ and $\\tau$?")

EXTRAS["08_oscillatory_rheology"] = _x(
    ["obtain G' and G'' from strain and stress waveforms", "interpret frequency sweeps: terminal zone, "
     "crossover, plateau", "find the linear range and flow point in amplitude sweeps",
     "determine a gel point (Winter-Chambon) and check the Cox-Merz rule"],
    "notebooks 05 and 07", "75 min",
    [md(r"""
## Inside the algorithm: moduli from waveforms by hand
Project the stress onto $\sin\omega t$ and $\cos\omega t$ over whole cycles (the first Fourier coefficient); for a
strain $\gamma_0\sin\omega t$ the in-phase part divided by $\gamma_0$ is $G'$, the quadrature part $G''$:
"""), code(r"""ns = t.size
in_phase = 2 / ns * np.sum(stress * np.sin(w * t))
quadrature = 2 / ns * np.sum(stress * np.cos(w * t))
print(f"by hand: G' = {in_phase / 0.01:.2f} Pa, G'' = {quadrature / 0.01:.2f} Pa;  library: {r['Gp']:.2f}, {r['Gpp']:.2f}")""")],
    ["Using only the 220 degC frequency sweep (unshifted), estimate the terminal slopes of G' and G''. Why are "
     "they closer to the theoretical values than at 150 degC?",
     "Determine the gel time using only two frequencies (1 and 31.6 rad/s) instead of four. How robust is the "
     "Winter-Chambon estimate?"],
    "**Implement it yourself:** find the crossover of the melt's G' and G'' by hand: locate the sign change of "
    "$\\ln G' - \\ln G''$ and interpolate linearly in log-log coordinates; compare with `oscillatory.crossover`.")

EXTRAS["09_relaxation_spectra"] = _x(
    ["fit a discrete relaxation spectrum to G' and G'' data", "explain why spectrum fitting is an ill-posed problem",
     "use regularisation and judge what a spectrum can and cannot tell", "predict other linear responses from a spectrum"],
    "notebooks 05, 07 and 08; least squares", "60-75 min",
    [md(r"""
## Inside the algorithm: non-negative least squares on a fixed grid
With the relaxation times fixed, $G'$ and $G''$ are *linear* in the $g_i$. Dividing each equation by the measured
value makes the residuals relative; `scipy.optimize.nnls` then finds the best $g_i \ge 0$:
"""), code(r"""from scipy.optimize import nnls
tau_grid = np.logspace(-5, 3, 25)
wt = w[:, None] * tau_grid
A = np.vstack([wt**2 / (1 + wt**2) / Gp[:, None], wt / (1 + wt**2) / Gpp[:, None]])
g_hand, _ = nnls(A, np.ones(2 * w.size), maxiter=5000)
lib = ve.fit_spectrum(w, Gp, Gpp, tau=tau_grid)
print(f"by hand: eta0 {np.sum(g_hand*tau_grid):.5g} Pa s;  library (same grid): {lib.eta0:.5g} Pa s")""")],
    ["Remove all data below 1 rad/s (reduced frequency) and refit. How do the fitted zero-shear viscosity and "
     "steady-state compliance change, and why?",
     "Fit spectra with 1, 2, 5 and 10 modes per decade. How does the misfit change, and from which density on "
     "does adding modes stop helping?"],
    "**Implement it yourself:** compute the steady-state compliance $J_e^0 = \\sum g_i\\tau_i^2/(\\sum g_i\\tau_i)^2$ from the "
    "fitted spectrum by hand, and check it against the long-time behaviour of the creep compliance from "
    "`ve.creep_from_relaxation` ($J(t) \\to J_e^0 + t/\\eta_0$).")


# =====================================================================================================
# Nonlinear rheology, materials and processing
NOTEBOOKS["06_thixotropy"] = [
    md(r"""
# 06 · Thixotropy

Stir a paint and it thins; leave it and it thickens again. A drilling mud gels when circulation stops, and
the pressure to restart it grows with the time it has rested. This time dependence - **thixotropy** - comes
from a microstructure (flocs, a particle network) that shear breaks down and rest rebuilds. The stress then
depends on the shear *history*, not only on the current shear rate.
"""),
    setup_cell(["from scipy import optimize", "from engrheo import datasets, thixotropy as th"]),
    md(r"""
## The structural-kinetics picture
A structure parameter $\lambda$ (1 = fully built, 0 = broken down) obeys
$d\lambda/dt = k_b(1 - \lambda) - k_d\dot\gamma\lambda$ - build-up at rest, breakdown by shear - and the stress depends on it,
$\tau = \lambda\tau_y + (\eta_\infty + \Delta\eta\,\lambda)\dot\gamma$.

## A paint: the hysteresis loop
The classic thixotropy test ramps the shear rate up and down:
"""),
    code(r"""paint = pd.read_csv(datasets.path("paint_hysteresis_loop.csv"), comment="#")
up, down = paint[paint.direction == "up"], paint[paint.direction == "down"]
plt.plot(up.shear_rate_1_s, up.shear_stress_Pa, "o-", ms=3, label="up")
plt.plot(down.shear_rate_1_s, down.shear_stress_Pa, "s-", ms=3, label="down")
plt.xlabel("shear rate (1/s)"); plt.ylabel("shear stress (Pa)"); plt.legend(); plt.show()
x, y = paint.shear_rate_1_s.to_numpy(), paint.shear_stress_Pa.to_numpy()    # the loop, in time order
area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))      # shoelace formula for a closed polygon
print(f"loop area {area:.0f} Pa/s")"""),
    md(r"""
The downward curve lies below the upward one: during the ramp the structure was broken down faster than
it could rebuild. The loop area is a popular thixotropy index - but it is not a material property. With
the true parameters of this paint, the model shows how the area depends on how fast the test is run:
"""),
    code(r"""p_paint = th.Params(3.0, 0.08, 0.4, 0.02, 0.08)
for t_ramp in (0.05, 0.2, 1, 5, 60, 600, 6000):
    print(f"ramp time {t_ramp:>7g} s: loop area {th.hysteresis_loop(p_paint, 200.0, t_ramp)['area']:6.0f} Pa/s")"""),
    md(r"""
The area peaks for ramps of a fraction of a second and falls on both sides: very fast ramps leave no time
for breakdown, very slow ramps keep the structure near equilibrium. Over the ramp times used in practice
the area changes by orders of magnitude - loop areas can only be compared between tests run in exactly
the same way. Step tests, below, separate the time scales properly.

## Cement paste: a step test and the structural parameters
"""),
    code(r"""cem = pd.read_csv(datasets.path("cement_paste_step_test.csv"), comment="#")
t, rate, tau = cem.time_s.to_numpy(), cem.shear_rate_1_s.to_numpy(), cem.shear_stress_Pa.to_numpy()
names = ["tau_y", "eta_inf", "d_eta", "k_build", "k_break"]

def model(p, n):
    prm = th.Params(*p)
    return th.simulate(prm, t[:n], rate[:n], lam0=th.equilibrium_structure(prm, rate[0]))["stress"]

def fit_first(n):
    use = rate[:n] > 0                                  # at rest the stress is zero: nothing to fit
    f = optimize.least_squares(lambda p: np.log(model(p, n)[use] / tau[:n][use]), [10, 0.5, 1, 0.05, 0.05],
                               bounds=(1e-6, np.inf), x_scale="jac")
    cov = np.linalg.inv(f.jac.T @ f.jac) * np.sum(f.fun**2) / (use.sum() - 5)
    return f.x, np.sqrt(np.diag(cov)) / f.x

for label, n in (("steps 10 -> 100 -> 10 1/s only (first 600 s)", 600), ("full protocol incl. rest and 1 1/s probe", t.size)):
    p, rel = fit_first(n)
    print(label + ":\n   " + ", ".join(f"{n_} = {v:.3g} (+/-{r:.0%})" for n_, v, r in zip(names, p, rel)))
print("true: tau_y = 25, eta_inf = 0.1, d_eta = 1.5, k_build = 0.01, k_break = 0.02")
p_fit, _ = fit_first(t.size)
plt.plot(t, tau, ".", ms=2, label="measured"); plt.plot(t, model(p_fit, t.size), label="fitted structural-kinetics model")
plt.xlabel("time (s)"); plt.ylabel("shear stress (Pa)"); plt.legend(); plt.show()"""),
    md(r"""
After the step up the stress jumps and then decays as the structure breaks down; after the step back it
drops below equilibrium and slowly recovers. Fitted to these steps alone, the model reproduces the data -
yet its parameters are uncertain by more than 100 %: scaling the structural stresses and the rate constants
together yields almost the same stress history. The rest period breaks that ambiguity: the jump in stress
when shearing resumes at 1 1/s measures how much structure was rebuilt, and with it the build-up rate.
With the full protocol all five parameters are determined to about 1 %. **Design thixotropy tests with
rest periods** - a lesson similar to the creep-recovery test of notebook 07.

## Gel strength after rest
Drilling engineers measure "gel strengths" - the stress needed to start flow after 10 s and 10 min of rest.
With the fitted model the structure recovers as $\lambda(t) = 1 - (1 - \lambda_0)e^{-k_b t}$:
"""),
    code(r"""prm = th.Params(*p_fit)
lam0 = th.equilibrium_structure(prm, 100.0)                     # after circulating at 100 1/s
for rest in (10, 600, 3600):
    lam = 1 - (1 - lam0) * np.exp(-prm.k_build * rest)
    print(f"after {rest:5d} s of rest: structure {lam:.2f}, stress to restart flow {lam * prm.tau_y:5.1f} Pa")"""),
    md(r"""
The restart stress grows with rest time - and with it the pump pressure needed to restart a pipeline
(notebook 15: $\Delta P_{start} = 4\tau_y L/D$). For cement paste this recovery also underlies the loss of
workability at rest; real cement adds irreversible hydration on top.
"""),
]

NOTEBOOKS["10_nonlinear_viscoelasticity"] = [
    md(r"""
# 10 · Nonlinear viscoelasticity

Stir a polymer solution with a rod and it climbs the rod instead of forming a vortex; start shearing it
suddenly and the stress overshoots before settling. Linear viscoelasticity (notebooks 07-09) cannot describe
this: it applies only to small deformations. **Constitutive models** such as the upper-convected Maxwell,
Giesekus and Phan-Thien-Tanner models extend the linear spectrum to large, fast deformations.
"""),
    setup_cell(["from scipy import optimize", "from engrheo import datasets, constitutive as cm"]),
    code(r"""sol = pd.read_csv(datasets.path("polymer_solution_startup.csv"), comment="#")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
for rate, grp in sol.groupby("shear_rate_1_s"):
    a1.semilogx(grp.time_s, grp.shear_stress_Pa / rate, label=f"{rate:g} 1/s")
    a2.loglog(grp.time_s, grp.N1_Pa, label=f"{rate:g} 1/s")
a1.set(xlabel="time (s)", ylabel="transient viscosity (Pa s)", title="Start-up of shear"); a1.legend(fontsize=8)
a2.set(xlabel="time (s)", ylabel="N1 (Pa)", title="First normal-stress difference"); plt.show()"""),
    md(r"""
Three nonlinear effects are visible: the viscosity falls with rate (shear thinning), the stress overshoots
at high rates before reaching steady state, and a **first normal-stress difference** $N_1$ builds up. A positive
$N_1$ pulls the fluid towards the rotation axis - the rod-climbing (Weissenberg) effect - and causes die swell.

## Linear spectrum plus one nonlinear parameter
The linear spectrum (from a frequency sweep) is known: $g$ = 60, 25, 8 Pa at $\tau$ = 0.02, 0.2, 2 s. The UCM model
uses it unchanged - and predicts neither shear thinning nor overshoot. The Giesekus model adds one parameter,
the mobility $\alpha$; fit it to all start-up curves at once:
"""),
    code(r"""g, tau = np.array([60.0, 25.0, 8.0]), np.array([0.02, 0.2, 2.0])
groups = list(sol.groupby("shear_rate_1_s"))

def misfit(alpha):
    modes = cm.modes_from_spectrum(g, tau, "giesekus", alpha=alpha)
    res = []
    for rate, grp in groups:
        r = cm.startup_shear(modes, rate, np.r_[0.0, grp.time_s.to_numpy()])
        res.append(np.log(r["stress"][1:] / grp.shear_stress_Pa.to_numpy()))
    return float(np.mean(np.concatenate(res) ** 2))

alpha = optimize.minimize_scalar(misfit, bounds=(0.01, 0.5), method="bounded", options={"xatol": 1e-4}).x
print(f"fitted Giesekus alpha = {alpha:.3f}  (true 0.25); rms misfit {np.sqrt(misfit(alpha))*100:.1f} %")
ucm, gie = cm.modes_from_spectrum(g, tau), cm.modes_from_spectrum(g, tau, "giesekus", alpha=alpha)
rate, grp = groups[-1]
tt = np.r_[0.0, grp.time_s.to_numpy()]
plt.semilogx(grp.time_s, grp.shear_stress_Pa, "ko", ms=3, label=f"data, {rate:g} 1/s")
plt.semilogx(tt[1:], cm.startup_shear(ucm, rate, tt)["stress"][1:], label="UCM (linear spectrum only)")
plt.semilogx(tt[1:], cm.startup_shear(gie, rate, tt)["stress"][1:], label=f"Giesekus, alpha = {alpha:.2f}")
plt.xlabel("time (s)"); plt.ylabel("shear stress (Pa)"); plt.legend(); plt.show()"""),
    md(r"""
With one extra parameter the Giesekus model captures overshoot and thinning at every rate. The fitted model
now predicts steady flow curves and normal stresses:
"""),
    code(r"""rates = np.logspace(-2, 3, 16)
st = cm.steady_shear(gie, rates)
eta0 = np.sum(g * tau)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
a1.loglog(rates, st["eta"]); a1.axhline(eta0, color="k", ls=":", lw=1)
a1.set(xlabel="shear rate (1/s)", ylabel="viscosity (Pa s)", title="Predicted flow curve")
a2.loglog(rates, st["N1"], label="N1"); a2.loglog(rates, -st["N2"], label="-N2")
a2.set(xlabel="shear rate (1/s)", ylabel="normal-stress difference (Pa)"); a2.legend(); plt.show()
print(f"N2/N1 at 10 1/s: {st['N2'][10]/st['N1'][10]:.3f}   (negative and small, as measured for polymer solutions)")"""),
    md(r"""
The Giesekus model predicts a small negative second normal-stress difference, as observed for polymer
solutions; the UCM model gives $N_2 = 0$. Each nonlinear model has a characteristic signature - Giesekus and
PTT differ most in extension (notebook 14) - so the choice of model should be tested in more than one flow.
"""),
]

NOTEBOOKS["11_laos"] = [
    md(r"""
# 11 · Large-amplitude oscillatory shear (LAOS)

Small oscillations probe a material without disturbing it; large ones deform it the way processing and use
do - spreading a cream, chewing a yoghurt. The response is no longer sinusoidal, and $G'$ and $G''$ alone
cannot describe it. LAOS analysis extracts the extra information from the **shape** of the stress signal:
Fourier harmonics, Lissajous curves and the Chebyshev decomposition.
"""),
    setup_cell(["from engrheo import datasets, laos, oscillatory, constitutive as cm"]),
    md("## Lissajous curves of a polymer solution (Giesekus model)"),
    code(r"""modes = cm.modes_from_spectrum([60.0, 25.0, 8.0], [0.02, 0.2, 2.0], "giesekus", alpha=0.25)
w = 2.0
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
I3 = {}
for g0 in (0.1, 1.0, 5.0, 20.0):
    r = cm.laos_shear(modes, g0, w, n_cycles=12)
    s = r["stress"]
    a1.plot(r["strain"] / g0, s / np.abs(s).max(), label=f"strain {g0:g}")
    a2.plot(g0 * w * np.cos(w * r["t"]) / (g0 * w), s / np.abs(s).max())
    I3[g0] = laos.harmonics(r["t"][:-1], r["strain"][:-1], s[:-1], w)["I3_I1"]
a1.set(xlabel="strain / amplitude", ylabel="stress / max", title="elastic Lissajous (stress vs strain)"); a1.legend(fontsize=8)
a2.set(xlabel="rate / max", title="viscous Lissajous (stress vs rate)"); plt.show()
print("relative third harmonic I3/I1:", {k: round(v, 4) for k, v in I3.items()})"""),
    md(r"""
At small amplitude the elastic Lissajous curve is an ellipse (linear response). At large amplitude it
distorts: the stress no longer follows the strain sinusoidally, and higher harmonics appear. The relative
third harmonic $I_3/I_1$ grows with the square of the amplitude at first and is the most common single measure
of nonlinearity.

## A stirred yoghurt
"""),
    code(r"""yo = pd.read_csv(datasets.path("yoghurt_laos.csv"), comment="#")
w_y = 2 * np.pi
rows = []
fig, ax = plt.subplots()
for g0, grp in yo.groupby("strain_amplitude"):
    c = laos.chebyshev(grp.time_s, grp.strain, grp.shear_stress_Pa, w_y)
    rows.append((g0, c["G_M"], c["G_L"], c["S"], c["T"], c["I3_I1"]))
    ax.plot(grp.strain / g0, grp.shear_stress_Pa / grp.shear_stress_Pa.abs().max(), label=f"{g0:g}")
ax.set(xlabel="strain / amplitude", ylabel="stress / max"); ax.legend(title="strain amplitude", fontsize=8); plt.show()
print(pd.DataFrame(rows, columns=["strain_amp", "G'_M (Pa)", "G'_L (Pa)", "S", "T", "I3/I1"]).round(3).to_string(index=False))"""),
    md(r"""
Read the table from small to large amplitude. At the smallest strain $G'_M$ and $G'_L$ agree: linear response.
At intermediate strains the large-strain modulus exceeds the minimum-strain modulus ($S > 0$): the network
**stiffens within each cycle** as its strands are stretched. At larger strains the sign reverses: the
large-strain modulus falls far below its linear value while the minimum-strain modulus stays high - within
each cycle the gel is stiff near zero strain but yields at the extremes ($S$ strongly negative). The
first-harmonic $G'$ alone would report only a gradual softening; the Chebyshev measures resolve stiffening
and then yielding, which is closer to what the mouth feels. The viscous measure $T$ is negative throughout:
the flow contribution thins within each cycle.
"""),
]


EXTRAS["06_thixotropy"] = _x(
    ["explain thixotropy with a structural-kinetics model", "interpret hysteresis loops and know their limits",
     "fit structural parameters to a step test", "estimate gel strength and restart stresses after rest"],
    "notebooks 03-04 and 15; exponential functions", "60 min",
    [md(r"""
## Inside the algorithm: the exact step response
At a constant shear rate the structure equation is linear, and its solution relaxes exponentially to
$\lambda_{eq} = k_b/(k_b + k_d\dot\gamma)$ with rate constant $k_b + k_d\dot\gamma$. For the step from 10 to 100 1/s:
"""), code(r"""k = prm.k_build + prm.k_break * 100.0
lam_eq, lam_start = prm.k_build / k, th.equilibrium_structure(prm, 10.0)
ts = np.array([200.0, 250.0, 300.0])
by_hand = lam_eq + (lam_start - lam_eq) * np.exp(-k * (ts - 200))
lib = th.simulate(prm, t, rate, lam0=lam_start)["structure"][[200, 250, 300]]      # rows at t = 200, 250, 300 s
print("by hand:", np.round(by_hand, 6), " library:", np.round(lib, 6))""")],
    ["A cement grout line (50 mm, 200 m) stops for 10 min, and for 1 h. Using the fitted cement-paste model and "
     "$\\Delta P = 4\\tau_{restart}L/D$, estimate the pressure needed to restart flow in each case.",
     "For the paint, find the ramp time that gives the largest loop area. Relate it to the build-up and breakdown "
     "time scales $1/k_b$ and $1/(k_d\\dot\\gamma)$."],
    "**Implement it yourself:** integrate the structure equation for the step test with the explicit Euler method "
    "(time step 1 s, then 10 s) and compare with the exact interval solution of `th.simulate`. When does Euler fail?")

EXTRAS["10_nonlinear_viscoelasticity"] = _x(
    ["recognise nonlinear viscoelastic effects: shear thinning, stress overshoot, normal stresses",
     "use differential constitutive models (UCM, Giesekus, PTT) built on a linear spectrum",
     "fit a nonlinear parameter and predict other flows"],
    "notebooks 07-09; ordinary differential equations", "75 min",
    [md(r"""
## Inside the algorithm: steady Giesekus viscosity in closed form
For a single Giesekus mode the steady shear viscosity has an analytical solution (Bird, Armstrong and
Hassager). Compare it with the long-time limit of the ODE solution:
"""), code(r"""a_, lam_, eta_ = 0.25, 0.2, 5.0
for Wi in (0.5, 5.0):
    chi = np.sqrt((np.sqrt(1 + 16*a_*(1-a_)*Wi**2) - 1) / (8*a_*(1-a_)*Wi**2))
    f = (1 - chi) / (1 + (1 - 2*a_) * chi)
    exact = eta_ * (1 - f)**2 / (1 + (1 - 2*a_) * f)
    ode = cm.steady_shear([cm.Mode(eta_, lam_, "giesekus", alpha=a_)], [Wi / lam_])["eta"][0]
    print(f"Wi = {Wi}: closed form {exact:.8f} Pa s, ODE {ode:.8f} Pa s")""")],
    ["At low shear rates the first normal-stress coefficient $\\Psi_1 = N_1/\\dot\\gamma^2$ tends to $2\\sum g_i\\tau_i^2$. Check "
     "this with the fitted model and with the data at 0.1 1/s.",
     "At which strain ($\\dot\\gamma\\,t$) does the stress overshoot occur at 10 and 100 1/s? Is it roughly constant?"],
    "**Implement it yourself:** integrate the single-mode UCM start-up equations "
    "$\\lambda\\,d\\tau_{xy}/dt = -\\tau_{xy} + \\eta\\dot\\gamma$, $\\lambda\\,d\\tau_{xx}/dt = -\\tau_{xx} + 2\\lambda\\dot\\gamma\\tau_{xy}$ with the explicit Euler "
    "method and compare with `cm.startup_shear`.")

EXTRAS["11_laos"] = _x(
    ["describe nonlinear oscillatory responses with Lissajous curves and Fourier harmonics",
     "compute and interpret the Chebyshev measures G'_M, G'_L, S and T",
     "distinguish intra-cycle stiffening from overall softening"],
    "notebooks 08 and 10; Fourier series", "60 min",
    [md(r"""
## Inside the algorithm: harmonics by least squares
Over one steady cycle, fit the stress with $\sin n\omega t$ and $\cos n\omega t$ for odd $n$; the coefficients divided by the
strain amplitude are $G'_n$ and $G''_n$:
"""), code(r"""g0, grp = list(yo.groupby("strain_amplitude"))[3]
tt, s = grp.time_s.to_numpy(), grp.shear_stress_Pa.to_numpy()
X = np.column_stack([f(n * w_y * tt) for n in (1, 3, 5) for f in (np.sin, np.cos)])
c = np.linalg.lstsq(X, s, rcond=None)[0] / g0
lib = laos.harmonics(tt, grp.strain, s, w_y, n_max=5)
print("by hand G'_1, G'_3:", np.round(c[[0, 2]], 3), " library:", np.round(lib["Gp"][:2], 3))""")],
    ["Simulate LAOS of the UCM model (the same spectrum, no Giesekus term) at strain 20. How large is $I_3/I_1$, and "
     "why is the UCM shear stress linear in strain at any amplitude?",
     "For the yoghurt, at which strain amplitude does $G'_L$ fall below half of its small-strain value? Compare with the "
     "flow point from the first-harmonic moduli (`oscillatory.moduli_from_waveforms` for each amplitude)."],
    "**Implement it yourself:** the area of the elastic Lissajous loop equals the energy dissipated per cycle, "
    "$\\pi\\gamma_0^2 G''_1$. Compute the loop area of one yoghurt cycle with the shoelace formula and compare.")


NOTEBOOKS["12_polymer_melts"] = [
    md(r"""
# 12 · Polymer melts and solutions

The flow of a polymer melt is governed by its molecules: their length (molar mass), how strongly they
entangle, and the breadth of the molar-mass distribution. This notebook connects rheological
measurements to molecular parameters - and shows why the industry's favourite single number, the melt
flow index, is not enough.
"""),
    setup_cell(["from scipy import optimize", "from engrheo import datasets, polymers, models"]),
    md(r"""
## Zero-shear viscosity and molar mass
Short chains flow like a Rouse fluid, $\eta_0 \propto M$; above a critical molar mass $M_c$ the chains entangle and
$\eta_0 \propto M^{3.4}$ - a doubling of the molar mass then raises the viscosity about tenfold.
"""),
    code(r"""pe = pd.read_csv(datasets.path("polyethylene_eta0_vs_mw.csv"), comment="#")
M, eta0 = pe.Mw_g_mol.to_numpy(), pe.eta0_Pa_s.to_numpy()

def resid(p):                                        # p = log10 Mc, log10 eta_c, exponent above Mc
    return np.log10(polymers.zero_shear_viscosity(M, 10**p[0], 10**p[1], p[2])) - np.log10(eta0)
fit = optimize.least_squares(resid, [4.0, 0.0, 3.0])
Mc, eta_c, a = 10**fit.x[0], 10**fit.x[1], fit.x[2]
MM = np.logspace(2.8, 5.8, 200)
plt.loglog(M, eta0, "o", label="PE grades at 190 degC")
plt.loglog(MM, polymers.zero_shear_viscosity(MM, Mc, eta_c, a), label=f"fit: M_c = {Mc/1e3:.1f} kg/mol, slope {a:.2f}")
plt.xlabel("Mw (g/mol)"); plt.ylabel("zero-shear viscosity (Pa s)"); plt.legend(); plt.show()
print(f"critical molar mass {Mc/1e3:.2f} kg/mol, exponent above M_c {a:.2f}   (true 4.0 kg/mol, 3.4)")
print(f"a 20 % higher Mw raises eta0 by a factor {1.2**a:.2f}")"""),
    md(r"""
The steep power law is why molar mass is controlled so tightly in polymer production: small changes in
chain length make large changes in processability.

## Entanglements from the plateau modulus
The rubbery plateau of a melt's $G'$ (notebook 05) measures the entanglement density: the plateau modulus $G_N^0$
gives the molar mass between entanglements, $M_e = \rho RT/G_N^0$. Estimate $G_N^0$ as $G'$ where $\tan\delta$ is smallest:
"""),
    code(r"""melt = pd.read_csv(datasets.path("polystyrene_frequency_sweeps.csv"), comment="#")
m170 = melt[melt.temperature_C == 220]                   # the hottest sweep reaches furthest into the plateau
i = np.argmin(m170.G_loss_Pa / m170.G_storage_Pa)
G_N0 = m170.G_storage_Pa.iloc[i]
print(f"G' at minimum tan(delta): {G_N0:.3g} Pa  ->  M_e = rho R T / G_N0 = {polymers.entanglement_molar_mass(G_N0, 950.0, 493.15):.1f} kg/mol")"""),
    md(r"""
The minimum of $\tan\delta$ gives only a rough estimate of the plateau - the plateau of this material is not
flat - and the prefactor (1 or 4/5) differs between textbooks, so $M_e$ is uncertain by tens of percent. For
polystyrene the literature value is about 13-18 kg/mol. Note that $M_c$ is typically 2-3 times $M_e$.

## The melt flow index - and its limits
The melt flow index (MFI or MFR) is the mass of polymer extruded through a standard die in 10 minutes
under a standard load. It is a single point on the flow curve, at a low shear rate:
"""),
    code(r"""for mfi in (0.3, 2.0, 20.0):
    c = polymers.mfi_conditions(mfi, 2.16, 740.0)       # polyethylene: 190 degC, 2.16 kg, melt density 740 kg/m3
    print(f"MFI {mfi:4.1f} g/10 min: wall stress {c['tau_w']/1e3:.1f} kPa, apparent rate {c['rate_apparent']:7.2f} 1/s, "
          f"apparent viscosity {c['eta_apparent']:7.0f} Pa s")
cond = polymers.mfi_conditions(2.0, 2.16, 740.0)
rr = np.logspace(-2, 4, 200)
eta_target = cond["eta_apparent"]
for name, n_, lam in (("narrow distribution", 0.55, 0.05), ("broad distribution", 0.3, 2.0)):
    # choose eta0 so both grades have the same apparent viscosity at the MFI shear rate
    base = models.viscosity("carreau", cond["rate_apparent"], 1.0, 0.0, lam, n_)
    eta0_ = eta_target / base
    plt.loglog(rr, models.viscosity("carreau", rr, eta0_, 0.0, lam, n_), label=f"{name}")
    print(f"{name}: viscosity at 1000 1/s (extrusion) {models.viscosity('carreau', 1000.0, eta0_, 0.0, lam, n_):7.1f} Pa s")
plt.axvline(cond["rate_apparent"], color="k", ls=":", label="MFI test"); plt.xlabel("shear rate (1/s)"); plt.ylabel("viscosity (Pa s)")
plt.legend(); plt.show()"""),
    md(r"""
Two grades with the same MFI can behave very differently in extrusion or injection moulding, where the
shear rates are hundreds of times higher: a broad molar-mass distribution shear-thins more strongly. The MFI
is a useful quality-control number for one grade, but comparing grades needs the flow curve.
"""),
]

NOTEBOOKS["13_suspensions_and_emulsions"] = [
    md(r"""
# 13 · Suspensions and emulsions

Cement paste, paints, mine tailings, chocolate, blood, mayonnaise: most engineering fluids are particles or
droplets in a liquid. Their viscosity rises dramatically as the particles crowd together, diverging at a
**maximum packing fraction**; at high concentrations they may develop a yield stress or even thicken under
shear.
"""),
    setup_cell(["from engrheo import datasets, suspensions as su"]),
    code(r"""gb = pd.read_csv(datasets.path("glass_beads_suspension.csv"), comment="#")
phi, eta_r = gb.volume_fraction.to_numpy(), gb.relative_viscosity.to_numpy()
fixed = su.fit_krieger_dougherty(phi, eta_r)                    # [eta] = 2.5 for spheres
free = su.fit_krieger_dougherty(phi, eta_r, fix_intrinsic=None)
pp = np.linspace(0, 0.58, 200)
plt.semilogy(phi, eta_r, "o", label="glass beads in oil")
plt.semilogy(pp, su.einstein(pp), ":", label="Einstein (dilute)")
plt.semilogy(pp, su.krieger_dougherty(pp, fixed["phi_max"]), label=f"Krieger-Dougherty, phi_max = {fixed['phi_max']:.3f}")
plt.xlabel("solid volume fraction"); plt.ylabel("relative viscosity"); plt.legend(); plt.show()
print(f"[eta] fixed at 2.5: phi_max = {fixed['phi_max']:.3f};  both free: phi_max = {free['phi_max']:.3f}, [eta] = {free['intrinsic']:.2f}")
print("true: phi_max = 0.61, [eta] = 2.5")"""),
    md(r"""
Einstein's result holds only in the dilute limit. Krieger-Dougherty captures the whole curve with one
parameter, the maximum packing fraction (about 0.58-0.64 for monodisperse spheres, depending on packing).

## Close to maximum packing, everything is sensitive
"""),
    code(r"""pm = fixed["phi_max"]
for p_ in (0.40, 0.50, 0.55, 0.58):
    print(f"phi = {p_:.2f}: eta_r = {su.krieger_dougherty(p_, pm):8.1f};  +0.01 in phi -> x{su.krieger_dougherty(p_ + 0.01, pm)/su.krieger_dougherty(p_, pm):.2f}")
for pm2 in (pm, pm + 0.05, pm + 0.10):
    print(f"phi_max = {pm2:.2f} (broader size distribution): eta_r at phi = 0.55 is {su.krieger_dougherty(0.55, pm2):6.1f}")"""),
    md(r"""
Near maximum packing a one-percent change in solids content changes the viscosity by tens of percent. The
same sensitivity is an opportunity: mixing particle sizes lets small particles fill the gaps between large
ones, raising $\phi_{max}$ (the Farris effect). That is how concrete, chocolate and ceramic slurries reach high
solids contents at a pumpable viscosity.

## When suspensions thicken: cornstarch
"""),
    code(r"""cs = pd.read_csv(datasets.path("cornstarch_flow_curve.csv"), comment="#")
rate, eta = cs.shear_rate_1_s.to_numpy(), cs.viscosity_Pa_s.to_numpy()
slope = np.gradient(np.log(eta), np.log(rate))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
a1.loglog(rate, eta, "o-"); a1.set(xlabel="shear rate (1/s)", ylabel="viscosity (Pa s)")
a2.semilogx(rate, slope, "o-"); a2.axhline(0, color="k", lw=0.8)
a2.set(xlabel="shear rate (1/s)", ylabel="d ln(eta) / d ln(rate)"); plt.show()
print(f"minimum viscosity {eta.min():.2f} Pa s at {rate[np.argmin(eta)]:.1f} 1/s; at the highest rate {eta[-1]:.1f} Pa s")"""),
    md(r"""
Dense suspensions of non-Brownian particles first shear-thin, then thicken sharply as frictional contacts
form between particles. For process equipment this is dangerous: pumping or mixing such a slurry faster
can raise the resistance abruptly, even jam the flow. Know where thickening starts before choosing
pump or mixer speeds.

**Emulsions** behave similarly with droplets instead of particles - but droplets deform, so emulsions can
exceed the packing limit of spheres: mayonnaise (over 75 % oil) is a soft solid with a yield stress because
its droplets are squeezed against each other.
"""),
]

NOTEBOOKS["14_extensional_rheology"] = [
    md(r"""
# 14 · Extensional rheology

Fibre spinning, film blowing, inkjet printing, spraying, swallowing and the break-up of a thread of saliva are
dominated not by shear but by **extension** - stretching. Polymer solutions that thin in shear can become
enormously resistant in extension (strain hardening), which stabilises fibres and filaments but can stop an
inkjet drop from detaching.
"""),
    setup_cell(["from engrheo import datasets, extensional as ex, constitutive as cm"]),
    md(r"""
## Shear versus extension
A Newtonian liquid's extensional viscosity is exactly three times its shear viscosity (Trouton ratio 3). A
polymer solution behaves very differently once the extension rate exceeds about half the inverse relaxation
time ($Wi = \lambda\dot\varepsilon > 0.5$), where stretching outpaces relaxation:
"""),
    code(r"""eta_p, lam = 10.0, 0.1
t = np.logspace(-3, 1, 200)
fig, ax = plt.subplots()
for model, extra in (("ucm", {}), ("giesekus", {"alpha": 0.1}), ("ptt", {"eps": 0.05})):
    for Wi in (0.1, 2.0):
        r = cm.startup_extension([cm.Mode(eta_p, lam, model, **extra)], Wi / lam, np.r_[0.0, t])
        ax.loglog(t, r["eta_E_plus"][1:] / eta_p, label=f"{model}, Wi = {Wi}", ls="-" if Wi > 1 else ":")
ax.axhline(3, color="k", lw=0.8)
ax.set(xlabel="time (s)", ylabel="eta_E+ / eta_0", ylim=(1, 1e4)); ax.legend(fontsize=7); plt.show()"""),
    md(r"""
At $Wi$ = 0.1 all three models approach the Trouton value of 3. At $Wi$ = 2 the UCM (Oldroyd-B) model's
extensional viscosity grows without bound - its dumbbells stretch infinitely - while Giesekus and PTT
saturate at a high but finite value, like real polymers with finite chain length. The large ratio of
extensional to shear viscosity is what lets a polymer solution form long, stable threads.

## Measuring it: capillary thinning (CaBER)
A drop is stretched between two plates into a liquid bridge, which then thins under surface tension.
The mid-filament diameter tells the story:
"""),
    code(r"""cb = pd.read_csv(datasets.path("polymer_solution_caber.csv"), comment="#")
t_c, D = cb.time_s.to_numpy(), cb.diameter_m.to_numpy()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
a1.semilogy(t_c * 1e3, D * 1e6, ".", ms=4); a1.set(xlabel="time (ms)", ylabel="diameter (um)", title="Capillary thinning")
ec = (t_c > 0.02) & (t_c < 0.18)                           # the straight (exponential) part on the semilog plot
fit = ex.fit_elastocapillary(t_c[ec], D[ec])
a1.semilogy(t_c[ec] * 1e3, fit["D_fit0"] * np.exp(-t_c[ec] / (3 * fit["lambda_E"])) * 1e6, "r-", label="elasto-capillary fit")
a1.legend()
sigma = 0.062
eta_E = ex.apparent_extensional_viscosity(t_c, D, sigma, X=0.7127)
ok = np.isfinite(eta_E) & (t_c < 0.2)
a2.semilogy(t_c[ok] * 1e3, eta_E[ok], ".", ms=4); a2.set(xlabel="time (ms)", ylabel="apparent extensional viscosity (Pa s)")
plt.show()
print(f"extensional relaxation time lambda_E = {fit['lambda_E']*1e3:.2f} ms (true 12 ms)")"""),
    md(r"""
A Newtonian liquid would thin linearly in time. This solution thins **exponentially** - the elasto-capillary
balance, in which surface tension is resisted by stretched polymer chains - with a rate set by the extensional
relaxation time. The apparent extensional viscosity grows by orders of magnitude as the filament thins
(strain hardening), until the chains are fully stretched and the filament finally drains and breaks. (The
factor $2X - 1$ with $X$ = 0.7127 is exact only for Newtonian thinning; for polymer solutions it is an
approximation, so the absolute values are indicative.)

For inkjet printing this relaxation time decides whether a drop detaches cleanly or trails a long ligament
that breaks into satellite droplets.
"""),
]

NOTEBOOKS["16_mixing_and_processing"] = [
    md(r"""
# 16 · Mixing and processing

The flow curves and viscoelastic parameters of the earlier notebooks feed directly into equipment design:
the power a mixer needs, the pressure a die requires, the size of an extrudate after it leaves the die.
This notebook works three such calculations and shows where non-Newtonian behaviour changes the answer.
"""),
    setup_cell(["from engrheo import flows, models, constitutive as cm"]),
    md(r"""
## Mixing a shear-thinning fluid
Shear rates in a stirred tank vary from point to point. Metzner and Otto showed that, in laminar flow, the
power follows the Newtonian curve if the viscosity is evaluated at an average shear rate $k_s N$ (impeller
speed $N$ in rev/s; $k_s \approx$ 10-13 for turbines). For the xanthan drink of notebook 03:
"""),
    code(r"""car = (35.0, 0.002, 12.0, 0.25)                             # Carreau fit of the xanthan solution
D_imp, rho, ks, Kp = 0.1, 1000.0, 11.0, 70.0                 # impeller diameter, density, Metzner-Otto constant, Po*Re
for N in (0.5, 1.0, 2.0, 4.0):
    rate = ks * N
    eta = models.viscosity("carreau", rate, *car)
    P = Kp * eta * N**2 * D_imp**3
    Re = rho * N * D_imp**2 / eta
    print(f"N = {N:3.1f} rev/s: average rate {rate:4.0f} 1/s, apparent viscosity {eta:6.3f} Pa s, Re = {Re:6.1f}, power {P*1e3:6.2f} mW")"""),
    md(r"""
Doubling the speed of a Newtonian mixer in laminar flow quadruples the power; for this shear-thinning drink
the power rises much less, because the fluid thins as it is stirred faster. The check on the Reynolds number
matters: Metzner-Otto holds only in laminar flow, which here ends at the higher speeds.

## Pressure in a sheet die
A polymer melt (the Carreau melt of notebook 02: $\eta_0$ = 3000 Pa·s, $\lambda$ = 0.3 s, $n$ = 0.3) is extruded through a
sheet die 1 m wide with a 1 mm gap and 50 mm land length at 100 kg/h. At process shear rates the melt is
described by a local power law, whose parameters depend on the wall shear rate - so the calculation iterates:
"""),
    code(r"""melt = (3000.0, 0.0, 0.3, 0.3)
W, H, L_die, rho_m = 1.0, 1e-3, 0.05, 750.0
Q = 100 / 3600 / rho_m
rate_w = 6 * Q / (W * H**2)                                   # start from the Newtonian wall rate
for it in range(6):
    n_loc = 1 + np.gradient(np.log(models.viscosity("carreau", rate_w * np.array([0.99, 1.0, 1.01]), *melt)), np.log([0.99, 1.0, 1.01]))[1]
    K_loc = models.stress("carreau", rate_w, *melt) / rate_w**n_loc
    res = flows.slit_power_law(K_loc, n_loc, W, H, Q=Q)
    print(f"iteration {it}: local n = {n_loc:.3f}, wall rate {res['rate_w']:7.1f} 1/s, pressure drop {res['dp_per_L']*L_die/1e5:6.2f} bar")
    if abs(res["rate_w"] / rate_w - 1) < 1e-6:
        break
    rate_w = res["rate_w"]"""),
    md(r"""
The iteration settles in a few steps. Using the zero-shear viscosity instead would overestimate the
pressure enormously - the melt at the wall is sheared at a rate where its viscosity is far lower.

## Extrudate swell
When an elastic melt leaves a die, the recovering normal stresses make the extrudate swell. Tanner's
estimate uses the ratio of the first normal-stress difference to the shear stress at the wall:
"""),
    code(r"""modes = cm.modes_from_spectrum([60.0, 25.0, 8.0], [0.02, 0.2, 2.0], "giesekus", alpha=0.25)   # notebook 10 solution
for rate in (1.0, 10.0, 100.0):
    st = cm.steady_shear(modes, [rate])
    tau = st["eta"][0] * rate
    print(f"wall rate {rate:5.0f} 1/s: N1/tau = {st['N1'][0]/tau:5.2f}, estimated swell ratio {flows.tanner_swell(st['N1'][0], tau):.2f}")"""),
    md(r"""
Swell grows with the elasticity of the flow - the ratio $N_1/\tau$, which rises with the Weissenberg number. Die
designers compensate by making the die smaller than the desired product. Tanner's formula is an estimate;
real swell also depends on die length (short dies swell more) and on the full relaxation after exit.
"""),
]


EXTRAS["12_polymer_melts"] = _x(
    ["relate zero-shear viscosity to molar mass (Rouse and entangled regimes)",
     "estimate the entanglement molar mass from the plateau modulus",
     "interpret the melt flow index and its limitations"],
    "notebooks 03, 05 and 08", "60 min",
    [md(r"""
## Inside the algorithm: the melt flow index test by hand
The load on the piston produces a pressure $mg/(\pi R_p^2)$; in the die the wall stress is that pressure times
$R/(2L)$, and the apparent rate is $4Q/(\pi R^3)$ with $Q$ = MFI / (density x 600 s):
"""), code(r"""mfi, load, rho_ = 2.0, 2.16, 740.0
p = load * 9.80665 / (np.pi * (9.55e-3 / 2)**2)
tau_hand = p * (2.095e-3 / 2) / (2 * 8.0e-3)
rate_hand = 4 * (mfi * 1e-3 / rho_ / 600) / (np.pi * (2.095e-3 / 2)**3)
lib = polymers.mfi_conditions(mfi, load, rho_)
print(f"by hand: {tau_hand:.1f} Pa, {rate_hand:.3f} 1/s;  library: {lib['tau_w']:.1f} Pa, {lib['rate_apparent']:.3f} 1/s")""")],
    ["Predict the zero-shear viscosity of a 150 kg/mol grade with the fitted relation. How uncertain is the "
     "prediction if the exponent is uncertain by +/-0.1?",
     "A grade must process at the same viscosity as before, but at a melt temperature 20 K higher (Arrhenius, "
     "E_a = 30 kJ/mol for polyethylene). By how much may its molar mass increase?"],
    "**Implement it yourself:** estimate the exponent above $M_c$ with an ordinary straight-line fit of "
    "$\\log\\eta_0$ against $\\log M$ for the grades above 10 kg/mol, and compare with the piecewise fit.")

EXTRAS["13_suspensions_and_emulsions"] = _x(
    ["predict suspension viscosity from the volume fraction (Einstein, Krieger-Dougherty)",
     "fit the maximum packing fraction and appreciate the sensitivity near it",
     "recognise shear thickening and its process risks"],
    "notebooks 03-04", "45-60 min",
    [md(r"""
## Inside the algorithm: fitting the maximum packing fraction
With $[\eta]$ fixed, Krieger-Dougherty has one parameter; scan $\phi_{max}$ and minimise the squared log residuals:
"""), code(r"""grid = np.linspace(phi.max() + 0.005, 0.8, 4000)
sse = [np.sum((np.log(su.krieger_dougherty(phi, g)) - np.log(eta_r))**2) for g in grid]
print(f"by hand: phi_max = {grid[int(np.argmin(sse))]:.4f};  library: {fixed['phi_max']:.4f}")""")],
    ["What solids fraction gives a relative viscosity of 10 with the fitted $\\phi_{max}$? And of 100?",
     "At a fixed solids content of 0.55, by what factor does the viscosity fall if a broader size distribution "
     "raises $\\phi_{max}$ from the fitted value to 0.70?"],
    "**Implement it yourself:** tabulate the Einstein, Batchelor and Krieger-Dougherty predictions at "
    "$\\phi$ = 0.02, 0.05, 0.1 and 0.2 and compare them with the data. Where does each fail?")

EXTRAS["14_extensional_rheology"] = _x(
    ["explain the Trouton ratio and strain hardening", "compare constitutive models in extension",
     "analyse a capillary-thinning experiment for the extensional relaxation time"],
    "notebooks 10 and 07", "60 min",
    [md(r"""
## Inside the algorithm: the elasto-capillary fit
In the elasto-capillary regime $D = D_1 e^{-t/(3\lambda_E)}$, a straight line of $\ln D$ against $t$ with slope $-1/(3\lambda_E)$:
"""), code(r"""slope, _ = np.polyfit(t_c[ec], np.log(D[ec]), 1)
print(f"by hand: lambda_E = {-1/(3*slope)*1e3:.3f} ms;  library: {fit['lambda_E']*1e3:.3f} ms")""")],
    ["A filament of glycerol ($\\eta$ = 1 Pa·s, $\\sigma$ = 0.063 N/m) starts at 3 mm diameter. How long does it take to "
     "break (Newtonian thinning)? Compare with the polymer solution.",
     "For an inkjet nozzle of 20 um radius, compare the capillary time $\\sqrt{\\rho R^3/\\sigma}$ ($\\rho$ = 1000 kg/m³) with the "
     "extensional relaxation time. What does the resulting Deborah number predict about satellite drops?"],
    "**Implement it yourself:** generate a Newtonian thinning curve with `ex.newtonian_thinning`, compute the "
    "apparent extensional viscosity from its slope by hand, and confirm the Trouton ratio of 3.")

EXTRAS["16_mixing_and_processing"] = _x(
    ["estimate laminar mixing power for shear-thinning fluids (Metzner-Otto)",
     "calculate die pressures with a local power law and iteration", "estimate extrudate swell from normal stresses"],
    "notebooks 03, 10 and 15", "60 min",
    [md(r"""
## Inside the algorithm: flow rate through a slit by integrating the profile
For a power-law fluid in a slit the velocity at distance $y$ from the mid-plane is
$u = \frac{n}{n+1}\left(\frac{\Delta P}{KL}\right)^{1/n}\left(h^{1+1/n} - y^{1+1/n}\right)$, $h = H/2$; integrate across the gap:
"""), code(r"""from scipy import integrate
G = res["dp_per_L"]
u = lambda y: n_loc/(n_loc+1) * (G/K_loc)**(1/n_loc) * ((H/2)**(1+1/n_loc) - y**(1+1/n_loc))
Q_int = 2 * W * integrate.quad(u, 0, H/2, epsabs=0, epsrel=1e-12)[0]
print(f"integrated {Q_int:.6e} m3/s, closed form {res['Q']:.6e} m3/s")""")],
    ["How does the mixing power of the xanthan drink scale with speed between 0.5 and 2 rev/s, compared with a "
     "Newtonian liquid? Estimate the exponent from the table.",
     "Double the die gap to 2 mm at the same throughput. By what factor does the pressure drop fall, and how "
     "does that compare with a Newtonian melt (factor 8)?"],
    "**Implement it yourself:** write Tanner's swell formula yourself and tabulate the swell ratio against "
    "$N_1/\\tau$ from 0 to 5. At what ratio does the extrudate swell by 50 %?")


NOTEBOOKS["17_rheometer_file_to_report"] = [
    md(r"""
# 17 · From a messy rheometer file to a report

**Problem.** A food-technology lab has measured a tomato ketchup four ways - an amplitude sweep, a
frequency sweep, and flow curves up and down - and exported everything in one file from a
German-language rheometer PC. The customer wants a one-page report: Is the ketchup gel-like at rest? What
is its yield stress? How does it flow when squeezed from a bottle, and does it recover?

This notebook uses nearly every tool of the course, starting where real work starts: with a file that is
not ready for analysis.
"""),
    setup_cell(["import io", "from engrheo import datasets, fitting, oscillatory"]),
    md("## 1. Look at the file"),
    code(r"""path = datasets.path("ketchup_rheometer_export.csv")
with open(path, encoding="utf-8") as fh:
    raw_lines = fh.read().splitlines()
print(f"{len(raw_lines)} lines")
print("\n".join(raw_lines[:14]))
print("...")
print("\n".join(raw_lines[49:57]))"""),
    md(r"""
Several things stand out: semicolons separate the columns, decimals use commas, a units row follows each
header, the file contains four separate tests with their own metadata, and a status column flags
problems. `pd.read_csv` on the whole file would fail; the tests must be split first.

## 2. Parse the blocks
"""),
    code(r"""def parse_export(lines):
    tests, i = {}, 0
    while i < len(lines):
        if lines[i].startswith("Test;"):
            name = lines[i].split(";")[1]
            meta = {}
            i += 1
            while not lines[i].startswith("Point;"):          # metadata lines of this test
                key, value = lines[i].split(";")[:2]
                meta[key] = value
                i += 1
            header, units = lines[i].split(";"), lines[i + 1].split(";")
            columns = [f"{h} ({u})" if u else h for h, u in zip(header, units)]
            j = i + 2
            while j < len(lines) and lines[j].strip():
                j += 1
            table = pd.read_csv(io.StringIO("\n".join(lines[i + 2:j])), sep=";", decimal=",", header=None,
                                names=columns, na_values=["---"])
            tests[name] = {"meta": meta, "data": table}
            i = j
        else:
            i += 1
    return tests

tests = parse_export(raw_lines)
for name, t in tests.items():
    print(f"{name:<20} {len(t['data']):3d} points  {t['meta']}  columns: {list(t['data'].columns)[1:4]}")"""),
    md(r"""
## 3. Check before analysing
"""),
    code(r"""for name, t in tests.items():
    flagged = t["data"][t["data"]["Status"] != "ok"]
    if len(flagged):
        print(f"{name}:"); print(flagged.to_string(index=False)); print()
up = tests["Flow curve (up)"]["data"]
consistency = np.nanmax(np.abs(up["Viscosity (mPa·s)"] / 1000 - up["Shear stress (Pa)"] / up["Shear rate (1/s)"])
                        / (up["Shear stress (Pa)"] / up["Shear rate (1/s)"]))
print(f"viscosity column (mPa s) consistent with stress/rate to {consistency:.1e}")"""),
    md(r"""
Two kinds of problem: at the lowest shear rates of the upward curve the instrument reports that steady
state was not reached - the stress was still decaying when the point was recorded, so it reads too high -
and one point of the downward curve failed completely. Both are excluded, and the decision is recorded. The
viscosity column is in mPa·s, not Pa·s - a factor of 1000 that the units row reveals.

## 4. At rest: amplitude and frequency sweeps
"""),
    code(r"""amp = tests["Amplitude sweep"]["data"]
strain = amp["Strain (%)"].to_numpy() / 100
Gp, Gpp = amp["Storage modulus (Pa)"].to_numpy(), amp["Loss modulus (Pa)"].to_numpy()
lve = oscillatory.lve_limit(strain, Gp)
flow_strain, flow_G = oscillatory.crossover(strain, Gp, Gpp)
tau_flowpoint = np.hypot(flow_G, flow_G) * flow_strain             # |G*| x strain at the crossover
frq = tests["Frequency sweep"]["data"]
w = frq["Angular frequency (rad/s)"].to_numpy()
tan_d = frq["Loss modulus (Pa)"] / frq["Storage modulus (Pa)"]
slope = np.polyfit(np.log(w), np.log(frq["Storage modulus (Pa)"]), 1)[0]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
a1.loglog(strain * 100, Gp, "o-", ms=3, label="G'"); a1.loglog(strain * 100, Gpp, "s-", ms=3, label="G''")
a1.axvline(lve * 100, color="k", ls=":"); a1.set(xlabel="strain (%)", ylabel="modulus (Pa)", title="Amplitude sweep, 1 Hz"); a1.legend()
a2.loglog(w, frq["Storage modulus (Pa)"], "o-", ms=3, label="G'"); a2.loglog(w, frq["Loss modulus (Pa)"], "s-", ms=3, label="G''")
a2.set(xlabel="angular frequency (rad/s)", title="Frequency sweep"); a2.legend(); plt.show()
print(f"linear range up to {lve*100:.2f} % strain; flow point at {flow_strain*100:.1f} % strain, stress {tau_flowpoint:.1f} Pa")
print(f"frequency sweep: G' > G'' everywhere: {bool(np.all(tan_d < 1))}; tan(delta) {tan_d.min():.2f}-{tan_d.max():.2f}; "
      f"G' ~ omega^{slope:.2f}")"""),
    md(r"""
At rest the ketchup is a **weak gel**: $G'$ exceeds $G''$ at all frequencies and depends only weakly on frequency.
The amplitude sweep shows how much deformation the structure tolerates before it yields.

## 5. When it flows: flow curves
"""),
    code(r"""def valid(block):
    d = tests[block]["data"]
    d = d[(d["Status"] == "ok") & d["Shear stress (Pa)"].notna()].sort_values("Shear rate (1/s)")
    return d["Shear rate (1/s)"].to_numpy(), d["Shear stress (Pa)"].to_numpy()

r_up, t_up = valid("Flow curve (up)")
r_dn, t_dn = valid("Flow curve (down)")
hb_up = fitting.fit_flow_curve(r_up, t_up, "herschel_bulkley")
hb_dn = fitting.fit_flow_curve(r_dn, t_dn, "herschel_bulkley")
print(hb_up); print(); print(hb_dn)
rr = np.logspace(-2, 2.6, 200)
plt.loglog(r_up, t_up, "o", ms=4, label="up (valid points)"); plt.loglog(r_dn, t_dn, "s", ms=4, label="down")
plt.loglog(rr, hb_up.predict_stress(rr), "C0-", lw=1); plt.loglog(rr, hb_dn.predict_stress(rr), "C1-", lw=1)
plt.xlabel("shear rate (1/s)"); plt.ylabel("shear stress (Pa)"); plt.legend(); plt.show()
x = np.r_[r_up, r_dn[::-1]]; y = np.r_[t_up, t_dn[::-1]]
loop = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
print(f"thixotropic loop area {loop:.0f} Pa/s; down-curve yield stress {hb_dn.params['tau_y']/hb_up.params['tau_y'] - 1:+.0%} relative to up")"""),
    md(r"""
The downward curve lies below the upward one - shear has broken down part of the structure (notebook 06).

## 6. How strong is it? Three yield-stress estimates
"""),
    code(r"""estimates = pd.DataFrame({
    "method": ["Herschel-Bulkley, up curve (extrapolated)", "Herschel-Bulkley, down curve (extrapolated)",
               "flow point of the amplitude sweep (G' = G'')"],
    "yield stress (Pa)": [hb_up.params["tau_y"], hb_dn.params["tau_y"], tau_flowpoint],
    "95 % interval (Pa)": [f"{hb_up.conf_int()['tau_y'][0]:.1f} - {hb_up.conf_int()['tau_y'][1]:.1f}",
                           f"{hb_dn.conf_int()['tau_y'][0]:.1f} - {hb_dn.conf_int()['tau_y'][1]:.1f}", "-"]})
print(estimates.round(1).to_string(index=False))"""),
    md(r"""
The estimates differ, and they should: each measures something different (notebook 04). The up-curve fit
describes the undisturbed product, the down curve the product after shearing, and the flow point the
stress at which the gel structure gives way in oscillation. A report must say which one it quotes.

## 7. The report
"""),
    code(r"""import tempfile
from pathlib import Path
out = Path(tempfile.gettempdir()) / "ketchup_report"
out.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(r_up, t_up, "o", label="measured (up)"); ax.loglog(rr, hb_up.predict_stress(rr), label="Herschel-Bulkley fit")
ax.set(xlabel="shear rate (1/s)", ylabel="shear stress (Pa)", title="Ketchup, 25 degC"); ax.legend()
fig.savefig(out / "flow_curve.png", dpi=150, bbox_inches="tight"); plt.close(fig)
p = hb_up.params; ci = hb_up.conf_int()
report = f'''# Rheology report - tomato ketchup, 25 degC

**At rest:** weak gel (G' > G'' from {w.min():g} to {w.max():g} rad/s, tan(delta) {tan_d.min():.2f}-{tan_d.max():.2f});
linear up to {lve*100:.1f} % strain; structure yields at {flow_strain*100:.0f} % strain ({tau_flowpoint:.0f} Pa).

**Flow (Herschel-Bulkley, upward flow curve, {len(r_up)} valid points):** yield stress {p['tau_y']:.1f} Pa
(95 % CI {ci['tau_y'][0]:.1f}-{ci['tau_y'][1]:.1f}), K = {p['K']:.2f} Pa s^n, n = {p['n']:.2f}.

**Thixotropy:** after shearing the yield stress is {abs(hb_dn.params['tau_y']/p['tau_y'] - 1):.0%} lower (loop area {loop:.0f} Pa/s).

**Data handling:** {int((up['Status'] != 'ok').sum())} upward points excluded (steady state not reached),
1 downward point excluded (measurement failed). Figure: flow_curve.png.
'''
(out / "report.md").write_text(report, encoding="utf-8")
print(report)"""),
    md(r"""
The report states results *with* their uncertainty, names the methods, and records every data-handling
decision - so that anyone can reproduce it from the raw file. The whole analysis is a script: when next
week's batch is measured, it runs again unchanged.

(The file is synthetic, generated from a Herschel-Bulkley ketchup with $\tau_y$ = 15 Pa, $K$ = 4.5 Pa·s$^n$ and
$n$ = 0.35 - compare the report with these values.)
"""),
]


EXTRAS["17_rheometer_file_to_report"] = _x(
    ["parse a multi-test rheometer export (separators, decimal commas, units, metadata)",
     "apply quality checks and document excluded data", "combine oscillatory and steady tests into conclusions",
     "write a reproducible report with uncertainties"],
    "notebooks 00, 02-04, 06 and 08", "75-90 min",
    [md(r"""
## Inside the algorithm: the yield stress at the flow point by hand
At the crossover $G' = G''$, so $|G^*| = \sqrt 2\,G'$; the stress amplitude there is $|G^*|\gamma_0$. Interpolate the crossover
on log scales between the two bracketing points:
"""), code(r"""d = np.log(Gp) - np.log(Gpp)
i = np.nonzero(np.sign(d[:-1]) != np.sign(d[1:]))[0][0]
f = d[i] / (d[i] - d[i + 1])
g_c = np.exp(np.log(strain[i]) + f * np.log(strain[i + 1] / strain[i]))
G_c = np.exp(np.log(Gp[i]) + f * np.log(Gp[i + 1] / Gp[i]))
print(f"by hand: flow point at {g_c*100:.2f} % strain, stress {np.sqrt(2)*G_c*g_c:.2f} Pa;  notebook: {tau_flowpoint:.2f} Pa")""")],
    ["Refit the upward flow curve *including* the two points flagged 'steady state not reached'. How much do the "
     "yield stress and its confidence interval change? Why is excluding them justified - and why must the exclusion "
     "be reported?",
     "The ketchup must flow from a squeezed bottle but not run off a plate. Which of the three yield-stress estimates "
     "is most relevant to each requirement? Compare all three with the true value (15 Pa)."],
    "**Implement it yourself:** turn the parsing and the checks into one reusable function `load_export(path)` that "
    "returns the tests as DataFrames with SI units (Pa·s instead of mPa·s, strain as a fraction) and a list of "
    "excluded points, and test it on the file.")


def apply_extras(name, cells):
    ex = EXTRAS[name]
    header = md("**What you will learn**\n" + "\n".join(f"- {o}" for o in ex["objectives"])
                + f"\n\n**Before you start:** {ex['prereq']}  ·  **Time:** about {ex['time']}")
    cells = [cells[0], header] + list(cells[1:])
    idx = [i for i, c in enumerate(cells) if c.cell_type == "markdown" and "## Exercises" in c.source]
    if idx:
        i = idx[0]
        n_items = len(re.findall(r"^\d+\. ", cells[i].source, flags=re.M))
        cells[i] = md(cells[i].source.rstrip() + f"\n{n_items + 1}. {ex['implement']}")
        return cells[:i] + list(ex["inside"]) + cells[i:]
    items = ex["exercises"] + [ex["implement"]]
    exercises = md("## Exercises\n" + "\n".join(f"{k}. {t}" for k, t in enumerate(items, 1)))
    return cells + list(ex["inside"]) + [exercises]


def write_all(run: bool = True, only=()):
    import nbclient

    for name in sorted(NOTEBOOKS):
        if only and not name.startswith(tuple(only)):
            continue
        cells = apply_extras(name, NOTEBOOKS[name])
        nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                                "language": "python"},
                                                 "language_info": {"name": "python"}})
        if run:
            nbclient.NotebookClient(nb, timeout=900, kernel_name="python3",
                                    resources={"metadata": {"path": str(HERE)}}).execute()
        nbformat.write(nb, HERE / f"{name}.ipynb")
        print("wrote", name)


if __name__ == "__main__":
    write_all(run="--no-run" not in sys.argv, only=[a for a in sys.argv[1:] if not a.startswith("--")])
