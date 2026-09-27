"""Build the worked solutions to every exercise of the course, as executed notebooks.

The exercise texts are taken from ../notebooks/build_notebooks.py, so questions and solutions cannot
drift apart; the build fails if a notebook's number of solutions differs from its number of exercises.

    python solutions/build_solutions.py            # build and execute all solution notebooks
    python solutions/build_solutions.py 05 12      # only those whose names start with 05 or 12
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("course", HERE.parent / "notebooks" / "build_notebooks.py")
course = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(course)
md, code = course.md, course.code
SOLUTIONS: dict[str, dict] = {}


def solutions(name, setup="", imports=()):
    """Register the solutions of one course notebook: a function returning one list of cells per exercise."""
    def deco(fn):
        SOLUTIONS[name] = {"setup": setup, "imports": list(imports), "answers": fn()}
        return fn
    return deco


def exercises(name) -> list[str]:
    cells = course.apply_extras(name, list(course.NOTEBOOKS[name]))
    text = [c.source for c in cells if c.cell_type == "markdown" and "## Exercises" in c.source][0]
    items = re.split(r"\n(?=\d+\. )", text.split("## Exercises")[1].strip())
    return [re.sub(r"^\d+\.\s*", "", it).strip() for it in items]


def notebook_cells(name) -> list:
    title = re.sub(r"^#\s*", "", course.NOTEBOOKS[name][0].source.splitlines()[0])
    entry = SOLUTIONS[name]
    ex = exercises(name)
    if len(ex) != len(entry["answers"]):
        raise ValueError(f"{name}: {len(ex)} exercises but {len(entry['answers'])} solutions")
    cells = [md(f"""
# Solutions · {title}

Worked solutions to the exercises of [notebook {name[:2]}](../notebooks/{name}.ipynb). Try each exercise
yourself before reading its solution - the learning happens in the attempt. Solutions are one way to
answer each question; other correct approaches exist.
"""), course.setup_cell(entry["imports"])] + ([code(entry["setup"])] if entry["setup"] else [])
    for i, (text, answer) in enumerate(zip(ex, entry["answers"]), 1):
        cells.append(md(f"## Exercise {i}\n\n{text}"))
        cells.extend(answer)
    return cells


def write_all(only=()):
    import nbclient

    for name in SOLUTIONS:
        if only and not name.startswith(tuple(only)):
            continue
        nb = new_notebook(cells=notebook_cells(name),
                          metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                    "language_info": {"name": "python"}})
        nbclient.NotebookClient(nb, timeout=900, kernel_name="python3",
                                resources={"metadata": {"path": str(HERE)}}).execute()
        path = HERE / f"{name[:2]}_solutions.ipynb"
        nbformat.write(nb, path)
        print("wrote", path.name)




# =====================================================================================================
@solutions("00_python_for_rheology_data", imports=["from engrheo import datasets, geometry"],
           setup=r"""df = pd.read_csv(datasets.path("xanthan_flow_curve.csv"), comment="#")
rate, eta = df.shear_rate_1_s.to_numpy(), df.viscosity_Pa_s.to_numpy()""")
def _():
    return [
        [code(r"""tau_min, _ = geometry.cone_plate(1e-6, 1.0, R=0.025, theta=np.radians(1.0))
eta_min = tau_min / rate
plt.loglog(rate, eta, "o", ms=4, label="data"); plt.loglog(rate, eta_min, "k--", label="low-torque limit (1 uN m)")
plt.xlabel("shear rate (1/s)"); plt.ylabel("viscosity (Pa s)"); plt.legend(); plt.show()
print(f"lowest measurable stress {tau_min*1e3:.1f} mPa; smallest margin above the limit: factor {np.min(eta / eta_min):.0f} "
      f"(at {rate[np.argmin(eta / eta_min)]:g} 1/s)")"""),
         md(r"""
The low-torque limit is a straight line of slope $-1$ on the viscosity plot (constant minimum stress divided by
the rate). Every point lies well above it, so the whole flow curve is trustworthy. The margin is smallest
at the lowest shear rates, where both the stress and the torque are smallest - which is where low-viscosity
samples usually hit the limit first.
""")],
        [code(r"""arr = np.loadtxt(datasets.path("xanthan_flow_curve.csv"), delimiter=",", skiprows=3)   # 2 comment lines + header
print(arr.shape, arr[:2])
print("same numbers as pandas:", np.allclose(arr, df.to_numpy()))"""),
         md(r"""
`np.loadtxt` returns a plain array: fast and simple for purely numeric files, but you must count the lines
to skip and remember which column is which. pandas keeps the column names, handles comment lines, mixed
types and missing values - more convenient for real exports (notebook 17).
""")],
        [code(r"""def loglog_interp(x0, x, y):
    return np.exp(np.interp(np.log(x0), np.log(x), np.log(y)))
print(f"viscosity at 50 1/s: {loglog_interp(50.0, rate, eta):.4f} Pa s")"""),
         md(r"""
Interpolating straight lines between the logarithms reproduces the notebook's log-log value, which follows a
power-law curve between neighbouring points instead of a straight line on linear axes.
""")],
    ]


# =====================================================================================================
@solutions("01_what_rheology_measures")
def _():
    return [
        [code(r"""V, D = 2.0, 0.05
newt = 8 * V / D
for n in (1.0, 0.5):
    print(f"n = {n}: wall shear rate {(3*n + 1)/(4*n) * newt:.0f} 1/s")"""),
         md(r"""
Shear thinning raises the wall shear rate by the factor $(3n+1)/(4n)$ - 25 % for $n$ = 0.5 - because the flatter
velocity profile has a steeper gradient at the wall. Evaluating the viscosity at $8V/D$ would therefore use
too low a rate (and, for a shear-thinning fluid, too high a viscosity).
""")],
        [code(r"""lam, V, D, L_die = 0.5, 0.5, 1e-3, 0.02
rate = 8 * V / D
t_res = L_die / V
print(f"shear rate ~ {rate:.0f} 1/s; Weissenberg number {lam*rate:.0f}; residence time {t_res*1e3:.0f} ms; Deborah number {lam/t_res:.1f}")"""),
         md(r"""
Both numbers are far above 1: the melt is sheared much faster than it can relax ($Wi \gg 1$), and it leaves
the die before it has relaxed ($De \gg 1$). Strong elastic effects are expected - large normal stresses and
pronounced die swell (notebook 16).
""")],
        [code(r"""def wall_shear_rate(V, D, n=1.0):
    return (3 * n + 1) / (4 * n) * 8 * V / D
Vs = np.array([0.1, 0.3, 1.0, 3.0])
print(pd.DataFrame({f"n = {n}": wall_shear_rate(Vs, 0.05, n) for n in (1.0, 0.6, 0.3)}, index=[f"{v} m/s" for v in Vs]).round(0))"""),
         md(r"""
The wall shear rate grows linearly with velocity and increases as the fluid becomes more shear-thinning -
by the factor $(3n+1)/(4n)$ = 1.58, i.e. 58 %, at $n$ = 0.3.
""")],
    ]


# =====================================================================================================
@solutions("02_rheometers_and_pitfalls", imports=["from engrheo import datasets, fitting, geometry, models"],
           setup=r"""melt = pd.read_csv(datasets.path("polymer_melt_capillary.csv"), comment="#")
melt["L_over_R"] = melt.die_length_mm / melt.die_radius_mm
Rd = melt.die_radius_mm.iloc[0] * 1e-3""")
def _():
    return [
        [code(r"""tau_w, app = [], []
for Q, grp in melt.groupby("flow_rate_m3_s"):
    tau_w.append(geometry.bagley(grp.L_over_R, grp.pressure_drop_Pa)["tau_w"]); app.append(4 * Q / (np.pi * Rd**3))
tau_w, app = np.array(tau_w), np.array(app)
_, rate_w, _ = geometry.capillary(2 * tau_w, app * np.pi * Rd**3 / 4, R=Rd, L=Rd)
single = melt[melt.L_over_R == 10]
raw_rate = (4 * single.flow_rate_m3_s / (np.pi * Rd**3)).to_numpy()
raw_tau = (single.pressure_drop_Pa * Rd / (2 * 10 * Rd)).to_numpy()
grid = np.logspace(np.log10(rate_w.min()), np.log10(rate_w.max()), 50)
true = models.viscosity("carreau", grid, 3000.0, 0.0, 0.3, 0.3)
for label, r_, t_ in (("raw single die", raw_rate, raw_tau), ("Bagley only", app, tau_w), ("Bagley + Rabinowitsch", rate_w, tau_w)):
    f = fitting.fit_flow_curve(r_, t_, "carreau", p0=[3000, 0.0, 0.3, 0.3])
    err = f.predict_viscosity(grid) / true - 1
    print(f"{label:<22}: eta0 = {f.params['eta0']:5.0f} Pa s, n = {f.params['n']:.3f}, lambda = {f.params['lam']:.3f} s | "
          f"largest viscosity error within the measured range {np.max(np.abs(err)):6.1%}")
print("true: eta0 = 3000 Pa s, n = 0.30, lambda = 0.30 s")"""),
         md(r"""
Judged by $\eta_0$, the raw data seem best - but that is a coincidence. These data start at 10 1/s, above the onset
of shear thinning ($1/\lambda$ ≈ 3 1/s), so $\eta_0$ and $\lambda$ are extrapolations that none of the data sets can determine.
The meaningful test is the viscosity **within the measured range**: there the raw single-die curve is wrong by
up to about 100 % (it counts the entrance pressure loss as die friction), the Bagley-corrected one by about
15 %, and the fully corrected one by about 2 %. Compare models where the data are; capillary rheometers are
poorly suited to measuring zero-shear viscosities.
""")],
        [code(r"""Ri, Ro, nu = 0.0125, 0.0136, 1.0e-6
d = Ro - Ri
Omega = np.sqrt(1700 * nu**2 / (Ri * d**3))
print(f"Taylor vortices above Omega = {Omega:.1f} rad/s ({Omega*60/(2*np.pi):.0f} rpm), i.e. a shear rate of about {Omega*Ri/d:.0f} 1/s")"""),
         md(r"""
For water, secondary flows set in at a shear rate of only about 100 1/s in this cell. Above it, the extra
dissipation of the vortices raises the torque and the "viscosity" appears to increase - an apparent shear
thickening that is purely an artefact. Low-viscosity fluids at high rates need narrower gaps or other
geometries.
""")],
        [code(r"""paste = pd.read_csv(datasets.path("toothpaste_plates_slip.csv"), comment="#")
grp = paste[paste.shear_stress_Pa == paste.shear_stress_Pa.unique()[2]]
slope, intercept = np.polyfit(1 / (grp.gap_mm * 1e-3), grp.apparent_shear_rate_1_s, 1)
lib = geometry.mooney(grp.gap_mm * 1e-3, grp.apparent_shear_rate_1_s, kind="plates")
print(f"stress {grp.shear_stress_Pa.iloc[0]} Pa: by hand true rate {intercept:.4f} 1/s, slip {slope/2*1e3:.4f} mm/s; "
      f"library {lib['true_rate']:.4f} 1/s, {lib['slip_velocity']*1e3:.4f} mm/s")"""),
         md(r"""
For parallel plates the apparent rim rate is the true rate plus $2u_s/h$; a straight line against $1/h$ has
intercept equal to the true rate and slope $2u_s$ - the same calculation as the library's.
""")],
    ]


# =====================================================================================================
@solutions("03_shear_thinning_fluids", imports=["from engrheo import datasets, fitting, models"],
           setup=r"""df = pd.read_csv(datasets.path("xanthan_flow_curve.csv"), comment="#")
rate, tau = df.shear_rate_1_s.to_numpy(), df.shear_stress_Pa.to_numpy()
best = fitting.fit_flow_curve(rate, tau, "carreau")""")
def _():
    return [
        [code(r"""for lo in (10.0, 1.0):
    m = rate >= lo
    pl = fitting.fit_flow_curve(rate[m], tau[m], "power_law")
    print(f"power law fitted from {lo:g} 1/s: n = {pl.params['n']:.3f}; predicted/measured viscosity at 0.01 1/s: "
          f"{pl.predict_viscosity(0.01) / (tau[0]/rate[0]):.1f}")"""),
         md(r"""
Fitting from 1 1/s instead of 10 1/s changes little: the power law still overestimates the viscosity at rest
almost fivefold. No choice of range can make a power law describe a plateau; a model with a zero-shear
viscosity is needed.
""")],
        [code(r"""d, drho, g = 0.3e-3, 50.0, 9.81
for name, eta in (("water", 1.0e-3), ("xanthan drink (at rest)", best.params["eta0"])):
    v = drho * g * d**2 / (18 * eta)
    print(f"{name:<24}: settling velocity {v:.2e} m/s = {v*3.6e6:.3g} mm/h, particle shear rate v/d = {v/d:.1e} 1/s")
print(f"onset of shear thinning 1/lambda = {1/best.params['lam']:.3f} 1/s")"""),
         md(r"""
In water the pulp would sink by about 9 m per hour; in the drink by a fraction of a millimetre per hour -
the shelf-life problem is solved. The shear rate around a settling particle is far below the onset of shear
thinning, so the plateau viscosity is the right one to use - and a power-law model, whose viscosity has no
plateau, would have predicted nonsense here.
""")],
        [code(r"""for f in (best, fitting.fit_flow_curve(rate, tau, "power_law")):
    ssr = np.sum(f.log_residuals**2)
    aic = f.n * np.log(ssr / f.n) + 2 * len(f.params)
    print(f"{f.model:<10}: AIC by hand {aic:.3f}, library {f.aic:.3f}")"""),
         md(r"""
AIC $= n\ln(\text{SSR}/n) + 2k$: the fit term rewards small residuals, the penalty term $2k$ charges for each
parameter. The Carreau model's two extra parameters cost 4 AIC units but reduce the residuals enormously.
""")],
    ]


# =====================================================================================================
@solutions("04_yield_stress_fluids", imports=["from engrheo import datasets, fitting, geometry, models"])
def _():
    return [
        [code(r"""mud = pd.read_csv(datasets.path("drilling_mud_fann.csv"), comment="#")
th = dict(zip(mud.rpm, mud.dial_reading_deg))
lsyp = (2 * th[3] - th[6]) * 0.4788
api = geometry.api_bingham(th[600], th[300])
tau_m, rate_m = geometry.fann35(mud.rpm.to_numpy(), mud.dial_reading_deg.to_numpy())
o = np.argsort(rate_m)
hb = fitting.fit_flow_curve(rate_m[o], tau_m[o], "herschel_bulkley")
for name, v in (("API yield point", api["tau_y"]), ("low-shear yield point 2 theta3 - theta6", lsyp),
                ("Herschel-Bulkley fit", hb.params["tau_y"]), ("true yield stress", 4.8)):
    print(f"{name:<38}: {v:5.2f} Pa")"""),
         md(r"""
The low-shear yield point, built from the two lowest readings, lands close to the Herschel-Bulkley yield
stress and the true value; the API yield point, extrapolated from the two highest readings, is more than
twice too large. That is why drilling practice increasingly reports LSYP or Herschel-Bulkley parameters
for hole cleaning and suspension calculations.
""")],
        [code(r"""choc = pd.read_csv(datasets.path("chocolate_casson.csv"), comment="#")
rc, tc = choc.shear_rate_1_s.to_numpy(), choc.shear_stress_Pa.to_numpy()
for lo in (0.0, 10.0):
    m = rc >= lo
    f = fitting.fit_flow_curve(rc[m], tc[m], "herschel_bulkley")
    print(f"HB fitted to rates >= {lo:4.0f} 1/s ({m.sum()} points): tau_y = {f.params['tau_y']:5.2f} +/- {f.se['tau_y']:.2f} Pa")"""),
         md(r"""
Dropping the low-rate points changes the extrapolated yield stress and makes it much more uncertain: the
further the data are from zero shear rate, the longer the extrapolation. An extrapolated yield stress is
only as good as the lowest shear rates measured.
""")],
        [code(r"""conc = pd.read_csv(datasets.path("concrete_flow_curve.csv"), comment="#")
mu_p, ty = np.polyfit(conc.shear_rate_1_s, conc.shear_stress_Pa, 1)
lib = fitting.fit_flow_curve(conc.shear_rate_1_s, conc.shear_stress_Pa, "bingham")
print(f"straight line (absolute residuals): tau_y = {ty:.1f} Pa, mu_p = {mu_p:.2f} Pa s")
print(f"library (relative residuals):       tau_y = {lib.params['tau_y']:.1f} Pa, mu_p = {lib.params['mu_p']:.2f} Pa s")
print("true: tau_y = 600 Pa, mu_p = 45 Pa s")"""),
         md(r"""
Ordinary least squares weights absolute errors, so the high-rate points (largest stresses) dominate the
slope; relative residuals weight every point equally in percentage terms. Here the stresses vary only by a
factor of about 2, so the two answers are close; for data spanning decades the choice matters much more
(notebook 03).
""")],
    ]


# =====================================================================================================
@solutions("05_temperature_and_tts", imports=["from engrheo import datasets, tts"],
           setup=r"""oil = pd.read_csv(datasets.path("engine_oil_viscosity_temperature.csv"), comment="#")
T, eta = oil.temperature_C.to_numpy() + 273.15, oil.viscosity_Pa_s.to_numpy()""")
def _():
    return [
        [code(r"""hot = T >= 333.15
b, a = np.polyfit(1 / T[hot], np.log(eta[hot]), 1)
pred0 = np.exp(a + b / 273.15)
print(f"Arrhenius from 60-120 degC: Ea = {b*tts.R_GAS/1000:.1f} kJ/mol")
print(f"viscosity at 0 degC: predicted {pred0:.3f} Pa s, measured {eta[0]:.3f} Pa s -> predicted/measured = {pred0/eta[0]:.2f}")"""),
         md(r"""
Extrapolated from the hot range, the Arrhenius law badly underestimates the cold-start viscosity: the
apparent activation energy grows as the oil cools, so the viscosity rises faster than the hot-range line
predicts. Cold-start behaviour must be measured cold - or described with an equation, like Vogel's, that
captures the curvature.
""")],
        [code(r"""C1, C2, T_ref = 8.86, 101.6, 443.15
for T_C in (220, 150):
    aT = 10 ** tts.wlf(T_C + 273.15, C1, C2, T_ref)
    print(f"{T_C} degC: relaxation times (and eta0) change by a factor {aT:.3g} relative to 170 degC; longest mode {20*aT:.3g} s")"""),
         md(r"""
Fifty kelvin hotter, the melt relaxes about a thousand times faster; twenty kelvin colder, about a hundred
times slower. Near the glass transition WLF behaviour is extremely steep: processing temperature is the
strongest lever on flow and relaxation, and a few degrees of temperature non-uniformity in a mould or die
create large differences in orientation and residual stress.
""")],
        [code(r"""def wlf_shift(T, C1, C2, T_ref):
    return -C1 * (T - T_ref) / (C2 + T - T_ref)

def wlf_convert(C1, C2, T_ref, T_new):
    C2n = C2 + T_new - T_ref
    return C1 * C2 / C2n, C2n

C1n, C2n = wlf_convert(8.86, 101.6, 443.15, 473.15)
temps = np.array([423.15, 443.15, 473.15, 493.15])
a_old = wlf_shift(temps, 8.86, 101.6, 443.15) - wlf_shift(473.15, 8.86, 101.6, 443.15)   # re-referenced to 473.15 K
print(f"new constants at 200 degC: C1 = {C1n:.3f}, C2 = {C2n:.1f} K")
print("max difference in log10 aT:", np.max(np.abs(a_old - wlf_shift(temps, C1n, C2n, 473.15))))"""),
         md(r"""
The WLF equation has the same form at every reference temperature; changing the reference only changes the
constants. $C_1C_2$ is invariant, which is why tabulated constants must always be quoted with their $T_{ref}$.
""")],
    ]


# =====================================================================================================
@solutions("06_thixotropy", imports=["from scipy import optimize", "from engrheo import datasets, thixotropy as th"],
           setup=r"""cem = pd.read_csv(datasets.path("cement_paste_step_test.csv"), comment="#")
t, rate, tau = cem.time_s.to_numpy(), cem.shear_rate_1_s.to_numpy(), cem.shear_stress_Pa.to_numpy()
def model(p):
    prm = th.Params(*p)
    return th.simulate(prm, t, rate, lam0=th.equilibrium_structure(prm, rate[0]))["stress"]
use = rate > 0
p_fit = optimize.least_squares(lambda p: np.log(model(p)[use] / tau[use]), [10, 0.5, 1, 0.05, 0.05], bounds=(1e-6, np.inf),
                               x_scale="jac").x
prm = th.Params(*p_fit)""")
def _():
    return [
        [code(r"""D, L = 0.05, 200.0
lam0 = th.equilibrium_structure(prm, 100.0)
for rest in (600, 3600):
    tau_r = prm.tau_y * (1 - (1 - lam0) * np.exp(-prm.k_build * rest))
    print(f"after {rest/60:3.0f} min: restart stress {tau_r:5.1f} Pa -> pressure {4*tau_r*L/D/1e5:5.2f} bar")"""),
         md(r"""
With this paste's build-up rate ($1/k_b$ ≈ 100 s) the structure is essentially rebuilt within ten minutes, so the
restart pressure is almost the same after 10 min and 1 h. Real cement also hydrates, so its yield stress keeps
rising for hours - the structural model covers only the reversible part.
""")],
        [code(r"""p_paint = th.Params(3.0, 0.08, 0.4, 0.02, 0.08)
ramps = np.logspace(-2, 3, 41)
areas = np.array([th.hysteresis_loop(p_paint, 200.0, tr)["area"] for tr in ramps])
best = ramps[np.argmax(areas)]
print(f"largest loop ({areas.max():.0f} Pa/s) for a ramp of {best:.2f} s")
print(f"breakdown time at the top rate 1/(k_break x 200) = {1/(0.08*200):.3f} s; build-up time 1/k_build = {1/0.02:.0f} s")"""),
         md(r"""
The loop is largest when the ramp lasts a few breakdown times: long enough for the structure to break down
during the ramp, too short for it to rebuild on the way down. The build-up time (50 s) is much longer, so for
ramps between seconds and minutes the down curve always lags - but the loop shrinks as the ramps get slower.
""")],
        [code(r"""def euler(dt):
    ts = np.arange(0, t[-1] + dt / 2, dt)
    r = np.interp(ts, t, rate, left=rate[0])
    lam = np.empty_like(ts); lam[0] = th.equilibrium_structure(prm, 10.0)
    for i in range(ts.size - 1):
        lam[i + 1] = lam[i] + dt * (prm.k_build * (1 - lam[i]) - prm.k_break * r[i] * lam[i])
    return ts, lam
exact = th.simulate(prm, t, rate, lam0=th.equilibrium_structure(prm, 10.0))["structure"]
for dt in (0.5, 1.0, 10.0):
    ts, lam = euler(dt)
    err = np.max(np.abs(np.interp(t, ts, lam) - exact))
    print(f"Euler, dt = {dt:4.1f} s: max error {err:.3g}   (dt x rate constant at 100 1/s = {dt*(prm.k_build + prm.k_break*100):.2f})")"""),
         md(r"""
Explicit Euler is stable only while the time step times the rate constant stays below 2. At 100 1/s the
rate constant is about 2 per second, so steps of 1 s sit at the edge and 10 s steps blow up. The structure
equation is **stiff** at high shear rates: the exact interval solution (or an implicit method) has no such
limit - one reason why the library integrates it exactly.
""")],
    ]


# =====================================================================================================
@solutions("07_linear_viscoelasticity", imports=["from scipy import optimize", "from engrheo import datasets",
                                                 "from engrheo import viscoelastic as ve"])
def _():
    return [
        [code(r"""mx, sls = ve.maxwell(1000.0, 1000.0), ve.standard_linear_solid(300.0, 700.0, 1.0)
for name, m in (("Maxwell", mx), ("standard linear solid", sls)):
    print(f"{name:<22}: J(10 s) = {m['creep'](10.0)*1e3:.2f} 1/kPa, J(100 s) = {m['creep'](100.0)*1e3:.2f} 1/kPa")"""),
         md(r"""
The Maxwell compliance keeps growing linearly - it flows - whereas the standard linear solid levels off at
$1/G_e$. A cross-linked rubber has a finite equilibrium modulus, so only the solid model (or a spectrum with
$G_e > 0$) can describe it.
""")],
        [code(r"""dough = pd.read_csv(datasets.path("dough_creep_recovery.csv"), comment="#")
tt, strain = dough.time_s.to_numpy(), dough.strain.to_numpy()
def history(t_, G1, eta1, G2, eta2):
    J = lambda s: np.where(s >= 0, ve.burgers_creep(np.maximum(s, 0), G1, eta1, G2, eta2), 0.0)
    return 50.0 * (J(t_) - np.where(t_ > 120, J(t_ - 120), 0.0))
p, _ = optimize.curve_fit(history, tt, strain, p0=[4e3, 5e5, 1e3, 1e4], bounds=(0, np.inf))
s120, s300 = strain[tt == 120][0], strain[-1]
permanent = 50 * 120 / p[1]
print(f"recovered after 180 s: measured {(s120 - s300)/s120:.1%}, Burgers {(s120 - history(np.array([300.0]), *p)[0])/s120:.1%}, "
      f"long-time limit {(s120 - permanent)/s120:.1%}")"""),
         md(r"""
After three minutes the dough has recovered almost all of the strain it ever will: the long-time limit is
barely higher, because the delayed elasticity (retardation time 25 s) has largely come back. The rest - about
15 % of the strain at 120 s - is permanent viscous flow.
""")],
        [code(r"""rng = np.random.default_rng(7)
mx = ve.maxwell(2000.0, 500.0)
t = np.linspace(0.01, 1.0, 40)
G = mx["relaxation"](t) * np.exp(rng.normal(0, 0.02, t.size))
slope, icpt = np.polyfit(t, np.log(G), 1)
print(f"G = {np.exp(icpt):.0f} Pa (true 2000), tau = {-1/slope:.4f} s (true {mx['tau']:.4f})")"""),
         md(r"""
For a single exponential the logarithm is a straight line, so ordinary linear regression gives both
parameters to about a percent. With several modes the log plot curves and this trick no longer works - the
spectrum methods of notebook 09 take over.
""")],
    ]


# =====================================================================================================
@solutions("08_oscillatory_rheology", imports=["from engrheo import datasets, oscillatory"])
def _():
    return [
        [code(r"""melt = pd.read_csv(datasets.path("polystyrene_frequency_sweeps.csv"), comment="#")
for T_C in (220, 150):
    g = melt[melt.temperature_C == T_C]
    low = g.omega_rad_s < g.omega_rad_s.min() * 10
    s1 = np.polyfit(np.log(g.omega_rad_s[low]), np.log(g.G_storage_Pa[low]), 1)[0]
    s2 = np.polyfit(np.log(g.omega_rad_s[low]), np.log(g.G_loss_Pa[low]), 1)[0]
    print(f"{T_C} degC, lowest decade: slope of G' {s1:.2f}, of G'' {s2:.2f}   (terminal theory 2 and 1)")"""),
         md(r"""
At 220 degC the melt relaxes about a thousand times faster than at 170 degC, so even the unshifted sweep reaches
deep into the terminal zone and shows slopes near 2 and 1. At 150 degC the same frequencies correspond to the
plateau region, where the slopes are small. The terminal slopes appear only when $\omega\tau_{max} \ll 1$.
""")],
        [code(r"""gel = pd.read_csv(datasets.path("food_gel_time_sweep.csv"), comment="#")
piv = gel.assign(tan_delta=gel.G_loss_Pa / gel.G_storage_Pa).pivot(index="time_min", columns="omega_rad_s", values="tan_delta")
all4 = oscillatory.gel_point(piv.index.to_numpy(), piv.to_numpy())
two = oscillatory.gel_point(piv.index.to_numpy(), piv[[1.0, 31.62]].to_numpy())
print(f"gel time: four frequencies {all4:.2f} min, two frequencies {two:.2f} min (true 23.4)")"""),
         md(r"""
Even two frequencies locate the crossing of the $\tan\delta$ curves well, because far-apart frequencies give
curves that cross at a large angle. More frequencies add robustness against noise; in practice three to five
are used.
""")],
        [code(r"""melt = pd.read_csv(datasets.path("polystyrene_frequency_sweeps.csv"), comment="#")
g = melt[melt.temperature_C == 170]
w, Gp, Gpp = g.omega_rad_s.to_numpy(), g.G_storage_Pa.to_numpy(), g.G_loss_Pa.to_numpy()
d = np.log(Gp) - np.log(Gpp)
idx = np.nonzero(np.sign(d[:-1]) != np.sign(d[1:]))[0]
if idx.size:
    i = idx[0]; f = d[i] / (d[i] - d[i + 1])
    print(f"by hand: crossover at {np.exp(np.log(w[i]) + f*np.log(w[i+1]/w[i])):.4g} rad/s; library: {oscillatory.crossover(w, Gp, Gpp)[0]:.4g} rad/s")
else:
    print("no crossover within the 170 degC sweep:", oscillatory.crossover(w, Gp, Gpp))"""),
         md(r"""
The crossover is found from the sign change of $\ln G' - \ln G''$ and a straight-line interpolation in log-log
coordinates - exactly the library's method. If a single sweep does not contain the crossover, the master
curve (which spans more decades) is needed.
""")],
    ]


# =====================================================================================================
@solutions("09_relaxation_spectra", imports=["from engrheo import datasets, tts", "from engrheo import viscoelastic as ve"],
           setup=r"""melt = pd.read_csv(datasets.path("polystyrene_frequency_sweeps.csv"), comment="#")
curves = {T_C + 273.15: (g.omega_rad_s.to_numpy(), g[["G_storage_Pa", "G_loss_Pa"]].to_numpy()) for T_C, g in melt.groupby("temperature_C")}
mc = tts.master_curve(curves, 443.15)
w = np.concatenate([mc["omega_reduced"][T_] for T_ in sorted(curves)])
Y = np.vstack([mc["y"][T_] for T_ in sorted(curves)])
o = np.argsort(w); w, Gp, Gpp = w[o], Y[o, 0], Y[o, 1]
g_true = np.array([2.7e6, 1.08e6, 6.3e5, 5.4e5, 4.95e5, 3.6e5, 1.35e5]); tau_true = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0, 20.0])""")
def _():
    return [
        [code(r"""full = ve.fit_spectrum(w, Gp, Gpp)
m = w >= 1.0
cut = ve.fit_spectrum(w[m], Gp[m], Gpp[m])
for name, f in (("all data", full), ("only omega >= 1 rad/s", cut)):
    print(f"{name:<22}: eta0 = {f.eta0:.3g} Pa s, Je0 = {ve.steady_state_compliance(f.g, f.tau):.3g} 1/Pa")
print(f"true: eta0 = {ve.zero_shear_viscosity(g_true, tau_true):.3g} Pa s, Je0 = {ve.steady_state_compliance(g_true, tau_true):.3g} 1/Pa")"""),
         md(r"""
Without the terminal zone the longest relaxation times are unknown: the zero-shear viscosity comes out about
8 % low and the steady-state compliance - which depends even more strongly on the slowest modes - more than
25 % low, although the fit to the remaining data is excellent. Integral properties are robust only if the data
reach the terminal region.
""")],
        [code(r"""decades = np.log10(1 / w.min()) + 1 - (np.log10(1 / w.max()) - 1)
for per in (1, 2, 5, 10):
    f = ve.fit_spectrum(w, Gp, Gpp, n_modes=int(np.ceil(per * decades)))
    print(f"{per:2d} modes per decade: rms misfit {100*f.rel_rms:.1f} %")"""),
         md(r"""
One mode per decade leaves a visibly rippled fit; the misfit falls steeply up to two modes per decade and
reaches the noise level of the data (2 %) at about five. Beyond that, extra modes only make the (already
ill-posed) spectrum less unique - there is nothing left to fit.
""")],
        [code(r"""f = ve.fit_spectrum(w, Gp, Gpp)
Je0 = np.sum(f.g * f.tau**2) / np.sum(f.g * f.tau) ** 2
print(f"longest fitted relaxation time with g > 0: {f.tau[f.g > 0].max():.0f} s")
t = np.r_[0.0, np.logspace(-6, 6, 2000)]                      # run far beyond the slowest mode
J = ve.creep_from_spectrum(f.g, f.tau, t)
late = t > 1e5
slope, icpt = np.polyfit(t[late], J[late], 1)
print(f"Je0 from the spectrum {Je0:.4g} 1/Pa; creep intercept {icpt:.4g} 1/Pa; 1/eta0: {1/f.eta0:.4g}, creep slope {slope:.4g}")"""),
         md(r"""
At long times the creep compliance is a straight line $J_e^0 + t/\eta_0$: its intercept is the steady-state
compliance and its slope the inverse zero-shear viscosity, both matching the formulas from the spectrum. "Long"
means long compared with the *slowest* mode - here the fitted spectrum contains a mode of about an hour, fitted
to the lowest-frequency data, so the straight line is reached only after days of creep.
""")],
    ]


# =====================================================================================================
@solutions("10_nonlinear_viscoelasticity", imports=["from engrheo import datasets, constitutive as cm"],
           setup=r"""sol = pd.read_csv(datasets.path("polymer_solution_startup.csv"), comment="#")
g, tau = np.array([60.0, 25.0, 8.0]), np.array([0.02, 0.2, 2.0])
gie = cm.modes_from_spectrum(g, tau, "giesekus", alpha=0.25)""")
def _():
    return [
        [code(r"""psi1_linear = 2 * np.sum(g * tau**2)
st = cm.steady_shear(gie, [0.01, 0.1])
d = sol[sol.shear_rate_1_s == 0.1]
print(f"2 sum g tau^2 = {psi1_linear:.3f} Pa s^2")
print(f"model Psi1 at 0.01 and 0.1 1/s: {st['N1'][0]/0.01**2:.3f}, {st['N1'][1]/0.1**2:.3f} Pa s^2")
print(f"data at 0.1 1/s (end of start-up, t = {d.time_s.iloc[-1]:.0f} s): {d.N1_Pa.iloc[-1]/0.1**2:.3f} Pa s^2")"""),
         md(r"""
At low rates the first normal-stress coefficient approaches its linear-viscoelastic limit $2\sum g_i\tau_i^2$, which
is dominated by the slowest mode (it weights $\tau^2$). At 0.1 1/s the data are close to that limit; the small
difference is noise plus the fact that the start-up had not quite reached steady state (the slowest mode needs
several times 2 s).
""")],
        [code(r"""for rate in (10.0, 100.0):
    d = sol[sol.shear_rate_1_s == rate]
    t_max = d.time_s.iloc[int(np.argmax(d.shear_stress_Pa))]
    print(f"{rate:5.0f} 1/s: stress maximum at t = {t_max:.3g} s, i.e. strain {rate * t_max:.1f}")"""),
         md(r"""
The times of the maxima differ by a factor of about six, but the strains - 4.5 and 7 strain units - are similar:
the overshoot is governed mainly by **strain**, marking the point where the microstructure has been deformed
enough to start yielding. (The data are sampled on a coarse time grid, so the strains are approximate.)
""")],
        [code(r"""eta, lam, rate = 100.0, 0.5, 2.0
dt = 1e-3
ts = np.arange(0, 3 + dt / 2, dt)
txy, txx = np.zeros(ts.size), np.zeros(ts.size)
for i in range(ts.size - 1):
    txy[i + 1] = txy[i] + dt / lam * (-txy[i] + eta * rate)
    txx[i + 1] = txx[i] + dt / lam * (-txx[i] + 2 * lam * rate * txy[i])
lib = cm.startup_shear([cm.Mode(eta, lam)], rate, ts)
print(f"max rel. difference: stress {np.max(np.abs(txy[1:]/lib['stress'][1:] - 1)):.1e}, N1 {np.max(np.abs(txx[50:]/lib['N1'][50:] - 1)):.1e}")"""),
         md(r"""
Explicit Euler agrees to about the step size divided by the relaxation time - first-order accuracy. The library
uses an adaptive implicit/explicit solver (LSODA) with tight tolerances, which also copes with stiff multi-mode
problems where Euler would need tiny steps.
""")],
    ]


# =====================================================================================================
@solutions("11_laos", imports=["from engrheo import datasets, laos, oscillatory, constitutive as cm"],
           setup=r"""yo = pd.read_csv(datasets.path("yoghurt_laos.csv"), comment="#")
w_y = 2 * np.pi""")
def _():
    return [
        [code(r"""ucm = cm.modes_from_spectrum([60.0, 25.0, 8.0], [0.02, 0.2, 2.0])
r = cm.laos_shear(ucm, 20.0, 2.0, n_cycles=12)
print(f"UCM at strain amplitude 20: I3/I1 = {laos.harmonics(r['t'][:-1], r['strain'][:-1], r['stress'][:-1], 2.0)['I3_I1']:.2e}")"""),
         md(r"""
Essentially zero. In simple shear the UCM equation for the shear stress, $\lambda\dot\tau_{xy} = -\tau_{xy} + \eta\dot\gamma + \lambda\dot\gamma\tau_{yy}$,
contains the normal stress $\tau_{yy}$ - which is always zero for UCM. The shear stress therefore obeys a *linear*
equation at any amplitude; the nonlinearity appears only in the normal stresses. Shear-stress harmonics
need a model such as Giesekus or PTT.
""")],
        [code(r"""rows = []
for g0, grp in yo.groupby("strain_amplitude"):
    c = laos.chebyshev(grp.time_s, grp.strain, grp.shear_stress_Pa, w_y)
    m = oscillatory.moduli_from_waveforms(grp.time_s, grp.strain, grp.shear_stress_Pa, w_y)
    rows.append((g0, c["G_L"], m["Gp"], m["Gpp"]))
t = pd.DataFrame(rows, columns=["strain_amp", "G'_L", "G'_1", "G''_1"])
print(t.to_string(index=False, formatters={"strain_amp": "{:g}".format, "G'_L": "{:.1f}".format,
                                           "G'_1": "{:.1f}".format, "G''_1": "{:.1f}".format}))
lin = t["G'_L"].iloc[0]
below = t[t["G'_L"] < lin / 2]
print(f"G'_L falls below half of its linear value ({lin/2:.0f} Pa) first at strain amplitude {below.strain_amp.iloc[0]:g}")
print(f"first-harmonic G' > G'' at every amplitude tested: {bool(np.all(t['G' + chr(39) + '_1'] > t['G' + chr(39)*2 + '_1']))}")"""),
         md(r"""
At 100 % strain the large-strain modulus has collapsed to about a quarter of its linear value - the gel yields at
the extremes of every cycle - yet the first-harmonic moduli still report $G' > G''$, a "solid", at every amplitude
tested: no flow point is reached. Averaged over the cycle, the stiff behaviour near zero strain masks the yielding
at large strain. The Chebyshev measures resolve that intra-cycle yielding; the first harmonic alone would miss it.
""")],
        [code(r"""g0, grp = list(yo.groupby("strain_amplitude"))[2]
x, y = grp.strain.to_numpy(), grp.shear_stress_Pa.to_numpy()
area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
Gpp1 = oscillatory.moduli_from_waveforms(grp.time_s, x, y, w_y)["Gpp"]
print(f"strain {g0}: loop area {area:.4f} J/m3, pi gamma0^2 G''_1 = {np.pi * g0**2 * Gpp1:.4f} J/m3")"""),
         md(r"""
The area of the elastic Lissajous loop is the energy dissipated per unit volume and cycle. Only the first
harmonic contributes to it (higher odd harmonics integrate to zero over a cycle), so it equals $\pi\gamma_0^2 G''_1$
even in the nonlinear regime - up to the discretisation of the sampled loop.
""")],
    ]


# =====================================================================================================
@solutions("12_polymer_melts", imports=["from scipy import optimize", "from engrheo import datasets, polymers"],
           setup=r"""pe = pd.read_csv(datasets.path("polyethylene_eta0_vs_mw.csv"), comment="#")
M, eta0 = pe.Mw_g_mol.to_numpy(), pe.eta0_Pa_s.to_numpy()
res = optimize.least_squares(lambda p: np.log10(polymers.zero_shear_viscosity(M, 10**p[0], 10**p[1], p[2])) - np.log10(eta0), [4.0, 0.0, 3.0])
Mc, eta_c, a = 10**res.x[0], 10**res.x[1], res.x[2]""")
def _():
    return [
        [code(r"""M_new = 150e3
for da in (0.0, -0.1, 0.1):
    print(f"exponent {a + da:.2f}: eta0(150 kg/mol) = {polymers.zero_shear_viscosity(M_new, Mc, eta_c, a + da):.3g} Pa s")"""),
         md(r"""
An uncertainty of 0.1 in the exponent changes the prediction by a factor $(M/M_c)^{0.1}$ - here almost 1.5 either
way, because 150 kg/mol is far above $M_c$. Power-law extrapolations over large ranges amplify small errors in
the exponent; interpolation within the measured grades is much safer.
""")],
        [code(r"""Ea, T1 = 30e3, 190 + 273.15
T2 = T1 + 20
aT = np.exp(Ea / 8.314 * (1 / T2 - 1 / T1))
print(f"20 K hotter: viscosity x {aT:.3f}; allowed Mw increase for the same viscosity: x {aT ** (-1 / a):.3f}")"""),
         md(r"""
Twenty kelvin lower the viscosity by about a quarter; because $\eta_0 \propto M^{3.4}$, only about 10 % more molar
mass uses that margin up. Temperature and molar mass trade off steeply against each other.
""")],
        [code(r"""m = M > 10e3
slope = np.polyfit(np.log10(M[m]), np.log10(eta0[m]), 1)[0]
print(f"straight line above 10 kg/mol: exponent {slope:.2f};  piecewise fit: {a:.2f}")"""),
         md(r"""
Both agree within the scatter. The piecewise fit uses all the data and also locates $M_c$; the straight line needs
you to choose the range - include points too close to $M_c$ and the slope is pulled down.
""")],
    ]


# =====================================================================================================
@solutions("13_suspensions_and_emulsions", imports=["from engrheo import datasets, suspensions as su"],
           setup=r"""gb = pd.read_csv(datasets.path("glass_beads_suspension.csv"), comment="#")
phi, eta_r = gb.volume_fraction.to_numpy(), gb.relative_viscosity.to_numpy()
pm = su.fit_krieger_dougherty(phi, eta_r)["phi_max"]""")
def _():
    return [
        [code(r"""for target in (10.0, 100.0):
    p_ = pm * (1 - target ** (-1 / (2.5 * pm)))
    print(f"relative viscosity {target:5.0f} at phi = {p_:.3f}  (check: {su.krieger_dougherty(p_, pm):.1f})")"""),
         md(r"""
Inverting Krieger-Dougherty: going from a tenfold to a hundredfold viscosity needs only about 0.1 more solids
(0.48 to 0.58) - the steep approach to maximum packing again.
""")],
        [code(r"""print(f"eta_r at phi = 0.55: phi_max {pm:.3f} -> {su.krieger_dougherty(0.55, pm):.1f};  phi_max 0.70 -> {su.krieger_dougherty(0.55, 0.70):.1f}"
      f"  (factor {su.krieger_dougherty(0.55, pm)/su.krieger_dougherty(0.55, 0.70):.1f})")"""),
         md(r"""
A broader size distribution that raises the maximum packing fraction to 0.70 cuts the viscosity at the same
solids content by a factor of more than two - or allows considerably more solids at the same viscosity.
""")],
        [code(r"""low = phi <= 0.2
table = pd.DataFrame({"phi": phi[low], "measured": eta_r[low], "Einstein": su.einstein(phi[low]),
                      "Batchelor": su.batchelor(phi[low]), "Krieger-Dougherty": su.krieger_dougherty(phi[low], pm)})
print(table.round(3).to_string(index=False))"""),
         md(r"""
All three agree in the dilute limit, within the 3 % scatter of the data (at $\phi$ = 0.1 Einstein even happens to lie
closest). From $\phi$ = 0.15 Einstein falls clearly behind (it ignores particle interactions), Batchelor holds a little
further, and only Krieger-Dougherty follows the data up to high concentrations (notebook 13).
""")],
    ]


# =====================================================================================================
@solutions("14_extensional_rheology", imports=["from engrheo import extensional as ex"])
def _():
    return [
        [code(r"""D0, sigma, eta = 3e-3, 0.063, 1.0
print(f"glycerol filament: break-up after {D0 * eta / (ex.C_NEWTONIAN * sigma):.3f} s (linear thinning)")"""),
         md(r"""
The viscous glycerol filament thins steadily and breaks after about a third of a second. The dilute polymer
solution, although its shear viscosity is thousands of times lower, survives for a comparable time: its
strain-hardening extensional viscosity, not its shear viscosity, controls the thinning.
""")],
        [code(r"""rho, R, sigma, lam_E = 1000.0, 20e-6, 0.062, 12e-3
t_cap = np.sqrt(rho * R**3 / sigma)
print(f"capillary time {t_cap*1e6:.1f} us; Deborah number lambda_E / t_cap = {lam_E / t_cap:.0f}")"""),
         md(r"""
The relaxation time is about a thousand times longer than the capillary time, so the polymer controls the
break-up: instead of pinching off quickly, the ink forms a long thread (often with beads along it) that
delays detachment. This suppresses the fast Rayleigh-Plateau break-up into satellite drops but can create long
tails; ink formulators tune polymer content to balance the two.
""")],
        [code(r"""t = np.linspace(0, 0.2, 41)
D = ex.newtonian_thinning(t, 1e-3, 0.03, 1.2)
dDdt = np.gradient(D, t)
eta_E = (2 * 0.7127 - 1) * 0.03 / -dDdt
print(f"apparent Trouton ratio by hand: {np.mean(eta_E) / 1.2:.4f}")"""),
         md(r"""
For Newtonian thinning the slope is constant, $-0.1418\sigma/\eta$, and the formula returns exactly $3\eta$: the
prefactor $(2X - 1)$ was chosen to make that so.
""")],
    ]


# =====================================================================================================
@solutions("15_pipe_flow_of_non_newtonian_fluids", imports=["from engrheo import datasets, fitting, flows"],
           setup=r"""conc = pd.read_csv(datasets.path("concrete_flow_curve.csv"), comment="#")
b = fitting.fit_flow_curve(conc.shear_rate_1_s, conc.shear_stress_Pa, "bingham").params""")
def _():
    return [
        [code(r"""Q, L = 30 / 3600, 100.0
for D in (0.100, 0.125, 0.150):
    s = flows.pipe_hb(b["tau_y"], b["mu_p"], 1.0, D / 2, Q=Q)
    total, yield_part = s["dp_per_L"] * L, 4 * b["tau_y"] * L / D
    print(f"D = {D*1000:.0f} mm: {total/1e5:6.1f} bar, of which yield stress {yield_part/1e5:5.1f} bar, viscous {(total - yield_part)/1e5:6.1f} bar")"""),
         md(r"""
The viscous part falls steeply with diameter (for a fixed flow rate roughly as $1/D^4$), the yield-stress part only
as $1/D$. Larger lines cut pumping pressure dramatically - at the cost of more concrete in the line and heavier
pipework.
""")],
        [code(r"""from engrheo import geometry
mud = pd.read_csv(datasets.path("drilling_mud_fann.csv"), comment="#")
tau_m, rate_m = geometry.fann35(mud.rpm.to_numpy(), mud.dial_reading_deg.to_numpy())
fast = rate_m >= 170; o = np.argsort(rate_m[fast])
pl = fitting.fit_flow_curve(rate_m[fast][o], tau_m[fast][o], "power_law").params
r = [flows.pressure_drop_power_law(q * 6.309e-5, 0.1086, 1000.0, 1200.0, pl["K"], pl["n"]) for q in (250, 500)]
print(f"regimes: {r[0]['regime']}, {r[1]['regime']}; pressure ratio {r[1]['dp']/r[0]['dp']:.2f}; laminar power-law scaling 2^n = {2**pl['n']:.2f}")"""),
         md(r"""
Between 250 gal/min (laminar) and 500 gal/min (turbulent) the flow changes regime, and the pressure loss rises
2.8-fold instead of the 1.4-fold that the laminar power-law scaling $Q^n$ would suggest. In turbulent flow the loss
grows roughly as $Q^{1.75}$-$Q^2$. Pump sizing must check the regime and use the matching friction law.
""")],
        [code(r"""R, Q, K, n, rho = 0.025, 1.5e-3, 0.5, 0.4, 1000.0
V, D = Q / (np.pi * R**2), 2 * R
Re = rho * V ** (2 - n) * D**n / (8 ** (n - 1) * K * ((3 * n + 1) / (4 * n)) ** n)
dp_f = 4 * (16 / Re) / D * rho * V**2 / 2
print(f"Re_MR by hand {Re:.3f} (library {flows.reynolds_metzner_reed(rho, V, D, K, n):.3f}); "
      f"f = 16/Re gives {dp_f:.3f} Pa/m, exact laminar solution {flows.pipe_power_law(K, n, R, Q=Q)['dp_per_L']:.3f} Pa/m")"""),
         md(r"""
The Metzner-Reed Reynolds number is *defined* so that the laminar friction factor keeps its Newtonian form
16/Re for power-law fluids - which is why the two pressure gradients agree exactly.
""")],
    ]


# =====================================================================================================
@solutions("16_mixing_and_processing", imports=["from engrheo import flows, models"])
def _():
    return [
        [code(r"""car = (35.0, 0.002, 12.0, 0.25)
N = np.array([0.5, 2.0])
P = 70.0 * models.viscosity("carreau", 11.0 * N, *car) * N**2 * 0.1**3
exp_ = np.log(P[1] / P[0]) / np.log(N[1] / N[0])
n_loc = 1 + np.log(models.viscosity("carreau", 22.0, *car) / models.viscosity("carreau", 5.5, *car)) / np.log(4)
print(f"power ~ N^{exp_:.2f} (Newtonian laminar: N^2); local power-law index n = {n_loc:.2f} -> theory N^(n+1) = N^{n_loc + 1:.2f}")"""),
         md(r"""
In laminar mixing $P \propto \eta_{app}N^2$ with $\eta_{app} \propto N^{n-1}$, so $P \propto N^{n+1}$ - for this strongly shear-thinning drink
the power rises little more than linearly with speed, instead of quadratically.
""")],
        [code(r"""K_loc, n_loc, W, L_die, Q = 3000 * 0.3 ** (0.3 - 1), 0.3, 1.0, 0.05, 100 / 3600 / 750
dp1 = flows.slit_power_law(K_loc, n_loc, W, 1e-3, Q=Q)["dp_per_L"] * L_die
dp2 = flows.slit_power_law(K_loc, n_loc, W, 2e-3, Q=Q)["dp_per_L"] * L_die
print(f"gap 1 mm: {dp1/1e5:.1f} bar, gap 2 mm: {dp2/1e5:.1f} bar -> factor {dp1/dp2:.2f} (power law 2^(2n+1) = {2**(2*n_loc+1):.2f}; Newtonian 8)")"""),
         md(r"""
For a power-law fluid the die pressure scales as $H^{-(2n+1)}$ at fixed throughput: doubling the gap of this melt
cuts the pressure only about threefold, not eightfold as for a Newtonian liquid. (The local $K$ and $n$ used here
are those of the Carreau melt at high rates, where it behaves as a power law.)
""")],
        [code(r"""def tanner(ratio):
    return 0.1 + (1 + 0.5 * (ratio / 2) ** 2) ** (1 / 6)
ratios = np.linspace(0, 10, 101)
sw = tanner(ratios)
print(f"swell 1.5 is reached at N1/tau = {ratios[np.argmax(sw >= 1.5)]:.1f}; check library: {flows.tanner_swell(8.0, 1.0):.3f} vs {tanner(8.0):.3f}")"""),
         md(r"""
Because of the 1/6 power, swell grows slowly: the normal stresses must be many times the shear stress before an
extrudate swells by half - which strongly elastic melts reach at high extrusion rates.
""")],
    ]


# =====================================================================================================
@solutions("17_rheometer_file_to_report", imports=["import io", "from engrheo import datasets, fitting"],
           setup=r"""lines = open(datasets.path("ketchup_rheometer_export.csv"), encoding="utf-8").read().splitlines()

def parse_export(lines):
    tests, i = {}, 0
    while i < len(lines):
        if lines[i].startswith("Test;"):
            name = lines[i].split(";")[1]; i += 1
            while not lines[i].startswith("Point;"):
                i += 1
            cols = [f"{h} ({u})" if u else h for h, u in zip(lines[i].split(";"), lines[i + 1].split(";"))]
            j = i + 2
            while j < len(lines) and lines[j].strip():
                j += 1
            tests[name] = pd.read_csv(io.StringIO("\n".join(lines[i + 2:j])), sep=";", decimal=",", header=None,
                                      names=cols, na_values=["---"])
            i = j
        else:
            i += 1
    return tests

tests = parse_export(lines)
up = tests["Flow curve (up)"].sort_values("Shear rate (1/s)")""")
def _():
    return [
        [code(r"""ok = up["Status"] == "ok"
for label, d in (("valid points only", up[ok]), ("including 'steady state not reached'", up)):
    f = fitting.fit_flow_curve(d["Shear rate (1/s)"], d["Shear stress (Pa)"], "herschel_bulkley")
    lo, hi = f.conf_int()["tau_y"]
    print(f"{label:<38}: tau_y = {f.params['tau_y']:5.2f} Pa (95 % CI {lo:.2f}-{hi:.2f}), scatter {100*f.sigma:.1f} %")"""),
         md(r"""
The two unreliable points pull the fitted yield stress up and inflate the scatter - and they sit exactly where the
yield stress is determined, at the lowest rates. Excluding them is justified because the instrument itself
reported that the stress had not settled; reporting the exclusion lets a reader judge that decision, and
repeat the analysis either way.
""")],
        [code(r"""print("true yield stress 15 Pa; estimates: HB up about 15 Pa, HB down about 13.5 Pa, flow point about 5 Pa (notebook 17)")"""),
         md(r"""
**Flowing from a squeezed bottle** is about the stress needed to make an undisturbed, rested product flow: the
upward-curve Herschel-Bulkley yield stress (≈ 15 Pa, matching the true value) is the relevant one. **Not running off a
plate** concerns a product that has just been sheared by squeezing and must hold its shape under gravity: the
lower, post-shear value from the downward curve is the conservative choice. The flow-point stress (≈ 5 Pa)
describes where the gel network first gives way in oscillation - it is lower because yielding in oscillation
starts at small strains, long before steady flow; it is the most sensitive indicator of structural change
between batches, not a design value for flow.
""")],
        [code(r"""def load_export(path):
    raw = open(path, encoding="utf-8").read().splitlines()
    out, excluded = {}, []
    for name, d in parse_export(raw).items():
        d = d.copy()
        if "Viscosity (mPa·s)" in d:
            d["Viscosity (Pa·s)"] = d.pop("Viscosity (mPa·s)") / 1000
        if "Strain (%)" in d:
            d["Strain (-)"] = d.pop("Strain (%)") / 100
        bad = (d["Status"] != "ok") | d.isna().any(axis=1)
        excluded += [f"{name}, point {int(p)}: {s}" for p, s in zip(d.loc[bad, "Point"], d.loc[bad, "Status"])]
        out[name] = d[~bad].reset_index(drop=True)
    return out, excluded

data, excluded = load_export(datasets.path("ketchup_rheometer_export.csv"))
print({k: len(v) for k, v in data.items()}); print("\n".join(excluded))"""),
         md(r"""
One function now turns any export of this format into clean SI tables plus an explicit list of excluded points -
the basis for a report that can be regenerated for every new batch.
""")],
    ]


if __name__ == "__main__":
    write_all(only=[a for a in sys.argv[1:] if not a.startswith("--")])
