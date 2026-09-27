import numpy as np
import pytest

from engrheo import datasets, fitting, flows, geometry, models


# ---------------------------------------------------------------- models
def test_models_interface():
    assert models.parameter_names("herschel_bulkley") == ("tau_y", "K", "n")
    assert models.stress("power_law", 2.0, 3.0, 0.5) == pytest.approx(3.0 * 2.0**0.5)
    assert models.stress("power_law", 2.0, K=3.0, n=0.5) == pytest.approx(3.0 * 2.0**0.5)
    with pytest.raises(KeyError):
        models.stress("unknown", 1.0, 1.0)
    with pytest.raises(TypeError):
        models.stress("power_law", 1.0, 1.0)                  # missing parameter
    with pytest.raises(TypeError):
        models.stress("power_law", 1.0, 1.0, n=0.5)            # mixed position and name
    with pytest.raises(TypeError):
        models.stress("power_law", 1.0, K=1.0)
    with pytest.raises(ValueError):
        models.stress("power_law", [1.0, -1.0], 1.0, 0.5)


def test_yield_stress_models_approach_tau_y():
    for name, p in (("bingham", (4.0, 0.1)), ("herschel_bulkley", (4.0, 1.0, 0.5)), ("casson", (4.0, 0.1))):
        assert models.stress(name, 1e-12, *p) == pytest.approx(4.0, rel=1e-4)


# ---------------------------------------------------------------- fitting
def test_fit_accepts_viscosity_and_reports():
    rate = np.logspace(-1, 2, 12)
    eta = models.viscosity("power_law", rate, 2.0, 0.5)
    fit = fitting.fit_flow_curve(rate, viscosity=eta, model="power_law")
    assert fit.params["n"] == pytest.approx(0.5, rel=1e-8)
    assert np.allclose(fit.predict_viscosity(rate), eta)
    assert "power_law fit" in str(fit) and set(fit.conf_int()) == {"K", "n"}


def test_fit_input_errors():
    rate = np.logspace(-1, 2, 12)
    tau = models.stress("power_law", rate, 2.0, 0.5)
    with pytest.raises(ValueError):
        fitting.fit_flow_curve(rate, tau, "power_law", viscosity=tau / rate)
    with pytest.raises(ValueError):
        fitting.fit_flow_curve(rate[:5], tau, "power_law")
    with pytest.raises(ValueError):
        fitting.fit_flow_curve(rate[:3], tau[:3], "carreau_yasuda")     # too few points
    with pytest.raises(KeyError):
        fitting.fit_flow_curve(rate, tau, "magic")


def test_compare_models_prefers_true_model():
    rng = np.random.default_rng(1)
    rate = np.logspace(-2, 3, 40)
    tau = models.stress("carreau", rate, 8.0, 0.01, 2.0, 0.35) * np.exp(rng.normal(0, 0.02, rate.size))
    fits, text = fitting.compare_models(rate, tau, ("power_law", "carreau", "cross"))
    assert fits[0].model in ("carreau", "cross") and fits[-1].model == "power_law"
    assert "delta AIC" in text


def test_at_bound_is_reported():
    rate = np.logspace(-1, 2, 15)
    fit = fitting.fit_flow_curve(rate, models.stress("power_law", rate, 2.0, 0.5), "herschel_bulkley")
    assert "tau_y" in fit.at_bound and "at a bound" in str(fit)


# ---------------------------------------------------------------- geometry
def test_geometry_newtonian_consistency():
    mu, R, h = 0.5, 0.02, 1e-3
    omega = np.logspace(-1, 1, 6)
    M = np.pi * mu * omega * R**4 / (2 * h)                      # exact Newtonian torque, parallel plates
    tau, rate, n = geometry.parallel_plate(M, omega, R, h)
    assert np.allclose(tau / rate, mu) and np.allclose(n, 1)
    tau0, _, _ = geometry.parallel_plate(M, omega, R, h, correct=False)
    assert np.allclose(tau0, tau)                               # Newtonian: correction changes nothing


def test_geometry_errors():
    with pytest.raises(ValueError):
        geometry.capillary([1.0, 2.0], [1.0, 2.0], 1e-3, 0.01)  # need 3 points for the slope
    with pytest.raises(ValueError):
        geometry.parallel_plate([3.0, 2.0, 1.0], [3.0, 2.0, 1.0], 0.02, 1e-3)   # rates not increasing


# ---------------------------------------------------------------- flows
def test_flows_arguments_and_limits():
    with pytest.raises(ValueError):
        flows.pipe_power_law(1.0, 0.5, 0.01)
    with pytest.raises(ValueError):
        flows.pipe_hb(1.0, 1.0, 0.5, 0.01, dp_per_L=1.0, Q=1.0)
    no_flow = flows.pipe_hb(10.0, 1.0, 0.5, 0.01, dp_per_L=100.0)   # tau_w = 0.5 < tau_y
    assert no_flow["Q"] == 0 and np.all(flows.velocity_hb([0, 0.005], 10.0, 1.0, 0.5, 0.01, 100.0) == 0)


def test_friction_regimes():
    assert flows.fanning_power_law(1000, 0.5) == pytest.approx(16 / 1000)
    turb = flows.fanning_power_law(1e5, 0.5)
    assert turb < 16 / 1e5 * 50 and turb > 0
    r = flows.pressure_drop_power_law(Q=0.01, D=0.05, L=10, rho=1000, K=0.1, n=0.6)
    assert r["regime"] in ("laminar", "turbulent") and r["dp"] > 0
    assert flows.fanning_bingham_laminar(500, 0) == pytest.approx(16 / 500)


def test_datasets_interface():
    with pytest.raises(KeyError):
        datasets.path("nope.csv")


# ---------------------------------------------------------------- inverse models, Couette simulation, Fann, data
def test_rate_from_stress_inverts_every_model():
    params = {"newtonian": (0.8,), "power_law": (3.0, 0.4), "cross": (12.0, 0.02, 3.0, 0.8), "carreau": (8.0, 0.01, 2.0, 0.35),
              "carreau_yasuda": (8.0, 0.01, 2.0, 1.2, 0.35), "sisko": (0.02, 5.0, 0.3), "bingham": (4.0, 0.05),
              "herschel_bulkley": (6.0, 1.5, 0.5), "casson": (3.0, 0.02)}
    rate = np.logspace(-2, 3, 11)
    for name, p in params.items():
        back = models.rate_from_stress(name, models.stress(name, rate, *p), *p)
        assert np.allclose(back, rate, rtol=1e-8), name
    assert models.rate_from_stress("bingham", 2.0, 4.0, 0.05) == 0.0            # below the yield stress
    with pytest.raises(TypeError):
        models.rate_from_stress("power_law", 1.0, 3.0)


def test_couette_speed_scalar_and_array():
    w1 = geometry.couette_speed("power_law", (2.0, 0.5), 1e-3, 0.0125, 0.0136, 0.0375)
    w2 = geometry.couette_speed("power_law", (2.0, 0.5), [1e-3, 2e-3], 0.0125, 0.0136, 0.0375)
    assert isinstance(w1, float) and w2.shape == (2,) and w2[1] > w2[0] and w2[0] == pytest.approx(w1)
    assert geometry.couette_speed("bingham", (5.0, 0.03), 1e-6, 0.017245, 0.018415, 0.038) == 0.0   # no flow


def test_fann_and_api_formulas():
    tau, rate = geometry.fann35([600, 300], [60.0, 40.0])
    assert rate[0] == pytest.approx(1021.4, rel=1e-4) and tau[0] == pytest.approx(60 * 0.5113, rel=1e-12)
    api = geometry.api_bingham(60.0, 40.0)
    assert api["PV_mPas"] == 20.0 and api["YP_lbf100ft2"] == 20.0
    pl = geometry.api_power_law(60.0, 40.0)
    assert 0 < pl["n"] < 1 and pl["K"] > 0


def test_mooney_kinds():
    with pytest.raises(ValueError):
        geometry.mooney([1e-3, 2e-3], [10.0, 9.0], kind="cone")
    cap = geometry.mooney([1e-3, 2e-3], [10.0 + 4e-3 / 1e-3, 10.0 + 4e-3 / 2e-3])
    assert cap["slip_velocity"] == pytest.approx(1e-3) and cap["true_rate"] == pytest.approx(10.0)


def test_bundled_data_readable():
    import pandas as pd

    for name in datasets.available():
        if name == "ketchup_rheometer_export.csv":          # deliberately NOT a plain CSV (notebook 17 parses it)
            with open(datasets.path(name), encoding="utf-8") as fh:
                assert sum(line.startswith("Test;") for line in fh) == 4
            continue
        df = pd.read_csv(datasets.path(name), comment="#")
        assert len(df) >= 6 and not df.isna().any().any(), name
        assert datasets.info(name)
