import numpy as np
import pytest

from engrheo import oscillatory, tts
from engrheo import viscoelastic as ve


# ---------------------------------------------------------------- viscoelastic
def test_spectrum_input_errors():
    with pytest.raises(ValueError):
        ve.moduli([1.0, 2.0], [1.0], 1.0)
    with pytest.raises(ValueError):
        ve.relaxation_modulus([1.0], [-1.0], 0.5)
    with pytest.raises(ValueError):
        ve.creep_from_relaxation(lambda s: np.exp(-s), [0.1, 0.2, 0.3])        # must start at 0


def test_models_limits():
    kv = ve.kelvin_voigt(1e3, 10.0)
    assert kv["creep"](1e6) == pytest.approx(1e-3)                              # long-time compliance 1/G
    Gp, Gpp = kv["moduli"]([1.0, 2.0])
    assert np.allclose(Gp, 1e3) and np.allclose(Gpp, [10.0, 20.0])
    assert ve.burgers_creep(0.0, 1e3, 1e4, 5e2, 1e3) == pytest.approx(1e-3)     # instantaneous 1/G1
    t = np.array([100.0, 200.0])
    slope = np.diff(ve.burgers_creep(t, 1e3, 1e4, 5e2, 1e3)) / np.diff(t)
    assert slope[0] == pytest.approx(1e-4, rel=1e-6)                           # long-time flow 1/eta1
    sls = ve.standard_linear_solid(2e3, 8e3, 0.05)
    assert sls["relaxation"](0.0) == pytest.approx(1e4) and sls["creep"](1e6) == pytest.approx(1 / 2e3)


def test_boltzmann_step_strain():
    mx = ve.maxwell(1e4, 500.0)
    t = np.linspace(0, 0.2, 50)
    s = ve.boltzmann_stress(mx["relaxation"], t, np.full(t.size, 0.1))
    assert np.allclose(s, 0.1 * mx["relaxation"](t))
    with pytest.raises(ValueError):
        ve.boltzmann_stress(mx["relaxation"], t[::-1], t)


def test_fit_spectrum_solid_and_methods():
    g, tau, Ge = np.array([3e3, 1e3]), np.array([0.01, 1.0]), 500.0
    w = np.logspace(-2, 3, 26)
    Gp, Gpp = ve.moduli(g, tau, w, Ge)
    fit = ve.fit_spectrum(w, Gp, Gpp, tau=tau, solid=True)
    assert fit.Ge == pytest.approx(Ge, rel=1e-8) and np.allclose(fit.g, g, rtol=1e-8)
    assert np.allclose(fit.moduli(w)[0], Gp) and fit.relaxation(1e9) == pytest.approx(Ge)
    with pytest.raises(ValueError):
        ve.fit_spectrum(w, Gp[:-1], Gpp)


# ---------------------------------------------------------------- oscillatory
def test_complex_and_crossover():
    d = oscillatory.complex_quantities([1.0, 2.0], [3.0, 4.0], [4.0, 3.0])
    assert np.allclose(d["G_star"], 5.0) and np.allclose(d["eta_star"], [5.0, 2.5])
    w = np.logspace(0, 2, 10)
    assert oscillatory.crossover(w, 10 * w, w) is None                          # G' > G'' everywhere
    assert oscillatory.lve_limit(np.logspace(-3, -1, 10), np.full(10, 100.0)) is None


def test_waveform_with_offset_and_phase():
    w = 2.0
    t = np.linspace(0, 3 * 2 * np.pi / w, 600, endpoint=False)
    strain = 0.05 * np.sin(w * t + 0.7) + 0.01                                  # phase offset and drift-free offset
    stress = 0.05 * (200 * np.sin(w * t + 0.7) + 80 * np.cos(w * t + 0.7)) + 3.0
    r = oscillatory.moduli_from_waveforms(t, strain, stress, w)
    assert r["Gp"] == pytest.approx(200, rel=1e-9) and r["Gpp"] == pytest.approx(80, rel=1e-9)


def test_gel_point_and_cox_merz_errors():
    with pytest.raises(ValueError):
        oscillatory.gel_point([1, 2, 3], np.ones((3, 1)))
    ratio = oscillatory.cox_merz_ratio([1.0, 10.0, 1e4], [5.0, 2.0, 1.0], [0.1, 100.0], [5.0, 5.0 * 10**-1.5])
    assert np.isnan(ratio[2]) and ratio[0] == pytest.approx(5.0 / (5.0 * 10 ** (-1.5 / 3)), rel=1e-9)


# ---------------------------------------------------------------- tts
def test_tts_interfaces():
    assert tts.arrhenius(300.0, 50e3, 320.0) > 0                               # colder -> slower -> aT > 1
    with pytest.raises(ValueError):
        tts.master_curve({300.0: ([1, 2, 3], [1, 2, 3])}, 310.0)
    w = np.logspace(-1, 1, 12)
    y = 1 / (1 + w)
    assert tts.shift_factor(w, y, w / 10**0.5, y) == pytest.approx(0.5, abs=1e-6)


def test_creep_from_spectrum_long_time_and_solid():
    g, tau = np.array([1e5, 3e4, 1e4]), np.array([1e-3, 0.1, 5.0])
    t = np.r_[0.0, np.logspace(-6, 3, 600)]
    J = ve.creep_from_spectrum(g, tau, t)
    assert J[0] == pytest.approx(1 / g.sum())
    assert J[-1] == pytest.approx(ve.steady_state_compliance(g, tau) + t[-1] / ve.zero_shear_viscosity(g, tau), rel=1e-5)
    Js = ve.creep_from_spectrum([1e4], [0.1], t, Ge=2e3)                     # a solid: bounded creep 1/Ge
    assert Js[-1] == pytest.approx(1 / 2e3, rel=1e-6)
