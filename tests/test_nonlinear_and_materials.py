import numpy as np
import pytest

from engrheo import constitutive as cm
from engrheo import extensional as ex
from engrheo import flows, laos, polymers, thixotropy
from engrheo import suspensions as su


def test_mode_validation_and_spectrum_modes():
    with pytest.raises(ValueError):
        cm.Mode(1.0, 1.0, "maxwell")
    with pytest.raises(ValueError):
        cm.Mode(-1.0, 1.0)
    with pytest.raises(ValueError):
        cm.Mode(1.0, 1.0, "giesekus", alpha=0.7)
    modes = cm.modes_from_spectrum([100.0, 0.0, 50.0], [0.1, 1.0, 2.0], "giesekus", alpha=0.2)
    assert len(modes) == 2 and modes[1].eta_p == pytest.approx(100.0) and modes[0].alpha == 0.2


def test_giesekus_shear_thinning_and_negative_N2():
    m = [cm.Mode(100.0, 1.0, "giesekus", alpha=0.3)]
    r = cm.steady_shear(m, [0.1, 10.0])
    assert r["eta"][1] < r["eta"][0] and r["N2"][1] < 0 < r["N1"][1]


def test_ptt_extension_bounded_and_overshoot():
    m = [cm.Mode(100.0, 1.0, "ptt", eps=0.1)]
    e = cm.startup_extension(m, 2.0, np.array([0.0, 50.0]))          # Wi = 2 > 0.5: UCM would diverge
    assert np.isfinite(e["eta_E_plus"][-1])
    su_ = cm.startup_shear([cm.Mode(100.0, 1.0, "giesekus", alpha=0.3)], 10.0, np.linspace(0, 10, 400))
    assert su_["stress"].max() > su_["stress"][-1] * 1.05              # stress overshoot at high Wi


def test_laos_outputs():
    r = cm.laos_shear([cm.Mode(100.0, 0.1)], 0.5, 5.0, n_cycles=10, points_per_cycle=128)
    assert r["t"].size == 129 and r["strain"].max() == pytest.approx(0.5, rel=1e-3)
    c = laos.chebyshev(r["t"][:-1], r["strain"][:-1], r["stress"][:-1], 5.0)
    assert abs(c["S"]) < 1e-6 and abs(c["T"]) < 1e-6                    # UCM shear stress is linear in strain


def test_thixotropy_interfaces():
    p = thixotropy.Params(10.0, 0.05, 0.5, 0.02, 0.05)
    with pytest.raises(ValueError):
        thixotropy.simulate(p, [0, 1, 2], [1.0, 1.0])
    res = thixotropy.simulate(p, np.linspace(0, 1000, 11), np.full(11, 2.0))
    assert res["structure"][-1] == pytest.approx(thixotropy.equilibrium_structure(p, 2.0), rel=1e-6)
    loop = thixotropy.hysteresis_loop(p, 50.0, 60.0)
    assert loop["area"] > 0 and loop["stress_up"].size == loop["stress_down"].size
    assert thixotropy.equilibrium_flow_curve(p, [0.0])[0] == pytest.approx(10.0)


def test_suspension_and_extension_edges():
    with pytest.raises(ValueError):
        su.krieger_dougherty(0.7, 0.64)
    with pytest.raises(ValueError):
        su.quemada([0.1, 0.7], 0.64)
    free = su.fit_krieger_dougherty([0.1, 0.2, 0.3, 0.4], su.krieger_dougherty(np.array([0.1, 0.2, 0.3, 0.4]), 0.6, 3.0),
                                    fix_intrinsic=None)
    assert free["phi_max"] == pytest.approx(0.6, rel=1e-5) and free["intrinsic"] == pytest.approx(3.0, rel=1e-5)
    t = np.linspace(0, 0.5, 51)
    eta = ex.apparent_extensional_viscosity(t, ex.newtonian_thinning(t, 1e-3, 0.03, 1.2), 0.03)
    assert np.isnan(eta[-1]) and np.isfinite(eta[0])                     # after break-up: NaN, not infinity


def test_polymers_and_processing():
    assert polymers.zero_shear_viscosity(1.5e4, 3e4, 1e3) == pytest.approx(500.0)          # Rouse: linear in M
    assert polymers.zero_shear_viscosity(6e4, 3e4, 1e3) == pytest.approx(1e3 * 2**3.4)
    with pytest.raises(ValueError):
        flows.slit_power_law(1.0, 0.5, 0.1, 1e-3)
    mix = flows.mixing_power_laminar(N=2.0, D=0.2, K=10.0, n=0.4, ks=11.0, Kp=70.0)
    assert mix["rate_avg"] == pytest.approx(22.0) and mix["power"] == pytest.approx(70 * 10 * 22**-0.6 * 4 * 0.008)
    assert flows.tanner_swell(2e4, 1e4) > 1.1
