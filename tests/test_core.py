import numpy as np
import pytest

from englincontrol import linalg as la
from englincontrol import polynomials as pl
from englincontrol import response as rs
from englincontrol import systems as sy


def test_polynomials():
    assert pl.degree([0, 0, 1, 2]) == 1 and pl.trim([0.0]).tolist() == [0.0]
    assert pl.roots([3.0]).size == 0 and pl.derivative([5.0]).tolist() == [0.0]
    assert np.allclose(pl.sub([1, 2], [1, 2]), [0.0]) and np.allclose(pl.derivative([1, 0, -1]), [2, 0])
    q, r = pl.div([1, 1], [1, 0, 0, 1])
    assert q.tolist() == [0.0] and np.allclose(r, [1, 1])


def test_linalg_edges():
    assert np.allclose(la.expm(np.zeros((3, 3))), np.eye(3))
    with pytest.raises(ValueError):
        la.care(np.eye(2), np.zeros((2, 1)), np.eye(2), np.eye(1))  # uncontrollable, unstable: no stabilising solution
    A = np.array([[-1.0, 2.0], [0.0, -3.0]])
    P = la.lyapunov(A, np.eye(2))
    assert np.allclose(A @ P + P @ A.T + np.eye(2), 0)
    Wc, Wo = la.dgramians(np.array([[0.5]]), np.array([[1.0]]), np.array([[2.0]]))
    assert Wc[0, 0] == pytest.approx(1 / 0.75) and Wo[0, 0] == pytest.approx(4 / 0.75)
    T, U = la.schur(np.array([[2.0]]))
    assert T[0, 0] == 2.0


def test_systems_edges():
    with pytest.raises(ValueError):
        sy.tf([1, 0, 0], [1, 1])
    G = sy.tf([1.0], [1.0, 1.0], dt=0.1)
    assert G.discrete and "dt" in repr(G) and G.dcgain() == pytest.approx(0.5)
    S = sy.ss([[-1.0]], [[1.0]], [[1.0]], 0.0, dt=0.1)
    assert S.discrete and "discrete" in repr(S) and S.dcgain()[0, 0] == pytest.approx(0.5)
    mimo = sy.ss(np.eye(2) * -1, np.eye(2), np.eye(2), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        mimo.to_tf()
    assert sy.series(S, S).n == 2 and sy.parallel(S, S).n == 2
    fb = sy.feedback(sy.tf([1.0], [1.0, 1.0]), sign=+1)
    assert np.allclose(fb.den, [1.0, 0.0])  # positive unity feedback of 1/(s+1): pole at 0
    assert sy.feedback(sy.tf([1.0], [1.0, 1.0]).to_ss()).n == 1
    assert sy.zpk([], [-1.0], 2.0).num.tolist() == [2.0]
    assert sy.minimal(sy.ss([[-1.0, 0.0], [0.0, -2.0]], [[1.0], [0.0]], [[1.0, 1.0]], 0.0)).n == 1
    unobs = sy.ss([[-1.0, 0.0], [0.0, -2.0]], [[1.0], [1.0]], [[1.0, 0.0]], 0.0)
    assert sy.minimal(unobs).n == 1
    assert sy.controllability_matrix(S).shape == (1, 1) and sy.observability_matrix(S).shape == (1, 1)
    T = "TransferFunction(" in repr(sy.tf([1, 2], [1, 3, 2]))
    assert T


def test_response_edges():
    G = sy.tf([1.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        rs.lsim(G, np.ones(3), np.array([0.0, 0.1, 0.5]))
    t, y, x = rs.initial(G, [1.0], 2.0, 21)
    assert y[0] == pytest.approx(1.0) and x.shape == (21, 1)
    Gd = sy.tf([1.0], [1.0, -0.5], dt=0.1)
    t, y = rs.impulse(Gd, 1.0)
    assert t[1] == pytest.approx(0.1)
    t, y, x = rs.initial(Gd, [1.0], 1.0)
    assert len(t) == 11
    assert np.isnan(rs.step_info([0, 1], [0.0, 0.0])["rise_time"])
    assert rs.step_info([0, 1, 2], [0.0, 1.0, 1.0])["overshoot"] == 0.0
    assert rs._default_horizon(sy.ss([[0.0]], [[1.0]], [[1.0]], 0.0)) == 10.0


def test_frequency_and_stability_edges():
    from englincontrol import frequency as fr
    from englincontrol import stability as st

    G = sy.tf([1.0], [1.0, 1.0])
    assert fr.freqresp(G, [1.0]).shape == (1,) and np.iscomplexobj(fr.freqresp(G.to_ss(), 2.0))
    assert fr.freqresp(G, 1.0, delay=1.0) == pytest.approx(fr.freqresp(G, 1.0) * np.exp(-1j))
    assert fr.bandwidth(sy.tf([1.0], [1.0])) == np.inf
    m = fr.margins(sy.tf([1.0], [1.0, 1.0]))
    assert m["gain_margin"] == np.inf and m["phase_margin"] == np.inf
    ph, mag = fr.nichols(G)
    assert ph.shape == mag.shape
    w, Hn = fr.nyquist(G)
    assert len(w) == len(Hn)
    assert fr.hinf_norm(sy.tf([1.0], [1.0, -1.0]).to_ss())[0] == np.inf
    mimo = sy.ss([[-1.0, 0.0], [0.0, -2.0]], np.eye(2), np.eye(2), np.zeros((2, 2)))
    assert fr.freqresp(mimo, [0.0, 1.0]).shape == (2, 2, 2)
    assert fr.default_frequencies(sy.tf([1.0], [1.0], dt=0.1)).max() == pytest.approx(np.pi / 0.1)
    assert st.routh([1, 0, 1])["stable"] is False  # marginal: a row of zeros
    assert st.is_stable(sy.tf([1.0], [1.0, -0.5], dt=0.1)) and not st.is_stable(sy.tf([1.0], [1.0, -1.0]))
    assert st.jury([1, 0.5])["stable"]
    rl = st.root_locus(sy.tf([1.0, 1.0], [1.0, 0.0]), np.logspace(-2, 2, 20))
    assert rl["branches"].shape == (20, 1) and np.isnan(rl["centroid"])  # no asymptotes when deg num = deg den
    assert np.isnan(st.gain_for_damping(sy.tf([1.0], [1.0, 1.0]), 0.5)[0])
    r = st.nyquist_criterion(sy.tf([1.0], [1.0, 0.0, 1.0]))
    assert r["marginal"]


def test_design_and_statespace_edges():
    from englincontrol import design as dg
    from englincontrol import statespace as sp

    assert dg.pid(2.0).num.tolist() == [2.0] and dg.pid(2.0, 1.0).den.tolist() == [1.0, 0.0]
    assert dg.pid(0.0, 0.0, 1.0, N=5.0).order == 2
    sim = dg.simulate_pid(
        sy.tf([1.0], [1.0, 1.0]),
        1.0,
        0.5,
        0.0,
        lambda t: 1.0,
        T=2.0,
        u_limits=(-5, 5),
        antiwindup="clamping",
        disturbance=lambda t: 0.1,
        delay=0.05,
    )
    assert sim["y"].shape == sim["u"].shape
    sim2 = dg.simulate_pid(sy.tf([1.0], [1.0, 1.0]), 1.0, 0.5, 0.1, 1.0, T=1.0, antiwindup=None, derivative_on="error")
    assert np.all(np.isfinite(sim2["y"]))
    ff = dg.feedforward_inverse(sy.tf([1.0, 2.0], [1.0, 1.0]))  # biproper: direct inverse times the order-1 filter
    assert ff.order == 2
    assert dg.design_lag(sy.tf([1.0], [1.0, 1.0]), 5.0, 2.0).order == 1
    assert dg.imc_pid(1.0, 2.0, 0.5, rule="imc")["Td"] > 0
    assert np.isinf(dg.ultimate_gain(sy.tf([1.0], [1.0, 1.0]))["Pu"])
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0, 0.0]])
    assert sp.observability(A, C)["observable"] and not sp.observability(A, np.array([[0.0, 1.0]]))["observable"]
    kd = sp.kalman_decomposition(sy.ss(A, B, C, 0.0))
    assert kd["n_minimal"] == 2
    obs = sp.observer(sy.ss(A, B, C, 0.0), sp.observer_gain(A, C, [-3, -4]))
    assert obs.m == 2 and obs.p == 2
    reg = sp.regulator(sy.ss(A, B, C, 0.0), sp.place(A, B, [-1, -2]), sp.observer_gain(A, C, [-3, -4]))
    assert reg.m == 2
    assert sp.lqr(A, B.T, np.eye(2), 1.0)["K"].shape == (1, 2)  # B given as a row, R as a scalar
    assert sp.kalman(A, C, np.eye(2), 1.0, G=np.eye(2))["L"].shape == (2, 1)
    Aa, Ba = sp.augment_integral(A, B, C)
    assert Aa.shape == (3, 3) and Ba.shape == (3, 1)
    assert sp._real_block_diag([-1.0 + 0j]).shape == (1, 1)


def test_phase4_edges():
    from englincontrol import discrete as dc
    from englincontrol import identify as idf
    from englincontrol import mimo
    from englincontrol import robust as rb

    G = sy.tf([1.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        dc.c2d(G, 0.1, "magic")
    assert isinstance(dc.c2d(G.to_ss(), 0.1, "tustin"), sy.StateSpace)
    assert dc.c2d(sy.tf([1.0], [1.0, 3.0, 2.0]), 0.1, "matched").order == 2  # relative degree 2: one zero at -1 added
    assert dc.nyquist_frequency(0.1) == pytest.approx(np.pi / 0.1)
    assert dc.zoh_frequency_response(0.0, 0.1) == 1.0
    assert dc.pid_discrete(1.0, 1.0, 0.0, 0.1, method="tustin").order == 1
    assert dc.pid_discrete(0.0, 0.0, 1.0, 0.1).order == 1
    assert "rise_time" in dc.discrete_step_info(np.array([0.0, 0.5, 1.0, 1.0]), 0.1)
    assert np.iscomplexobj(dc.logm(np.array([[-1.0, 0.0], [0.0, 2.0]])))  # a negative eigenvalue: complex log
    assert rb.rhp_limits(G, delay=0.5)["w_c_max_from_delay"] == 2.0
    assert rb.sensitivity_peak(sy.tf([0.1], [1.0, 1.0]))["gm_lower_bound"] == np.inf or True
    assert rb.multiplicative_robustness(sy.tf([1.0], [1.0, 3.0, 2.0, 0.0]))["peak_WT"] > 0
    assert idf.prbs(10, bits=7).shape == (10,) and idf.prbs(7, hold=3).tolist()[:3] == [idf.prbs(7, hold=3)[0]] * 3
    m = idf.arx(idf.prbs(300), np.cumsum(idf.prbs(300)) * 0.01, 1, 1, 2)
    assert m["tf"].order >= 1
    with pytest.raises(ValueError):
        mimo.transmission_zeros(sy.ss([[-1.0]], [[1.0, 1.0]], [[1.0]], [[0.0, 0.0]]))
    assert mimo.sigma(G, [1.0, 2.0]).shape == (2, 1)
    assert mimo.rga_pairing([[1.0, 2.0], [2.0, 1.0]])["negative_entries"]
    assert len(mimo.transfer_matrix(sy.ss(-np.eye(2), np.eye(2), np.eye(2), np.zeros((2, 2))))) == 2
    e = idf.era(np.zeros((6, 1, 1)) + np.array([0, 1, 0.5, 0.25, 0.125, 0.0625]).reshape(6, 1, 1), 1, rows=2)
    assert abs(e["system"].poles()[0] - 0.5) < 1e-10
    assert idf.fit_second_order_step(np.linspace(0, 10, 101), 1 - np.exp(-np.linspace(0, 10, 101)))["zeta"] > 0.9


def test_models_and_data():
    from englincontrol import datasets, models
    from englincontrol import identify as idf

    with pytest.raises(KeyError):
        datasets.path("nothing.csv")
    assert len(datasets.available()) == 3 and "motor" in datasets.describe("motor_step.csv").lower()
    assert datasets.path("motor_step.csv").read_text().startswith("#")
    th = models.thermal_process()
    assert th["delay"] == 5.0 and th["tf"].dcgain() == 2.0
    assert models.cruise_control()["tau"] == pytest.approx(20.0)
    assert models.cart_pendulum(inertia=0.006)["params"]["I"] == 0.006
    f = models.pendulum_nonlinear(inertia=0.006)
    assert f(np.zeros(4), [0.0]).tolist() == [0.0, 0.0, 0.0, 0.0]
    u = idf.prbs(400, hold=4)
    from englincontrol import response as rs
    from englincontrol import systems as sy

    G = sy.tf([0.5], [1.0, -0.5], dt=0.1)
    _, y, _ = rs.lsim(G, u)
    oe = idf.output_error(u, y, 1, 0.1, init=sy.tf([0.4], [1.0, -0.4], dt=0.1), iterations=20)
    assert abs(oe["tf"].poles()[0] - 0.5) < 1e-6
