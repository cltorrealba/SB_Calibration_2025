import numpy as np
import pandas as pd
from sb_calibration.calibration import objective, optimize


def make_df(times, X=None, G=None, F=None, E=None):
    d = {"time_h": np.asarray(times, dtype=float)}
    if X is not None: d["biomass_viable_gL"] = np.asarray(X, dtype=float)
    if G is not None: d["Glucose"] = np.asarray(G, dtype=float)
    if F is not None: d["Fructose"] = np.asarray(F, dtype=float)
    if E is not None: d["Ethanol"] = np.asarray(E, dtype=float)
    return pd.DataFrame(d)


def test_objective_per_assay_vs_per_point_balance():
    # Assay A has many points; B has few. Per_assay should reduce A's dominance.
    mats = {
        "A": make_df(np.linspace(0, 10, 51), X=np.linspace(1.0, 6.0, 51)),  # 51 pts
        "B": make_df([0.0, 10.0], X=[1.0, 6.0]),  # 2 pts
    }

    def sim_fn(p, t_meas, temp_segs, pulses, x0):
        t = np.asarray(t_meas, dtype=float)
        # simple linear sim with slope controlled by p[0]
        slope = float(p[0])
        Xsim = np.column_stack([
            1.0 + slope * t,   # X
            np.full_like(t, 0.1),  # N
            np.full_like(t, 10.0), # G
            np.zeros_like(t),      # F
            np.zeros_like(t),      # E
        ])
        return t, Xsim

    stds = objective.compute_global_stds(mats)
    p = np.array([0.5, 0, 0, 0, 0], dtype=float)
    sse_point = objective.sse_for_experiments_real(p, mats, stds=stds, simulate_fn=sim_fn, balance="per_point")
    sse_assay = objective.sse_for_experiments_real(p, mats, stds=stds, simulate_fn=sim_fn, balance="per_assay")

    assert isinstance(sse_point, float) and isinstance(sse_assay, float)
    # With per_assay averaging, the loss should be closer to per-assay mean than per-point sum.
    assert sse_assay <= sse_point


def test_convergence_on_synthetic_quadratic(tmp_path):
    # Two-parameter synthetic: optimum at p=[2.0, 3.0]
    target = np.array([2.0, 3.0, 0, 0, 0], dtype=float)

    # Measurements set to a constant so the unique optimum is at delta=0 (p=[2,3])
    mats = {"A": make_df([0.0, 1.0, 2.0], X=[1.0, 1.0, 1.0])}

    def sim_fn(p, t_meas, temp_segs, pulses, x0):
        t = np.asarray(t_meas, dtype=float)
        # Predict X ~ 1.0 + 0.05*(p0-2)^2 + 0.05*(p1-3)^2 (constant in time)
        delta = (p[0]-2.0)**2 + (p[1]-3.0)**2
        xval = 1.0 + 0.05*delta
        Xsim = np.column_stack([
            np.full_like(t, xval),
            np.full_like(t, 0.1),
            np.full_like(t, 10.0),
            np.zeros_like(t),
            np.zeros_like(t),
        ])
        return t, Xsim

    p0 = np.array([0.5, 0.5, 0, 0, 0], dtype=float)
    bounds = [(1e-3, 10.0)] * 5
    out = tmp_path/"pbest_quad.npz"

    best, score, meta = optimize.calibrate_full(
        mats,
        p0,
        bounds,
        sim_fn,
        mode="multistart",
        n_starts=8,
        local_maxiter=50,
        out_path=str(out),
        sse_balance="per_assay",
    )

    assert out.exists()
    assert isinstance(score, float)
    # With constant measurements, the unique optimum is delta=0 at p=[2,3]
    delta = (best[0]-2.0)**2 + (best[1]-3.0)**2
    assert delta <= 0.25
