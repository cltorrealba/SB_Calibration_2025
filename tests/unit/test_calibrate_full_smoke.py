import numpy as np
import pandas as pd
from sb_calibration.calibration import optimize


def make_df(times, X=None, G=None):
    d = {"time_h": np.asarray(times, dtype=float)}
    if X is not None: d["biomass_viable_gL"] = np.asarray(X, dtype=float)
    if G is not None: d["Glucose"] = np.asarray(G, dtype=float)
    return pd.DataFrame(d)


def test_calibrate_full_runs_quickly(tmp_path):
    mats = {"A": make_df([0.0, 1.0, 2.0], X=[1.0, 2.0, 3.0], G=[10.0, 9.0, 8.0])}

    def sim_fn(p, t_meas, temp_segs, pulses, x0):
        t_sim = np.asarray(t_meas, dtype=float)
        Xsim = np.tile(np.array([1.0, 0.1, 10.0, 0.0, 0.0]), (len(t_sim),1))
        return t_sim, Xsim

    p0 = np.ones(5)
    bounds = [(1e-3, 10.0)] * 5
    out = tmp_path / "pbest_full.npz"
    pbest, score, meta = optimize.calibrate_full(mats, p0, bounds, sim_fn, n_starts=4, local_maxiter=10, out_path=str(out))
    assert out.exists()
    assert isinstance(pbest, np.ndarray)
    assert isinstance(score, float)
