import numpy as np
import pandas as pd
from sb_calibration.calibration import objective


def make_df(times, X=None, N=None, G=None, F=None, E=None):
    d = {"time_h": np.asarray(times, dtype=float)}
    if X is not None: d["biomass_viable_gL"] = np.asarray(X, dtype=float)
    if N is not None: d["YAN"] = np.asarray(N, dtype=float)
    if G is not None: d["Glucose"] = np.asarray(G, dtype=float)
    if F is not None: d["Fructose"] = np.asarray(F, dtype=float)
    if E is not None: d["Ethanol"] = np.asarray(E, dtype=float)
    return pd.DataFrame(d)


def test_compute_global_stds_simple():
    mats = {
        "A": make_df([0,1,2], X=[1.0,2.0,3.0], N=[100.0,110.0,105.0]),
        "B": make_df([0,1], G=[10.0,12.0], F=[5.0,5.5]),
    }
    stds = objective.compute_global_stds(mats)
    assert set(stds.keys()) == {"X","N","G","F","E"}
    assert stds["E"] == 1.0  # no ethanol data -> fallback
    assert stds["X"] > 0


def test_sse_for_experiments_real_with_synthetic_simulator():
    # tiny mats with one assay
    mats = {
        "A": make_df([0.0, 1.0, 2.0], X=[1.0, 2.0, 3.0], G=[10.0, 9.0, 8.0])
    }

    def sim_fn(p, t_meas, temp_segs, pulses, x0):
        # return t grid and Xsim with trivial linear trajectories
        t_sim = np.asarray(t_meas, dtype=float)
        Xsim = np.vstack([np.array([1.0, 0.1, 10.0, 0.0, 0.0]) + 0.5 * (t_sim[:,None]),
                          ]).reshape(len(t_sim),5)
        return t_sim, Xsim

    stds = objective.compute_global_stds(mats)
    sse = objective.sse_for_experiments_real(np.ones(5), mats, stds=stds, simulate_fn=sim_fn)
    assert isinstance(sse, float)
    assert sse >= 0.0
