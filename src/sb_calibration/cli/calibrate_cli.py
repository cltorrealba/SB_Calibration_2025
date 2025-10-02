"""CLI entrypoint for calibration orchestration (minimal, testable).

This module exposes `run_calibration` which can be invoked from tests or a simple CLI.
"""
from typing import Optional
import numpy as np
from ..calibration.optimize import calibrate_full
from ..model.zenteno import simulate_on_grid


def run_calibration(mats=None, out_path: str = "mats/pbest_checkpoint.npz"):
    """Run calibration using the real simulator. If `mats` is None, a small
    synthetic mats dict will be used for a quick smoke run.
    """
    if mats is None:
        # small synthetic mats to keep smoke runs fast
        import pandas as pd
        mats = {"A": pd.DataFrame({"time_h": np.linspace(0, 10, 6), "biomass_viable_gL": np.linspace(0.5, 1.0, 6)})}

    # tiny defaults for a smoke run; for real calibration increase n_starts and local_maxiter
    # zenteno model expects 14 parameters
    p0 = np.ones(14)
    bounds = [(1e-6, 1e3)] * 14
    pbest, score, meta = calibrate_full(mats, p0, bounds, simulate_on_grid, mode="multistart", n_starts=4, local_maxiter=20, out_path=out_path)
    return pbest, score


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="mats/pbest_checkpoint.npz")
    args = parser.parse_args()
    run_calibration(None, out_path=args.out)
