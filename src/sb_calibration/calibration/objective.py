import numpy as np
from typing import Dict, List, Tuple, Optional


def sse_for_experiment(y_meas: np.ndarray, y_sim: np.ndarray) -> float:
    """Compute sum of squared errors, handling NaNs gracefully."""
    y_meas = np.asarray(y_meas, dtype=float)
    y_sim = np.asarray(y_sim, dtype=float)
    mask = ~np.isnan(y_meas) & ~np.isnan(y_sim)
    if mask.sum() == 0:
        return 0.0
    diff = (y_sim[mask] - y_meas[mask])
    return float(np.nansum(diff * diff))


def compute_global_stds(mats: Dict[str, 'pd.DataFrame']) -> Dict[str, float]:
    """Compute global std per observable (X,N,G,F,E) from a dict of dataframes.

    This mirrors the legacy behaviour: collects values across assays and returns
    a dict with a positive std (1.0 fallback).
    """
    import pandas as pd

    vals = {"X": [], "N": [], "G": [], "F": [], "E": []}
    N_SCALE = 1e-3
    for code, df in mats.items():
        if "biomass_viable_gL" in df:
            vals["X"].extend(pd.to_numeric(df["biomass_viable_gL"], errors="coerce").dropna().astype(float).tolist())
        if "YAN" in df:
            vals["N"].extend((pd.to_numeric(df["YAN"], errors="coerce").dropna().astype(float) * N_SCALE).tolist())
        if "Glucose" in df:
            vals["G"].extend(pd.to_numeric(df["Glucose"], errors="coerce").dropna().astype(float).tolist())
        if "Fructose" in df:
            vals["F"].extend(pd.to_numeric(df["Fructose"], errors="coerce").dropna().astype(float).tolist())
        if "Ethanol" in df:
            vals["E"].extend(pd.to_numeric(df["Ethanol"], errors="coerce").dropna().astype(float).tolist())

    stds: Dict[str, float] = {}
    for k, arr in vals.items():
        if len(arr) == 0:
            stds[k] = 1.0
        else:
            s = float(np.nanstd(np.asarray(arr, dtype=float)))
            stds[k] = s if s > 0 else 1.0
    return stds


def sse_for_experiments_real(p_real: np.ndarray,
                             mats: Dict[str, 'pd.DataFrame'],
                             pulses_by_assay: Optional[Dict[str, List[Tuple[float, float]]]] = None,
                             x0_by_assay: Optional[Dict[str, np.ndarray]] = None,
                             weights: Optional[Dict[str, float]] = None,
                             stds: Optional[Dict[str, float]] = None,
                             verbose: bool = False,
                             simulate_fn=None) -> float:
    """Compute normalized SSE across multiple assays.

    This is a lightweight port of the legacy function. The caller should pass
    a `simulate_fn(p_real, t_meas, temp_segments, pulses, x0)` compatible
    with the legacy `simulate_on_grid`. If simulate_fn is None, a RuntimeError
    is raised.
    """
    if simulate_fn is None:
        raise RuntimeError("simulate_fn must be provided to sse_for_experiments_real")

    if weights is None:
        weights = {"X": 1.0, "N": 1.0, "G": 1.0, "F": 1.0, "E": 1.0}
    if stds is None:
        stds = {k: 1.0 for k in ["X", "N", "G", "F", "E"]}

    total = 0.0
    N_SCALE = 1e-3
    for code, df in mats.items():
        t_meas = df["time_h"].astype(float).to_numpy()
        temp_segs_C = None
        # try to find temperature-like column
        for c in ("Temperature_C", "temperature", "temp_c", "temperatura"):
            if c in df.columns:
                temp_segs_C = df
                break
        pulses = (pulses_by_assay or {}).get(code, None)
        x0 = (x0_by_assay or {}).get(code, None)

        t_sim, Xsim = simulate_fn(p_real, t_meas, temp_segs_C, pulses, x0)
        t_sim = np.asarray(t_sim, dtype=float)

        X_interp = np.vstack([
            np.interp(t_meas, t_sim, Xsim[:, 0]),
            np.interp(t_meas, t_sim, Xsim[:, 1]),
            np.interp(t_meas, t_sim, Xsim[:, 2]),
            np.interp(t_meas, t_sim, Xsim[:, 3]),
            np.interp(t_meas, t_sim, Xsim[:, 4]),
        ]).T

        sse = 0.0
        # X
        if "biomass_viable_gL" in df.columns:
            y = df["biomass_viable_gL"].astype(float).to_numpy()
            m = ~np.isnan(y)
            if m.any():
                sse += weights.get("X", 1.0) * np.nansum(((X_interp[m, 0] - y[m]) / stds["X"]) ** 2)
        # N
        if "YAN" in df.columns:
            y = pd_to_numeric_safe(df["YAN"]).astype(float) * N_SCALE
            m = ~np.isnan(y)
            if m.any():
                sse += weights.get("N", 1.0) * np.nansum(((X_interp[m, 1] - y[m]) / stds["N"]) ** 2)
        # G
        if "Glucose" in df.columns:
            y = df["Glucose"].astype(float).to_numpy()
            m = ~np.isnan(y)
            if m.any():
                sse += weights.get("G", 1.0) * np.nansum(((X_interp[m, 2] - y[m]) / stds["G"]) ** 2)
        # F
        if "Fructose" in df.columns:
            y = df["Fructose"].astype(float).to_numpy()
            m = ~np.isnan(y)
            if m.any():
                sse += weights.get("F", 1.0) * np.nansum(((X_interp[m, 3] - y[m]) / stds["F"]) ** 2)
        # E
        if "Ethanol" in df.columns:
            y = df["Ethanol"].astype(float).to_numpy()
            m = ~np.isnan(y)
            if m.any():
                sse += weights.get("E", 1.0) * np.nansum(((X_interp[m, 4] - y[m]) / stds["E"]) ** 2)

        total += sse
    if verbose:
        print(f"SSE={total:.4e}")
    return float(total)


def pd_to_numeric_safe(s):
    import pandas as pd
    return pd.to_numeric(s, errors="coerce").to_numpy()


