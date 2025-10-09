import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time


def sse_for_experiment(y_meas: np.ndarray, y_sim: np.ndarray) -> float:
    """Compute sum of squared errors, handling NaNs gracefully."""
    y_meas = np.asarray(y_meas, dtype=float)
    y_sim = np.asarray(y_sim, dtype=float)
    mask = ~np.isnan(y_meas) & ~np.isnan(y_sim)
    if mask.sum() == 0:
        return 0.0
    diff = (y_sim[mask] - y_meas[mask])
    return float(np.nansum(diff * diff))


def compute_global_stds(mats: Dict[str, Any]) -> Dict[str, float]:
    """Compute global std per observable (X,N,G,F,E) from a dict of dataframes.

    This mirrors the legacy behaviour: collects values across assays and returns
    a dict with a positive std (1.0 fallback).
    """
    import pandas as pd

    vals = {"X": [], "N": [], "G": [], "F": [], "E": [], "S": []}
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
        if "SugarTotal_exp" in df:
            vals["S"].extend(pd.to_numeric(df["SugarTotal_exp"], errors="coerce").dropna().astype(float).tolist())

    stds: Dict[str, float] = {}
    for k, arr in vals.items():
        if len(arr) == 0:
            stds[k] = 1.0
        else:
            s = float(np.nanstd(np.asarray(arr, dtype=float)))
            stds[k] = s if s > 0 else 1.0
    return stds


def sse_for_experiments_real(p_real: np.ndarray,
                             mats: Dict[str, Any],
                             pulses_by_assay: Optional[Dict[str, List[Tuple[float, float]]]] = None,
                             x0_by_assay: Optional[Dict[str, np.ndarray]] = None,
                             weights: Optional[Dict[str, float]] = None,
                             stds: Optional[Dict[str, float]] = None,
                             verbose: bool = False,
                             simulate_fn=None,
                             balance: str = "per_assay",
                             resample_dt_h: Optional[float] = None,
                             sim_progress: bool = False) -> float:
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
        stds = {k: 1.0 for k in ["X", "N", "G", "F", "E", "S"]}

    total = 0.0
    N_SCALE = 1e-3

    def _maybe_resample(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if resample_dt_h is None or len(t) <= 1:
            return t, y
        t = np.asarray(t, float); y = np.asarray(y, float)
        if not np.isfinite(t).any():
            return t, y
        t0 = float(np.nanmin(t))
        grid = np.arange(t0, float(np.nanmax(t)) + 1e-9, float(resample_dt_h))
        idx = np.searchsorted(t, grid, side="left")
        idx[idx == len(t)] = len(t) - 1
        return t[idx], y[idx]

    def _series_loss(sim: np.ndarray, y: np.ndarray, std: float, w: float) -> Optional[float]:
        m = ~np.isnan(y)
        if not m.any():
            return None
        err2 = ((sim[m] - y[m]) / (std if std > 0 else 1.0)) ** 2
        # Use mean to avoid bias from number of points
        return float(np.nanmean(err2) * w)
    for code, df in mats.items():
        t_sim_start = time.time()
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
        if sim_progress:
            try:
                dt = time.time() - t_sim_start
                print(f"[SIM] OK assay={code}  nT={len(t_sim)}  dt={dt:0.2f}s")
            except Exception:
                pass
        t_sim = np.asarray(t_sim, dtype=float)

        X_interp = np.vstack([
            np.interp(t_meas, t_sim, Xsim[:, 0]),
            np.interp(t_meas, t_sim, Xsim[:, 1]),
            np.interp(t_meas, t_sim, Xsim[:, 2]),
            np.interp(t_meas, t_sim, Xsim[:, 3]),
            np.interp(t_meas, t_sim, Xsim[:, 4]),
        ]).T

        per_var_losses: List[float] = []
        # X
        if "biomass_viable_gL" in df.columns:
            y = df["biomass_viable_gL"].astype(float).to_numpy()
            if resample_dt_h:
                t_rs, y_rs = _maybe_resample(t_meas, y)
                sim_rs = np.interp(t_rs, t_sim, Xsim[:, 0])
                per = _series_loss(sim_rs, y_rs, stds.get("X", 1.0), weights.get("X", 1.0))
            else:
                per = _series_loss(X_interp[:, 0], y, stds.get("X", 1.0), weights.get("X", 1.0))
            if per is not None:
                per_var_losses.append(per)
        # N
        if "YAN" in df.columns:
            y = pd_to_numeric_safe(df["YAN"]).astype(float) * N_SCALE
            if resample_dt_h:
                t_rs, y_rs = _maybe_resample(t_meas, y)
                sim_rs = np.interp(t_rs, t_sim, Xsim[:, 1])
                per = _series_loss(sim_rs, y_rs, stds.get("N", 1.0), weights.get("N", 1.0))
            else:
                per = _series_loss(X_interp[:, 1], y, stds.get("N", 1.0), weights.get("N", 1.0))
            if per is not None:
                per_var_losses.append(per)
        # G
        if "Glucose" in df.columns:
            y = df["Glucose"].astype(float).to_numpy()
            if resample_dt_h:
                t_rs, y_rs = _maybe_resample(t_meas, y)
                sim_rs = np.interp(t_rs, t_sim, Xsim[:, 2])
                per = _series_loss(sim_rs, y_rs, stds.get("G", 1.0), weights.get("G", 1.0))
            else:
                per = _series_loss(X_interp[:, 2], y, stds.get("G", 1.0), weights.get("G", 1.0))
            if per is not None:
                per_var_losses.append(per)
        # F
        if "Fructose" in df.columns:
            y = df["Fructose"].astype(float).to_numpy()
            if resample_dt_h:
                t_rs, y_rs = _maybe_resample(t_meas, y)
                sim_rs = np.interp(t_rs, t_sim, Xsim[:, 3])
                per = _series_loss(sim_rs, y_rs, stds.get("F", 1.0), weights.get("F", 1.0))
            else:
                per = _series_loss(X_interp[:, 3], y, stds.get("F", 1.0), weights.get("F", 1.0))
            if per is not None:
                per_var_losses.append(per)
        # E
        if "Ethanol" in df.columns:
            y = df["Ethanol"].astype(float).to_numpy()
            if resample_dt_h:
                t_rs, y_rs = _maybe_resample(t_meas, y)
                sim_rs = np.interp(t_rs, t_sim, Xsim[:, 4])
                per = _series_loss(sim_rs, y_rs, stds.get("E", 1.0), weights.get("E", 1.0))
            else:
                per = _series_loss(X_interp[:, 4], y, stds.get("E", 1.0), weights.get("E", 1.0))
            if per is not None:
                per_var_losses.append(per)
        # Optional S = G+F when available
        if "SugarTotal_exp" in df.columns:
            y = df["SugarTotal_exp"].astype(float).to_numpy()
            if resample_dt_h:
                t_rs, y_rs = _maybe_resample(t_meas, y)
                sim_rs = np.interp(t_rs, t_sim, Xsim[:, 2] + Xsim[:, 3])
                per = _series_loss(sim_rs, y_rs, stds.get("S", 1.0), weights.get("S", 1.0))
            else:
                per = _series_loss(X_interp[:, 2] + X_interp[:, 3], y, stds.get("S", 1.0), weights.get("S", 1.0))
            if per is not None:
                per_var_losses.append(per)

        if not per_var_losses:
            continue
        if balance == "per_assay":
            total += float(np.mean(per_var_losses))
        else:  # "per_point" or fallback
            total += float(np.sum(per_var_losses))
    if verbose:
        print(f"SSE={total:.4e}")
    return float(total)


def pd_to_numeric_safe(s):
    import pandas as pd
    return pd.to_numeric(s, errors="coerce").to_numpy()


