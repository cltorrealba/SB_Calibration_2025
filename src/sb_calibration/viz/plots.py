from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as _pd
_SUGAR_DATASET = None  # lazy-loaded sugar-density-derived totals


def _load_sugar_dataset() -> Optional[_pd.DataFrame]:
    global _SUGAR_DATASET
    if _SUGAR_DATASET is not None:
        return _SUGAR_DATASET
    try:
        path = os.path.join("sugar_density_out", "sugar_density_dataset.csv")
        if os.path.exists(path):
            _SUGAR_DATASET = _pd.read_csv(path)
        else:
            _SUGAR_DATASET = _pd.DataFrame()
    except Exception:
        _SUGAR_DATASET = _pd.DataFrame()
    return _SUGAR_DATASET


def _pick_col(df, candidates: List[str]) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols:
            return cols[name.lower()]
    return None


def _meas_columns(df) -> Dict[str, Optional[str]]:
    return {
        "time": _pick_col(df, ["time_h", "tiempo_h", "t_h", "t"]),
        "X": _pick_col(df, ["biomass_viable_gL", "biomass_total_gL", "x_gl", "x"]),
        "N": _pick_col(df, ["yan", "n_mg_l", "n_mgl", "nitrogeno_mg_l"]),
        "G": _pick_col(df, ["glucose", "glucose_gl", "g_gl", "g"]),
        "F": _pick_col(df, ["fructose", "fructose_gl", "f_gl", "f"]),
        "E": _pick_col(df, ["ethanol", "alcohol", "ethanol_gl", "e_gl", "e"]),
        "T": _pick_col(df, ["temperature_c", "temperature", "temp_c", "temperatura"]),
        "D": _pick_col(df, ["densidad", "density"]),
    }


def plot_fit_for_assay(
    assay_code: str,
    df,
    p_real: np.ndarray,
    simulate_fn,
    pulses: Optional[List[Tuple[float, float]]] = None,
    x0: Optional[np.ndarray] = None,
    out_path: Optional[str] = None,
) -> Optional[str]:
    """Plot measured vs simulated (X,N,G,F,E) for one assay.

    - df: canonical matrix with 'time_h' and measured columns.
    - p_real: parameter vector used for simulation.
    - simulate_fn: callable(p, t_meas, temp_segments, pulses, x0)->(t_sim, Xsim)
    - pulses: optional pulses list [(t_h, dN_gL)]
    - out_path: file to save plot; returns the path if saved.
    """
    cols = _meas_columns(df)
    if cols["time"] is None:
        return None
    t_meas = np.asarray(df[cols["time"]], dtype=float)
    # temperature segments: pass the dataframe itself (simulate_fn should handle DataFrame input)
    temp_segments = df[[cols["time"], cols["T"]]] if cols["T"] is not None else df

    t_sim, Xsim = simulate_fn(p_real, t_meas, temp_segments, pulses, x0)
    # Build a smooth plotting grid combining dense base grid + pulses + measurement times
    t_meas_f = np.asarray(t_meas, dtype=float)
    tmax_meas = np.nanmax(t_meas_f) if np.isfinite(np.nanmax(t_meas_f)) else (t_sim[-1] if len(t_sim)>0 else 0.0)
    tmax = max(float(tmax_meas), float(t_sim[-1]) if len(t_sim)>0 else 0.0)
    n_dense = int(min(1000, max(200, (tmax/0.25) if tmax>0 else 200)))  # ~cada 0.25 h
    base = np.linspace(0.0, tmax, num=n_dense)
    extra = []
    # include pulses times for visible discontinuities
    if pulses:
        try:
            extra.extend([float(tp) for tp,_ in pulses if 0.0 <= float(tp) <= tmax])
        except Exception:
            pass
    # include measurement times
    try:
        extra.extend([float(tt) for tt in t_meas_f[np.isfinite(t_meas_f)] if 0.0 <= float(tt) <= tmax])
    except Exception:
        pass
    t_line = np.unique(np.concatenate([base, np.asarray(extra, dtype=float) if extra else base]))
    # Interpolate simulation onto plotting grid and clip negatives to zero for visualization
    X_line = np.empty((len(t_line), Xsim.shape[1]), dtype=float)
    for j in range(Xsim.shape[1]):
        X_line[:, j] = np.interp(t_line, t_sim, Xsim[:, j])
    X_line = np.maximum(X_line, 0.0)
    m_line = np.ones_like(t_line, dtype=bool)
    sim_map = {"X": 0, "N": 1, "G": 2, "F": 3, "E": 4}
    order = ["X", "N", "G", "F", "E"]
    pos_map = {"X": 0, "N": 1, "G": 2, "F": 3, "E": 4, "S": 5}
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12), squeeze=False)
    axes = axes.flatten()

    # Plot X,N,G,F,E in fixed positions if present; collect anchors for X0 and E_final
    anchor_X0 = None  # (t, y)
    anchor_Ef = None  # (t, y)
    for var in order:
        if cols[var] is None:
            continue
        ax = axes[pos_map[var]]
        y_meas = np.asarray(df[cols[var]], dtype=float)
        if var == "N":
            y_meas = y_meas * 1e-3
        # For 24xxx, show only the FINAL ethanol point to avoid confusion with initial lab reading
        t_plot = t_meas
        y_plot = y_meas
        if var == "E":
            try:
                if str(assay_code).isdigit():
                    mf = np.isfinite(t_meas) & np.isfinite(y_meas)
                    if np.any(mf):
                        idxs = np.where(mf)[0]
                        j_last = int(idxs[-1])
                        t_plot = np.array([float(t_meas[j_last])])
                        y_plot = np.array([float(y_meas[j_last])])
            except Exception:
                pass
        ax.plot(t_plot, y_plot, "o", label=f"{var} meas", alpha=0.7)
        ax.plot(t_line[m_line], X_line[:, sim_map[var]], "-", label=f"{var} sim", alpha=0.9)
        ax.set_xlabel("time [h]")
        ax.set_ylabel(f"{var} [g/L]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        # capture anchors using the same arrays used to plot (ensures units/time alignment)
        if var == "X":
            m = np.isfinite(t_meas) & np.isfinite(y_meas)
            if m.any():
                j0 = int(np.where(m)[0][0])
                anchor_X0 = (float(t_meas[j0]), float(y_meas[j0]))
        if var == "E":
            m = np.isfinite(t_meas) & np.isfinite(y_meas)
            if m.any():
                j1 = int(np.where(m)[0][-1])
                anchor_Ef = (float(t_meas[j1]), float(y_meas[j1]))

    # Sixth subplot: S = G+F
    axS = axes[pos_map["S"]]
    has_G = cols["G"] is not None
    has_F = cols["F"] is not None
    if has_G or has_F:
        g = np.asarray(df[cols["G"]], dtype=float) if has_G else 0.0
        f = np.asarray(df[cols["F"]], dtype=float) if has_F else 0.0
        yS = g + f
        axS.plot(t_meas, yS, "o", label="S=G+F meas", alpha=0.7)
    ssim_line = X_line[:, sim_map["G"]] + X_line[:, sim_map["F"]]
    axS.plot(t_line[m_line], ssim_line, "-", label="S sim", alpha=0.9)
    # Overlay density-based points or estimates
    try:
        ds = _load_sugar_dataset()
        plotted_density_points = False
        if ds is not None and not ds.empty and 'assay' in ds.columns:
            sub = ds[ds['assay'].astype(str) == str(assay_code)]
            if not sub.empty:
                tds = np.asarray(sub.get('time_h', sub.get('t', sub.index)), dtype=float)
                sds = np.asarray(sub.get('total_sugar'), dtype=float)
                mds = ~(np.isnan(tds) | np.isnan(sds))
                if mds.any():
                    axS.plot(tds[mds], sds[mds], 'x', label='S from density', alpha=0.6)
                    plotted_density_points = True
        if not plotted_density_points and cols['D'] is not None and ds is not None and not ds.empty:
            den_global = np.asarray(ds['density'], dtype=float)
            sug_global = np.asarray(ds['total_sugar'], dtype=float)
            mg = ~(np.isnan(den_global) | np.isnan(sug_global))
            if mg.any():
                coef = np.polyfit(den_global[mg], sug_global[mg], deg=3)
                den_local = np.asarray(df[cols['D']], dtype=float)
                sl = np.polyval(coef, den_local)
                ml = ~(np.isnan(t_meas) | np.isnan(sl))
                if ml.any():
                    axS.plot(t_meas[ml], sl[ml], 'x', label='S from density (est)', alpha=0.5)
    except Exception:
        pass
    axS.set_xlabel("time [h]")
    axS.set_ylabel("S [g/L]")
    axS.grid(True, alpha=0.3)
    axS.legend()

    # Overlay legacy-style anchor markers using captured points
    if anchor_X0 is not None and cols["X"] is not None:
        axX = axes[pos_map["X"]]
        axX.plot([anchor_X0[0]], [anchor_X0[1]], marker='*', color='k', ms=9, zorder=5, label='X0 anchor')
        axX.legend()
    if anchor_Ef is not None and cols["E"] is not None:
        axE = axes[pos_map["E"]]
        axE.plot([anchor_Ef[0]], [anchor_Ef[1]], marker='*', color='k', ms=9, zorder=5, label='E_final anchor')
        # annotate numeric value for clarity
        try:
            axE.annotate(f"{anchor_Ef[1]:.2f} g/L @ {anchor_Ef[0]:.0f}h",
                         xy=(anchor_Ef[0], anchor_Ef[1]), xytext=(5, -12), textcoords="offset points",
                         fontsize=9, color='k', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))
        except Exception:
            pass
        axE.legend()

    fig.suptitle(f"Assay {assay_code} — fit")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        return out_path
    return None
