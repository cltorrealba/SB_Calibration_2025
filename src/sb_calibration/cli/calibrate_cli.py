"""CLI entrypoint for calibration orchestration (minimal, testable).

This module exposes `run_calibration` which can be invoked from tests or a simple CLI.
"""
from typing import Optional
import os
import numpy as np
from ..calibration.optimize import calibrate_full
from ..model.zenteno import simulate_on_grid, load_parameters_from_excel
from ..calibration.config import CalibrationConfig
# Lazy import inside plotting block to avoid hard dependency at import time
try:
    from ..viz.plots import plot_fit_for_assay  # type: ignore
except Exception:
    plot_fit_for_assay = None  # will be checked before plotting
from ..calibration.pulses import build_pulses_from_chem, pulses_from_mats_yan_diff, pulses_from_2024_insumos
from ..preprocess.calibration_preprocess import (
    process_all as pp_process_all,
    attach_temperature_to_results as pp_attach_T,
    build_calibration_matrices as pp_build_mats,
)


def prebuild_2024_from_split(
    split_choice: str | None,
    split_file: str,
    temps_dir: str | None,
    cache: bool = True,
    verbose: bool = False,
) -> list[str]:
    """Build and persist 24xxx mats listed in the split file. Returns list of built IDs.

    - split_choice: 'train' | 'valid' | 'all' | None. If None, defaults to 'all'.
    - split_file: path to CSV with column 'assay' and 'split'.
    - temps_dir: directory containing Data <ID>.xlsx
    - cache: whether to persist to mats/assay=<ID>.parquet
    """
    try:
        import pandas as pd
        import os
        from sb_calibration.preprocess.sw_2024 import build_mats_for_assay_2024
    except Exception:
        # Dependencies not ready; do nothing
        return []

    if not os.path.exists(split_file):
        if verbose:
            print(f"[PREBUILD] Split file not found: {split_file}")
        return []
    try:
        sdf = pd.read_csv(split_file)
    except Exception:
        if verbose:
            print(f"[PREBUILD] Failed to read split file: {split_file}")
        return []
    if split_choice is None:
        split_choice = "all"
    if split_choice != "all":
        sdf = sdf[sdf.get("split").str.lower() == split_choice.lower()]
    if sdf.empty or "assay" not in sdf.columns:
        if verbose:
            print("[PREBUILD] No assays found in split selection")
        return []

    built: list[str] = []
    # numeric IDs only (e.g., 24028)
    codes = [str(a) for a in sdf["assay"].astype(str).tolist() if str(a).isdigit()]
    if not codes:
        if verbose:
            print("[PREBUILD] No numeric 24xxx assays to build")
        return []
    for code in sorted(set(codes)):
        try:
            dfmat = build_mats_for_assay_2024(code, temps_dir or "Datos Experimentales")
            if dfmat is None:
                if verbose:
                    print(f"[PREBUILD] Skipped {code}: builder returned None")
                continue
            if cache:
                try:
                    os.makedirs("mats", exist_ok=True)
                    dfmat.to_parquet(f"mats/assay={code}.parquet")
                    if verbose:
                        print(f"[PREBUILD] Cached mats/assay={code}.parquet")
                except Exception:
                    pass
            built.append(code)
        except Exception:
            if verbose:
                print(f"[PREBUILD] Error building {code} (continuing)")
            continue
    return built


def run_calibration(mats=None, out_path: str = "mats/pbest_checkpoint.npz", cfg: CalibrationConfig | None = None,
                    file_path: str | None = None, assays: list[str] | None = None,
                    temps_dir: str | None = None, use_smoothed_biomass: bool = False,
                    method: str = "Radau", jacobian: str = "analytic", exclude: list[str] | None = None,
                    chem_file: str | None = None, weights: dict[str, float] | None = None,
                    plot: bool = False, plots_dir: str = "mats/plots",
                    plot_from: str | None = None,
                    write_summary: bool = False,
                    summary_out: str | None = None,
                    split: str | None = None, split_file: str | None = None,
                    cache_2024: bool = True,
                    preview_p0_before: bool = False,
                    p0_excel: str | None = None,
                    p0_set: int = 3,
                    pulses_csv_path: str | None = None,
                    verbose: bool = False,
                    eval_print_every: int | None = None,
                    iter_print_every: int | None = None,
                    de_maxiter: int | None = None,
                    de_popsize: int | None = None,
                    de_tol: float | None = None,
                    # Objective extras
                    sugar_depletion_penalty_w: float = 0.0,
                    sugar_threshold: float = 1.0,
                    yan_offset_mgl: float = 20.0,
                    # Lag-phase controls
                    lag_mode: str = "none",
                    lag_tau_h: float = 12.0,
                    lag_sensitivity: float = 0.06,
                    lag_floor: float = 0.0,
                    # Optional bounds overrides for yields
                    yxg_lb: float | None = None,
                    yxg_ub: float | None = None,
                    yxf_lb: float | None = None,
                    yxf_ub: float | None = None,
                    yxn_lb: float | None = None,
                    yxn_ub: float | None = None,
                    sim_progress: bool | None = None):
    """Run calibration using the real simulator. If `mats` is None, a small
    synthetic mats dict will be used for a quick smoke run.
    """
    import time as _time, json as _json, sys as _sys, platform as _platform
    t_start = _time.time()
    pulses_by_assay = None
    built_2024_cached: list[str] = []
    # helper to map a generic pulses table (CSV/DF) into {assay: [(t_h, dN_gL), ...]}
    def _pulses_from_table(df_any):
        try:
            import pandas as pd
            dfp = df_any.copy()
            cols = {str(c).strip().lower(): c for c in dfp.columns}
            assay_col = cols.get("assay") or cols.get("ensayo_norm") or cols.get("ensayo")
            time_col = cols.get("time_h") or cols.get("t_h") or cols.get("t")
            val_gl = cols.get("dn_gl") or cols.get("dngl") or cols.get("delta_g_l")
            val_mgl = cols.get("deltayan_mgl") or cols.get("delta_mg_l") or cols.get("yan_delta_mgl")
            if assay_col is None or time_col is None or (val_gl is None and val_mgl is None):
                return None
            out: dict[str, list[tuple[float, float]]] = {}
            for k, g in dfp.groupby(assay_col):
                t = pd.to_numeric(g[time_col], errors="coerce").to_numpy(dtype=float)
                if val_gl is not None:
                    v = pd.to_numeric(g[val_gl], errors="coerce").to_numpy(dtype=float)
                else:
                    v = pd.to_numeric(g[val_mgl], errors="coerce").to_numpy(dtype=float) * 1e-3
                mask = ~(np.isnan(t) | np.isnan(v))
                pairs = sorted([(float(tt), float(vv)) for tt, vv in zip(t[mask], v[mask])])
                if pairs:
                    out[str(k)] = pairs
            return out
        except Exception:
            return None
    if mats is None:
        if file_path is not None:
            # Build mats from preprocess pipeline
            # Optionally override temps directory at runtime
            if temps_dir:
                # override module-level const used in preprocess
                import sb_calibration.preprocess.calibration_preprocess as cpp
                cpp.TEMPS_DIR = temps_dir
                if verbose:
                    print(f"[INFO] TEMPS_DIR set to: {cpp.TEMPS_DIR}")
            # If split specified and assays not provided, try reading split CSV
            assays_from_split = None
            if split and not assays:
                import os
                import pandas as pd
                split_path = split_file or "splits/assay_split.csv"
                if os.path.exists(split_path):
                    try:
                        sdf = pd.read_csv(split_path)
                        take = sdf[sdf.get("split").str.lower() == split.lower()]
                        assays_from_split = sorted(set(take["assay"].astype(str).tolist())) if not take.empty else None
                        assays = assays_from_split
                        if verbose:
                            print(f"[INFO] Split '{split}' loaded from {split_path}: {assays_from_split}")
                    except Exception:
                        assays_from_split = None
                        assays = assays
            results_dict, combined_df = pp_process_all(file_path, assays=assays)
            results_with_T = pp_attach_T(results_dict)
            mats = pp_build_mats(results_with_T, use_smoothed_biomass=use_smoothed_biomass)
            # exclusions
            if exclude:
                mats = {k:v for k,v in mats.items() if k not in set(exclude)}
            # If split included assays not present in mats (e.g., 24xxx from 2024), try building from Data <ID>.xlsx
            if assays_from_split:
                missing = [a for a in assays_from_split if a not in mats]
                if missing:
                    from sb_calibration.preprocess.sw_2024 import build_mats_for_assay_2024
                    for code in missing:
                        # numeric-only assays (2024) are expected here
                        try:
                            dfmat = build_mats_for_assay_2024(code, temps_dir or "Datos Experimentales")
                            if dfmat is not None:
                                mats[code] = dfmat
                                # persist for future runs
                                if cache_2024:
                                    try:
                                        import os
                                        os.makedirs("mats", exist_ok=True)
                                        outp = f"mats/assay={code}.parquet"
                                        dfmat.to_parquet(outp)
                                        built_2024_cached.append(code)
                                        if verbose:
                                            print(f"[CACHE] Saved 2024 assay {code} -> {outp}")
                                    except Exception:
                                        pass
                                continue
                        except Exception:
                            pass
                        # fallback to a precomputed parquet if present
                        try:
                            import pandas as pd
                            mats[code] = pd.read_parquet(f"mats/assay={code}.parquet")
                            if verbose:
                                print(f"[LOAD] Loaded existing parquet for 2024 assay {code}")
                        except Exception:
                            # ignore if not present; user can pre-generate or adjust split
                            pass
            # build pulses from chemistry file if provided
            if chem_file:
                import pandas as pd
                try:
                    if chem_file.lower().endswith((".xls", ".xlsx")):
                        cdf = pd.read_excel(chem_file)
                    else:
                        cdf = pd.read_csv(chem_file)
                except Exception:
                    cdf = None
                if cdf is not None:
                    # prefer builder; fallback to generic table parser
                    pulses_by_assay = build_pulses_from_chem(cdf) or _pulses_from_table(cdf)
                    # filter to present assays and apply exclusions
                    present = set(mats.keys())
                    if exclude:
                        present = present - set(exclude)
                    pulses_by_assay = {k: v for k, v in (pulses_by_assay or {}).items() if k in present}
                    if verbose:
                        print(f"[INFO] Pulses loaded for assays: {sorted(pulses_by_assay.keys())}")
            # fallback to mats/pulses_YAN.csv if no chem_file or failed to load
            if pulses_by_assay is None or len(pulses_by_assay) == 0:
                try:
                    import os, pandas as pd
                    default_csv = pulses_csv_path or os.path.join("mats", "pulses_YAN.csv")
                    if os.path.exists(default_csv):
                        pdf = pd.read_csv(default_csv)
                        pulses_by_assay = _pulses_from_table(pdf)
                        # filter to present assays
                        if pulses_by_assay:
                            present = set(mats.keys())
                            pulses_by_assay = {k: v for k, v in pulses_by_assay.items() if k in present}
                            if verbose:
                                print(f"[INFO] Pulses loaded from {default_csv}")
                except Exception:
                    pass
            # ultimate fallback: construct automatically like legacy
            if pulses_by_assay is None or len(pulses_by_assay) == 0:
                try:
                    # 1) SBxxx from mats YAN deltas (select main pulse)
                    auto_sb = pulses_from_mats_yan_diff(mats, select_main=True)
                    # 2) 24xxx from 2024 insumos (read original Excel)
                    auto_24 = {}
                    for code in mats.keys():
                        if str(code).isdigit():
                            lst = pulses_from_2024_insumos(str(code), temps_dir or "Datos Experimentales")
                            if lst:
                                auto_24[str(code)] = lst
                    # combine, prefer 24xxx specific if available
                    combined = dict(auto_sb)
                    combined.update(auto_24)
                    if combined:
                        pulses_by_assay = combined
                        if verbose:
                            ks = sorted(pulses_by_assay.keys())
                            print(f"[PULSES][AUTO] Construidos automáticamente para: {ks}")
                except Exception:
                    pass
        else:
            # small synthetic mats to keep smoke runs fast
            import pandas as pd
            mats = {"A": pd.DataFrame({"time_h": np.linspace(0, 10, 6), "biomass_viable_gL": np.linspace(0.5, 1.0, 6)})}

    # Load p0 from excel if available; fallback to ones. Also expose helper to read pbest from checkpoint if needed elsewhere.
    def _load_params_from_checkpoint(path: str):
        try:
            if path and os.path.exists(path):
                data = np.load(path)
                parr = data.get("pbest")
                if parr is None:
                    parr = data.get("p") or data.get("params")
                return parr
        except Exception:
            return None
        return None
    try:
        src_excel = p0_excel or "zenteno_parameters.xlsx"
        p0 = load_parameters_from_excel(src_excel, param_set=p0_set)
        # Append lag_tau_h default (12 h) if not present in Excel p0 (older sheets)
        try:
            p0_arr = np.asarray(p0, dtype=float).ravel()
            if p0_arr.size < 15:
                p0_arr = np.concatenate([p0_arr, np.array([12.0], dtype=float)])  # lag_tau_h default
            if p0_arr.size < 16:
                p0_arr = np.concatenate([p0_arr, np.array([0.06], dtype=float)])  # lag_sensitivity default
            p0 = p0_arr
        except Exception:
            p0 = np.asarray(p0, dtype=float)
        if verbose:
            print(f"[P0] loaded from {src_excel} (set={p0_set})")
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar p0 desde Excel ({src_excel}, set={p0_set}). Corrige la ruta o set. Detalle: {e}")
    # Physical-ish bounds per parameter (order must match zenteno_model unpacking):
    # 1) mu0, 2) betaG0, 3) betaF0, 4) Kn0, 5) Kg0, 6) Kf0, 7) Kig0, 8) Kie0, 9) Kd0,
    # 10) Yxn, 11) Yxg, 12) Yxf, 13) Yeg, 14) Yef, 15) lag_tau_h (h@20°C), 16) lag_sensitivity
    bounds = [
        (1e-4, 2.0),   # mu0
        (1e-6, 2.0),   # betaG0
        (1e-6, 2.0),   # betaF0
        (1e-4, 200.0), # Kn0
        (1e-4, 200.0), # Kg0
        (1e-4, 200.0), # Kf0
        (1e-4, 200.0), # Kig0
        (1e-4, 200.0), # Kie0
        (1e-6, 1.0),   # Kd0
        (1.0, 100.0),  # Yxn
        (0.05, 10.0),  # Yxg
        (0.05, 10.0),  # Yxf
        (0.05, 20.0),  # Yeg
        (0.05, 20.0),  # Yef
        (2.0, 96.0),   # lag_tau_h (hours)
        (0.0, 0.25),   # lag_sensitivity
    ]
    # Apply optional overrides to yields bounds
    def _apply_override(idx: int, lb: float | None, ub: float | None):
        cur_lb, cur_ub = bounds[idx]
        new_lb = cur_lb if lb is None else float(lb)
        new_ub = cur_ub if ub is None else float(ub)
        # ensure order
        if new_lb > new_ub:
            new_lb, new_ub = new_ub, new_lb
        bounds[idx] = (new_lb, new_ub)
    # indices: 9=Yxn, 10=Yxg, 11=Yxf
    _apply_override(10, yxg_lb, yxg_ub)
    _apply_override(11, yxf_lb, yxf_ub)
    _apply_override(9, yxn_lb, yxn_ub)
    if cfg is None:
        cfg = CalibrationConfig()
        cfg.n_starts = 4
        cfg.local_maxiter = 20
    # wrap simulate_on_grid with tolerances
    def sim_wrapped(p, t_meas, temp_segs, pulses, x0):
        return simulate_on_grid(
            p, t_meas, temp_segs, pulses, x0,
            method=method, rtol=cfg.rtol, atol_vec=cfg.atol_vec(), jacobian=jacobian, verbose=verbose,
            lag_mode=lag_mode, lag_tau_h=lag_tau_h, lag_sensitivity=lag_sensitivity, lag_floor=lag_floor
        )

    # Helper: derive X0 with sugar-from-density fallback for 24xxx (or any mat with Densidad)
    def _derive_x0_with_density(code, df) -> np.ndarray:
        import numpy as _np
        # First valid values across the series (not strictly the first row)
        def first_valid(col):
            try:
                s = _np.asarray(pd.to_numeric(df[col], errors="coerce"), dtype=float)
                idx = _np.where(_np.isfinite(s))[0]
                return float(s[idx[0]]) if idx.size > 0 else _np.nan
            except Exception:
                return _np.nan
        # X
        try:
            import pandas as pd  # local to avoid global import issues
        except Exception:
            pd = None  # type: ignore
        X0 = first_valid("biomass_viable_gL") if 'pd' in locals() and pd is not None else _np.nan
        N0_mgL = first_valid("YAN") if 'pd' in locals() and pd is not None else _np.nan
        G0 = first_valid("Glucose") if 'pd' in locals() and pd is not None else _np.nan
        F0 = first_valid("Fructose") if 'pd' in locals() and pd is not None else _np.nan
        E0 = first_valid("Ethanol") if 'pd' in locals() and pd is not None else _np.nan
        # Convert YAN to g/L
        N0 = (N0_mgL * 1e-3) if _np.isfinite(N0_mgL) else _np.nan
        # If G/F missing or clearly zero while density exists, estimate total sugar from density and split 50/50
        need_split = (not _np.isfinite(G0)) or (not _np.isfinite(F0)) or ((G0 + F0) <= 0)
        is_numeric_assay = str(code).isdigit()
        # For 24xxx (numeric IDs), Ethanol initial must be 0.0 (lab only provides final E).
        if is_numeric_assay:
            E0 = 0.0
        if 'pd' in locals() and pd is not None and "Densidad" in getattr(df, 'columns', []):
            try:
                dens = pd.to_numeric(df["Densidad"], errors="coerce").to_numpy(dtype=float)
                m = _np.isfinite(dens)
                if m.any():
                    # take earliest valid density
                    j = int(_np.where(m)[0][0])
                    d0 = float(dens[j])
                    # load sugar-density dataset and fit cubic model globally
                    import os as _os
                    import pandas as _pd
                    ds_path = _os.path.join("sugar_density_out", "sugar_density_dataset.csv")
                    S0 = _np.nan
                    if _os.path.exists(ds_path):
                        dsd = _pd.read_csv(ds_path)
                        if ("density" in dsd.columns) and ("total_sugar" in dsd.columns):
                            den_global = _np.asarray(dsd["density"], dtype=float)
                            sug_global = _np.asarray(dsd["total_sugar"], dtype=float)
                            mg = _np.isfinite(den_global) & _np.isfinite(sug_global)
                            if mg.any():
                                coef = _np.polyfit(den_global[mg], sug_global[mg], deg=3)
                                S0 = float(_np.polyval(coef, d0))
                    if _np.isfinite(S0) and S0 > 0:
                        mismatch = (_np.isfinite(G0) and _np.isfinite(F0)) and (abs((G0+F0) - S0) / max(S0, 1e-12) > 0.05)
                        # Apply override for numeric (24xxx) on mismatch; for SB only when missing/zero
                        if need_split or (is_numeric_assay and mismatch):
                            G0 = S0 * 0.5
                            F0 = S0 * 0.5
            except Exception:
                pass
        # Fallbacks with DEFAULT_X0
        try:
            from sb_calibration.model.zenteno import DEFAULT_X0 as _DEF
        except Exception:
            _DEF = _np.array([1.0, 0.2, 200.0, 200.0, 0.0], dtype=float)
        x0_vec = _np.array([
            X0 if _np.isfinite(X0) and X0 >= 0 else _DEF[0],
            N0 if _np.isfinite(N0) and N0 >= 0 else _DEF[1],
            G0 if _np.isfinite(G0) and G0 >= 0 else _DEF[2],
            F0 if _np.isfinite(F0) and F0 >= 0 else _DEF[3],
            E0 if _np.isfinite(E0) and E0 >= 0 else _DEF[4],
        ], dtype=float)
        return x0_vec

    # Optional: generate preview plots with p0 before running optimization
    if preview_p0_before and plot_fit_for_assay is not None and file_path is not None and mats is not None:
        try:
            import os
            base_dir = os.path.dirname(out_path) if out_path else "."
            prev_dir = os.path.join(base_dir, "preview_p0")
            os.makedirs(prev_dir, exist_ok=True)
            if verbose:
                print(f"[PREVIEW] Generating p0 plots in {prev_dir}")
            for code, df in mats.items():
                pulses = (pulses_by_assay or {}).get(code)
                try:
                    # use the same X0 logic as calibration
                    try:
                        x0_vec = _derive_x0_with_density(code, df)
                    except Exception:
                        from sb_calibration.model.zenteno import DEFAULT_X0 as _DEF
                        x0_vec = _DEF
                    plot_fit_for_assay(code, df, p0, sim_wrapped, pulses=pulses, x0=x0_vec, out_path=f"{prev_dir}/{code}.png")
                except Exception:
                    pass
        except Exception:
            pass

    # If plot_from is provided, load pbest from NPZ and skip optimization
    if plot_from is not None:
        try:
            data = np.load(plot_from)
            pbest = data.get("pbest")
            score = float(data.get("score")) if "score" in data else float("nan")
            if pbest is None:
                raise ValueError("El archivo NPZ no contiene 'pbest'")
            if verbose:
                print(f"[PLOT-FROM] Loaded pbest from {plot_from} (score={score if np.isfinite(score) else 'NA'})")
        except Exception as e:
            raise RuntimeError(f"No se pudo cargar pbest desde {plot_from}: {e}")
    else:
        pbest, score, meta = calibrate_full(
            mats,
            p0,
            bounds,
            sim_wrapped,
            mode=cfg.mode,
            pulses_by_assay=pulses_by_assay,
            x0_by_assay={k: _derive_x0_with_density(k, v) for k, v in mats.items()},
            weights=weights,
            sse_balance=getattr(cfg, 'sse_balance', 'per_assay'),
            sse_resample_dt_h=getattr(cfg, 'sse_resample_dt_h', None),
            n_starts=cfg.n_starts,
            local_maxiter=cfg.local_maxiter,
            patience_starts=cfg.patience_starts,
            patience_evals=cfg.patience_evals,
            min_improvement_rel=cfg.min_improvement_rel,
            out_path=out_path,
            verbose=verbose,
            eval_print_every=(eval_print_every if eval_print_every is not None else 50),
            iter_print_every=(iter_print_every if iter_print_every is not None else 0),
            de_maxiter=(de_maxiter if de_maxiter is not None else 60),
            de_popsize=(de_popsize if de_popsize is not None else 12),
            de_tol=(de_tol if de_tol is not None else 1e-6),
            sim_progress=bool(sim_progress) if sim_progress is not None else False,
            # new objective options
            sugar_depletion_penalty_w=sugar_depletion_penalty_w,
            sugar_threshold=sugar_threshold,
            yan_offset_mgl=yan_offset_mgl,
        )
    if verbose and plot_from is None:
        print(f"[RESULT] SSE={score:.4e}  out={out_path}")

    # Optional: write a JSON summary of the run (works for optimize and plot-from)
    if write_summary:
        try:
            # choose npz path and json output path
            npz_path = (plot_from if plot_from is not None else out_path)
            base = (summary_out if (summary_out and len(str(summary_out))>0) else (str(npz_path).rsplit('.',1)[0] + ".json"))
            # collect basic environment
            env = {
                "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                "duration_s": round(_time.time() - t_start, 3),
                "python": _sys.version.split('\n')[0],
                "platform": _platform.platform(),
            }
            try:
                import numpy as _np
                env["numpy"] = str(_np.__version__)
            except Exception:
                pass
            # simulation config
            sim_cfg = {
                "method": method,
                "jacobian": jacobian,
                "rtol": getattr(cfg, 'rtol', None),
                "atol": {
                    "X": getattr(cfg, 'atol_x', None),
                    "N": getattr(cfg, 'atol_n', None),
                    "G": getattr(cfg, 'atol_g', None),
                    "F": getattr(cfg, 'atol_f', None),
                    "E": getattr(cfg, 'atol_e', None),
                }
            }
            # optimizer config (when applicable)
            opt_cfg = {
                "mode": getattr(cfg, 'mode', None),
                "n_starts": getattr(cfg, 'n_starts', None),
                "local_maxiter": getattr(cfg, 'local_maxiter', None),
                "patience_starts": getattr(cfg, 'patience_starts', None),
                "patience_evals": getattr(cfg, 'patience_evals', None),
                "min_improvement_rel": getattr(cfg, 'min_improvement_rel', None),
                "de_maxiter": (de_maxiter if de_maxiter is not None else 60),
                "de_popsize": (de_popsize if de_popsize is not None else 12),
                "de_tol": (de_tol if de_tol is not None else 1e-6),
                "eval_print_every": (eval_print_every if eval_print_every is not None else 50),
                "iter_print_every": (iter_print_every if iter_print_every is not None else 0),
                "sim_progress": bool(sim_progress) if sim_progress is not None else False,
                # extras
                "sugar_penalty": sugar_depletion_penalty_w,
                "sugar_threshold": sugar_threshold,
                "yan_offset_mgl": yan_offset_mgl,
                "lag": {"mode": lag_mode, "tau_h": lag_tau_h, "sensitivity": lag_sensitivity, "floor": lag_floor},
            }
            # data context
            assays_list = sorted([str(k) for k in (mats or {}).keys()])
            data_ctx = {
                "file": file_path,
                "split": split,
                "split_file": split_file,
                "assays_used": assays_list,
                "n_assays": len(assays_list),
                "exclude": exclude,
                "temps_dir": temps_dir,
                "pulses_present_for": sorted(list((pulses_by_assay or {}).keys())),
            }
            # weights and bounds
            wb = {
                "weights": weights,
                "bounds": bounds,
                "sse_balance": getattr(cfg, 'sse_balance', 'per_assay'),
                "sse_resample_dt_h": getattr(cfg, 'sse_resample_dt_h', None),
                "overrides": {
                    "lb_yxg": yxg_lb,
                    "ub_yxg": yxg_ub,
                    "lb_yxf": yxf_lb,
                    "ub_yxf": yxf_ub,
                    "lb_yxn": yxn_lb,
                    "ub_yxn": yxn_ub,
                }
            }
            # parameter names in the expected order
            param_names = [
                "mu0","betaG0","betaF0","Kn0","Kg0","Kf0","Kig0","Kie0",
                "Kd0","Yxn","Yxg","Yxf","Yeg","Yef","lag_tau_h","lag_sensitivity"
            ]

            # result section
            # try to load score/pbest from npz when plotting-from
            score_out = None
            pbest_out = pbest
            try:
                import numpy as _np
                if plot_from is not None and npz_path is not None:
                    with _np.load(npz_path) as _d:
                        if 'score' in _d:
                            _s = _d['score']
                            try:
                                score_out = float(_s)
                            except Exception:
                                score_out = None
                        if (pbest_out is None) and ('pbest' in _d):
                            pbest_out = _d['pbest']
                else:
                    score_out = float(score)
            except Exception:
                pass

            result = {
                "npz_path": npz_path,
                "score": score_out,
                "pbest": (
                    [float(x) for x in (pbest_out.tolist() if hasattr(pbest_out, 'tolist') else list(pbest_out))]
                    if pbest_out is not None else None
                ),
            }

            # attach parameter mapping when available
            if result["pbest"] is not None and len(result["pbest"]) == len(param_names):
                result["params"] = {
                    "names": param_names,
                    "values": {n: result["pbest"][i] for i, n in enumerate(param_names)},
                    "bounds": {n: list(bounds[i]) if bounds and i < len(bounds) else None for i, n in enumerate(param_names)},
                }
            summary = {
                "env": env,
                "simulation": sim_cfg,
                "optimizer": opt_cfg,
                "data": data_ctx,
                "objective": wb,
                "result": result,
                "source": ("plot-from" if plot_from is not None else "optimize"),
            }
            # write JSON
            import os as _os
            _os.makedirs(_os.path.dirname(base) or ".", exist_ok=True)
            with open(base, "w", encoding="utf-8") as f:
                _json.dump(summary, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"[SUMMARY] wrote {base}")
        except Exception as e:
            if verbose:
                print(f"[SUMMARY] failed: {e}")
    # optional plotting
    if plot and plot_fit_for_assay is not None:
        for code, df in mats.items():
            pulses = (pulses_by_assay or {}).get(code)
            try:
                # Build x0 using the same density-based logic
                try:
                    x0_vec = _derive_x0_with_density(code, df)
                except Exception:
                    from sb_calibration.model.zenteno import DEFAULT_X0 as _DEF
                    x0_vec = _DEF
                plot_fit_for_assay(code, df, pbest, sim_wrapped, pulses=pulses, x0=x0_vec, out_path=f"{plots_dir}/{code}.png")
            except Exception:
                # keep calibration robust even if plotting fails for some assay
                pass

    # brief summary of cached 2024 builds (if any)
    try:
        if 'built_2024_cached' in locals() and built_2024_cached:
            print(f"[CACHE] Guardados {len(built_2024_cached)} ensayos 2024 en mats/: {', '.join(built_2024_cached)}")
    except Exception:
        pass

    return pbest, score


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="mats/pbest_checkpoint.npz")
    parser.add_argument("--file", default=None, help="Ruta al Excel BDD para construir mats")
    parser.add_argument("--assays", default=None, help="Lista separada por comas de ensayos a incluir (e.g., SB005,SB006)")
    parser.add_argument("--use-smoothed-biomass", action="store_true", help="Usar MA(3) de biomasa viable/muerta para las matrices")
    parser.add_argument("--temps-dir", default=None, help="Carpeta con los archivos de temperatura por ensayo")
    parser.add_argument("--chem-file", default=None, help="Archivo (Excel o CSV) con la planilla de química para inferir pulsos de YAN")
    parser.add_argument("--method", default="Radau", choices=["Radau","BDF"], help="Método stiff para solve_ivp")
    parser.add_argument("--jacobian", default="analytic", choices=["analytic","numeric","none"], help="Tipo de jacobiano a usar en la integración")
    parser.add_argument("--exclude", default=None, help="Lista separada por comas de ensayos a excluir (e.g., SB001,SB002)")
    parser.add_argument("--mode", default="multistart", choices=["multistart","de"])
    parser.add_argument("--sse-balance", default="per_assay", choices=["per_assay","per_point","none"], help="Modo de balance para la SSE (por defecto: per_assay)")
    parser.add_argument("--sse-resample-dt-h", type=float, default=None, help="Submuestreo temporal (horas) para la SSE; e.g., 6.0. Por defecto: None")
    parser.add_argument("--split", default=None, choices=["train","valid"], help="Seleccionar ensayos según splits precomputados si --assays no se entrega")
    parser.add_argument("--split-file", default="splits/assay_split.csv", help="Ruta al CSV de splits para --split")
    parser.add_argument("--n-starts", type=int, default=8)
    parser.add_argument("--local-maxiter", type=int, default=200)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol-x", type=float, default=1e-3)
    parser.add_argument("--atol-n", type=float, default=1e-2)
    parser.add_argument("--atol-g", type=float, default=1e-2)
    parser.add_argument("--atol-f", type=float, default=1e-2)
    parser.add_argument("--atol-e", type=float, default=1e-3)
    parser.add_argument("--plot", action="store_true", help="Guardar gráficos sim vs. meas por ensayo")
    parser.add_argument("--plots-dir", default="mats/plots", help="Carpeta de salida para los gráficos")
    parser.add_argument("--plot-from", default=None, help="Ruta a un .npz con pbest para generar gráficos sin recalibrar")
    # Summary flags: default ON for optimizer runs, OFF for plot-from
    summary_group = parser.add_mutually_exclusive_group()
    summary_group.add_argument("--write-summary", action="store_true", help="Forzar escritura de resumen JSON (también en plot-from)")
    summary_group.add_argument("--no-summary", action="store_true", help="Desactivar la escritura del resumen JSON")
    parser.add_argument("--summary-out", default=None, help="Ruta del JSON de salida (por defecto, junto al .npz)")
    parser.add_argument("--preview-p0-before", action="store_true", help="Generar plots con p0 antes de calibrar (preview)")
    parser.add_argument("--p0-excel", default="zenteno_parameters.xlsx", help="Excel con parámetros base para p0 (por defecto: zenteno_parameters.xlsx)")
    parser.add_argument("--p0-set", type=int, default=3, help="Set de parámetros dentro del Excel de p0 (por defecto: 3)")
    parser.add_argument("--no-cache-2024", action="store_true", help="No persistir a parquet las matrices 24xxx construidas on-the-fly")
    parser.add_argument("--verbose", action="store_true", help="Imprimir pasos detallados de preprocesado y calibración")
    parser.add_argument("--prebuild-2024", action="store_true", help="Preconstruir matrices 24xxx listadas en el split antes de calibrar")
    parser.add_argument("--prebuild-split", default=None, choices=["train","valid","all"], help="Split objetivo para preconstrucción (por defecto: 'all' o --split si se entrega)")
    parser.add_argument("--prebuild-only", action="store_true", help="Solo preconstruir 24xxx y salir (no ejecutar calibración)")
    parser.add_argument("--w-x", type=float, default=1.0, help="Peso para X en la SSE")
    parser.add_argument("--w-n", type=float, default=1.0, help="Peso para N en la SSE")
    parser.add_argument("--w-g", type=float, default=1.0, help="Peso para G en la SSE")
    parser.add_argument("--w-f", type=float, default=1.0, help="Peso para F en la SSE")
    parser.add_argument("--w-e", type=float, default=1.0, help="Peso para E en la SSE")
    parser.add_argument("--w-s", type=float, default=1.0, help="Peso para S=G+F o SugarTotal_exp en la SSE (si existe)")
    parser.add_argument("--pulses-csv", default=os.path.join("mats", "pulses_YAN.csv"), help="Ruta a CSV de pulsos (assay,time_h,dN_gL) si no se usa --chem-file")
    parser.add_argument("--eval-print-every", type=int, default=25, help="Imprimir progreso cada N evaluaciones de la SSE (default: 25)")
    parser.add_argument("--iter-print-every", type=int, default=5, help="Imprimir progreso dentro de L-BFGS-B cada N iteraciones (default: 5; 0=off)")
    parser.add_argument("--de-maxiter", type=int, default=120, help="Iteraciones máximas de Differential Evolution (default: 120)")
    parser.add_argument("--de-popsize", type=int, default=18, help="Tamaño de población DE (default: 18)")
    parser.add_argument("--de-tol", type=float, default=1e-6, help="Tolerancia de convergencia DE (default: 1e-6)")
    # Bounds overrides for yields (optional, to constrain biomass growth)
    parser.add_argument("--lb-yxg", type=float, default=None, help="Override lower bound for Yxg (biomass yield on glucose)")
    parser.add_argument("--ub-yxg", type=float, default=None, help="Override upper bound for Yxg (biomass yield on glucose)")
    parser.add_argument("--lb-yxf", type=float, default=None, help="Override lower bound for Yxf (biomass yield on fructose)")
    parser.add_argument("--ub-yxf", type=float, default=None, help="Override upper bound for Yxf (biomass yield on fructose)")
    parser.add_argument("--lb-yxn", type=float, default=None, help="Override lower bound for Yxn (biomass yield on nitrogen)")
    parser.add_argument("--ub-yxn", type=float, default=None, help="Override upper bound for Yxn (biomass yield on nitrogen)")
    parser.add_argument("--sim-progress", action="store_true", help="Imprimir [SIM] OK por ensayo tras cada simulación (más verboso)")
    # New objective/lag options
    parser.add_argument("--sugar-penalty", type=float, default=0.0, help="Peso de penalización por depleción temprana/tardía de azúcar (0=off)")
    parser.add_argument("--sugar-threshold", type=float, default=1.0, help="Umbral de azúcar total (g/L) para definir 'todo consumido'")
    parser.add_argument("--yan-offset-mgl", type=float, default=20.0, help="Offset a restar a YAN medido (mg/L); truncado a 0 (default: 20 mg/L)")
    parser.add_argument("--lag-mode", default="none", choices=["none","exp","logistic"], help="Modo de fase lag inicial")
    parser.add_argument("--lag-tau-h", type=float, default=12.0, help="Escala base de tiempo de lag (h) a 20°C")
    parser.add_argument("--lag-sensitivity", type=float, default=0.06, help="Sensibilidad de lag a temperatura (exp(s*(20-Tc)))")
    parser.add_argument("--lag-floor", type=float, default=0.0, help="Mínimo multiplicador de reacción durante lag (0..1)")
    args = parser.parse_args()
    cfg = CalibrationConfig(
        mode=args.mode,
        n_starts=args.n_starts,
        local_maxiter=args.local_maxiter,
        sse_balance=args.sse_balance,
        sse_resample_dt_h=args.sse_resample_dt_h,
        rtol=args.rtol,
        atol_x=args.atol_x,
        atol_n=args.atol_n,
        atol_g=args.atol_g,
        atol_f=args.atol_f,
        atol_e=args.atol_e,
    )
    assays = [s.strip() for s in args.assays.split(',')] if args.assays else None
    exclude = [s.strip() for s in args.exclude.split(',')] if args.exclude else None
    weights = {"X": args.w_x, "N": args.w_n, "G": args.w_g, "F": args.w_f, "E": args.w_e, "S": args.w_s}
    # Optional prebuild step
    if args.prebuild_2024 or args.prebuild_only:
        split_choice = args.prebuild_split or (args.split if args.split else "all")
        built = prebuild_2024_from_split(split_choice, args.split_file, args.temps_dir, cache=(not args.no_cache_2024), verbose=args.verbose)
        if built:
            print(f"[PREBUILD] Construidos {len(built)} ensayos 2024: {', '.join(built)}")
        else:
            print("[PREBUILD] No se construyeron ensayos 2024 (verifica split y Data <ID>.xlsx)")
        if args.prebuild_only:
            raise SystemExit(0)

    # Effective summary behavior: default ON for optimizer (no --plot-from), OFF for plot-from.
    if args.plot_from is None:
        write_summary_effective = True
    else:
        write_summary_effective = False
    if getattr(args, 'write_summary', False):
        write_summary_effective = True
    if getattr(args, 'no_summary', False):
        write_summary_effective = False

    run_calibration(
        None,
        out_path=args.out,
        cfg=cfg,
        file_path=args.file,
        assays=assays,
        temps_dir=args.temps_dir,
        use_smoothed_biomass=args.use_smoothed_biomass,
        method=args.method,
        jacobian=args.jacobian,
        exclude=exclude,
        chem_file=args.chem_file,
        weights=weights,
        plot=args.plot,
        plots_dir=args.plots_dir,
        plot_from=args.plot_from,
        write_summary=write_summary_effective,
        summary_out=args.summary_out,
        split=args.split,
        split_file=args.split_file,
        cache_2024=(not args.no_cache_2024),
        preview_p0_before=args.preview_p0_before,
        p0_excel=args.p0_excel,
        p0_set=args.p0_set,
        pulses_csv_path=args.pulses_csv,
        verbose=args.verbose,
        eval_print_every=args.eval_print_every,
        iter_print_every=args.iter_print_every,
        de_maxiter=args.de_maxiter,
        de_popsize=args.de_popsize,
        de_tol=args.de_tol,
        # new objective/lag options
        sugar_depletion_penalty_w=args.sugar_penalty,
        sugar_threshold=args.sugar_threshold,
        yan_offset_mgl=args.yan_offset_mgl,
        lag_mode=args.lag_mode,
        lag_tau_h=args.lag_tau_h,
        lag_sensitivity=args.lag_sensitivity,
        lag_floor=args.lag_floor,
        # bounds overrides
        yxg_lb=args.lb_yxg, yxg_ub=args.ub_yxg,
        yxf_lb=args.lb_yxf, yxf_ub=args.ub_yxf,
        yxn_lb=args.lb_yxn, yxn_ub=args.ub_yxn,
        sim_progress=args.sim_progress,
    )
