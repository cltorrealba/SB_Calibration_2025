"""Preview plots CLI: genera gráficos sim vs. mediciones por ensayo
usando p0 (o pbest si existe) antes/después de calibrar.

Ejemplos:
  python -m sb_calibration.cli.preview_plots_cli --file "Procesos_I+D_2025_3.xlsx" \
    --temps-dir "Datos Experimentales" --split train --outdir mats/preview_p0 --use-p0

  python -m sb_calibration.cli.preview_plots_cli --file "Procesos_I+D_2025_3.xlsx" \
    --temps-dir "Datos Experimentales" --split train --outdir mats/preview_best --use-best --checkpoint mats/pbest_checkpoint.npz
"""
from __future__ import annotations
import os
import numpy as np
from typing import Optional, List

from sb_calibration.model.zenteno import simulate_on_grid, DEFAULT_X0, load_parameters_from_excel
from sb_calibration.viz.plots import plot_fit_for_assay
from sb_calibration.preprocess.calibration_preprocess import (
    process_all as pp_process_all,
    attach_temperature_to_results as pp_attach_T,
    build_calibration_matrices as pp_build_mats,
)
from sb_calibration.calibration.pulses import build_pulses_from_chem, pulses_from_mats_yan_diff, pulses_from_2024_insumos


def _read_assays_from_split(split_file: str, split_choice: Optional[str]) -> Optional[List[str]]:
    try:
        import pandas as pd
        if os.path.exists(split_file):
            sdf = pd.read_csv(split_file)
            if split_choice in ("train", "valid"):
                sdf = sdf[sdf.get("split").str.lower() == split_choice.lower()]
            if not sdf.empty and "assay" in sdf.columns:
                return sorted(set(sdf["assay"].astype(str).tolist()))
    except Exception:
        pass
    return None


def main():
    import argparse
    p = argparse.ArgumentParser(description="Generar plots sim vs. medición por ensayo (p0 o pbest)")
    p.add_argument("--file", required=True, help="Excel BDD para construir matrices SBxxx (y opcionalmente 24xxx si se agregan desde Data <ID>.xlsx)")
    p.add_argument("--temps-dir", default=None, help="Carpeta con Data <ID>.xlsx")
    p.add_argument("--assays", default=None, help="Lista separada por comas de ensayos a incluir")
    p.add_argument("--split", default=None, choices=["train","valid"], help="Usar ensayos desde splits/assay_split.csv")
    p.add_argument("--split-file", default="splits/assay_split.csv", help="Ruta al CSV de split")
    p.add_argument("--use-p0", action="store_true", help="Usar p0 manual en lugar de pbest")
    p.add_argument("--use-best", action="store_true", help="Usar pbest desde --checkpoint")
    p.add_argument("--checkpoint", default="mats/pbest_checkpoint.npz", help="Ruta al checkpoint con pbest")
    p.add_argument("--p0-excel", default="zenteno_parameters.xlsx", help="Excel con parámetros base (por defecto: zenteno_parameters.xlsx)")
    p.add_argument("--p0-set", type=int, default=3, help="Set de parámetros a usar desde --p0-excel (por defecto: 3)")
    p.add_argument("--yan-offset-mgl", type=float, default=20.0, help="Restar este offset (mg/L) a YAN para graficar (>=0)")
    p.add_argument("--ignore-lag-param", action="store_true", help="Si el checkpoint trae 15 parámetros, descartar el último (lag_tau_h) y usar el valor por defecto para lag en el simulador")
    p.add_argument("--chem-file", default=None, help="Archivo de química (Excel/CSV) para inferir pulsos de YAN")
    p.add_argument("--pulses-csv", default=os.path.join("mats","pulses_YAN.csv"), help="CSV de pulsos si no hay planilla de química (assay,time_h,dN_gL)")
    p.add_argument("--outdir", default="mats/preview", help="Carpeta de salida para los plots")
    p.add_argument("--method", default="Radau", choices=["Radau","BDF"], help="Método stiff")
    p.add_argument("--jacobian", default="analytic", choices=["analytic","numeric","none"], help="Jacobian mode")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol-x", type=float, default=1e-3)
    p.add_argument("--atol-n", type=float, default=1e-2)
    p.add_argument("--atol-g", type=float, default=1e-2)
    p.add_argument("--atol-f", type=float, default=1e-2)
    p.add_argument("--atol-e", type=float, default=1e-3)
    p.add_argument("--no-cache-2024", action="store_true", help="No persistir los 24xxx construidos a mats/assay=<ID>.parquet")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    assays = [s.strip() for s in args.assays.split(',')] if args.assays else None
    if assays is None and args.split:
        assays = _read_assays_from_split(args.split_file, args.split)

    # build mats from BDD (SBxxx)
    if args.temps_dir:
        import sb_calibration.preprocess.calibration_preprocess as cpp
        cpp.TEMPS_DIR = args.temps_dir
    # If --file es un CSV de metadata, intenta usar el Excel BDD por defecto para SBxxx
    bdd_file = args.file
    if str(args.file).lower().endswith('.csv'):
        default_bdd = "Procesos_I+D_2025_3.xlsx"
        if os.path.exists(default_bdd):
            if args.verbose:
                print(f"[INFO] Metadata CSV detectado. Usando BDD='{default_bdd}' para SBxxx y Data <ID>.xlsx para 24xxx")
            bdd_file = default_bdd
        else:
            if args.verbose:
                print("[WARN] Metadata CSV detectado pero no se encontró el Excel BDD por defecto. Intentando continuar con CSV (puede no incluir SBxxx)")
    results_dict, _ = pp_process_all(bdd_file, assays=assays)
    results_T = pp_attach_T(results_dict)
    mats = pp_build_mats(results_T, use_smoothed_biomass=False)

    # If split/assays requested include 24xxx not present, build from Data <ID>.xlsx
    assays_requested = assays or list(mats.keys())
    numeric_missing = []
    try:
        if assays_requested:
            for a in assays_requested:
                if str(a).isdigit() and str(a) not in mats:
                    numeric_missing.append(str(a))
    except Exception:
        pass
    if numeric_missing:
        from sb_calibration.preprocess.sw_2024 import build_mats_for_assay_2024
        for code in sorted(set(numeric_missing)):
            try:
                dfmat = build_mats_for_assay_2024(code, args.temps_dir or "Datos Experimentales")
                if dfmat is not None:
                    mats[code] = dfmat
                    # optional cache
                    if not args.no_cache_2024:
                        try:
                            import pandas as pd
                            os.makedirs("mats", exist_ok=True)
                            dfmat.to_parquet(f"mats/assay={code}.parquet")
                            if args.verbose:
                                print(f"[CACHE] mats/assay={code}.parquet")
                        except Exception:
                            pass
                elif args.verbose:
                    print(f"[2024] builder devolvió None para {code}")
            except Exception as e:
                # fallback: load precomputed parquet if present
                try:
                    import pandas as pd
                    mats[code] = pd.read_parquet(f"mats/assay={code}.parquet")
                    if args.verbose:
                        print(f"[LOAD] mats/assay={code}.parquet")
                except Exception:
                    if args.verbose:
                        print(f"[2024] no se pudo construir/cargar {code}: {e}")

    # choose parameters (p0 or pbest) with robust checkpoint loading
    def _load_params():
        src = None
        parr = None
        # Prefer checkpoint if requested
        if args.use_best and os.path.exists(args.checkpoint):
            try:
                data = np.load(args.checkpoint)
                parr = data.get("pbest")
                if parr is None:
                    # backward compatibility with older checkpoints
                    parr = data.get("p") or data.get("params")
                if parr is not None:
                    if args.ignore_lag_param and len(parr) >= 15:
                        parr = parr[:14]
                    src = f"checkpoint:{args.checkpoint}"
            except Exception as e:
                parr = None
        # Fallback to Excel (required if no checkpoint or checkpoint invalid)
        if parr is None:
            parr = load_parameters_from_excel(args.p0_excel, param_set=args.p0_set)
            src = f"excel:{args.p0_excel}[set={args.p0_set}]"
        return parr, src

    try:
        p, p_src = _load_params()
    except Exception as e:
        raise RuntimeError(f"No se pudieron cargar parámetros ni desde checkpoint ni desde Excel: {e}")
    if args.verbose:
        print(f"[PARAMS] source={p_src}")

    # build pulses from chem-file; fallback to CSV
    pulses_by_assay = {}
    if args.chem_file:
        import pandas as pd
        try:
            if args.chem_file.lower().endswith((".xls",".xlsx")):
                cdf = pd.read_excel(args.chem_file)
            else:
                cdf = pd.read_csv(args.chem_file)
            pulses_by_assay = build_pulses_from_chem(cdf) or {}
            if args.verbose and pulses_by_assay:
                print(f"[PULSES] assays con pulsos: {sorted(pulses_by_assay.keys())}")
        except Exception as e:
            if args.verbose:
                print(f"[PULSES] no se pudieron construir pulsos: {e}")
    # fallback: CSV generic parser
    if not pulses_by_assay and args.pulses_csv and os.path.exists(args.pulses_csv):
        try:
            import pandas as pd
            dfp = pd.read_csv(args.pulses_csv)
            # generic parse similar a calibrate_cli
            cols = {str(c).strip().lower(): c for c in dfp.columns}
            assay_col = cols.get("assay") or cols.get("ensayo_norm") or cols.get("ensayo")
            time_col = cols.get("time_h") or cols.get("t_h") or cols.get("t")
            val_gl = cols.get("dn_gl") or cols.get("dngl") or cols.get("delta_g_l") or cols.get("dn_gl_1")
            val_mgl = cols.get("deltayan_mgl") or cols.get("delta_mg_l") or cols.get("yan_delta_mgl")
            if assay_col and time_col and (val_gl or val_mgl):
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
                # filter to present assays
                present = set(mats.keys())
                pulses_by_assay = {k: v for k, v in out.items() if k in present}
                if args.verbose:
                    print(f"[PULSES] desde CSV: {args.pulses_csv}")
        except Exception as e:
            if args.verbose:
                print(f"[PULSES] error al leer CSV {args.pulses_csv}: {e}")
    # ultimate fallback: build automatically like legacy
    if not pulses_by_assay:
        try:
            # SBxxx from mats YAN deltas (select main)
            auto_sb = pulses_from_mats_yan_diff(mats, select_main=True)
            # 24xxx from insumos
            auto_24 = {}
            for code in mats.keys():
                if str(code).isdigit():
                    lst = pulses_from_2024_insumos(str(code), args.temps_dir or "Datos Experimentales")
                    if lst:
                        auto_24[str(code)] = lst
            combined = dict(auto_sb); combined.update(auto_24)
            # filter to present assays
            present = set(mats.keys())
            pulses_by_assay = {k: v for k, v in combined.items() if k in present}
            if args.verbose and pulses_by_assay:
                print(f"[PULSES][AUTO] construidos para: {sorted(pulses_by_assay.keys())}")
        except Exception as e:
            if args.verbose:
                print(f"[PULSES][AUTO] fallo construcción automática: {e}")

    def sim_wrapped(p_real, t_meas, temp_segs, pulses, x0):
        from types import SimpleNamespace
        atol_vec = np.array([args.atol_x, args.atol_n, args.atol_g, args.atol_f, args.atol_e], dtype=float)
        # Read lag options via env (optional) for preview runs
        lag_mode = os.environ.get("SB_LAG_MODE", "none")
        try:
            lag_tau_h = float(os.environ.get("SB_LAG_TAU_H", "12.0"))
            lag_sens = float(os.environ.get("SB_LAG_SENS", "0.06"))
            lag_floor = float(os.environ.get("SB_LAG_FLOOR", "0.0"))
        except Exception:
            lag_tau_h, lag_sens, lag_floor = 12.0, 0.06, 0.0
        return simulate_on_grid(p_real, t_meas, temp_segs, pulses, x0, method=args.method, rtol=args.rtol, atol_vec=atol_vec, jacobian=args.jacobian, verbose=args.verbose, lag_mode=lag_mode, lag_tau_h=lag_tau_h, lag_sensitivity=lag_sens, lag_floor=lag_floor)

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
            pd = None
        X0 = first_valid("biomass_viable_gL") if pd is not None else _np.nan
        N0_mgL = first_valid("YAN") if pd is not None else _np.nan
        G0 = first_valid("Glucose") if pd is not None else _np.nan
        F0 = first_valid("Fructose") if pd is not None else _np.nan
        E0 = first_valid("Ethanol") if pd is not None else _np.nan
        # Convert YAN to g/L
        try:
            yoff = float(args.yan_offset_mgl)
        except Exception:
            yoff = 0.0
        if _np.isfinite(N0_mgL):
            N0_corr_mgL = max(0.0, float(N0_mgL) - yoff)
            N0 = N0_corr_mgL * 1e-3
        else:
            N0 = _np.nan
        # If G/F missing or clearly zero while density exists, estimate total sugar from density and split 50/50
        need_split = (not _np.isfinite(G0)) or (not _np.isfinite(F0)) or ((G0 + F0) <= 0)
        is_numeric_assay = str(code).isdigit()
        # IMPORTANT: For numeric 24xxx, Ethanol initial must start at 0.0 unless an explicit near-t0 lab value exists.
        # Since 24xxx only inject a final lab E, the first_valid(E) is actually the final value. Force E0=0.0.
        if is_numeric_assay:
            E0 = 0.0
        if pd is not None and "Densidad" in df.columns:
            try:
                dens = pd.to_numeric(df["Densidad"], errors="coerce").to_numpy(dtype=float)
                t = pd.to_numeric(df.get("time_h", _np.arange(len(dens))), errors="coerce").to_numpy(dtype=float)
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
                        # Apply override as follows:
                        # - For numeric 24xxx assays: if missing OR if mismatch >5%, override to 50/50
                        # - For SBxxx (non-numeric): only override when missing/zero (do NOT override measured values by mismatch)
                        mismatch = (_np.isfinite(G0) and _np.isfinite(F0)) and (abs((G0+F0) - S0) / max(S0, 1e-12) > 0.05)
                        if need_split or (is_numeric_assay and mismatch):
                            G0 = S0 * 0.5
                            F0 = S0 * 0.5
            except Exception:
                pass
        # Fallbacks with DEFAULT_X0
        x0_vec = _np.array([
            X0 if _np.isfinite(X0) and X0 >= 0 else DEFAULT_X0[0],
            N0 if _np.isfinite(N0) and N0 >= 0 else DEFAULT_X0[1],
            G0 if _np.isfinite(G0) and G0 >= 0 else DEFAULT_X0[2],
            F0 if _np.isfinite(F0) and F0 >= 0 else DEFAULT_X0[3],
            E0 if _np.isfinite(E0) and E0 >= 0 else DEFAULT_X0[4],
        ], dtype=float)
        return x0_vec

    os.makedirs(args.outdir, exist_ok=True)
    count = 0
    for code, df in mats.items():
        out = os.path.join(args.outdir, f"{code}.png")
        pulses = pulses_by_assay.get(code)
        # Build x0: first-valid strategy + sugar-from-density 50/50 split if needed
        try:
            import pandas as pd  # ensure pandas available locally
            x0_vec = _derive_x0_with_density(code, df)
        except Exception:
            x0_vec = DEFAULT_X0
        try:
            # Optional YAN correction for plotting
            df_plot = df
            try:
                if args.yan_offset_mgl and (args.yan_offset_mgl != 0.0) and ("YAN" in df.columns):
                    import pandas as pd
                    df_plot = df.copy()
                    yan = pd.to_numeric(df_plot["YAN"], errors="coerce").astype(float)
                    df_plot["YAN"] = np.maximum(0.0, yan - float(args.yan_offset_mgl))
            except Exception:
                df_plot = df
            plot_fit_for_assay(code, df_plot, p, sim_wrapped, pulses=pulses, x0=x0_vec, out_path=out)
            count += 1
            if args.verbose:
                print(f"[PLOT] {out}")
        except Exception as e:
            if args.verbose:
                print(f"[PLOT] fallo {code}: {e}")
            continue
    print(f"Listo: {count} plots en {args.outdir}")


if __name__ == "__main__":
    main()
