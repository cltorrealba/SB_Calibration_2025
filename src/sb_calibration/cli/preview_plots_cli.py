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
    results_dict, _ = pp_process_all(args.file, assays=assays)
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

    # choose parameters (p0 or pbest)
    p = None
    if args.use_best and os.path.exists(args.checkpoint):
        try:
            data = np.load(args.checkpoint)
            p = data.get("pbest")
        except Exception:
            p = None
    if p is None:
        # fallback to parameters from excel if available; else ones
        try:
            p = load_parameters_from_excel(args.p0_excel, param_set=args.p0_set)
            if args.verbose:
                print(f"[P0] cargado desde {args.p0_excel} (set={args.p0_set})")
        except Exception:
            p = np.ones(14, dtype=float)
            if args.verbose:
                print("[P0] usando vector de unos (14)")

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
        return simulate_on_grid(p_real, t_meas, temp_segs, pulses, x0, method=args.method, rtol=args.rtol, atol_vec=atol_vec, jacobian=args.jacobian, verbose=args.verbose)

    os.makedirs(args.outdir, exist_ok=True)
    count = 0
    for code, df in mats.items():
        out = os.path.join(args.outdir, f"{code}.png")
        pulses = pulses_by_assay.get(code)
        # Build x0 from first experimental row when possible
        try:
            t0_row = df.iloc[0]
            X0 = float(t0_row.get("biomass_viable_gL", np.nan))
            N0_mgL = float(t0_row.get("YAN", np.nan))
            G0 = float(t0_row.get("Glucose", np.nan))
            F0 = float(t0_row.get("Fructose", np.nan))
            E0 = float(t0_row.get("Ethanol", np.nan))
            N0 = (N0_mgL * 1e-3) if np.isfinite(N0_mgL) else np.nan  # mg/L -> g/L
            x0_vec = np.array([
                X0 if np.isfinite(X0) else DEFAULT_X0[0],
                N0 if np.isfinite(N0) else DEFAULT_X0[1],
                G0 if np.isfinite(G0) else DEFAULT_X0[2],
                F0 if np.isfinite(F0) else DEFAULT_X0[3],
                E0 if np.isfinite(E0) else DEFAULT_X0[4],
            ], dtype=float)
        except Exception:
            x0_vec = DEFAULT_X0
        try:
            plot_fit_for_assay(code, df, p, sim_wrapped, pulses=pulses, x0=x0_vec, out_path=out)
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
