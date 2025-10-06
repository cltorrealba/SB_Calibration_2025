"""CLI entrypoint for calibration orchestration (minimal, testable).

This module exposes `run_calibration` which can be invoked from tests or a simple CLI.
"""
from typing import Optional
import numpy as np
from ..calibration.optimize import calibrate_full
from ..model.zenteno import simulate_on_grid
from ..calibration.config import CalibrationConfig
from ..viz.plots import plot_fit_for_assay
from ..calibration.pulses import build_pulses_from_chem
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
                    split: str | None = None, split_file: str | None = None,
                    cache_2024: bool = True,
                    verbose: bool = False):
    """Run calibration using the real simulator. If `mats` is None, a small
    synthetic mats dict will be used for a quick smoke run.
    """
    pulses_by_assay = None
    built_2024_cached: list[str] = []
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
                    pulses_by_assay = build_pulses_from_chem(cdf)
                    # filter to present assays and apply exclusions
                    present = set(mats.keys())
                    if exclude:
                        present = present - set(exclude)
                    pulses_by_assay = {k: v for k, v in pulses_by_assay.items() if k in present}
                    if verbose:
                        print(f"[INFO] Pulses loaded for assays: {sorted(pulses_by_assay.keys())}")
        else:
            # small synthetic mats to keep smoke runs fast
            import pandas as pd
            mats = {"A": pd.DataFrame({"time_h": np.linspace(0, 10, 6), "biomass_viable_gL": np.linspace(0.5, 1.0, 6)})}

    # tiny defaults for a smoke run; for real calibration increase n_starts and local_maxiter
    # zenteno model expects 14 parameters
    p0 = np.ones(14)
    bounds = [(1e-6, 1e3)] * 14
    if cfg is None:
        cfg = CalibrationConfig()
        cfg.n_starts = 4
        cfg.local_maxiter = 20
    # wrap simulate_on_grid with tolerances
    def sim_wrapped(p, t_meas, temp_segs, pulses, x0):
        return simulate_on_grid(p, t_meas, temp_segs, pulses, x0, method=method, rtol=cfg.rtol, atol_vec=cfg.atol_vec(), jacobian=jacobian)

    pbest, score, meta = calibrate_full(
        mats,
        p0,
        bounds,
        sim_wrapped,
        mode=cfg.mode,
        pulses_by_assay=pulses_by_assay,
        weights=weights,
        n_starts=cfg.n_starts,
        local_maxiter=cfg.local_maxiter,
        patience_starts=cfg.patience_starts,
        patience_evals=cfg.patience_evals,
        min_improvement_rel=cfg.min_improvement_rel,
        out_path=out_path,
    )
    if verbose:
        print(f"[RESULT] SSE={score:.4e}  out={out_path}")
    # optional plotting
    if plot:
        for code, df in mats.items():
            pulses = (pulses_by_assay or {}).get(code)
            try:
                plot_fit_for_assay(code, df, pbest, sim_wrapped, pulses=pulses, x0=None, out_path=f"{plots_dir}/{code}.png")
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
    args = parser.parse_args()
    cfg = CalibrationConfig(
        mode=args.mode,
        n_starts=args.n_starts,
        local_maxiter=args.local_maxiter,
        rtol=args.rtol,
        atol_x=args.atol_x,
        atol_n=args.atol_n,
        atol_g=args.atol_g,
        atol_f=args.atol_f,
        atol_e=args.atol_e,
    )
    assays = [s.strip() for s in args.assays.split(',')] if args.assays else None
    exclude = [s.strip() for s in args.exclude.split(',')] if args.exclude else None
    weights = {"X": args.w_x, "N": args.w_n, "G": args.w_g, "F": args.w_f, "E": args.w_e}
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

    run_calibration(None, out_path=args.out, cfg=cfg, file_path=args.file, assays=assays, temps_dir=args.temps_dir, use_smoothed_biomass=args.use_smoothed_biomass, method=args.method, jacobian=args.jacobian, exclude=exclude, chem_file=args.chem_file, weights=weights, plot=args.plot, plots_dir=args.plots_dir, split=args.split, split_file=args.split_file, cache_2024=(not args.no_cache_2024), verbose=args.verbose)
