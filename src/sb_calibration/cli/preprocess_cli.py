"""Preprocess CLI: construye matrices para 2024 y 2025.

Usos típicos:
 - Preconstruir 24xxx (2024) desde split: Data <ID>.xlsx -> mats/assay=<ID>.parquet
 - Construir SBxxx (2025) desde BDD + temperaturas -> mats/assay=SBxxx.parquet
"""
from __future__ import annotations
import os
from typing import List, Optional


def _read_assays_from_split(split_file: str, split_choice: Optional[str], verbose: bool = False) -> List[str]:
    try:
        import pandas as pd
    except Exception:
        return []
    if not os.path.exists(split_file):
        if verbose:
            print(f"[SPLIT] Archivo no encontrado: {split_file}")
        return []
    try:
        sdf = pd.read_csv(split_file)
    except Exception as e:
        if verbose:
            print(f"[SPLIT] Error leyendo {split_file}: {e}")
        return []
    if split_choice and split_choice.lower() in ("train", "valid"):
        sdf = sdf[sdf.get("split").str.lower() == split_choice.lower()]
    if sdf.empty or "assay" not in sdf.columns:
        if verbose:
            print("[SPLIT] Sin ensayos válidos en el split")
        return []
    assays = [str(a) for a in sdf["assay"].astype(str).tolist()]
    if verbose:
        print(f"[SPLIT] Ensayos seleccionados ({split_choice or 'all'}): {assays}")
    return assays


def build_2024_mats(assays: List[str], temps_dir: Optional[str], out_dir: str, fmt: str = "parquet", verbose: bool = False) -> List[str]:
    built: List[str] = []
    try:
        from sb_calibration.preprocess.sw_2024 import build_mats_for_assay_2024
        import pandas as pd
    except Exception:
        return built
    os.makedirs(out_dir, exist_ok=True)
    for code in sorted(set([a for a in assays if a.isdigit()])):
        try:
            dfmat = build_mats_for_assay_2024(code, temps_dir or "Datos Experimentales")
            if dfmat is None:
                if verbose:
                    print(f"[2024] No se pudo construir {code} (sin datos suficientes)")
                continue
            path = os.path.join(out_dir, f"assay={code}.{('csv' if fmt=='csv' else 'parquet')}")
            if fmt == "csv":
                dfmat.to_csv(path, index=False)
            else:
                dfmat.to_parquet(path)
            if verbose:
                print(f"[2024] Guardado {path}")
            built.append(code)
        except Exception as e:
            if verbose:
                print(f"[2024] Error en {code}: {e}")
            continue
    return built


def build_2025_mats(bdd_file: str, assays: Optional[List[str]], temps_dir: Optional[str], out_dir: str, use_smoothed_biomass: bool = False, fmt: str = "parquet", verbose: bool = False) -> List[str]:
    try:
        from sb_calibration.preprocess.calibration_preprocess import (
            process_all as pp_process_all,
            attach_temperature_to_results as pp_attach_T,
            build_calibration_matrices as pp_build_mats,
        )
        import sb_calibration.preprocess.calibration_preprocess as cpp
        import pandas as pd
        import numpy as np
    except Exception:
        return []
    if temps_dir:
        cpp.TEMPS_DIR = temps_dir
        if verbose:
            print(f"[2025] TEMPS_DIR = {cpp.TEMPS_DIR}")
    results, _ = pp_process_all(bdd_file, assays=assays)
    results_T = pp_attach_T(results)
    mats = pp_build_mats(results_T, use_smoothed_biomass=use_smoothed_biomass)
    os.makedirs(out_dir, exist_ok=True)
    built: List[str] = []
    for code, df in mats.items():
        try:
            path = os.path.join(out_dir, f"assay={code}.{('csv' if fmt=='csv' else 'parquet')}")
            if fmt == "csv":
                df.to_csv(path, index=False)
            else:
                df.to_parquet(path)
            if verbose:
                print(f"[2025] Guardado {path}")
            built.append(code)
        except Exception as e:
            if verbose:
                print(f"[2025] Error en {code}: {e}")
            continue
    return built


def main():
    import argparse
    p = argparse.ArgumentParser(description="Preprocesado — construir matrices 2024 (24xxx) y 2025 (SBxxx)")
    p.add_argument("--out", default="mats", help="Carpeta de salida (por defecto: mats)")
    p.add_argument("--format", default="parquet", choices=["parquet","csv"], help="Formato de salida")
    p.add_argument("--temps-dir", default=None, help="Carpeta con Data <ID>.xlsx (2024)")
    p.add_argument("--split", default=None, choices=["train","valid","all"], help="Split a utilizar para seleccionar ensayos")
    p.add_argument("--split-file", default="splits/assay_split.csv", help="Ruta al CSV de split")
    p.add_argument("--assays", default=None, help="Lista separada por comas de ensayos a construir (mezclados 24xxx/SBxxx)")
    p.add_argument("--prebuild-2024", action="store_true", help="Construir matrices 24xxx desde Data <ID>.xlsx")
    p.add_argument("--build-2025", action="store_true", help="Construir matrices SBxxx desde la BDD")
    p.add_argument("--file", default=None, help="Ruta al Excel BDD para 2025 (requerido si --build-2025)")
    p.add_argument("--use-smoothed-biomass", action="store_true", help="Usar MA(3) para biomasa en matrices 2025")
    p.add_argument("--verbose", action="store_true", help="Imprimir pasos detallados de construcción y guardado")
    args = p.parse_args()

    assays = None
    if args.assays:
        assays = [s.strip() for s in args.assays.split(',') if s.strip()]
    else:
        assays = _read_assays_from_split(args.split_file, args.split, verbose=args.verbose)

    if args.prebuild_2024:
        built24 = build_2024_mats(assays or [], args.temps_dir, args.out, fmt=args.format, verbose=args.verbose)
        print(f"[2024] Construidos: {len(built24)} -> {', '.join(built24) if built24 else '-'}")

    if args.build_2025:
        if not args.file:
            raise SystemExit("--file es requerido para --build-2025")
        # Filtra solo SBxxx si la lista viene mezclada
        assays25 = None
        if assays:
            assays25 = [a for a in assays if a.upper().startswith('SB')]
        built25 = build_2025_mats(args.file, assays25, args.temps_dir, args.out, use_smoothed_biomass=args.use_smoothed_biomass, fmt=args.format, verbose=args.verbose)
        print(f"[2025] Construidos: {len(built25)} -> {', '.join(built25) if built25 else '-'}")

    if not args.prebuild_2024 and not args.build_2025:
        print("Nada que hacer: usa --prebuild-2024 y/o --build-2025")


if __name__ == "__main__":
    main()
