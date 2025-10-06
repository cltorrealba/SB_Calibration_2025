"""Pulses utility CLI

Funciones:
- template: crea un CSV de ejemplo con el esquema esperado (assay,time_h,dN_gL)
- from-chem: genera un CSV a partir de un archivo de química (Excel/CSV)
- verify: valida y resume un CSV de pulsos existente
"""
from __future__ import annotations
import os
from typing import Dict, List, Tuple, Optional
import numpy as np


def _flatten_pulses(pulses_by_assay: Dict[str, List[Tuple[float, float]]]) -> "tuple[list[str], list[float], list[float]]":
    assays: list[str] = []
    t: list[float] = []
    dn: list[float] = []
    for code, pairs in (pulses_by_assay or {}).items():
        for (th, dN) in pairs:
            assays.append(str(code))
            t.append(float(th))
            dn.append(float(dN))
    return assays, t, dn


def cmd_template(out_path: str) -> str:
    import pandas as pd
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.DataFrame({
        "assay": ["SB007", "SB007", "SB010"],
        "time_h": [24.0, 72.0, 48.0],
        "dN_gL": [0.15, 0.10, 0.20],
    })
    df.to_csv(out_path, index=False)
    return out_path


def cmd_from_chem(chem_file: str, out_path: str, verbose: bool = False) -> Optional[str]:
    import pandas as pd
    from sb_calibration.calibration.pulses import build_pulses_from_chem
    # lee química
    if chem_file.lower().endswith((".xls", ".xlsx")):
        cdf = pd.read_excel(chem_file)
    else:
        cdf = pd.read_csv(chem_file)
    pulses = build_pulses_from_chem(cdf) or {}
    if not pulses:
        if verbose:
            print("[PULSES] No se detectaron pulsos desde el archivo de química")
        return None
    assays, t, dn = _flatten_pulses(pulses)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame({"assay": assays, "time_h": t, "dN_gL": dn}).to_csv(out_path, index=False)
    if verbose:
        print(f"[PULSES] Guardado {out_path} con {len(t)} pulsos")
    return out_path


def _parse_generic_table(df_any):
    import pandas as pd
    dfp = df_any.copy()
    cols = {str(c).strip().lower(): c for c in dfp.columns}
    assay_col = cols.get("assay") or cols.get("ensayo_norm") or cols.get("ensayo")
    time_col = cols.get("time_h") or cols.get("t_h") or cols.get("t")
    val_gl = cols.get("dn_gl") or cols.get("dngl") or cols.get("delta_g_l") or cols.get("dn_gl_1")
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


def cmd_verify(csv_path: str, verbose: bool = False) -> dict:
    import pandas as pd
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    pulses = _parse_generic_table(df) or {}
    summary = {k: len(v) for k, v in pulses.items()}
    if verbose:
        print("[PULSES] Resumen por ensayo:")
        for k in sorted(summary):
            print(f" - {k}: {summary[k]} pulsos")
    return summary


def main():
    import argparse
    p = argparse.ArgumentParser(description="Utilidades para pulsos de YAN (CSV ↔ química)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_t = sub.add_parser("template", help="Crear CSV de ejemplo: assay,time_h,dN_gL")
    p_t.add_argument("--out", default=os.path.join("mats", "pulses_YAN.csv"))

    p_c = sub.add_parser("from-chem", help="Construir CSV desde archivo de química")
    p_c.add_argument("--chem-file", required=True)
    p_c.add_argument("--out", default=os.path.join("mats", "pulses_YAN.csv"))
    p_c.add_argument("--verbose", action="store_true")

    p_v = sub.add_parser("verify", help="Validar/resumir un CSV de pulsos")
    p_v.add_argument("--csv", default=os.path.join("mats", "pulses_YAN.csv"))
    p_v.add_argument("--verbose", action="store_true")

    args = p.parse_args()
    if args.cmd == "template":
        out = cmd_template(args.out)
        print(f"Creado template en {out} (edítalo con tus pulsos)")
    elif args.cmd == "from-chem":
        out = cmd_from_chem(args.chem_file, args.out, verbose=args.verbose)
        if out:
            print(f"Listo: {out}")
        else:
            print("No se generó CSV (sin pulsos detectados)")
    elif args.cmd == "verify":
        summary = cmd_verify(args.csv, verbose=args.verbose)
        total = sum(summary.values())
        print(f"CSV válido. Ensayos: {len(summary)}  Pulsos totales: {total}")


if __name__ == "__main__":
    main()
