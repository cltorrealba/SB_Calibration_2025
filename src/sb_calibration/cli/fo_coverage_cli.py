"""CLI to report coverage of measurements used by the objective function.

Outputs a CSV with rows per assay and columns:
- counts per variable (X,N,G,F,E,S)
- time min/max

This helps verify that plotted points match those entering the FO, including SugarTotal_exp when present.
"""
from __future__ import annotations
import os
from typing import Dict, Any, List


def _load_mats_for_assays(assays: List[str]) -> Dict[str, Any]:
    mats: Dict[str, Any] = {}
    import pandas as pd
    for code in assays:
        # prefer cached parquet
        pq = f"mats/assay={code}.parquet"
        df = None
        if os.path.exists(pq):
            try:
                df = pd.read_parquet(pq)
            except Exception:
                df = None
        # If cached df is missing SugarTotal_exp, try to rebuild on the fly for numeric assays
        need_rebuild = (df is None) or ("SugarTotal_exp" not in getattr(df, 'columns', []))
        if need_rebuild:
            try:
                if str(code).isdigit():
                    from sb_calibration.preprocess.sw_2024 import build_mats_for_assay_2024
                    rebuilt = build_mats_for_assay_2024(str(code), "Datos Experimentales")
                    if rebuilt is not None:
                        df = rebuilt
                        # optionally refresh cache
                        try:
                            os.makedirs("mats", exist_ok=True)
                            df.to_parquet(pq)
                        except Exception:
                            pass
            except Exception:
                pass
        if df is not None:
            mats[str(code)] = df
    return mats


def _infer_split_assays(split_csv: str, which: str | None) -> List[str]:
    import pandas as pd
    if not os.path.exists(split_csv):
        return []
    try:
        df = pd.read_csv(split_csv)
    except Exception:
        return []
    if "assay" not in df.columns:
        return []
    if which is None or which.lower() == "all":
        return [str(a) for a in df["assay"].astype(str).tolist()]
    m = (df.get("split", pd.Series(["all"] * len(df))).astype(str).str.lower() == which.lower())
    return [str(a) for a in df.loc[m, "assay"].astype(str).tolist()]


def build_coverage(mats: Dict[str, Any]) -> "tuple[list[dict[str, Any]], list[str]]":
    rows: list[dict[str, Any]] = []
    cols = [
        "assay",
        "n_X", "n_N", "n_G", "n_F", "n_E", "n_S",
        "t_min_h", "t_max_h",
    ]
    for code, df in mats.items():
        try:
            import numpy as np
            t = df.get("time_h")
            tmin = float(np.nanmin(t)) if t is not None else float("nan")
            tmax = float(np.nanmax(t)) if t is not None else float("nan")
            def n_non_nan(col: str, scale: float = 1.0) -> int:
                if col not in df.columns:
                    return 0
                v = df[col]
                try:
                    import pandas as pd
                    vv = pd.to_numeric(v, errors="coerce").to_numpy(dtype=float) * scale
                except Exception:
                    vv = np.asarray(v, dtype=float) * scale
                return int(np.isfinite(vv).sum())
            row = {
                "assay": str(code),
                "n_X": n_non_nan("biomass_viable_gL"),
                "n_N": n_non_nan("YAN", scale=1e-3),
                "n_G": n_non_nan("Glucose"),
                "n_F": n_non_nan("Fructose"),
                "n_E": n_non_nan("Ethanol"),
                "n_S": n_non_nan("SugarTotal_exp"),
                "t_min_h": tmin,
                "t_max_h": tmax,
            }
            rows.append(row)
        except Exception:
            continue
    return rows, cols


def main(argv: list[str] | None = None) -> str | None:
    import argparse
    parser = argparse.ArgumentParser(description="FO coverage report (counts per variable and time range)")
    parser.add_argument("--assays", nargs="*", help="Assay codes to include; if empty, use split file")
    parser.add_argument("--split-file", default="splits/assay_split.csv", help="CSV with columns assay, split")
    parser.add_argument("--split", default="train", help="Which split to use if --assays empty: train|valid|all")
    parser.add_argument("--out", default="mats/fo_coverage_report.csv", help="Output CSV path")
    args = parser.parse_args(argv)

    assays = list(args.assays) if args.assays and len(args.assays) > 0 else _infer_split_assays(args.split_file, args.split)
    if not assays:
        print("[COVERAGE] No assays to process")
        return None
    mats = _load_mats_for_assays(assays)
    rows, cols = build_coverage(mats)
    if not rows:
        print("[COVERAGE] No rows produced")
        return None
    import pandas as pd
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(args.out, index=False)
    print(f"[COVERAGE] Wrote {args.out} with {len(df)} rows")
    return args.out


if __name__ == "__main__":
    main()
