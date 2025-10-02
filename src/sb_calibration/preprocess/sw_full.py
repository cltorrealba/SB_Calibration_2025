"""SW-specific preprocessing helpers (reimplementation to continue migration).

This module provides a small set of utilities that mirror responsibilities from
the original `SW_Preprocess_data.py`. It intentionally re-uses functions from
`calibration_preprocess` and adds a couple of SW-specific helpers that are
easy to unit test.
"""
from typing import Dict, Optional, List
import numpy as np
import pandas as pd

from .calibration_preprocess import extract_assay, process_one_assay


def process_multiple_from_df(df_bdd: pd.DataFrame, assays: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Process a BDD-like DataFrame (already loaded) and return per-assay processed DFs.

    This function is a thin wrapper allowing tests to run without Excel IO.
    """
    # Ensure Ensayo_norm exists
    if "Ensayo_norm" not in df_bdd.columns and "Ensayo" in df_bdd.columns:
        # lazy import to avoid circulars
        from .calibration_preprocess import normalize_ensayo
        df_bdd = df_bdd.copy()
        df_bdd["Ensayo_norm"] = df_bdd["Ensayo"].apply(normalize_ensayo)

    codes = sorted(df_bdd["Ensayo_norm"].dropna().unique()) if assays is None else assays
    results: Dict[str, pd.DataFrame] = {}
    for code in codes:
        wide = extract_assay(df_bdd, code)
        if wide.empty:
            continue
        processed = process_one_assay(wide, code)
        results[code] = processed
    return results


def infer_inoculum_gL_from_concentration(first_total_conc: float, slope: float = 0.03692948069886653, intercept: float = 1.8737767179767686) -> float:
    """Infer inoculum (g/L) given a measured concentration value using the legacy conversion.

    This is a small helper used in the original preprocessing anchoring step.
    """
    total_gL_raw = slope * float(first_total_conc) + intercept
    # Remove intercept as original did to get adjusted total
    adjusted = max(total_gL_raw - intercept, 0.0)
    return adjusted
