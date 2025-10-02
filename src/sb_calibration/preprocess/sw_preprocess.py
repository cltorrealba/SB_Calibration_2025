"""Adapter for legacy SW_Preprocess_data functionality.

This module provides a small compatibility layer so existing callers can be
repointed to the new `preprocess` package. It intentionally delegates to
`calibration_preprocess` functions for now and can be expanded with the
original SW-specific parsing logic later.
"""
from typing import Dict, Optional, List
import pandas as pd

from .calibration_preprocess import (
    extract_assay,
    process_one_assay,
    process_all,
    attach_temperature_to_results,
    build_calibration_matrices,
)


def process_multiple(file_path: str, assays: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Process the main BDD and return per-assay processed dataframes.

    For now this delegates to `process_all` imported from the calibration_preprocess
    module. Keeps the same signature as the legacy script's top-level function.
    """
    results, combined = process_all(file_path, assays=assays)
    return results


def build_and_export_matrices(results: Dict[str, pd.DataFrame], out_dir: str) -> None:
    mats = build_calibration_matrices(results, use_smoothed_biomass=True)
    for code, df in mats.items():
        path = f"{out_dir}/calib_matrix_{code}.csv"
        df.to_csv(path, index=False)
