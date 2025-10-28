import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sb_calibration.preprocess import calibration_preprocess as cp


def test_load_temperature_table_and_interp(tmp_path):
    # Prepare temporary folder and Excel file matching SB2ID for SB003 -> 25026
    tmpdir = tmp_path
    fname = "Data 25026.xlsx"
    fpath = tmpdir / fname

    # Build temperature table: 3 timestamps, hourly, temps [20, 21, 22]
    t0 = datetime(2025, 9, 1, 8, 0, 0)
    ts = [t0 + timedelta(hours=i) for i in range(3)]
    temps = [20.0, 21.0, 22.0]
    dfT = pd.DataFrame({cp.TEMP_DATE_COL: ts, cp.TEMP_VALUE_COL: temps})
    # Write to Excel
    with pd.ExcelWriter(str(fpath), engine="openpyxl") as writer:
        dfT.to_excel(writer, sheet_name=cp.TEMP_SHEET, index=False)

    # Point module to temp dir
    old_dir = cp.TEMPS_DIR
    cp.TEMPS_DIR = str(tmpdir)
    try:
        df_loaded = cp._load_temperature_table_for_assay("SB003")
        assert df_loaded is not None
        assert "ts_temp" in df_loaded.columns and "temp_C" in df_loaded.columns

        # Create a sample assay df with timestamps between the provided ones
        sample_ts = [t0 + timedelta(minutes=30), t0 + timedelta(hours=1, minutes=30)]
        df_assay = pd.DataFrame({"timestamp": sample_ts})
        interp = cp._interp_temperature_for_assay(df_assay, "SB003")
        # First should be ~20.5, second ~21.5
        assert np.isclose(interp[0], 20.5, atol=1e-6)
        assert np.isclose(interp[1], 21.5, atol=1e-6)
    finally:
        cp.TEMPS_DIR = old_dir


def test_interp_returns_nans_when_no_file(tmp_path):
    # Ensure that when file doesn't exist, we get NaNs
    old_dir = cp.TEMPS_DIR
    cp.TEMPS_DIR = str(tmp_path)
    try:
        df_assay = pd.DataFrame({"timestamp": [pd.Timestamp.now()]})
        interp = cp._interp_temperature_for_assay(df_assay, "SB003")
        assert np.isnan(interp).all()
    finally:
        cp.TEMPS_DIR = old_dir
