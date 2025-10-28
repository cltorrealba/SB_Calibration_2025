import pandas as pd
import numpy as np
from datetime import datetime, time
from sb_calibration.preprocess import calibration_preprocess as cp


def _write_temp_excel(path, sheet_name, df):
    with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def test_separate_date_and_time_columns(tmp_path):
    # Create file with separate Fecha and Hora columns
    t0 = datetime(2025, 9, 1, 8, 0, 0)
    dates = [t0.date(), t0.date()]
    times = [time(8, 0, 0), time(9, 0, 0)]
    temps = [20.0, 21.0]
    dfT = pd.DataFrame({cp.TEMP_DATE_COL: dates, "Hora": times, cp.TEMP_VALUE_COL: temps})
    fpath = tmp_path / "Data 25026.xlsx"
    _write_temp_excel(fpath, cp.TEMP_SHEET, dfT)

    old_dir = cp.TEMPS_DIR
    cp.TEMPS_DIR = str(tmp_path)
    try:
        df_loaded = cp._load_temperature_table_for_assay("SB003")
        assert df_loaded is not None
        assert len(df_loaded) == 2

        # interpolate for a timestamp between 8:00 and 9:00
        sample_ts = [t0 + pd.Timedelta(minutes=30)]
        interp = cp._interp_temperature_for_assay(pd.DataFrame({"timestamp": sample_ts}), "SB003")
        # Expect an interpolated value between 20 and 21 (inclusive)
        assert 20.0 <= float(interp[0]) <= 21.0
    finally:
        cp.TEMPS_DIR = old_dir


def test_european_date_format(tmp_path):
    # Dates in day-first format (e.g., 01/09/2025)
    t0 = datetime(2025, 9, 1, 8, 0, 0)
    dfT = pd.DataFrame({cp.TEMP_DATE_COL: ["01/09/2025 08:00", "01/09/2025 09:00"], cp.TEMP_VALUE_COL: [20.0, 21.0]})
    fpath = tmp_path / "Data 25026.xlsx"
    _write_temp_excel(fpath, cp.TEMP_SHEET, dfT)

    old_dir = cp.TEMPS_DIR
    cp.TEMPS_DIR = str(tmp_path)
    try:
        df_loaded = cp._load_temperature_table_for_assay("SB003")
        assert df_loaded is not None and len(df_loaded) == 2
    finally:
        cp.TEMPS_DIR = old_dir


def test_missing_columns_fallback(tmp_path):
    # File exists but missing expected columns -> loader returns None
    dfT = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
    fpath = tmp_path / "Data 25026.xlsx"
    _write_temp_excel(fpath, cp.TEMP_SHEET, dfT)

    old_dir = cp.TEMPS_DIR
    cp.TEMPS_DIR = str(tmp_path)
    try:
        df_loaded = cp._load_temperature_table_for_assay("SB003")
        assert df_loaded is None
    finally:
        cp.TEMPS_DIR = old_dir
