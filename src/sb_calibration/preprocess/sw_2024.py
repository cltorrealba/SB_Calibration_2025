from __future__ import annotations
import os
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd


def _col_like(df: pd.DataFrame, *cands: str) -> Optional[str]:
    for c in df.columns:
        l = str(c).strip().lower()
        for cand in cands:
            if l == cand.lower():
                return c
    for c in df.columns:
        l = str(c).strip().lower()
        for cand in cands:
            if cand.lower() in l:
                return c
    return None


def _build_time_h_from_any(df: pd.DataFrame) -> Optional[np.ndarray]:
    # Try 'time_h' directly
    c = _col_like(df, 'time_h', 'tiempo_h', 'horas', 't_h')
    if c is not None:
        return pd.to_numeric(df[c], errors='coerce').to_numpy(dtype=float)
    # Try timestamp
    ts_c = _col_like(df, 'timestamp', 'fecha', 'datetime', 'fecha_muestra')
    if ts_c is not None:
        ts = pd.to_datetime(df[ts_c], errors='coerce')
        if ts.notna().any():
            t0 = ts.min()
            return ((ts - t0).dt.total_seconds() / 3600.0).to_numpy(dtype=float)
    return None


def _read_temperature_table(xlsx_path: str) -> Optional[pd.DataFrame]:
    try:
        # Try common sheet name used in legacy
        dfT = pd.read_excel(xlsx_path, sheet_name='Manual Temperaturas')
    except Exception:
        # Fallback: first sheet that contains a temp-like column
        try:
            xl = pd.ExcelFile(xlsx_path)
            for s in xl.sheet_names:
                df = xl.parse(s)
                if _col_like(df, 'temperatura', 'temperature_c', 'temp_c') is not None:
                    dfT = df
                    break
            else:
                return None
        except Exception:
            return None

    c_temp = _col_like(dfT, 'temperatura', 'temperature_c', 'temp_c')
    if c_temp is None:
        return None
    t = _build_time_h_from_any(dfT)
    if t is None:
        return None
    temp_c = pd.to_numeric(dfT[c_temp], errors='coerce').to_numpy(dtype=float)
    m = ~(np.isnan(t) | np.isnan(temp_c))
    if not m.any():
        return None
    df = pd.DataFrame({'time_h': t[m], 'Temperature_C': temp_c[m]})
    df = df.sort_values('time_h').drop_duplicates(subset=['time_h'])
    return df


def _read_signals_any(xlsx_path: str) -> Dict[str, pd.DataFrame]:
    """Scan sheets to find columns for YAN, Glucose, Fructose, Ethanol and return per-signal dataframes with time_h and value."""
    out: Dict[str, pd.DataFrame] = {}
    try:
        xl = pd.ExcelFile(xlsx_path)
    except Exception:
        return out
    targets = {
        'YAN': ('YAN',),
        'Glucose': ('glucose', 'glucosa'),
        'Fructose': ('fructose', 'fructosa'),
        'Ethanol': ('ethanol', 'alcohol'),
    }
    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet)
        except Exception:
            continue
        t = _build_time_h_from_any(df)
        if t is None:
            continue
        for key, cands in targets.items():
            if key in out:
                continue
            col = _col_like(df, *cands)
            if col is None:
                continue
            val = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
            m = ~(np.isnan(t) | np.isnan(val))
            if not m.any():
                continue
            dd = pd.DataFrame({'time_h': t[m], key: val[m]})
            dd = dd.sort_values('time_h').drop_duplicates(subset=['time_h'])
            out[key] = dd
        # Stop early if all found
        if set(out.keys()) == set(targets.keys()):
            break
    return out


def build_mats_for_assay_2024(assay_id: str, temps_dir: str = 'Datos Experimentales') -> Optional[pd.DataFrame]:
    """Build a calibration matrix for a 24xxx assay directly from its experimental Excel file.

    Best-effort flexible parsing: finds time and values for Temperature_C, YAN, Glucose, Fructose, Ethanol.
    Aligns signals onto a common time grid (from Temperature if available; otherwise union of available times).
    Returns a DataFrame with standard columns or None if the file cannot be parsed.
    """
    xlsx_path = os.path.join(temps_dir, f"Data {assay_id}.xlsx")
    if not os.path.exists(xlsx_path):
        return None

    dfT = _read_temperature_table(xlsx_path)
    sigs = _read_signals_any(xlsx_path)

    if dfT is not None:
        base_t = dfT['time_h'].to_numpy(dtype=float)
    else:
        # choose any available time as base (prefers YAN)
        for k in ('YAN', 'Glucose', 'Fructose', 'Ethanol'):
            if k in sigs:
                base_t = sigs[k]['time_h'].to_numpy(dtype=float)
                break
        else:
            return None

    base_t = np.asarray(base_t, dtype=float)
    base_t = np.unique(np.clip(base_t, a_min=0.0, a_max=None))
    mat = pd.DataFrame({'time_h': base_t})

    if dfT is not None:
        mat['Temperature_C'] = np.interp(base_t, dfT['time_h'].to_numpy(dtype=float), dfT['Temperature_C'].to_numpy(dtype=float))
    else:
        mat['Temperature_C'] = np.nan

    for k, dfk in sigs.items():
        mat[k] = np.interp(base_t, dfk['time_h'].to_numpy(dtype=float), dfk[k].to_numpy(dtype=float))

    # Standard columns expected by downstream code
    for c in ['biomass_viable_gL', 'biomass_dead_gL', 'AMMONIA', 'PAN', 'Glycerol']:
        if c not in mat.columns:
            mat[c] = np.nan

    # Reorder columns to canonical order when possible
    cols = [
        'time_h', 'biomass_viable_gL', 'biomass_dead_gL', 'YAN', 'AMMONIA', 'PAN',
        'Fructose', 'Glucose', 'Glycerol', 'Ethanol', 'Temperature_C'
    ]
    mat = mat.reindex(columns=cols)
    return mat
