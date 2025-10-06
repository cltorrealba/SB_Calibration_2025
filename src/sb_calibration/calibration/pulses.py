from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def _col_like(df: pd.DataFrame, *cands: str) -> Optional[str]:
    # exact lower match
    for c in df.columns:
        l = str(c).strip().lower()
        for cand in cands:
            if l == cand.lower():
                return c
    # contains fallback
    for c in df.columns:
        l = str(c).strip().lower()
        for cand in cands:
            if cand.lower() in l:
                return c
    return None


def _extract_assay_from_string(s: str) -> Optional[str]:
    import re
    m = re.search(r"(SB\d{3})", str(s))
    return m.group(1) if m else None


def build_pulses_from_chem(chem_df: pd.DataFrame) -> Dict[str, List[Tuple[float, float]]]:
    """Construye dict {assay: [(t_h, dN_gL), ...]} desde una planilla química flexible.

    - Intenta detectar columnas de código de muestra, YAN y tiempo (time_h o timestamp).
    - Si hay timestamp y no hay time_h, deriva time_h relativo.
    - Para cada fila 'actual', calcula delta YAN vs. fila anterior y lo convierte a dN_gL (mg/L → g/L).
    - Mapea el código de ensayo desde un string que contenga 'SBxxx'.
    """
    if chem_df is None or len(chem_df) == 0:
        return {}

    col_code = _col_like(chem_df, "Código", "codigo", "sample_id", "muestra")
    col_yan = _col_like(chem_df, "YAN", "yan", "nitrogeno", "n_asm")
    col_time = _col_like(chem_df, "time_h", "tiempo_h", "horas", "t_h")

    if col_yan is None:
        return {}

    dfc = chem_df.copy()
    # derive time_h from timestamp if needed
    if col_time is None:
        col_ts = _col_like(chem_df, "timestamp", "fecha", "datetime", "fecha_muestra")
        if col_ts is not None:
            ts = pd.to_datetime(dfc[col_ts], errors="coerce")
            t0 = ts.min()
            dfc["__time_h__"] = (ts - t0).dt.total_seconds() / 3600.0
            col_time = "__time_h__"
        else:
            dfc["__time_h__"] = 0.0
            col_time = "__time_h__"

    dfc[col_yan] = pd.to_numeric(dfc[col_yan], errors="coerce")
    dfc[col_time] = pd.to_numeric(dfc[col_time], errors="coerce")

    pulses: Dict[str, List[Tuple[float, float]]] = {}
    for i in range(len(dfc)):
        yan_cur = float(dfc.iloc[i][col_yan]) if pd.notna(dfc.iloc[i][col_yan]) else np.nan
        t_cur = float(dfc.iloc[i][col_time]) if pd.notna(dfc.iloc[i][col_time]) else 0.0
        if i > 0:
            yan_prev = float(dfc.iloc[i-1][col_yan]) if pd.notna(dfc.iloc[i-1][col_yan]) else np.nan
        else:
            yan_prev = np.nan

        if np.isfinite(yan_cur) and np.isfinite(yan_prev):
            dYAN_mgL = max(yan_cur - yan_prev, 0.0)
        else:
            dYAN_mgL = 0.0

        dN_gL = dYAN_mgL / 1000.0  # mg/L → g/L
        # map assay code
        if col_code is not None:
            assay = _extract_assay_from_string(dfc.iloc[i][col_code]) or "UNKNOWN"
        else:
            assay = "UNKNOWN"
        pulses.setdefault(assay, []).append((t_cur, dN_gL))

    # ordenar por tiempo por ensayo
    for k in list(pulses.keys()):
        pulses[k] = sorted(pulses[k], key=lambda z: z[0])
    return pulses
