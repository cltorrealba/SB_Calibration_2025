from __future__ import annotations
import os
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

    # Group by assay (legacy behavior): compute ΔYAN on consecutive rows PER ENSAYO
    pulses: Dict[str, List[Tuple[float, float]]] = {}
    # derive assay code column into a normalized SBxxx when possible
    if col_code is not None:
        dfc["__assay__"] = dfc[col_code].apply(_extract_assay_from_string)
    else:
        dfc["__assay__"] = None
    # if no assay info, treat as a single stream (UNKNOWN)
    if dfc["__assay__"].isna().all():
        dfc["__assay__"] = "UNKNOWN"
    for ass, grp in dfc.groupby("__assay__"):
        g = grp.sort_values(col_time)
        y = pd.to_numeric(g[col_yan], errors="coerce").to_numpy(dtype=float)
        t = pd.to_numeric(g[col_time], errors="coerce").to_numpy(dtype=float)
        if len(y) < 2:
            continue
        dy = np.diff(y)
        tm = 0.5 * (t[1:] + t[:-1])
        lst: List[Tuple[float, float]] = []
        for dyi, ti in zip(dy, tm):
            if np.isfinite(dyi) and dyi > 0 and np.isfinite(ti):
                lst.append((float(ti), float(dyi) / 1000.0))  # mg/L → g/L
        if lst:
            dfp = pd.DataFrame(lst, columns=["t", "dN"]).groupby("t", as_index=False).agg({"dN": "sum"}).sort_values("t")
            pulses[str(ass)] = list(dfp.itertuples(index=False, name=None))

    # ordenar por tiempo por ensayo
    for k in list(pulses.keys()):
        pulses[k] = sorted(pulses[k], key=lambda z: z[0])
    return pulses


def pulses_from_mats_yan_diff(
    mats: Dict[str, pd.DataFrame],
    select_main: bool = True,
    mid_win: Tuple[float, float] = (24.0, 72.0),
) -> Dict[str, List[Tuple[float, float]]]:
    """Emula el enfoque legacy a partir de las series YAN en las mats.

    - Para cada ensayo con columna YAN (mg/L) y time_h, calcula ΔYAN>0 y lo mapea a ΔN (g/L).
    - La hora del pulso se fija en el punto medio entre mediciones consecutivas (t_{i-1}, t_i).
    - Si select_main=True, se reduce a un solo pulso principal por ensayo (máximo ΔN en ventana [24,72] h,
      o en global si no hay en ventana).
    """
    out: Dict[str, List[Tuple[float, float]]] = {}
    for code, df in (mats or {}).items():
        if "time_h" not in df.columns or "YAN" not in df.columns:
            continue
        t = pd.to_numeric(df["time_h"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df["YAN"], errors="coerce").to_numpy(dtype=float)
        m = ~(np.isnan(t) | np.isnan(y))
        t, y = t[m], y[m]
        if len(y) < 2:
            continue
        order = np.argsort(t)
        t = t[order]; y = y[order]
        dy = np.diff(y)
        tm = 0.5 * (t[1:] + t[:-1])
        lst = [(float(ti), float(dyi) / 1000.0) for dyi, ti in zip(dy, tm) if np.isfinite(dyi) and dyi > 0 and np.isfinite(ti)]
        if not lst:
            continue
        if not select_main:
            dfp = pd.DataFrame(lst, columns=["t", "dN"]).groupby("t", as_index=False).agg({"dN": "sum"}).sort_values("t")
            out[str(code)] = list(dfp.itertuples(index=False, name=None))
            continue
        # seleccionar un pulso principal como en legacy
        lo, hi = mid_win
        mid = [p for p in lst if lo <= p[0] <= hi]
        if mid:
            main = max(mid, key=lambda z: z[1])
        else:
            main = max(lst, key=lambda z: z[1])
        if main[1] <= 0:
            main = max(lst, key=lambda z: z[0])
        out[str(code)] = [main]
    return out


def pulses_from_2024_insumos(assay_id: str, temps_dir: str = "Datos Experimentales") -> List[Tuple[float, float]]:
    """Construye pulsos para un ensayo 24xxx leyendo 'Data <ID>.xlsx'.

    Basado en legacy SW_Preprocess_data:
    - Lee 'Insumos Operacionales' (FDA, kg → mg YAN via 0.2) y 'Otros Insumos' (Vitaferm, g → mg YAN via 0.08)
    - Lee 'Manual Densidades' para mapear densidad de aplicación -> time_h
    - Obtiene volumen V_L desde 'Antecedentes' para expresar en mg/L; retorna ΔN en g/L (mg/L→g/L)
    """
    xlsx_path = os.path.join(temps_dir, f"Data {assay_id}.xlsx")
    if not os.path.exists(xlsx_path):
        return []
    try:
        xl = pd.ExcelFile(xlsx_path)
    except Exception:
        return []

    # Volumen (L)
    V_L = None
    try:
        ant = pd.read_excel(xl, sheet_name="Antecedentes")
        for col in ("ant_vino_estimado_l", "ant_volumen_l", "volumen_l"):
            if col in ant.columns:
                V_L = pd.to_numeric(ant[col].iloc[0], errors="coerce")
                if pd.notna(V_L) and V_L > 0:
                    V_L = float(V_L)
                    break
    except Exception:
        pass

    # Densidades -> (time_h, dens)
    def _read_dens():
        try:
            dman = pd.read_excel(xl, sheet_name="Manual Densidades")
        except Exception:
            return None
        ts = None
        for c in ("medicion_fecha", "fecha", "timestamp"):
            if c in dman.columns:
                ts = pd.to_datetime(dman[c], errors="coerce"); break
        if ts is None:
            return None
        cD = None
        for c in dman.columns:
            if str(c).strip().lower() in ("densidad", "density"):
                cD = c; break
        if cD is None:
            return None
        ok = ts.notna() & pd.to_numeric(dman[cD], errors="coerce").notna()
        df = pd.DataFrame({"__dt__": ts[ok], "dens": pd.to_numeric(dman[cD], errors="coerce")[ok]})
        df = df.sort_values("__dt__")
        if df.empty:
            return None
        t0 = df["__dt__"].iloc[0]
        df["time_h"] = (df["__dt__"] - t0).dt.total_seconds() / 3600.0
        return df[["time_h", "dens"]]

    dens_df = _read_dens()
    def _map_density_to_time(dens_appl: float) -> float:
        if dens_df is None or pd.isna(dens_appl):
            return np.nan
        j = int((dens_df["dens"] - float(dens_appl)).abs().idxmin())
        return float(dens_df.iloc[j]["time_h"])

    rows: List[Tuple[float, float]] = []
    # FDA en Insumos Operacionales
    try:
        ops = pd.read_excel(xl, sheet_name="Insumos Operacionales")
        ops.columns = [str(c).strip() for c in ops.columns]
        if {"insumo", "cantidad"}.issubset(ops.columns):
            sub_fda = ops.loc[ops["insumo"].astype(str).str.strip().str.lower() == "fda"].copy()
            for _, r in sub_fda.iterrows():
                kg = pd.to_numeric(r.get("cantidad"), errors="coerce")
                if pd.isna(kg) or kg <= 0:
                    continue
                dens_appl = pd.to_numeric(r.get("densidad"), errors="coerce")
                if pd.isna(dens_appl):
                    dens_appl = pd.to_numeric(r.get("densidad_aplicacion"), errors="coerce")
                t_h = _map_density_to_time(dens_appl)
                if pd.isna(t_h):
                    continue
                yan_mg_total = float(kg) * 1e6 * 0.2
                yan_mgL = (yan_mg_total / V_L) if (V_L and V_L > 0) else np.nan
                if pd.isna(yan_mgL):
                    continue
                rows.append((float(t_h), float(yan_mgL) / 1000.0))  # mg/L→g/L
    except Exception:
        pass

    # Vitaferm en Otros Insumos
    try:
        otros = pd.read_excel(xl, sheet_name="Otros Insumos")
        otros.columns = [str(c).strip() for c in otros.columns]
        if {"nombre", "cantidad"}.issubset(otros.columns):
            sub_vita = otros.loc[otros["nombre"].astype(str).str.strip().str.lower() == "vitaferm"].copy()
            for _, r in sub_vita.iterrows():
                g = pd.to_numeric(r.get("cantidad"), errors="coerce")
                if pd.isna(g) or g <= 0:
                    continue
                dens_appl = pd.to_numeric(r.get("densidad_aplicacion"), errors="coerce")
                t_h = _map_density_to_time(dens_appl)
                if pd.isna(t_h):
                    continue
                yan_mg_total = float(g) * 1000.0 * 0.08
                yan_mgL = (yan_mg_total / V_L) if (V_L and V_L > 0) else np.nan
                if pd.isna(yan_mgL):
                    continue
                rows.append((float(t_h), float(yan_mgL) / 1000.0))
    except Exception:
        pass

    if not rows:
        return []
    dfp = pd.DataFrame(rows, columns=["t", "dN"]).groupby("t", as_index=False).agg({"dN": "sum"}).sort_values("t")
    return list(dfp.itertuples(index=False, name=None))
