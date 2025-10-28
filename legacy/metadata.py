import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import warnings, os
warnings.filterwarnings("ignore", message="Parsing dates in", category=UserWarning)

# --- NUEVO: configuración mínima para cargar temperaturas 2025 ---
try:
    # Reusar constantes si existen en el módulo original
    from Calibration_data_preprocess import SB2ID, TEMPS_DIR, TEMP_SHEET, TEMP_DATE_COL, TEMP_VALUE_COL
except Exception:
    SB2ID = {}
    TEMPS_DIR = "Datos Experimentales"
    TEMP_SHEET = "Manual Temperaturas"
    TEMP_DATE_COL = "medicion_fecha"
    TEMP_VALUE_COL = "temperatura"

def _load_raw_temperature_2025(assay_code: str) -> Optional[pd.DataFrame]:
    """Lee archivo de temperatura bruto para ensayos 2025 (Calibration)."""
    ens_id = SB2ID.get(assay_code)
    if ens_id is None:
        return None
    fpath = os.path.join(TEMPS_DIR, f"Data {ens_id}.xlsx")
    if not os.path.isfile(fpath):
        return None
    try:
        dfT = pd.read_excel(fpath, sheet_name=TEMP_SHEET)
    except Exception:
        return None
    if TEMP_DATE_COL not in dfT.columns or TEMP_VALUE_COL not in dfT.columns:
        return None
    ts = pd.to_datetime(dfT[TEMP_DATE_COL], errors="coerce")
    val = pd.to_numeric(dfT[TEMP_VALUE_COL], errors="coerce")
    ok = ts.notna() & val.notna()
    if not ok.any():
        return None
    return pd.DataFrame({"ts": ts[ok], "temp_C": val[ok]})

def _ensure_temperature_column_2025(results_2025: Dict[str, Any]) -> Dict[str, Any]:
    """Inserta columna Temperature_C (interpolada linealmente) si falta en DF de año 2025."""
    out = {}
    for assay, df in results_2025.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            out[assay] = df
            continue
        if "Temperature_C" in df.columns and df["Temperature_C"].notna().any():
            out[assay] = df
            continue
        # Intentar cargar planilla de temperatura
        dfT = _load_raw_temperature_2025(assay)
        if dfT is None or dfT.empty:
            # Crear columna NaN (se diagnosticará luego)
            df2 = df.copy()
            df2["Temperature_C"] = np.nan
            out[assay] = df2
            print(f"[META-TEMP] 2025/{assay}: sin datos externos -> Temperature_C NaN")
            continue
        # Interpolar a eje interno (time_days o idx)
        df2 = df.copy()
        # Eje relativo de temperaturas
        dfT = dfT.sort_values("ts")
        t0 = dfT["ts"].iloc[0]
        dfT["th_rel"] = (dfT["ts"] - t0).dt.total_seconds()/3600.0
        if "timestamp" in df2.columns and df2["timestamp"].notna().any():
            samp_ts = pd.to_datetime(df2["timestamp"], errors="coerce")
            th_samp = (samp_ts - t0).dt.total_seconds()/3600.0
        elif "time_hours" in df2.columns:
            # Ya tiene horas internas -> alineamos escalando al rango de th_rel
            th_samp = pd.to_numeric(df2["time_hours"], errors="coerce")
        elif "time_days" in df2.columns:
            th_samp = pd.to_numeric(df2["time_days"], errors="coerce")*24.0
        else:
            th_samp = pd.Series(np.arange(len(df2), dtype=float))
        # Interpolación (clip a rango)
        x = dfT["th_rel"].to_numpy()
        y = dfT["temp_C"].to_numpy()
        xs = pd.to_numeric(th_samp, errors="coerce").to_numpy()
        xs = np.clip(xs, x.min(), x.max())
        temp_interp = np.interp(xs, x, y)
        df2["Temperature_C"] = temp_interp
        out[assay] = df2
        print(f"[META-TEMP] 2025/{assay}: temp rows={len(dfT)} mean={temp_interp.mean():.3f}")
    return out

def _extract_df_2025_entry(entry: Any):
    """
    Ahora 2025 = datos NO estandarizados (Calibration_data_preprocess):
    dict {assay: DataFrame} directamente.
    """
    return entry if isinstance(entry, pd.DataFrame) else None

def _extract_df_2024_entry(entry: Any):
    """
    Ahora 2024 = datos estandarizados (SW_Preprocess_data process_multiple):
    soporta:
      - dict bundle {'data': df, 'chem_df': df2, ...}
      - DataFrame directo
      - dict {assay_code: df}
    """
    if isinstance(entry, pd.DataFrame):
        return entry
    if isinstance(entry, dict):
        if "data" in entry and isinstance(entry["data"], pd.DataFrame):
            return entry["data"]
        for v in entry.values():
            if isinstance(v, pd.DataFrame):
                return v
    if isinstance(entry, (list, tuple)) and entry and isinstance(entry[0], pd.DataFrame):
        return entry[0]
    return None

def _first_valid_timestamp(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    for col in ["Fecha y hora", "timestamp", "__ts__"]:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce", utc=True)
            s = s.dropna()
            if not s.empty:
                return s.min()
    return None

def _compute_mean_temperature(df: pd.DataFrame) -> float:
    for col in ["Temperature_C", "Temperature"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                return float(s.mean())
    return float("nan")

def _extract_chem_df_for_2025(assay: str, chem_pack: Any):
    """
    chem_pack puede ser:
      - dict {assay: chem_df}
      - dict anidado con clave 'chem_df'
      - None
    """
    if chem_pack is None:
        return None
    if isinstance(chem_pack, dict):
        if assay in chem_pack:
            cand = chem_pack[assay]
            if isinstance(cand, pd.DataFrame):
                return cand
            if isinstance(cand, dict) and isinstance(cand.get("chem_df"), pd.DataFrame):
                return cand["chem_df"]
        # también soportar estructura process_multiple: {code: {'chem_df': df}}
        if isinstance(chem_pack.get(assay), dict):
            inner = chem_pack[assay]
            if isinstance(inner.get("chem_df"), pd.DataFrame):
                return inner["chem_df"]
    return None

def _compute_first_addition_time(chem_df: Optional[pd.DataFrame]) -> float:
    """
    Devuelve el menor time_h donde haya adición (nombre_insumo no NaN).
    """
    if chem_df is None or chem_df.empty:
        return float("nan")
    if "time_h" not in chem_df.columns:
        return float("nan")
    if "nombre_insumo" in chem_df.columns:
        sub = chem_df[chem_df["nombre_insumo"].notna()].copy()
    else:
        sub = chem_df.copy()
    if sub.empty:
        return float("nan")
    t = pd.to_numeric(sub["time_h"], errors="coerce")
    t = t.dropna()
    return float(t.min()) if not t.empty else float("nan")

# === NUEVO: Mapeo explícito Código (muestra adición) -> Ensayo 2025 ===
CODE_TO_ASSAY_2025 = {
    "ING25-SB003-3 (25026)": "SB003",
    "ING25-SB004-3  (25027)": "SB004",
    "ING25-SB005-3  (25028)": "SB005",
    "ING25-SB006-3  (25029)": "SB006",
    "ING25-SB007-6 (25085)": "SB007",
    "ING25-SB008-6 (25086)": "SB008",
    "ING25-SB009-4  Post Nutri": "SB009",
    "ING25-SB010-4  Post Nutri": "SB010",
    "ING25-SB011-5 (25170)": "SB011",
    "ING25-SB012-5 (25171)": "SB012",
}
ASSAY_TO_CODE_2025 = {v: k for k, v in CODE_TO_ASSAY_2025.items()}

def _get_df(entry: Any):
    if isinstance(entry, pd.DataFrame):
        return entry
    if isinstance(entry, dict) and isinstance(entry.get("data"), pd.DataFrame):
        return entry["data"]
    return None

def _get_chem_df_2024(entry: Any):
    if isinstance(entry, dict) and isinstance(entry.get("chem_df"), pd.DataFrame):
        return entry["chem_df"]
    return None

def _assay_t0_timestamp(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    # T0 para referencia de timing (prioridad: timestamp, Fecha y hora)
    for col in ["timestamp", "Fecha y hora", "__ts__"]:
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce", utc=True).dropna()
            if not ts.empty:
                return ts.min()
    return None

def _compute_tnut_2025(assay: str, df_assay: pd.DataFrame, chem_df25: Optional[pd.DataFrame]) -> float:
    if chem_df25 is None or chem_df25.empty:
        return np.nan
    code_target = ASSAY_TO_CODE_2025.get(assay)
    if not code_target:
        return np.nan
    if "Código" not in chem_df25.columns:
        return np.nan
    subset = chem_df25[chem_df25["Código"].astype(str).str.strip() == code_target.strip()].copy()
    if subset.empty:
        return np.nan
    if "timestamp" not in subset.columns:
        return np.nan
    ts_add = pd.to_datetime(subset["timestamp"], errors="coerce", utc=True).dropna()
    if ts_add.empty:
        return np.nan
    add_time = ts_add.min()
    t0 = _assay_t0_timestamp(df_assay)
    if t0 is None:
        return np.nan
    dt_h = (add_time - t0).total_seconds()/3600.0
    # Seguridad: ignorar negativos
    return float(dt_h) if dt_h >= 0 else np.nan

def _compute_tnut_2024_from_bundle(entry: Any) -> float:
    """
    Devuelve el primer tiempo de adición (time_h) para 2024
    (regla original restaurada: considerar adiciones con time_h > 0).
    """
    chem = _get_chem_df_2024(entry)
    if chem is None or chem.empty or "time_h" not in chem.columns:
        return np.nan
    t = pd.to_numeric(chem["time_h"], errors="coerce")
    t = t[(t > 0) & np.isfinite(t)]  # revertido desde (>12) a (>0)
    if t.empty:
        return np.nan
    return float(t.min())

def make_meta(results_2025: Dict[str, Any],
              results_2024: Dict[str, Any],
              chem_df_2025: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    rows = []
    debug = True
    def _dbg(m):
        if debug: print(f"[META-DBG] {m}")
    def t_bin(T):
        if T < 16: return "low"
        if T > 19: return "high"
        return "med"
    # NUEVO helpers de redondeo
    def _round_T_design(T):
        return int(np.rint(T)) if np.isfinite(T) else np.nan
    # CAMBIO: redondear tnut a múltiplos de 6 horas
    def _round_tnut_hours(tnut_h: float):
        if not np.isfinite(tnut_h) or tnut_h < 0:
            return np.nan
        return round(tnut_h / 6.0) * 6.0  # múltiplos de 6 h

    # 2025 primero (RICH)
    _dbg(f"Procesando año 2025 (ensayos={len(results_2025)})")
    for assay, raw in results_2025.items():
        df = _get_df(raw)
        if df is None or df.empty:
            _dbg(f"2025/{assay}: DF vacío")
            continue
        Tc = np.nan
        if "Temperature_C" in df:
            ts = pd.to_numeric(df["Temperature_C"], errors="coerce")
            if ts.notna().any():
                Tc = float(ts.mean())
        tnut_h = _compute_tnut_2025(assay, df, chem_df_2025)
        T_design = _round_T_design(Tc)
        tnut_h_rounded = _round_tnut_hours(tnut_h)
        if np.isfinite(T_design):
            if np.isfinite(tnut_h_rounded):
                condition = f"RICH_T{T_design}_A{int(tnut_h_rounded)}h"
            else:
                condition = f"RICH_T{T_design}"
        else:
            condition = "RICH_Tnan"
        batch = "lotNA"
        if "timestamp" in df:
            ts_assay = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna()
            if not ts_assay.empty: batch = ts_assay.min().strftime("lot%Y%m%d")
        elif "Fecha y hora" in df:
            ts_assay = pd.to_datetime(df["Fecha y hora"], errors="coerce", utc=True).dropna()
            if not ts_assay.empty: batch = ts_assay.min().strftime("lot%Y%m%d")
        rows.append(dict(
            assay=assay,
            year=2025,
            T_mean=Tc,
            T_bin=t_bin(Tc if np.isfinite(Tc) else 18.0),
            tnut=tnut_h,
            condition_id=condition,
            batch_id=batch
        ))
        _dbg(f"2025/{assay}: T_mean={Tc if np.isfinite(Tc) else 'NaN'} T_design={T_design if np.isfinite(T_design) else 'NaN'} "
             f"tnut_raw_h={tnut_h if np.isfinite(tnut_h) else 'NaN'} tnut_round_h={tnut_h_rounded if np.isfinite(tnut_h_rounded) else 'NaN'} cond={condition}")

    # 2024 (CCD)
    _dbg(f"Procesando año 2024 (ensayos={len(results_2024)})")
    for assay, entry in results_2024.items():
        df = _get_df(entry)
        if df is None or df.empty:
            _dbg(f"2024/{assay}: DF vacío")
            continue
        Tc = np.nan
        if "Temperature_C" in df:
            ts = pd.to_numeric(df["Temperature_C"], errors="coerce")
            if ts.notna().any():
                Tc = float(ts.mean())
        tnut_h = _compute_tnut_2024_from_bundle(entry)
        T_design = _round_T_design(Tc)
        tnut_h_rounded = _round_tnut_hours(tnut_h)
        if np.isfinite(T_design):
            if np.isfinite(tnut_h_rounded):
                condition = f"CCD_T{T_design}_tnut{int(tnut_h_rounded)}h"
            else:
                condition = f"CCD_T{T_design}"
        else:
            condition = "CCD_Tnan"
        batch = "lotNA"
        if "Fecha y hora" in df:
            ts_assay = pd.to_datetime(df["Fecha y hora"], errors="coerce", utc=True).dropna()
            if not ts_assay.empty: batch = ts_assay.min().strftime("lot%Y%m%d")
        rows.append(dict(
            assay=str(assay),
            year=2024,
            T_mean=Tc,
            T_bin=t_bin(Tc if np.isfinite(Tc) else 18.0),
            tnut=tnut_h,
            condition_id=condition,
            batch_id=batch
        ))
        _dbg(f"2024/{assay}: T_mean={Tc if np.isfinite(Tc) else 'NaN'} T_design={T_design if np.isfinite(T_design) else 'NaN'} "
             f"tnut_raw_h={tnut_h if np.isfinite(tnut_h) else 'NaN'} tnut_round_h={tnut_h_rounded if np.isfinite(tnut_h_rounded) else 'NaN'} cond={condition}")

    meta = pd.DataFrame(rows)
    if meta.empty:
        _dbg("Sin filas generadas.")
        return meta
    meta = meta.sort_values(["year", "condition_id", "assay"]).reset_index(drop=True)
    meta["replicate_id"] = meta.groupby(["year", "condition_id"]).cumcount() + 1
    _dbg(f"Meta construida: filas={len(meta)} NaN_T={int(meta['T_mean'].isna().sum())} NaN_tnut={int(meta['tnut'].isna().sum())}")
    return meta

# %% MAIN
if __name__ == "__main__":
    print("[META] Construyendo metadata 2024 (SW) + 2025 (Calibration)...")
    results_2024_raw = {}
    results_2025_raw = {}
    chem_df25 = None

    # 2024
    try:
        from SW_Preprocess_data import process_multiple as proc_sw
        codes_2024 = [str(i) for i in range(24018, 24032)]
        base_dir_2024 = "Datos Experimentales"
        pack24 = proc_sw(codes_2024, base_dir_2024)
        for code, bundle in pack24.items():
            df_main = bundle.get("data")
            if isinstance(df_main, pd.DataFrame) and not df_main.empty:
                results_2024_raw[code] = bundle
        print(f"[META] Ensayos 2024 cargados: {len(results_2024_raw)}")
    except Exception as e:
        print(f"[META] Aviso 2024: {e}")

    # 2025
    try:
        import Calibration_data_preprocess as cdp
        res25, chem_df25 = cdp.process_all()   # ahora retorna chem_df25
        from Calibration_data_preprocess import attach_temperature_to_results
        res25 = attach_temperature_to_results(res25)
        # Asegurar columna 'timestamp' en resultados si existe en chem_df25 (no se modifica DF original si ya la tiene)
        for code, df in res25.items():
            results_2025_raw[code] = df
        print(f"[META] Ensayos 2025 cargados: {len(results_2025_raw)}")
        if isinstance(chem_df25, pd.DataFrame):
            print(f"[META] chem_df25 filas: {len(chem_df25)} columnas: {list(chem_df25.columns)}")
    except Exception as e:
        print(f"[META] Aviso 2025: {e}")

    # Completar temperaturas faltantes 2025
    need_fix = [k for k, v in results_2025_raw.items()
                if isinstance(v, pd.DataFrame) and (("Temperature_C" not in v.columns) or v["Temperature_C"].isna().all())]
    if need_fix:
        print(f"[META-TEMP] Reintentando temperatura en {len(need_fix)} ensayos 2025.")
        subset = {k: results_2025_raw[k] for k in need_fix}
        fixed = _ensure_temperature_column_2025(subset)
        for k, df_new in fixed.items():
            results_2025_raw[k] = df_new

    # Metadata
    meta_df = make_meta(results_2025=results_2025_raw,
                        results_2024=results_2024_raw,
                        chem_df_2025=chem_df25)

    if meta_df.empty:
        print("[META] Sin datos para metadata.")
    else:
        print(f"[META] Filas metadata: {len(meta_df)}")
        print("\n[META] Resumen por condición:")
        print(meta_df.groupby(["year","condition_id"])
                     .size()
                     .reset_index(name="n_reps")
                     .to_string(index=False))
        print("\n[META] Preview:")
        print(meta_df.head(15).to_string(index=False))
        try:
            meta_df.to_csv("metadata_2024_2025.csv", index=False)
            print("\n[META] Exportado a metadata_2024_2025.csv")
        except Exception as e:
            print(f"[META] Export CSV error: {e}")
            print(f"[META] Export CSV error: {e}")
