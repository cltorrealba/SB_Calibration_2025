# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd


# =========================
# CONFIG / FLAGS
# =========================
# Redondeo consistente de tiempos
ROUND_TIME_DECIMALS = 6

# Reglas de recorte de laboratorio
ENFORCE_NONNEGATIVE_LAB = True          # descarta LAB con time_h < 0
ENFORCE_WITHIN_OPER_WINDOW = True       # recorta LAB a [0, t_end] si hay ventana operacional

# Política del inóculo si cae antes de t0
# 'keep'       -> insertar fila negativa (tal cual)
# 'clip_to_0'  -> insertar a time_h = 0.0
# 'drop'       -> no insertar si es negativo
INOCULUM_NEGATIVE_POLICY = "clip_to_0"


# =========================
# Debug helpers
# =========================
def _is_debug() -> bool:
    return os.environ.get("CALIB_PREPROC_DEBUG", "0") == "1"


def _dprint(*a, **kw):
    if _is_debug() or kw.pop("force", False):
        print(*a, **kw)


def _nn_summary(df: pd.DataFrame, cols: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in cols:
        if c in df.columns:
            out[c] = int(df[c].notna().sum())
    return out


# =========================
# Lista oficial de variables (ES)
# =========================
ALL_VARS: List[str] = [
    "Acetaldehído",
    "Acidez total",
    "Acidez volátil",
    "Ácido acético",
    "Ácido l-málico",
    "Ácido pirúvico",
    "Ácido tartárico",
    "Alcohol",
    "Amoníaco",
    "Antocianinas",
    "Azúcar Total mr",
    "Biomasa Total",
    "Biomasa viable",
    "Brix",
    "Densidad",
    "DO 280",
    "DO 420",
    "DO 520",
    "DO 620",
    "Fecha y hora",
    "Fructosa",
    "Glicerol",
    "Glucosa",
    "Índice de brotación",
    "PAN",
    "pH",
    "Polifenoles",
    "Polifenoles totales",
    "SO2 Libre",
    "SO2 Total",
    "Sulfito libre",
    "Sulfito total",
    "Taninos",
    "YAN",
]


# =========================
# Utilidades de columnas/parseo
# =========================
def _find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Retorna el nombre de la primera columna cuyo nombre contenga alguno de los patrones (case-insensitive)."""
    cols = [str(c) for c in df.columns]
    low = [c.lower() for c in cols]
    for patt in candidates:
        p = patt.lower()
        for i, l in enumerate(low):
            if p in l:
                return cols[i]
    return None


def _parse_dt_safely(series_like, *, prefer_excel_serial=False, dayfirst=True, tag=""):
    """
    Intenta parsear una serie de tiempos considerando estos casos:
    - Fechas ISO / string (pd.to_datetime)
    - Fechas seriales Excel (origin=1899-12-30)
    Heurística: elegimos el método que entregue un rango temporal razonable y más valores válidos.
    """
    # 1) parseo ISO
    s_raw = pd.Series(series_like)
    s_iso = pd.to_datetime(s_raw, errors="coerce", utc=True, dayfirst=dayfirst)
    iso_ok = s_iso.notna().mean() if len(s_iso) else 0.0
    iso_range_ok = 1.0 if (s_iso.min() is not pd.NaT and s_iso.max() is not pd.NaT and (s_iso.max() - s_iso.min()).total_seconds() >= 0) else 0.0

    # 2) parseo Excel serial
    # detecta si parece número excel (muchos valores numéricos > 40000)
    s_num = pd.to_numeric(s_raw, errors="coerce")
    frac_num = s_num.notna().mean() if len(s_num) else 0.0
    looks_excel = (frac_num > 0.5) and (s_num.dropna().median() > 40000)
    if looks_excel or prefer_excel_serial:
        s_xl = pd.to_datetime(s_num, errors="coerce", utc=True, origin="1899-12-30", unit="D")
    else:
        s_xl = pd.Series([pd.NaT] * len(s_raw), dtype="datetime64[ns, UTC]")
    xl_ok = s_xl.notna().mean() if len(s_xl) else 0.0
    xl_range_ok = 1.0 if (s_xl.min() is not pd.NaT and s_xl.max() is not pd.NaT and (s_xl.max() - s_xl.min()).total_seconds() >= 0) else 0.0

    _dprint(f"[DT-PARSE] ISO ok={iso_ok:.2f}, rangeOK={iso_range_ok:.2f} | XL ok={xl_ok:.2f}, rangeOK={xl_range_ok:.2f} | looks_excel={looks_excel}, frac_num={frac_num:.2f}")
    # regla de decisión
    choose_xl = False
    if looks_excel and xl_ok > 0 and xl_range_ok > 0:
        choose_xl = True
    elif iso_ok == 0 and xl_ok > 0:
        choose_xl = True
    elif prefer_excel_serial and xl_ok >= iso_ok:
        choose_xl = True

    if choose_xl:
        _dprint(f"[DT-PARSE] Se elige parseo Excel serial .")
        s = s_xl
    else:
        _dprint(f"[DT-PARSE] Se elige parseo ISO/dayfirst .")
        s = s_iso

    if len(s) and s.notna().any():
        _dprint(f"[DT-PARSE] Rango: {s.min()} → {s.max()}")
    return s


def _parse_datetime_col(df: pd.DataFrame, *candidates: str, prefer_excel_serial=False, tag=""):
    """
    Busca la primera columna existente en df dentro de 'candidates' y la parsea a tz-aware (UTC) con heurística.
    """
    for c in candidates:
        if c in df.columns:
            return _parse_dt_safely(df[c], prefer_excel_serial=prefer_excel_serial, dayfirst=True, tag=tag)
    return None


# =========================
# Interpolación SOLO operacional
# =========================
def _only_interpolate_operational(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpola SOLO variables operacionales (Temperature_C y Densidad/density).
    NO interpola variables químicas.
    """
    if "Temperature_C" in df.columns:
        df["Temperature_C"] = df["Temperature_C"].interpolate(method="linear", limit_direction="both")
    if "density" in df.columns:
        df["density"] = df["density"].interpolate(method="linear", limit_direction="both")
    if "Densidad" in df.columns:
        df["Densidad"] = df["Densidad"].interpolate(method="linear", limit_direction="both")
    return df


# =========================
# Laboratorio → señales
# =========================
def _extract_signals_from_lab(lab_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae señales desde 'Laboratorio':
    - Nombre: variable_text -> fallback template_variable_text.
    - Valor: valor_numeric -> parseo de 'valor'.
    - Mapea a ALL_VARS (en ES).
    - Agrega por timestamp (__dt__).
    """
    if lab_df is None or len(lab_df) == 0:
        _dprint("[LAB] vacío")
        return pd.DataFrame()
    if "__dt__" not in lab_df.columns:
        _dprint("[LAB] no tiene __dt__ (timestamp)")
        return pd.DataFrame()

    # Columna de nombre
    if "variable_text" in lab_df.columns and lab_df["variable_text"].astype(str).str.strip().ne("").any():
        name_col = "variable_text"
    elif "template_variable_text" in lab_df.columns and lab_df["template_variable_text"].astype(str).str.strip().ne("").any():
        name_col = "template_variable_text"
    else:
        _dprint("[LAB] no encontré columna de nombre (variable_text/template_variable_text)")
        return pd.DataFrame()

    def _norm(s):
        if s is None:
            return ""
        s = str(s).strip().lower()
        s = (
            s.replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
            .replace("ñ", "n")
        )
        s = s.replace("-", " ")
        s = re.sub(r"\s+", " ", s)
        return s

    MAP = {
        "acetaldehido": "Acetaldehído",
        "acidez total": "Acidez total",
        "acidez volatil": "Acidez volátil",
        "acido acetico": "Ácido acético",
        "acido l malico": "Ácido l-málico",
        "acido malico": "Ácido l-málico",
        "acido piruvico": "Ácido pirúvico",
        "acido tartarico": "Ácido tartárico",
        "alcohol": "Alcohol",
        "amoniaco": "Amoníaco",
        "antocianinas": "Antocianinas",
        "azucar total mr": "Azúcar Total mr",
        "biomasa total": "Biomasa Total",
        "biomasa viable": "Biomasa viable",
        "brix": "Brix",
        "densidad": "Densidad",
        "do 280": "DO 280",
        "do280": "DO 280",
        "do 420": "DO 420",
        "do420": "DO 420",
        "do 520": "DO 520",
        "do520": "DO 520",
        "do 620": "DO 620",
        "do620": "DO 620",
        "fructosa": "Fructosa",
        "glicerol": "Glicerol",
        "glucosa": "Glucosa",
        "indice de brotacion": "Índice de brotación",
        "pan": "PAN",
        "ph": "pH",
        "polifenoles totales": "Polifenoles totales",
        "polifenoles": "Polifenoles",
        "so2 libre": "SO2 Libre",
        "so2 total": "SO2 Total",
        "sulfito libre": "Sulfito libre",
        "sulfito total": "Sulfito total",
        "taninos": "Taninos",
        "yan": "YAN",
        "fecha y hora": "Fecha y hora",
    }

    def _parse_val(row) -> float:
        vn = row.get("valor_numeric", None)
        if vn is not None:
            try:
                v = float(vn)
                if not pd.isna(v):
                    return v
            except Exception:
                pass
        vs = row.get("valor", None)
        if vs is None:
            return np.nan
        s = str(vs).strip().replace("\xa0", " ").replace(",", ".")
        for tok in ("%", "≈", "~", "<", ">"):
            s = s.replace(tok, "")
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if not m:
            return np.nan
        try:
            return float(m.group(0))
        except Exception:
            return np.nan

    rows = []
    miss_map = {}
    for _, row in lab_df.iterrows():
        ts = row.get("__dt__")
        if pd.isna(ts):
            continue
        raw_name = str(row.get(name_col, "")).strip()
        nm = _norm(raw_name)

        out = None
        for patt, target in sorted(MAP.items(), key=lambda kv: -len(kv[0])):
            if patt in nm:
                out = target
                break
        if out is None:
            miss_map[nm] = miss_map.get(nm, 0) + 1
            continue

        v = _parse_val(row)
        if pd.isna(v):
            continue

        rows.append({"__dt__": ts, out: v})

    if miss_map:
        _dprint("[LAB] Variables no mapeadas (normalizadas -> count):", sorted(miss_map.items())[:20])

    if not rows:
        _dprint("[LAB] tras parseo, no se generaron filas de señales")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("__dt__")
    agg = {c: "mean" for c in df.columns if c != "__dt__"}
    out = df.groupby("__dt__", as_index=False).agg(agg)
    _dprint("[LAB] señales construidas. Rangos __dt__", out["__dt__"].min(), "→", out["__dt__"].max())
    _dprint("[LAB] columnas:", list(out.columns))
    _dprint("[LAB] no-nulos:", _nn_summary(out, [c for c in out.columns if c != "__dt__"]))
    return out


# =========================
# Biomasa: suavizado e inóculo
# =========================
def _add_smoothed_biomass_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea columnas de biomasa:
      - 'biomass_viable_gL', 'biomass_total_gL'
      - 'biomass_viable_smoothed_gL' = media móvil centrada (ventana=3) solo donde hay datos observados
    """
    s_viab = df.get("Biomasa viable")
    s_tot = df.get("Biomasa Total")

    if s_viab is not None:
        df["biomass_viable_gL"] = s_viab
        sv = s_viab.rolling(window=3, center=True, min_periods=1).mean()
        df["biomass_viable_smoothed_gL"] = sv.where(s_viab.notna())

    if s_tot is not None:
        df["biomass_total_gL"] = s_tot

    return df


def _estimate_inoculum_biomass(xls: pd.ExcelFile, t0) -> Tuple[Optional[float], Optional[float]]:
    """
    Estima biomasa inicial desde:
      - Antecedentes.ant_vino_estimado_l (volumen L)
      - Insumos Operacionales (insumo 'Levadura'): cantidad + unidad
    Retorna (inoc_time_h, inoc_gL) o (None, None).
    """
    try:
        ant = pd.read_excel(xls, sheet_name="Antecedentes")
    except Exception:
        return (None, None)

    V_L = None
    for col in ["ant_vino_estimado_l", "ant_volumen_l", "volumen_l"]:
        if col in ant.columns:
            V_L = pd.to_numeric(ant[col].iloc[0], errors="coerce")
            break

    try:
        ops = pd.read_excel(xls, sheet_name="Insumos Operacionales")
    except Exception:
        return (None, None)
    if ops is None or ops.empty:
        return (None, None)

    ops2 = ops.copy()
    ops2.columns = [str(c).strip() for c in ops2.columns]

    name_col = "insumo" if "insumo" in ops2.columns else None
    if name_col is None:
        return (None, None)

    dt_col = None
    for c in ["fecha_proceso_format", "fecha", "fecha_aplicacion_format"]:
        if c in ops2.columns:
            dt_col = c
            break

    mask_lev = ops2[name_col].astype(str).str.contains("levadura", case=False, na=False)
    lev = ops2.loc[mask_lev].copy()
    _dprint("[INOC] Filas con 'Levadura':", len(lev))
    if lev.empty:
        return (None, None)

    lev = lev.sort_values(dt_col if dt_col else lev.index).iloc[0:1]
    if dt_col:
        lev["_dt_raw"] = pd.to_datetime(lev[dt_col], errors="coerce", utc=True)
        _dprint("[INOC] 5 fechas parseadas (utc) en 'Levadura':", lev["_dt_raw"].astype(str).head(5).tolist())

    qty = None
    unit = None
    for c in ["cantidad", "Cantidad"]:
        if c in lev.columns:
            qty = pd.to_numeric(lev[c].iloc[0], errors="coerce")
            break
    for c in ["unidad", "unidad_text", "Unidad", "unidad_medida"]:
        if c in lev.columns:
            unit = str(lev[c].iloc[0]).strip().lower()
            break
    _dprint("[INOC] qty -> col='cantidad', valor=", qty, "| unidad -> col='unidad', valor='", unit, "'", sep="")

    if qty is None or pd.isna(qty):
        return (None, None)

    inoc_gL = None
    rule = None
    if unit in {"g/hl", "g / hl", "g por hl", "g/hL", "g por hL"}:
        inoc_gL = float(qty) / 100.0
        rule = "g/hL → g/L"
    elif unit in {"mg/l", "ppm"}:
        inoc_gL = float(qty) / 1000.0
        rule = "mg/L → g/L"
    elif unit in {"g", "gr", "gramo", "gramos"}:
        if V_L and V_L > 0:
            inoc_gL = float(qty) / float(V_L)
            rule = "g totales / V_L"
    elif unit in {"(kg)","kg", "kilogramo", "kilogramos"}:
        if V_L and V_L > 0:
            inoc_gL = float(qty) * 1000.0 / float(V_L)
            rule = "kg totales / V_L"
    else:
        if unit and ("g" in unit and "hl" in unit):
            inoc_gL = float(qty) / 100.0
            rule = "g*hL detectado → g/L"

    _dprint(f"[INOC] Regla usada: {rule} | inoc_gL (pre-check) = {inoc_gL}")

    inoc_dt = None
    if dt_col and dt_col in lev.columns:
        inoc_dt = pd.to_datetime(lev[dt_col].iloc[0], errors="coerce", utc=True)

    if inoc_dt is not None and not pd.isna(inoc_dt):
        inoc_time_h = float(((inoc_dt - pd.to_datetime(t0, utc=True)).total_seconds()) / 3600.0)
    else:
        inoc_time_h = 0.0

    _dprint("[INOC] t0(utc)=", pd.to_datetime(t0, utc=True), "| inoc_dt(utc)=", inoc_dt, sep="")
    _dprint("[INOC] inoc_time_h (horas) =", inoc_time_h)
    if inoc_gL is None or inoc_gL <= 0:
        return (inoc_time_h, None)

    return (inoc_time_h, float(inoc_gL))


# =========================
# Pipeline principal
# =========================
def process_all(file_path: str, assays: Optional[List[str]] = None):
    """
    Devuelve:
      - results_dict: {assay_code: DataFrame} con:
        time_h, Temperature_C, Densidad, TODAS las de ALL_VARS, 'Fecha y hora'
      - chem_df: tabla plana de YAN (para pulsos)
    """
    xls = pd.ExcelFile(file_path)

    # Assay code
    try:
        ant = pd.read_excel(xls, sheet_name="Antecedentes")
        assay_code = str(ant.get("ant_cubada").iloc[0]).strip() if "ant_cubada" in ant.columns else "ASSAY"
    except Exception:
        assay_code = "ASSAY"
    _dprint(f"[ASSAY] code: {assay_code}")

    # Carga hojas base
    lab = pd.read_excel(xls, sheet_name="Laboratorio")
    tman = pd.read_excel(xls, sheet_name="Manual Temperaturas")
    dman = pd.read_excel(xls, sheet_name="Manual Densidades")
    _dprint("[LOAD] filas -> lab:", len(lab), "temp:", len(tman), "dens:", len(dman))

    # Parse timestamps (tz-aware)
    lab = lab.copy()
    tman = tman.copy()
    dman = dman.copy()

    dt_lab = _parse_datetime_col(lab, "create_fecha", "fecha", "fecha_muestra")
    if dt_lab is not None:
        lab["__dt__"] = dt_lab
    tman["__dt__"] = _parse_datetime_col(tman, "medicion_fecha")
    dman["__dt__"] = _parse_datetime_col(dman, "medicion_fecha")

    _dprint("[TS] lab dt:", (lab["__dt__"].min() if "__dt__" in lab.columns else None), "→",
            (lab["__dt__"].max() if "__dt__" in lab.columns else None))
    _dprint("[TS] tman dt:", (tman["__dt__"].min() if "__dt__" in tman.columns else None), "→",
            (tman["__dt__"].max() if "__dt__" in tman.columns else None))
    _dprint("[TS] dman dt:", (dman["__dt__"].min() if "__dt__" in dman.columns else None), "→",
            (dman["__dt__"].max() if "__dt__" in dman.columns else None))

    # Nombres operacionales
    cT = _find_col(tman, "temperatura", "temp", "t (c)", "tempertura")
    if cT:
        tman.rename(columns={cT: "Temperature_C"}, inplace=True)
        tman["Temperature_C"] = pd.to_numeric(tman["Temperature_C"], errors="coerce")

    cD = _find_col(dman, "densidad")
    if cD:
        dman.rename(columns={cD: "density"}, inplace=True)
        dman["density"] = pd.to_numeric(dman["density"], errors="coerce")

    # t0 = mínimo timestamp de TEMP/DENS; si no hay, usar LAB
    ts_candidates = []
    for df_ in (tman, dman):
        if df_ is not None and "__dt__" in df_.columns:
            ts_candidates.append(df_["__dt__"].dropna())
    if not ts_candidates and lab is not None and "__dt__" in lab.columns:
        ts_candidates.append(lab["__dt__"].dropna())
    if not ts_candidates:
        _dprint("[T0] no hay timestamps en ninguna hoja → vacío")
        return {}, pd.DataFrame()

    t_all = pd.concat(ts_candidates)
    t0 = t_all.min()
    _dprint("[T0] =", t0)

    # Señales individuales → time_h
    temp_df = None
    if "__dt__" in tman.columns and "Temperature_C" in tman.columns:
        temp_df = tman[["__dt__", "Temperature_C"]].dropna().copy()
        temp_df["time_h"] = (temp_df["__dt__"] - t0).dt.total_seconds() / 3600.0
        temp_df.drop(columns="__dt__", inplace=True)
        _dprint("[TEMP] rows:", len(temp_df), "time_h min/max:",
                temp_df["time_h"].min(), temp_df["time_h"].max())

    dens_df = None
    if "__dt__" in dman.columns and "density" in dman.columns:
        dens_df = dman[["__dt__", "density"]].dropna().copy()
        dens_df["time_h"] = (dens_df["__dt__"] - t0).dt.total_seconds() / 3600.0
        dens_df.drop(columns="__dt__", inplace=True)
        _dprint("[DENS] rows:", len(dens_df), "time_h min/max:",
                dens_df["time_h"].min(), dens_df["time_h"].max())

    # Extraer señales de laboratorio (químicos) agregadas por timestamp
    sig = _extract_signals_from_lab(lab) if lab is not None else pd.DataFrame()
    if not sig.empty:
        sig["time_h"] = (sig["__dt__"] - t0).dt.total_seconds() / 3600.0
        _dprint("[LAB→sig] rows:", len(sig), "time_h min/max:",
                sig["time_h"].min(), sig["time_h"].max())
        _dprint("[LAB→sig] nn per col (top):",
                {k: int(v) for k, v in sig.drop(columns=["time_h", "__dt__"], errors="ignore")
                 .notna().sum().sort_values(ascending=False).head(10).items()})

    # ========= PRE0 (lab < t0 → fila 0) =========
    # Guardamos payload para asegurar relleno en la fila 0 tras el merge.
    pre0_payload: Dict[str, float] = {}
    if not sig.empty and "time_h" in sig.columns:
        neg_mask = sig["time_h"] < 0
        pos_mask = sig["time_h"] >= 0
        _dprint("[PRE0] negativos:", int(neg_mask.sum()), "| positivos:", int(pos_mask.sum()))
        if neg_mask.any():
            closest_idx = sig.loc[neg_mask, "time_h"].idxmax()  # más cercano a 0
            pre0_row = sig.loc[[closest_idx]].copy()
            pre0_row["time_h"] = 0.0
            # payload = todas las columnas químicas no nulas de ese muestreo
            for c in pre0_row.columns:
                if c in ("time_h", "__dt__"):
                    continue
                v = pre0_row[c].iloc[0]
                if pd.notna(v):
                    pre0_payload[c] = v
            _dprint("[PRE0] idx plegado:", int(closest_idx),
                    "| __dt__:", sig.loc[closest_idx, "__dt__"],
                    "| keys payload:", sorted(list(pre0_payload.keys()))[:12],
                    ("... (+%d)" % (len(pre0_payload) - 12) if len(pre0_payload) > 12 else ""))

            # combinamos: positivos + fila 0 de pre0 y colapsamos por time_h
            sig_pos = sig.loc[pos_mask].copy()
            sig_fold = pd.concat([sig_pos, pre0_row], ignore_index=True)

            def _first_valid(s):
                s2 = s.dropna()
                return s2.iloc[0] if len(s2) else np.nan

            sig = sig_fold.groupby("time_h", as_index=False).agg(_first_valid)
            _dprint("[PRE0] aplicado. ¿existe time_h=0 en sig?:",
                    bool((np.isclose(sig["time_h"], 0.0)).any()))

    # quitamos __dt__ definitivamente
    if not sig.empty and "__dt__" in sig.columns:
        sig = sig.drop(columns="__dt__")

    # Recorte temporal de laboratorio al rango operacional (0 ≤ time_h ≤ t_end)
    t_end = None
    if temp_df is not None and len(temp_df) > 0:
        t_end = float(temp_df["time_h"].max())
    if dens_df is not None and len(dens_df) > 0:
        t_end = max(t_end or 0.0, float(dens_df["time_h"].max()))

    if not sig.empty:
        _dprint("[SIG] time_h range:", sig["time_h"].min(), "→", sig["time_h"].max(),
                "| negativos:", int((sig["time_h"] < 0).sum()),
                "| positivos:", int((sig["time_h"] >= 0).sum()))
    if t_end is not None and not sig.empty:
        before = len(sig)
        sig = sig.loc[(sig["time_h"] >= -1e-6) & (sig["time_h"] <= t_end + 1e-6)].copy()
        _dprint(f"[CLIP] sig: {before} -> {len(sig)} (0 ≤ time_h ≤ {t_end})")
    elif t_end is None and not sig.empty:
        q95 = float(sig["time_h"].quantile(0.95))
        before = len(sig)
        sig = sig.loc[(sig["time_h"] >= -1e-6) & (sig["time_h"] <= q95)].copy()
        _dprint(f"[CLIP] sig by q95: {before} -> {len(sig)} (0 ≤ time_h ≤ {q95})")

    # Redondeo simétrico para emparejar
    def _round_time_h(df: Optional[pd.DataFrame], col="time_h", nd=6) -> Optional[pd.DataFrame]:
        if df is None or col not in df.columns:
            return df
        df[col] = np.round(df[col].astype(float), nd)
        return df
    temp_df = _round_time_h(temp_df, "time_h", 6)
    dens_df = _round_time_h(dens_df, "time_h", 6)
    sig = _round_time_h(sig, "time_h", 6)

    # Construcción de grilla de tiempos: unión de TEMP, DENS y LAB
    series_times = []
    if temp_df is not None and len(temp_df) > 0:
        series_times.append(temp_df["time_h"])
    if dens_df is not None and len(dens_df) > 0:
        series_times.append(dens_df["time_h"])
    if not sig.empty:
        series_times.append(sig["time_h"])

    if not series_times:
        _dprint("[GRID] sin tiempos -> vacío")
        return {}, pd.DataFrame()

    t_grid = np.unique(np.concatenate([st.dropna().values for st in series_times]))
    merged = pd.DataFrame({"time_h": t_grid})
    merged["time_h"] = np.round(merged["time_h"].astype(float), 6)
    _dprint("[GRID] len:", len(merged), "range:", merged["time_h"].min(), "→", merged["time_h"].max())

    # Insertar laboratorio (left join exacto por time_h)
    if not sig.empty:
        lab_times_unique = np.unique(sig["time_h"].dropna().values)
        hits = np.intersect1d(lab_times_unique, merged["time_h"].values).size
        _dprint(f"[LAB MERGE] tiempos únicos lab: {len(lab_times_unique)} | "
                f"matchean con grilla: {hits} | no matchean: {len(lab_times_unique)-hits}")
        merged = merged.merge(sig, on="time_h", how="left")
        _dprint("[MERGE] tras lab: nn:",
                {k: int(v) for k, v in merged.drop(columns=["time_h"], errors="ignore")
                 .notna().sum().sort_values(ascending=False).head(10).items()})

    # Interpolación SOLO operacional (T y D)
    def _interp_onto_grid(src: Optional[pd.DataFrame], col: str, grid: pd.DataFrame) -> pd.Series:
        if src is None or len(src) == 0 or col not in src.columns:
            return pd.Series(index=grid.index, dtype=float)
        s = src[["time_h", col]].dropna().sort_values("time_h")
        if s.empty:
            return pd.Series(index=grid.index, dtype=float)
        out = pd.Series(index=grid.index, dtype=float)
        out.loc[:] = np.interp(x=grid["time_h"].values, xp=s["time_h"].values, fp=s[col].values)
        return out

    if temp_df is not None and "Temperature_C" in temp_df.columns:
        merged["Temperature_C"] = _interp_onto_grid(temp_df, "Temperature_C", merged)
    if dens_df is not None and "density" in dens_df.columns:
        merged["density"] = _interp_onto_grid(dens_df, "density", merged)

    _dprint("[MERGE] tras interp T/D: nn Temperature_C:",
            int(merged["Temperature_C"].notna().sum()) if "Temperature_C" in merged.columns else 0,
            " nn density:",
            int(merged.get("density", pd.Series(dtype=float)).notna().sum()))

    # Consolidación Densidad
    if "density" in merged.columns:
        if "Densidad" not in merged.columns:
            merged["Densidad"] = merged["density"]
        else:
            mask = merged["Densidad"].isna() & merged["density"].notna()
            merged.loc[mask, "Densidad"] = merged.loc[mask, "density"]
        del merged["density"]

    # Asegurar ALL_VARS y 'Fecha y hora'
    for col in ALL_VARS:
        if col == "Fecha y hora":
            merged[col] = (pd.to_datetime(t0, utc=True) + pd.to_timedelta(merged["time_h"], unit="h"))
        elif col not in merged.columns:
            merged[col] = np.nan

    # ===== [FILL@t=0] rellenar con payload pre-t0 en la PRIMERA FILA =====
    if pre0_payload:
        idx0 = np.where(np.isclose(merged["time_h"].values, 0.0, atol=1e-6))[0]
        if idx0.size:
            i0 = int(idx0[0])
            filled_keys = []
            for k, v in pre0_payload.items():
                if k in merged.columns and pd.isna(merged.loc[i0, k]):
                    merged.loc[i0, k] = v
                    filled_keys.append(k)
            if filled_keys:
                _dprint("[FILL@t=0] columnas rellenadas en fila 0:", sorted(filled_keys)[:12],
                        ("... (+%d)" % (len(filled_keys) - 12) if len(filled_keys) > 12 else ""))

    # NO rellenar químicos; solo operacionales ya fueron interpolados
    merged = _only_interpolate_operational(merged)

    # Biomasa: suavizado (sobre datos existentes)  ---------------------------
    merged = _add_smoothed_biomass_columns(merged)

    # --- Inóculo SIEMPRE para la fila time_h=0 si está vacía ----------------
    #     (independiente de si existen datos de biomasa en otros tiempos)
    inoc_t_h, inoc_gL = _estimate_inoculum_biomass(xls, t0)
    _dprint("[BIO] Estimación inóculo -> inoc_time_h=", inoc_t_h, ", inoc_gL=", inoc_gL)

    # Asegura que exista una fila en t=0
    if not np.isclose(merged["time_h"].values, 0.0, atol=1e-6).any():
        new0 = pd.DataFrame({"time_h": [0.0]})
        merged = pd.concat([merged, new0], ignore_index=True)

    # Recalcula Fecha y hora para cualquier nueva fila en 0
    merged["Fecha y hora"] = (pd.to_datetime(t0, utc=True) +
                              pd.to_timedelta(merged["time_h"], unit="h"))

    # Si tenemos inoc_gL, rellenar biomasa en t=0 SOLO si está vacía
    if inoc_gL is not None and np.isfinite(inoc_gL):
        idx0 = np.where(np.isclose(merged["time_h"].values, 0.0, atol=1e-6))[0]
        if idx0.size:
            i0 = int(idx0[0])

            # crea columnas si no existen
            for k in ["Biomasa viable", "Biomasa Total",
                      "biomass_viable_gL", "biomass_total_gL",
                      "biomass_viable_smoothed_gL"]:
                if k not in merged.columns:
                    merged[k] = np.nan

            filled = []
            # reglas de llenado: sólo si NaN en t=0
            if pd.isna(merged.loc[i0, "Biomasa viable"]):
                merged.loc[i0, "Biomasa viable"] = float(inoc_gL); filled.append("Biomasa viable")
            if pd.isna(merged.loc[i0, "Biomasa Total"]):
                merged.loc[i0, "Biomasa Total"] = float(inoc_gL);  filled.append("Biomasa Total")
            if pd.isna(merged.loc[i0, "biomass_viable_gL"]):
                merged.loc[i0, "biomass_viable_gL"] = float(inoc_gL); filled.append("biomass_viable_gL")
            if pd.isna(merged.loc[i0, "biomass_total_gL"]):
                merged.loc[i0, "biomass_total_gL"] = float(inoc_gL);  filled.append("biomass_total_gL")
            # el suavizado en t=0 = valor del inóculo si estaba vacío
            if pd.isna(merged.loc[i0, "biomass_viable_smoothed_gL"]):
                merged.loc[i0, "biomass_viable_smoothed_gL"] = float(inoc_gL); filled.append("biomass_viable_smoothed_gL")

            if filled:
                _dprint("[BIO@t=0] columnas rellenadas con inóculo:", filled)

    merged = merged.sort_values("time_h").reset_index(drop=True)
    _dprint("[OUT] nn top:",
            _nn_summary(merged, ["Temperature_C","Densidad","Glucosa","Fructosa","Alcohol","YAN","Brix","pH",
                                 "Biomasa viable","Biomasa Total","biomass_viable_smoothed_gL"]))


    results_dict = {assay_code: merged}

    # chem_df YAN (para pulsos)
    chem_df = pd.DataFrame()
    if lab is not None and "__dt__" in lab.columns:
        name_col = "variable_text" if "variable_text" in lab.columns else None
        if (name_col is None or lab[name_col].astype(str).str.strip().eq("").all()) and "template_variable_text" in lab.columns:
            name_col = "template_variable_text"
        if name_col is not None:
            yan_mask = lab[name_col].astype(str).str.contains("YAN", case=False, na=False)
            sub = lab.loc[yan_mask].copy()
            if not sub.empty:
                sub["time_h"] = (sub["__dt__"] - t0).dt.total_seconds() / 3600.0
                sub.loc[sub["time_h"] < 0, "time_h"] = 0.0  # consistencia con plegado a 0
                sub["assay"] = assay_code
                if "valor_numeric" in sub.columns and sub["valor_numeric"].notna().any():
                    sub["valor"] = pd.to_numeric(sub["valor_numeric"], errors="coerce")
                chem_df = sub[["assay", "time_h", "valor", name_col]].rename(columns={name_col: "variable_text"})

    return results_dict, chem_df


# =========================
# Compat / Calibrador
# =========================
def attach_temperature_to_results(results_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    return results_dict


def build_calibration_matrices(
    results_dict: Dict[str, pd.DataFrame],
    use_smoothed_biomass: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Construye matrices por ensayo para el calibrador.
    - Incluye SIEMPRE todas las variables ALL_VARS (aunque sean NaN) + operacionales y biomasa.
    - Biomasa viable: usa 'biomass_viable_smoothed_gL' si use_smoothed_biomass y existe; si no, 'biomass_viable_gL'.
    """
    mats: Dict[str, pd.DataFrame] = {}

    for assay, df in results_dict.items():
        cols = ["time_h", "Temperature_C", "Densidad"]

        # Todas las variables químicas declaradas (excepto Fecha y hora / Densidad que ya van)
        chem_cols = [c for c in ALL_VARS if c not in ("Fecha y hora", "Densidad")]
        cols.extend([c for c in chem_cols if c in df.columns])

        # Biomasa
        bio_col = None
        if use_smoothed_biomass and "biomass_viable_smoothed_gL" in df.columns:
            bio_col = "biomass_viable_smoothed_gL"
        elif "biomass_viable_gL" in df.columns:
            bio_col = "biomass_viable_gL"
        if bio_col:
            cols.append(bio_col)

        # Asegura columnas / orden
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan

        m = df[cols].copy()

        # Diagnóstico adicional: negativos en la matriz resultante
        neg = int((m["time_h"] < 0).sum())
        if neg and _is_debug():
            _dprint(f"[MATS] '{assay}' con {neg} filas time_h < 0 (revisa políticas ENFORCE_* e INOCULUM_NEGATIVE_POLICY)")

        mats[assay] = m

    return mats
