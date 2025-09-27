# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import sys
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

# =========================
# CONFIG / FLAGS
# =========================
ROUND_TIME_DECIMALS = 6

# Reglas de recorte de laboratorio
ENFORCE_NONNEGATIVE_LAB = True
ENFORCE_WITHIN_OPER_WINDOW = True

# Política del inóculo si cae antes de t0
INOCULUM_NEGATIVE_POLICY = "clip_to_0"  # keep | clip_to_0 | drop

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
    "Acetaldehído","Acidez total","Acidez volátil","Ácido acético","Ácido l-málico",
    "Ácido pirúvico","Ácido tartárico","Alcohol","Amoníaco","Antocianinas","Azúcar Total mr",
    "Biomasa Total","Biomasa viable","Brix","Densidad","DO 280","DO 420","DO 520","DO 620",
    "Fecha y hora","Fructosa","Glicerol","Glucosa","Índice de brotación","PAN","pH",
    "Polifenoles","Polifenoles totales","SO2 Libre","SO2 Total","Sulfito libre","Sulfito total",
    "Taninos","YAN",
]

# =========================
# Utilidades de columnas/parseo
# =========================
def _find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    cols = [str(c) for c in df.columns]
    low = [c.lower() for c in cols]
    for patt in candidates:
        p = patt.lower()
        for i, l in enumerate(low):
            if p in l:
                return cols[i]
    return None

def _parse_dt_safely(series_like, *, prefer_excel_serial=False, dayfirst=True, tag=""):
    s_raw = pd.Series(series_like)

    s_iso = pd.to_datetime(s_raw, errors="coerce", utc=True, dayfirst=dayfirst)
    iso_ok = s_iso.notna().mean() if len(s_iso) else 0.0
    iso_range_ok = 1.0 if (s_iso.notna().any() and (s_iso.max() - s_iso.min()).total_seconds() >= 0) else 0.0

    s_num = pd.to_numeric(s_raw, errors="coerce")
    frac_num = s_num.notna().mean() if len(s_num) else 0.0
    looks_excel = (frac_num > 0.5) and (s_num.dropna().median() > 40000)
    if looks_excel or prefer_excel_serial:
        s_xl = pd.to_datetime(s_num, errors="coerce", utc=True, origin="1899-12-30", unit="D")
    else:
        s_xl = pd.Series([pd.NaT] * len(s_raw), dtype="datetime64[ns, UTC]")
    xl_ok = s_xl.notna().mean() if len(s_xl) else 0.0
    xl_range_ok = 1.0 if (s_xl.notna().any() and (s_xl.max() - s_xl.min()).total_seconds() >= 0) else 0.0

    _dprint(f"[DT-PARSE] ISO ok={iso_ok:.2f}, rangeOK={iso_range_ok:.2f} | XL ok={xl_ok:.2f}, rangeOK={xl_range_ok:.2f} | looks_excel={looks_excel}, frac_num={frac_num:.2f}")

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
    for c in candidates:
        if c in df.columns:
            return _parse_dt_safely(df[c], prefer_excel_serial=prefer_excel_serial, dayfirst=True, tag=tag)
    return None

# =========================
# Interpolación SOLO operacional
# =========================
def _only_interpolate_operational(df: pd.DataFrame) -> pd.DataFrame:
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
    if lab_df is None or len(lab_df) == 0:
        _dprint("[LAB] vacío")
        return pd.DataFrame()
    if "__dt__" not in lab_df.columns:
        _dprint("[LAB] no tiene __dt__ (timestamp)")
        return pd.DataFrame()

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
        s = (s.replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u").replace("ñ","n"))
        s = s.replace("-", " ")
        s = re.sub(r"\s+", " ", s)
        return s

    MAP = {
        "acetaldehido":"Acetaldehído","acidez total":"Acidez total","acidez volatil":"Acidez volátil","acido acetico":"Ácido acético",
        "acido l malico":"Ácido l-málico","acido malico":"Ácido l-málico","acido piruvico":"Ácido pirúvico","acido tartarico":"Ácido tartárico",
        "alcohol":"Alcohol","amoniaco":"Amoníaco","antocianinas":"Antocianinas","azucar total mr":"Azúcar Total mr","biomasa total":"Biomasa Total",
        "biomasa viable":"Biomasa viable","brix":"Brix","densidad":"Densidad","do 280":"DO 280","do280":"DO 280","do 420":"DO 420",
        "do420":"DO 420","do 520":"DO 520","do520":"DO 520","do 620":"DO 620","do620":"DO 620","fructosa":"Fructosa","glicerol":"Glicerol",
        "glucosa":"Glucosa","indice de brotacion":"Índice de brotación","pan":"PAN","ph":"pH","polifenoles totales":"Polifenoles totales",
        "polifenoles":"Polifenoles","so2 libre":"SO2 Libre","so2 total":"SO2 Total","sulfito libre":"Sulfito libre","sulfito total":"Sulfito total",
        "taninos":"Taninos","yan":"YAN","fecha y hora":"Fecha y hora",
    }

    def _parse_val(row) -> float:
        vn = row.get("valor_numeric", None)
        if vn is not None:
            try:
                v = float(vn)
                if not pd.isna(v): return v
            except Exception:
                pass
        vs = row.get("valor", None)
        if vs is None: return np.nan
        s = str(vs).strip().replace("\xa0"," ").replace(",", ".")
        for tok in ("%", "≈", "~", "<", ">"): s = s.replace(tok, "")
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if not m: return np.nan
        try: return float(m.group(0))
        except Exception: return np.nan

    rows = []
    miss_map = {}
    for _, row in lab_df.iterrows():
        ts = row.get("__dt__")
        if pd.isna(ts): continue
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
        if pd.isna(v): continue
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
    s_viab = df.get("Biomasa viable")
    s_tot  = df.get("Biomasa Total")
    if s_viab is not None:
        df["biomass_viable_gL"] = s_viab
        sv = s_viab.rolling(window=3, center=True, min_periods=1).mean()
        df["biomass_viable_smoothed_gL"] = sv.where(s_viab.notna())
    if s_tot is not None:
        df["biomass_total_gL"] = s_tot
    return df

def _estimate_inoculum_biomass(xls: pd.ExcelFile, t0) -> Tuple[Optional[float], Optional[float]]:
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
            qty = pd.to_numeric(lev[c].iloc[0], errors="coerce"); break
    for c in ["unidad", "unidad_text", "Unidad", "unidad_medida"]:
        if c in lev.columns:
            unit = str(lev[c].iloc[0]).strip().lower(); break
    _dprint("[INOC] qty -> col='cantidad', valor=", qty, "| unidad -> col='unidad', valor='", unit, "'", sep="")

    if qty is None or pd.isna(qty): return (None, None)

    inoc_gL = None; rule = None
    if unit in {"g/hl", "g / hl", "g por hl", "g/hl", "g por hl"}:
        inoc_gL = float(qty) / 100.0; rule = "g/hL → g/L"
    elif unit in {"mg/l", "ppm"}:
        inoc_gL = float(qty) / 1000.0; rule = "mg/L → g/L"
    elif unit in {"g", "gr", "gramo", "gramos"}:
        if V_L and V_L > 0:
            inoc_gL = float(qty) / float(V_L); rule = "g totales / V_L"
    elif unit in {"(kg)", "kg", "kilogramo", "kilogramos"}:
        if V_L and V_L > 0:
            inoc_gL = float(qty) * 1000.0 / float(V_L); rule = "kg totales / V_L"
    else:
        if unit and ("g" in unit and "hl" in unit):
            inoc_gL = float(qty) / 100.0; rule = "g*hL detectado → g/L"

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
# Helpers de inyección (inicio/fin)
# =========================
def _norm_txt(s: str) -> str:
    if s is None: return ""
    s = str(s).strip().lower()
    s = (s.replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u").replace("ñ","n"))
    return s

def _pick_val(vn, vs):
    if pd.notna(vn):
        try:
            x = float(vn)
            if np.isfinite(x): return x
        except Exception:
            pass
    if pd.isna(vs): return np.nan
    s = str(vs).strip().replace(",", ".")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else np.nan

_MAP_NORM_TO_ES = {
    "acetaldehido":"Acetaldehído","acidez total":"Acidez total","acidez volatil":"Acidez volátil","acido acetico":"Ácido acético",
    "acido l malico":"Ácido l-málico","acido malico":"Ácido l-málico","acido piruvico":"Ácido pirúvico","acido tartarico":"Ácido tartárico",
    "alcohol":"Alcohol","amoniaco":"Amoníaco","antocianinas":"Antocianinas","azucar total mr":"Azúcar Total mr",
    "biomasa total":"Biomasa Total","biomasa viable":"Biomasa viable","brix":"Brix","densidad":"Densidad",
    "do 280":"DO 280","do280":"DO 280","do 420":"DO 420","do420":"DO 420","do 520":"DO 520","do520":"DO 520","do 620":"DO 620","do620":"DO 620",
    "fructosa":"Fructosa","glicerol":"Glicerol","glucosa":"Glucosa","indice de brotacion":"Índice de brotación","pan":"PAN","ph":"pH",
    "polifenoles totales":"Polifenoles totales","polifenoles":"Polifenoles","so2 libre":"SO2 Libre","so2 total":"SO2 Total",
    "sulfito libre":"Sulfito libre","sulfito total":"Sulfito total","taninos":"Taninos","yan":"YAN","fecha y hora":"Fecha y hora",
}

def _inject_initial_lab_from_fermentacion(merged: pd.DataFrame, lab_df: pd.DataFrame, t0_utc: pd.Timestamp, all_vars: List[str]) -> pd.DataFrame:
    try:
        if lab_df is None or lab_df.empty: return merged
        if "__dt__" not in lab_df.columns or "muestreo_text" not in lab_df.columns: return merged

        mtxt = lab_df["muestreo_text"].astype(str)
        mask_fer = mtxt.apply(lambda x: "durante la fermentacion" in _norm_txt(x))
        lab_fer = lab_df.loc[mask_fer].copy()
        if lab_fer.empty:
            _dprint("[FER-INIT] No hay '3. Durante la fermentación'"); return merged

        lab_fer["__dt__"] = pd.to_datetime(lab_fer["__dt__"], errors="coerce", utc=True)
        lab_fer = lab_fer.dropna(subset=["__dt__"]).sort_values("__dt__")
        if lab_fer.empty:
            _dprint("[FER-INIT] Filtrado por dt dejó vacío."); return merged

        dt0 = lab_fer["__dt__"].iloc[0]
        sub = lab_fer.loc[lab_fer["__dt__"] == dt0]
        name_col = "variable_text" if "variable_text" in sub.columns else ("template_variable_text" if "template_variable_text" in sub.columns else None)
        if name_col is None:
            _dprint("[FER-INIT] No hay columnas de nombre de variable."); return merged

        row_vals = {}
        for _, r in sub.iterrows():
            nm = _norm_txt(str(r.get(name_col, "")).strip())
            target = None
            for patt, tgt in sorted(_MAP_NORM_TO_ES.items(), key=lambda kv: -len(kv[0])):
                if patt in nm: target = tgt; break
            if target is None: continue
            v = _pick_val(r.get("valor_numeric"), r.get("valor"))
            if pd.notna(v): row_vals[target] = float(v)

        if not row_vals:
            _dprint("[FER-INIT] No se pudieron extraer valores en dt0 fermentación."); return merged

        if not np.isclose(merged["time_h"].values, 0.0, atol=1e-6).any():
            merged = pd.concat([merged, pd.DataFrame({"time_h":[0.0]})], ignore_index=True)

        merged["Fecha y hora"] = (pd.to_datetime(t0_utc, utc=True) + pd.to_timedelta(merged["time_h"], unit="h"))

        i0 = np.where(np.isclose(merged["time_h"].values, 0.0, atol=1e-6))[0]
        if i0.size:
            i0 = int(i0[0])
            filled = []
            for col in all_vars:
                if col == "Fecha y hora": continue
                if col in row_vals:
                    if col not in merged.columns: merged[col] = np.nan
                    if pd.isna(merged.loc[i0, col]):
                        merged.loc[i0, col] = row_vals[col]; filled.append(col)
            _dprint(f"[FER-INIT] Punto inicial alimentado (dt={dt0}) → columnas:", filled)
        return merged
    except Exception as e:
        _dprint("[FER-INIT] EXC:", e)
        return merged

def _inject_final_lab_from_descube(merged: pd.DataFrame, lab_df: pd.DataFrame, all_vars: List[str]) -> pd.DataFrame:
    """
    Copia los valores de la PRIMERA muestra cuyo muestreo_text == '4. Descube'
    a la fila con time_h MÁXIMO (último punto operacional).
    SOLO escribe donde haya NaN en esa última fila.
    """
    try:
        if lab_df is None or lab_df.empty: return merged
        if "__dt__" not in lab_df.columns or "muestreo_text" not in lab_df.columns: return merged

        mtxt = lab_df["muestreo_text"].astype(str)
        mask_desc = mtxt.apply(lambda x: "descube" in _norm_txt(x))
        lab_desc = lab_df.loc[mask_desc].copy()
        if lab_desc.empty:
            _dprint("[DESCUBE] No hay '4. Descube'"); return merged

        lab_desc["__dt__"] = pd.to_datetime(lab_desc["__dt__"], errors="coerce", utc=True)
        lab_desc = lab_desc.dropna(subset=["__dt__"]).sort_values("__dt__")
        if lab_desc.empty:
            _dprint("[DESCUBE] Filtrado por dt dejó vacío."); return merged

        dt_end = lab_desc["__dt__"].iloc[0]  # misma política que inicio: tomamos la primera ocurrencia
        sub = lab_desc.loc[lab_desc["__dt__"] == dt_end]
        name_col = "variable_text" if "variable_text" in sub.columns else ("template_variable_text" if "template_variable_text" in sub.columns else None)
        if name_col is None:
            _dprint("[DESCUBE] No hay columnas de nombre de variable."); return merged

        row_vals = {}
        for _, r in sub.iterrows():
            nm = _norm_txt(str(r.get(name_col, "")).strip())
            target = None
            for patt, tgt in sorted(_MAP_NORM_TO_ES.items(), key=lambda kv: -len(kv[0])):
                if patt in nm: target = tgt; break
            if target is None: continue
            v = _pick_val(r.get("valor_numeric"), r.get("valor"))
            if pd.notna(v): row_vals[target] = float(v)

        if not row_vals:
            _dprint("[DESCUBE] No se pudieron extraer valores en dt_end descube."); return merged

        # índice del último punto operacional (máximo time_h)
        j_last = int(merged["time_h"].idxmax())
        filled = []
        for col in all_vars:
            if col == "Fecha y hora": continue
            if col in row_vals:
                if col not in merged.columns: merged[col] = np.nan
                if pd.isna(merged.loc[j_last, col]):
                    merged.loc[j_last, col] = row_vals[col]; filled.append(col)
        _dprint(f"[DESCUBE] Punto final alimentado (dt={dt_end}) en time_h={merged.loc[j_last, 'time_h']} → columnas:", filled)
        return merged
    except Exception as e:
        _dprint("[DESCUBE] EXC:", e)
        return merged

# =========================
# Pipeline principal
# =========================
def process_all(file_path: str, assays: Optional[List[str]] = None):
    xls = pd.ExcelFile(file_path)

    # ---- Código de ensayo
    try:
        ant = pd.read_excel(xls, sheet_name="Antecedentes")
        assay_code = str(ant.get("ant_cubada").iloc[0]).strip() if "ant_cubada" in ant.columns else "ASSAY"
    except Exception:
        assay_code = "ASSAY"
    _dprint(f"[ASSAY] code: {assay_code}")

    # ---- Carga hojas
    lab  = pd.read_excel(xls, sheet_name="Laboratorio")
    tman = pd.read_excel(xls, sheet_name="Manual Temperaturas")
    dman = pd.read_excel(xls, sheet_name="Manual Densidades")
    _dprint("[LOAD] filas -> lab:", len(lab), "temp:", len(tman), "dens:", len(dman))

    # ---- Parse de fechas
    lab = lab.copy();  tman = tman.copy(); dman = dman.copy()
    lab_dt  = _parse_datetime_col(lab,  "create_fecha", "fecha", "fecha_muestra", prefer_excel_serial=True, tag="lab")
    tman_dt = _parse_datetime_col(tman, "medicion_fecha", prefer_excel_serial=False, tag="tman")
    dman_dt = _parse_datetime_col(dman, "medicion_fecha", prefer_excel_serial=False, tag="dman")
    if lab_dt is not None:  lab["__dt__"]  = lab_dt
    if tman_dt is not None: tman["__dt__"] = tman_dt
    if dman_dt is not None: dman["__dt__"] = dman_dt

    _dprint("[TS] lab dt:",  (lab["__dt__"].min() if "__dt__" in lab.columns else None), "→", (lab["__dt__"].max() if "__dt__" in lab.columns else None))
    _dprint("[TS] tman dt:", (tman["__dt__"].min() if "__dt__" in tman.columns else None), "→", (tman["__dt__"].max() if "__dt__" in tman.columns else None))
    _dprint("[TS] dman dt:", (dman["__dt__"].min() if "__dt__" in dman.columns else None), "→", (dman["__dt__"].max() if "__dt__" in dman.columns else None))

    # ---- Columnas operacionales
    cT = _find_col(tman, "temperatura", "temp", "t (c)", "tempertura")
    if cT:
        tman.rename(columns={cT: "Temperature_C"}, inplace=True)
        tman["Temperature_C"] = pd.to_numeric(tman["Temperature_C"], errors="coerce")

    cD = _find_col(dman, "densidad")
    if cD:
        dman.rename(columns={cD: "density"}, inplace=True)
        dman["density"] = pd.to_numeric(dman["density"], errors="coerce")

    # ---- t0: mínimo TEMP/DENS; si no hay, LAB
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

    # ---- Operacionales → time_h
    temp_df = None
    if "__dt__" in tman.columns and "Temperature_C" in tman.columns:
        temp_df = tman[["__dt__", "Temperature_C"]].dropna().copy()
        temp_df["time_h"] = (temp_df["__dt__"] - t0).dt.total_seconds() / 3600.0
        temp_df.drop(columns="__dt__", inplace=True)
        _dprint("[TEMP] rows:", len(temp_df), "time_h min/max:", temp_df["time_h"].min(), temp_df["time_h"].max())

    dens_df = None
    if "__dt__" in dman.columns and "density" in dman.columns:
        dens_df = dman[["__dt__", "density"]].dropna().copy()
        dens_df["time_h"] = (dens_df["__dt__"] - t0).dt.total_seconds() / 3600.0
        dens_df.drop(columns="__dt__", inplace=True)
        _dprint("[DENS] rows:", len(dens_df), "time_h min/max:", dens_df["time_h"].min(), dens_df["time_h"].max())

    # ---- LAB (químicos) → time_h
    sig = _extract_signals_from_lab(lab) if lab is not None else pd.DataFrame()
    if not sig.empty:
        sig["time_h"] = (sig["__dt__"] - t0).dt.total_seconds() / 3600.0
        sig.drop(columns="__dt__", inplace=True)
        neg = int((sig["time_h"] < 0).sum()); pos = int((sig["time_h"] >= 0).sum())
        _dprint("[PRE0] negativos:", neg, " | positivos:", pos)
        _dprint("[SIG] time_h range:", sig["time_h"].min(), "→", sig["time_h"].max(), "| negativos:", neg, "| positivos:", pos)

    # ---- Recorte LAB a [0, t_end]
    t_end = None
    if temp_df is not None and len(temp_df) > 0: t_end = float(temp_df["time_h"].max())
    if dens_df is not None and len(dens_df) > 0: t_end = max(t_end or 0.0, float(dens_df["time_h"].max()))
    if not sig.empty and t_end is not None:
        before = len(sig)
        sig = sig.loc[(sig["time_h"] >= -1e-6) & (sig["time_h"] <= t_end + 1e-6)].copy()
        _dprint(f"[CLIP] sig: {before} -> {len(sig)} (0 ≤ time_h ≤ {t_end})")

    # ---- Grilla de tiempos
    def _round_time_h(df: Optional[pd.DataFrame], col="time_h", nd=6) -> Optional[pd.DataFrame]:
        if df is None or col not in df.columns: return df
        df[col] = np.round(df[col].astype(float), nd); return df

    # Use configured decimals instead of hardcoded 6
    temp_df = _round_time_h(temp_df, "time_h", ROUND_TIME_DECIMALS)
    dens_df = _round_time_h(dens_df, "time_h", ROUND_TIME_DECIMALS)
    sig     = _round_time_h(sig, "time_h", ROUND_TIME_DECIMALS)

    series_times = []
    if temp_df is not None and len(temp_df) > 0: series_times.append(temp_df["time_h"])
    if dens_df is not None and len(dens_df) > 0: series_times.append(dens_df["time_h"])
    if not sig.empty: series_times.append(sig["time_h"])
    if not series_times:
        _dprint("[GRID] sin tiempos -> vacío"); return {}, pd.DataFrame()

    t_grid = np.unique(np.concatenate([st.dropna().values for st in series_times]))
    merged = pd.DataFrame({"time_h": t_grid})
    merged["time_h"] = np.round(merged["time_h"].astype(float), ROUND_TIME_DECIMALS)
    _dprint("[GRID] len:", len(merged), "range:", merged["time_h"].min(), "→", merged["time_h"].max())

    # ---- Merge LAB por time_h exacto
    if not sig.empty:
        lab_times_unique = np.unique(sig["time_h"].dropna().values)
        hits = np.intersect1d(lab_times_unique, merged["time_h"].values).size
        _dprint(f"[LAB MERGE] tiempos únicos lab: {len(lab_times_unique)} | matchean con grilla: {hits} | no matchean: {len(lab_times_unique)-hits}")
        merged = merged.merge(sig, on="time_h", how="left")
        _dprint("[MERGE] tras lab: nn:", {k: int(v) for k, v in merged.drop(columns=["time_h"], errors="ignore").notna().sum().sort_values(ascending=False).head(10).items()})

    # ---- Interp. SOLO operacionales
    def _interp_onto_grid(src: Optional[pd.DataFrame], col: str, grid: pd.DataFrame) -> pd.Series:
        if src is None or len(src) == 0 or col not in src.columns: return pd.Series(index=grid.index, dtype=float)
        s = src[["time_h", col]].dropna().sort_values("time_h")
        if s.empty: return pd.Series(index=grid.index, dtype=float)
        out = pd.Series(index=grid.index, dtype=float)
        out.loc[:] = np.interp(x=grid["time_h"].values, xp=s["time_h"].values, fp=s[col].values)
        return out

    if temp_df is not None and "Temperature_C" in temp_df.columns:
        merged["Temperature_C"] = _interp_onto_grid(temp_df, "Temperature_C", merged)
    if dens_df is not None and "density" in dens_df.columns:
        merged["density"] = _interp_onto_grid(dens_df, "density", merged)

    _dprint("[MERGE] tras interp T/D: nn Temperature_C:",
            int(merged["Temperature_C"].notna().sum()) if "Temperature_C" in merged.columns else 0,
            " nn density:", int(merged.get("density", pd.Series(dtype=float)).notna().sum()))

    # ---- Consolidación Densidad
    if "density" in merged.columns:
        if "Densidad" not in merged.columns:
            merged["Densidad"] = merged["density"]
        else:
            mask = merged["Densidad"].isna() & merged["density"].notna()
            merged.loc[mask, "Densidad"] = merged.loc[mask, "density"]
        del merged["density"]

    # ---- Asegurar ALL_VARS y 'Fecha y hora'
    for col in ALL_VARS:
        if col == "Fecha y hora":
            # FIX: don't wrap to_timedelta with to_datetime; directly add to timestamp
            merged[col] = (pd.to_datetime(t0, utc=True) + pd.to_timedelta(merged["time_h"], unit="h"))
        elif col not in merged.columns:
            merged[col] = np.nan

    # ---- Inyección INICIO desde '3. Durante la fermentación'
    merged = _inject_initial_lab_from_fermentacion(merged=merged, lab_df=lab, t0_utc=t0, all_vars=ALL_VARS)

    # ---- NO rellenamos químicos; solo operacionales ya fueron interpolados
    merged = _only_interpolate_operational(merged)

    # ---- Biomasa (suavizado) + Inóculo si falta en t=0
    merged = _add_smoothed_biomass_columns(merged)
    inoc_time_h, inoc_gL = _estimate_inoculum_biomass(xls, t0)
    _dprint(f"[BIO] Estimación inóculo -> inoc_time_h={inoc_time_h}, inoc_gL={inoc_gL}")

    if not np.isclose(merged["time_h"].values, 0.0, atol=1e-6).any():
        merged = pd.concat([merged, pd.DataFrame({"time_h":[0.0]})], ignore_index=True)

    merged["Fecha y hora"] = (pd.to_datetime(t0, utc=True) + pd.to_timedelta(merged["time_h"], unit="h"))

    if inoc_gL is not None and np.isfinite(inoc_gL):
        idx0 = np.where(np.isclose(merged["time_h"].values, 0.0, atol=1e-6))[0]
        if idx0.size:
            i0 = int(idx0[0])
            for k in ["Biomasa viable","Biomasa Total","biomass_viable_gL","biomass_total_gL","biomass_viable_smoothed_gL"]:
                if k not in merged.columns: merged[k] = np.nan
            filled = []
            if pd.isna(merged.loc[i0, "Biomasa viable"]):
                merged.loc[i0, "Biomasa viable"] = float(inoc_gL); filled.append("Biomasa viable")
            if pd.isna(merged.loc[i0, "Biomasa Total"]):
                merged.loc[i0, "Biomasa Total"] = float(inoc_gL); filled.append("Biomasa Total")
            if pd.isna(merged.loc[i0, "biomass_viable_gL"]):
                merged.loc[i0, "biomass_viable_gL"] = float(inoc_gL); filled.append("biomass_viable_gL")
            if pd.isna(merged.loc[i0, "biomass_total_gL"]):
                merged.loc[i0, "biomass_total_gL"] = float(inoc_gL); filled.append("biomass_total_gL")
            if pd.isna(merged.loc[i0, "biomass_viable_smoothed_gL"]):
                merged.loc[i0, "biomass_viable_smoothed_gL"] = float(inoc_gL); filled.append("biomass_viable_smoothed_gL")
            if filled: _dprint("[BIO@t=0] columnas rellenadas con inóculo:", filled)

    # ---- NUEVO: Inyección FIN desde '4. Descube'
    merged = _inject_final_lab_from_descube(merged=merged, lab_df=lab, all_vars=ALL_VARS)

    merged = merged.sort_values("time_h").reset_index(drop=True)
    _dprint("[OUT] nn top:",
            _nn_summary(merged, ["Temperature_C","Densidad","Glucosa","Fructosa","Alcohol","YAN","Brix","pH",
                                 "Biomasa viable","Biomasa Total","biomass_viable_smoothed_gL"]))

    results_dict = {assay_code: merged}

    # ---- chem_df YAN (mediciones + pulsos de nutrientes)
    chem_df = pd.DataFrame()
    chem_rows = []

    # (A) Mediciones de laboratorio de YAN (se mantienen)
    if lab is not None and "__dt__" in lab.columns:
        name_col = "variable_text" if "variable_text" in lab.columns else None
        if (name_col is None or lab[name_col].astype(str).str.strip().eq("").all()) and "template_variable_text" in lab.columns:
            name_col = "template_variable_text"
        if name_col is not None:
            yan_mask = lab[name_col].astype(str).str.contains("YAN", case=False, na=False)
            sub = lab.loc[yan_mask].copy()
            if not sub.empty:
                sub["time_h"] = (pd.to_datetime(sub["__dt__"], errors="coerce", utc=True) - t0).dt.total_seconds() / 3600.0
                sub.loc[sub["time_h"] < 0, "time_h"] = 0.0
                sub["assay"] = assay_code
                if "valor_numeric" in sub.columns and sub["valor_numeric"].notna().any():
                    sub["valor"] = pd.to_numeric(sub["valor_numeric"], errors="coerce")
                for _, r in sub.iterrows():
                    chem_rows.append({
                        "assay": assay_code,
                        "time_h": float(r.get("time_h", np.nan)),
                        "valor": r.get("valor", np.nan),
                        "variable_text": "YAN",
                        "nombre_insumo": np.nan  # medición de laboratorio
                    })

    # NUEVO: lookup densidad→time_h para mapear densidad_aplicacion
    dens_lookup = merged[["time_h", "Densidad"]].dropna()
    def _map_density_to_time(d_appl):
        if dens_lookup.empty or pd.isna(d_appl):
            return np.nan
        idx = (dens_lookup["Densidad"] - d_appl).abs().idxmin()
        return float(dens_lookup.loc[idx, "time_h"])

    # (B) Pulsos de FDA y Vitaferm
    # Obtener volumen (V_L) para convertir a mg/L
    V_L = None
    try:
        # 'ant' ya fue leído arriba (si falló queda en except); reintentar si no existe
        if 'ant' not in locals() or isinstance(ant, Exception):
            ant = pd.read_excel(xls, sheet_name="Antecedentes")
        for col_vol in ["ant_vino_estimado_l", "ant_volumen_l", "volumen_l"]:
            if col_vol in ant.columns:
                V_L = pd.to_numeric(ant[col_vol].iloc[0], errors="coerce")
                if pd.notna(V_L) and V_L > 0:
                    break
    except Exception:
        V_L = None

    # Determinar fecha de inóculo (levadura) para filtrar Vitaferm (>0.5 días después)
    inoculum_dt = None
    try:
        ops_inoc = pd.read_excel(xls, sheet_name="Insumos Operacionales")
        ops_inoc.columns = [str(c).strip() for c in ops_inoc.columns]
        if "insumo" in ops_inoc.columns:
            mask_lev = ops_inoc["insumo"].astype(str).str.contains("levadura", case=False, na=False)
            if mask_lev.any():
                # Preferencia de columna de fecha
                for cand in ["fecha_proceso_format", "fecha", "fecha_aplicacion_format"]:
                    if cand in ops_inoc.columns:
                        dt_series = pd.to_datetime(ops_inoc.loc[mask_lev, cand], errors="coerce", utc=True)
                        dt_series = dt_series.dropna()
                        if not dt_series.empty:
                            inoculum_dt = dt_series.sort_values().iloc[0]
                        break
    except Exception:
        pass
    if inoculum_dt is None:
        inoculum_dt = pd.to_datetime(t0, utc=True)

    # (B1) FDA en "Insumos Operacionales"
    try:
        ops = pd.read_excel(xls, sheet_name="Insumos Operacionales")
        ops.columns = [str(c).strip() for c in ops.columns]
        needed_cols_ops = {"insumo", "cantidad"}
        if needed_cols_ops.issubset(set(ops.columns)):
            dt_col_ops = None
            for c in ["fecha_proceso_format", "fecha", "fecha_aplicacion_format"]:
                if c in ops.columns:
                    dt_col_ops = c
                    break
            # Eliminada restricción 'etapa' == 'Durante'
            mask_fda = (ops["insumo"].astype(str).str.strip().str.lower() == "fda")
            sub_fda = ops.loc[mask_fda].copy()
            if not sub_fda.empty:
                if dt_col_ops:
                    sub_fda["_dt_utc"] = pd.to_datetime(sub_fda[dt_col_ops], errors="coerce", utc=True)
                for _, r in sub_fda.iterrows():
                    kg = pd.to_numeric(r.get("cantidad"), errors="coerce")
                    if pd.isna(kg) or kg <= 0:
                        continue
                    dens_appl = pd.to_numeric(r.get("densidad"), errors="coerce")
                    if pd.isna(dens_appl):
                        dens_appl = pd.to_numeric(r.get("densidad_aplicacion"), errors="coerce")
                    time_h_pulse = _map_density_to_time(dens_appl)
                    if pd.isna(time_h_pulse):
                        continue
                    yan_mg_total = kg * 1e6 * 0.2
                    yan_val = (yan_mg_total / V_L) if (V_L and V_L > 0) else np.nan
                    chem_rows.append({
                        "assay": assay_code,
                        "time_h": time_h_pulse,
                        "valor": yan_val,
                        "variable_text": "YAN",
                        "nombre_insumo": "FDA"
                    })
    except Exception:
        pass

    # (B2) Vitaferm en "Otros Insumos"
    try:
        otros = pd.read_excel(xls, sheet_name="Otros Insumos")
        otros.columns = [str(c).strip() for c in otros.columns]
        needed_cols_otros = {"nombre", "cantidad"}
        if needed_cols_otros.issubset(set(otros.columns)):
            dt_col_v = None
            for c in ["fecha_aplicacion_format", "fecha", "fecha_proceso_format"]:
                if c in otros.columns:
                    dt_col_v = c
                    break
            mask_vita = otros["nombre"].astype(str).str.strip().str.lower() == "vitaferm"
            sub_vita = otros.loc[mask_vita].copy()
            if not sub_vita.empty:
                if dt_col_v:
                    sub_vita["_dt_utc"] = pd.to_datetime(sub_vita[dt_col_v], errors="coerce", utc=True)
                for _, r in sub_vita.iterrows():
                    # Eliminado filtro de >0.5 días desde inóculo
                    g = pd.to_numeric(r.get("cantidad"), errors="coerce")
                    if pd.isna(g) or g <= 0:
                        continue
                    dens_appl = pd.to_numeric(r.get("densidad_aplicacion"), errors="coerce")
                    time_h_pulse = _map_density_to_time(dens_appl)
                    if pd.isna(time_h_pulse):
                        continue
                    yan_mg_total = g * 1000.0 * 0.08
                    yan_val = (yan_mg_total / V_L) if (V_L and V_L > 0) else np.nan
                    chem_rows.append({
                        "assay": assay_code,
                        "time_h": time_h_pulse,
                        "valor": yan_val,
                        "variable_text": "YAN",
                        "nombre_insumo": "Vitaferm"
                    })
    except Exception:
        pass

    if chem_rows:
        chem_df = pd.DataFrame(chem_rows).sort_values("time_h").reset_index(drop=True)

    return results_dict, chem_df

# =========================
# Compat / Calibrador
# =========================
def attach_temperature_to_results(results_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    return results_dict

def build_calibration_matrices(results_dict: Dict[str, pd.DataFrame], use_smoothed_biomass: bool = True) -> Dict[str, pd.DataFrame]:
    mats: Dict[str, pd.DataFrame] = {}
    for assay, df in results_dict.items():
        cols = ["time_h", "Temperature_C", "Densidad"]
        chem_cols = [c for c in ALL_VARS if c not in ("Fecha y hora", "Densidad")]
        cols.extend([c for c in chem_cols if c in df.columns])
        bio_col = None
        if use_smoothed_biomass and "biomass_viable_smoothed_gL" in df.columns:
            bio_col = "biomass_viable_smoothed_gL"
        elif "biomass_viable_gL" in df.columns:
            bio_col = "biomass_viable_gL"
        if bio_col: cols.append(bio_col)
        for c in cols:
            if c not in df.columns: df[c] = np.nan
        m = df[cols].copy()
        if _is_debug():
            neg = int((m["time_h"] < 0).sum())
            if neg: _dprint(f"[MATS] '{assay}' con {neg} filas time_h < 0")
        mats[assay] = m
    return mats

def process_multiple(codes: List[str], directory: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Procesa múltiples archivos 'Data <codigo>.xlsx'.
    Retorna:
        {
          codigo: {
             'data': DataFrame principal,
             'chem_df': DataFrame YAN (puede estar vacío),
             'assay_code': código interno devuelto por process_all,
             'file_path': ruta del archivo procesado
          },
          ...
        }
    Solo incluye códigos cuyo archivo existe y se procesa sin excepción.
    """
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for code in codes:
        file_path = os.path.join(directory, f"Data {code}.xlsx")
        if not os.path.isfile(file_path):
            _dprint(f"[BATCH] Archivo no encontrado: {file_path}")
            continue
        try:
            results_dict, chem_df = process_all(file_path)
            if not results_dict:
                _dprint(f"[BATCH] Sin resultados para {code}")
                continue
            assay_code, df_main = next(iter(results_dict.items()))
            out[code] = {
                "data": df_main,
                "chem_df": chem_df,
                "assay_code": assay_code,
                "file_path": file_path,
            }
            _dprint(f"[BATCH] OK {code} -> assay_code={assay_code}, filas={len(df_main)}")
        except Exception as e:
            _dprint(f"[BATCH] Error procesando {code}: {e}")
    return out

# =========================
# MAIN (procesa 24018–24031)
# =========================
if __name__ == "__main__":
    codes = [str(i) for i in range(24018, 24032)]  # 24018 … 24031
    script_dir = os.path.dirname(__file__)
    default_data_dir = os.path.join(script_dir, "Datos Experimentales")
    data_dir = os.environ.get("CALIB_DATA_DIR", default_data_dir)
    # Parse argumentos (ej: --data-dir=PATH o PATH posicional)
    for arg in sys.argv[1:]:
        if arg == "--wdir":
            continue
        if arg.startswith("--data-dir="):
            data_dir = arg.split("=", 1)[1]
        elif not arg.startswith("--"):
            data_dir = arg  # primer posicional
    if not os.path.isdir(data_dir):
        print(f"[MAIN] Carpeta de datos no encontrada: {data_dir}")
        print(f"[MAIN] Cree la carpeta o pase ruta con --data-dir=...  (default era: {default_data_dir})")
        sys.exit(1)
    print(f"[MAIN] Procesando códigos: {', '.join(codes)} en '{data_dir}'")
    results = process_multiple(codes, data_dir)

    if not results:
        print("[MAIN] No se generaron resultados (verifique existencia de archivos).")
    else:
        for code, bundle in results.items():
            df_main = bundle.get("data", pd.DataFrame())
            chem_df = bundle.get("chem_df", pd.DataFrame())
            out_main = os.path.join(data_dir, f"preproc_{code}.csv")
            df_main.to_csv(out_main, index=False)
            print(f"[MAIN] {code}: data filas={len(df_main)} → {out_main}")
            if not chem_df.empty:
                out_chem = os.path.join(data_dir, f"chem_{code}.csv")
                chem_df.to_csv(out_chem, index=False)
                print(f"[MAIN] {code}: chem_df filas={len(chem_df)} → {out_chem}")
                out_chem = os.path.join(data_dir, f"chem_{code}.csv")
                chem_df.to_csv(out_chem, index=False)
                print(f"[MAIN] {code}: chem_df filas={len(chem_df)} → {out_chem}")
        print("[MAIN] Listo.")
