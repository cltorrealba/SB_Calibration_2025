"""
Construye una matriz por ensayo para calibración:
columnas = [time_h, BiomasaViable_gL, BiomasaMuerta_gL, YAN, AMMONIA, PAN, Fructose, Glucose, Glycerol, Ethanol, Temperature_C]
- time_h en HORAS desde t0 del ensayo (float)
- biomasa: usa *_adj por defecto, o *_ma3 si use_smoothed_biomass=True
- Ethanol en g/L (conv. desde % v/v usando densidad 0.78924 g/mL @ 20°C)
"""
import os, re, math
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
from sklearn.isotonic import IsotonicRegression

# ===================== CONFIG =====================
FILE_PATH = r"C:\Users\ctorrealba\OneDrive - Viña Concha y Toro S.A\Documentos\Proyectos I+D\PI-4497\Resultados\2025\SB_Calibration_2025\Procesos_I+D_2025_3.xlsx"   # <-- EDITA
SHEET_BDD = "BDD_Maestra"

# Carpeta con planillas de temperatura "Data <ID>.xlsx"
# (relativa al script o ruta absoluta; ajusta a tu estructura)
TEMPS_DIR = "Datos Experimentales"          # <-- EDITA: carpeta donde guardas "Data 25026.xlsx", etc.
TEMP_SHEET = "Manual Temperaturas"
TEMP_DATE_COL = "medicion_fecha"    # columna de fecha-hora (timestamp)
TEMP_VALUE_COL = "temperatura"      # columna de temperatura en °C

# Homologación código SBxxx -> ID Ensayo (nombre de archivo "Data <ID>.xlsx")
SB2ID = {
    "SB003": 25026,
    "SB004": 25027,
    "SB005": 25028,
    "SB006": 25029,
    "SB007": 25085,
    "SB008": 25086,
    "SB009": 25150,
    "SB010": 25151,
    "SB011": 25170,
    "SB012": 25171,
}

# Inóculo (g) por ensayo; volumen del reactor (L)
INOCULUM_G = {
    "SB003": 72, "SB004": 72,
    "SB007": 144, "SB008": 144, "SB009": 144,
    "SB010": 144, "SB011": 144, "SB012": 144,
}
REACTOR_VOL_L = 240.0

# Parámetros de “corte” por N (ajustables)
PAN_THR = 12.0
AMM_THR = 2.0
RUN_LEN = 2          # puntos consecutivos que cumplen el umbral

# Calibración experimental (de tu curva de PS)
SLOPE = 0.03692948069886653
INTERCEPT = 1.8737767179767686

# Suavizado (solo biomasa)
SMOOTH_WINDOW = 3    # Ventana elegida

# Conversión EtOH % v/v -> g/L (densidad a 20°C)
ETHANOL_DENSITY_G_ML = 0.78924  # g/mL a 20 °C
# =================== FIN CONFIG ===================


# ---------- utilidades base ----------
def normalize_ensayo(code: object) -> str:
    s = str(code).upper().strip()
    m = re.search(r"SB\s*0*?(\d{1,3})", s)
    if m:
        return f"SB{int(m.group(1)):03d}"
    m2 = re.search(r"SB[^0-9]*([0-9]{1,3})", s)
    if m2:
        return f"SB{int(m2.group(1)):03d}"
    return s

def moving_average_centered(a, w=3):
    s = pd.Series(a, dtype=float)
    return s.rolling(window=w, center=True, min_periods=1).mean().values

def load_bdd(path=FILE_PATH, sheet=SHEET_BDD):
    df = pd.read_excel(path, sheet_name=sheet)
    df["Ensayo_norm"] = df["Ensayo"].apply(normalize_ensayo)
    return df

def _as_float(series_like):
    return pd.to_numeric(pd.Series(series_like), errors="coerce").to_numpy(dtype=float)


# ---------- Normalización de nombres "ID Análisis" ----------
ANALYTE_MAP = {
    # Biomasa
    "CONCENTRATION": "Concentration",
    "VIABILITY": "Viability",
    # Nitrógeno
    "YAN": "YAN",
    "AMMONIA": "AMMONIA",
    "PAN": "PAN",
    # Azúcares y metabolitos
    "FRUCTOSE": "Fructose",
    "GLUCOSE": "Glucose",
    "GLYCEROL": "Glycerol",
    "ETANOL": "Ethanol",
    "ETHANOL": "Ethanol",
    # Otros posibles
    "PYRUVIC ACID": "Pyruvic_Acid",
    "L-MALIC ACID": "L_Malic_Acid",
    "PESO SECO": "Peso_Seco",
    "PESO HUMEDO": "Peso_Humedo",
}

def normalize_analysis_name(name: object) -> str:
    if name is None:
        return ""
    s = str(name).strip().upper()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("_", " ")
    return ANALYTE_MAP.get(s, s.title())


# ---------- Detección y construcción de timestamp ----------
def _find_timestamp_columns(columns):
    cols = [c for c in columns]
    patterns = [
        r"fecha\s*y\s*hora", r"fecha_y_hora", r"datetime", r"timestamp",
        r"fecha", r"date", r"hora", r"time"
    ]
    ordered = []
    for pat in patterns:
        for c in cols:
            if re.search(pat, str(c), flags=re.I):
                if c not in ordered:
                    ordered.append(c)
    return ordered

def _build_timestamp_series(sub: pd.DataFrame) -> Optional[pd.Series]:
    cand = _find_timestamp_columns(sub.columns)
    if not cand:
        return None
    for name in cand:
        # (pd >= 2.0) infer_datetime_format ya no es necesario
        s = pd.to_datetime(sub[name], errors="coerce", dayfirst=True)
        if s.notna().any():
            # Si hay columna 'Hora' separada, intentar combinar
            if re.search(r"fecha$", str(name), flags=re.I):
                for hname in cand:
                    if re.search(r"(hora|time)$", str(hname), flags=re.I):
                        h = pd.to_datetime(sub[hname].astype(str), errors="coerce", format="%H:%M:%S")
                        if h.notna().any():
                            s_comb = s.copy()
                            mask = s.notna() & h.notna()
                            s_comb.loc[mask] = pd.to_datetime(
                                s.loc[mask].dt.date.astype(str) + " " + h.loc[mask].dt.time.astype(str),
                                errors="coerce"
                            )
                            s = s_comb
                            break
            return s
    return None


# ---------- Extracción y armado "wide" por ensayo ----------
def extract_assay(df_bdd: pd.DataFrame, assay_code: str) -> pd.DataFrame:
    """
    Devuelve 'wide' por ensayo con columnas de interés en float y
    eje temporal 'time_days' desde la primera muestra.
    Si hay timestamp, también devuelve columna 'timestamp' (datetime64).
    Ahora incluye 'Código' alineado por muestra.
    """
    sub = df_bdd[df_bdd["Ensayo_norm"] == assay_code].copy()
    if sub.empty:
        return pd.DataFrame()

    ts_series = _build_timestamp_series(sub)

    keep = [
        "Concentration", "Viability", "YAN", "AMMONIA", "PAN",
        "Fructose", "Glucose", "Glycerol", "Ethanol"
    ]

    id_cols = [c for c in sub.columns if "ID Análisis" in c]
    if not id_cols:
        return pd.DataFrame()
    val_cols = [c.replace("ID Análisis", "Valor") for c in id_cols]

    frames = []
    for idc, vc in zip(id_cols, val_cols):
        # Propagamos 'Código' para poder reinyectarlo tras el pivot
        cols_take = ["Ensayo_norm", "Código", idc, vc]
        take = [c for c in cols_take if c in sub.columns]  # por si faltara 'Código'
        tmp = sub[take].rename(columns={idc: "Analisis", vc: "Valor"})
        tmp["Analisis_norm"] = tmp["Analisis"].apply(normalize_analysis_name)
        if ts_series is not None:
            tmp["__ts__"] = ts_series.values
        frames.append(tmp)

    long_df = pd.concat(frames, ignore_index=True).dropna(subset=["Analisis_norm", "Valor"])
    key = long_df[long_df["Analisis_norm"].isin(keep)].copy()
    if key.empty:
        return pd.DataFrame()

    # --- Con timestamp ---
    if "__ts__" in key.columns and key["__ts__"].notna().any():
        key["__ts__"] = pd.to_datetime(key["__ts__"], errors="coerce")
        key = key.dropna(subset=["__ts__"])

        # Mapa (Ensayo_norm, __ts__) -> Código
        if "Código" in key.columns:
            code_map = (key[["Ensayo_norm", "__ts__", "Código"]]
                        .dropna(subset=["Código"])
                        .drop_duplicates(subset=["Ensayo_norm", "__ts__"]))
        else:
            code_map = None

        wide = key.pivot_table(
            index=["Ensayo_norm", "__ts__"],
            columns="Analisis_norm",
            values="Valor",
            aggfunc="first"
        ).reset_index()

        wide = wide.sort_values("__ts__").reset_index(drop=True)
        if code_map is not None:
            wide = wide.merge(code_map, on=["Ensayo_norm", "__ts__"], how="left")

        t0 = wide["__ts__"].iloc[0]
        wide["time_days"] = (wide["__ts__"] - t0).dt.total_seconds() / 86400.0
        wide["timestamp"] = wide["__ts__"]
        wide["idx"] = np.arange(len(wide))

    # --- Sin timestamp (fallback por índice) ---
    else:
        # Misma lógica de idx que tenías, y mapeamos (Ensayo_norm, idx) -> Código
        key["idx"] = key.groupby(["Ensayo_norm", "Analisis_norm"]).cumcount()

        if "Código" in key.columns:
            code_map = (key[["Ensayo_norm", "idx", "Código"]]
                        .dropna(subset=["Código"])
                        .drop_duplicates(subset=["Ensayo_norm", "idx"]))
        else:
            code_map = None

        wide = key.pivot_table(
            index=["Ensayo_norm", "idx"],
            columns="Analisis_norm",
            values="Valor",
            aggfunc="first"
        ).reset_index()

        wide = wide.sort_values("idx").reset_index(drop=True)
        if code_map is not None:
            wide = wide.merge(code_map, on=["Ensayo_norm", "idx"], how="left")

        wide["time_days"] = wide["idx"].astype(float)  # fallback

    # Tipado numérico de columnas clave
    for col in keep:
        if col in wide.columns:
            wide[col] = pd.to_numeric(wide[col], errors="coerce")

    # Reordenar para que 'Código' quede visible cerca del frente
    front = ["Ensayo_norm"]
    if "__ts__" in wide.columns: front.append("__ts__")
    if "idx" in wide.columns: front.append("idx")
    if "timestamp" in wide.columns: front.append("timestamp")
    if "Código" in wide.columns: front.append("Código")
    others = [c for c in wide.columns if c not in front]
    wide = wide[front + others]

    return wide

# ---------- Isotónica y helpers numéricos ----------
def safe_interpolate_nan(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return y
    mask = ~np.isnan(y)
    if mask.sum() == 0:
        return np.zeros_like(y, dtype=float)
    first = np.argmax(mask)
    last  = n - 1 - np.argmax(mask[::-1])
    y[:first] = y[first]
    y[last+1:] = y[last]
    idx = np.arange(n, dtype=float)
    y_interp = y.copy()
    holes = ~mask
    if holes.any():
        y_interp[holes] = np.interp(idx[holes], idx[mask], y[mask])
    return y_interp

def correct_with_isotonic(conc, pan, amm, pan_thr=PAN_THR, amm_thr=AMM_THR, run_len=RUN_LEN) -> np.ndarray:
    conc = pd.to_numeric(pd.Series(conc), errors="coerce").to_numpy(dtype=float)
    pan  = pd.to_numeric(pd.Series(pan ), errors="coerce").to_numpy(dtype=float)
    amm  = pd.to_numeric(pd.Series(amm ), errors="coerce").to_numpy(dtype=float)

    n = len(conc)
    if n == 0:
        return conc

    valid = (~np.isnan(pan)) & (~np.isnan(amm))
    below = (pan <= pan_thr) & (amm <= amm_thr) & valid
    cut_idx, count = None, 0
    for i, v in enumerate(below):
        count = count + 1 if v else 0
        if count >= run_len:
            cut_idx = i - run_len + 1
            break
    if cut_idx is None:
        cut_idx = n - 1

    y_seg = conc[:cut_idx+1]
    y_seg = safe_interpolate_nan(y_seg)

    x_seg = np.arange(len(y_seg), dtype=float)
    if len(y_seg) == 1:
        y_iso = y_seg.copy()
    else:
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        y_iso = iso.fit_transform(x_seg, y_seg)

    conc_corr = np.empty_like(conc, dtype=float)
    conc_corr[:cut_idx+1] = y_iso
    conc_corr[cut_idx+1:] = y_iso[-1]
    return conc_corr


# ---------- Pipeline por ensayo ----------
def process_one_assay(wide: pd.DataFrame, assay_code: str) -> pd.DataFrame:
    """Devuelve dataframe con columnas *_raw, *_adj y *_ma3 para el ensayo (robusto a texto/NaN)."""
    wide = wide.copy()
    for c in ["Concentration","Viability","YAN","AMMONIA","PAN"]:
        if c not in wide.columns:
            wide[c] = np.nan

    conc = _as_float(wide["Concentration"])
    viab = _as_float(wide["Viability"])
    pan  = _as_float(wide["PAN"])
    amm  = _as_float(wide["AMMONIA"])

    # 1) Corrección isotónica (hasta corte)
    conc_corr = correct_with_isotonic(conc, pan, amm)

    # 2) Ratio muerto y separación
    with np.errstate(divide='ignore', invalid='ignore'):
        dead_ratio = np.divide(conc - viab, conc, out=np.zeros_like(conc, dtype=float), where=conc>0)
    dead_ratio = np.clip(dead_ratio, 0.0, 1.0)

    total_gL_raw = SLOPE * conc_corr + INTERCEPT
    dead_gL_raw  = total_gL_raw * dead_ratio
    viable_gL_raw= total_gL_raw - dead_gL_raw

    # 3) Corrección suave: quitar INTERCEPT y repartir por fracciones
    with np.errstate(divide='ignore', invalid='ignore'):
        frac_viab = np.where(total_gL_raw > 0, viable_gL_raw/total_gL_raw, 0.5)
    frac_dead = 1.0 - frac_viab

    total_adj = np.maximum(total_gL_raw - INTERCEPT, 0.0)
    viable_adj= total_adj * frac_viab
    dead_adj  = total_adj * frac_dead

    # 4) Anclaje t0 al inóculo
    inoc_g = INOCULUM_G.get(assay_code, None)
    inoc_gL = (inoc_g / REACTOR_VOL_L) if (inoc_g is not None and not pd.isna(inoc_g)) else 0.0
    f0 = float(frac_viab[0]) if total_gL_raw[0] > 0 else 0.5
    viable_adj[0] = inoc_gL * f0
    dead_adj[0]   = inoc_gL * (1.0 - f0)
    total_adj[0]  = viable_adj[0] + dead_adj[0]

    out = wide.copy()
    out["total_gL_raw"]  = total_gL_raw
    out["viable_gL_raw"] = viable_gL_raw
    out["dead_gL_raw"]   = dead_gL_raw

    out["total_gL_adj"]  = total_adj
    out["viable_gL_adj"] = viable_adj
    out["dead_gL_adj"]   = dead_adj

    # 5) Suavizado MA(3) SOLO para biomasa
    out["viable_gL_ma3"] = pd.Series(out["viable_gL_adj"]).rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean().values
    out["dead_gL_ma3"]   = pd.Series(out["dead_gL_adj"]).rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean().values
    out["total_gL_ma3"]  = pd.Series(out["total_gL_adj"]).rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean().values

    # YAN crudo; NO suavizamos N
    if "YAN" in out.columns:
        out["YAN"] = pd.to_numeric(out["YAN"], errors="coerce")

    # Tiempos
    if "time_days" not in out.columns and "idx" in out.columns:
        out["time_days"] = out["idx"].astype(float)
    out["time_hours"] = out["time_days"] * 24.0

    out.insert(0, "Ensayo", assay_code)
    return out


# ---------- Orquestado ----------
def process_all(file_path=FILE_PATH, assays=None):
    """Procesa todos los ensayos solicitados. Devuelve (results_dict, combined_df)."""
    bdd = load_bdd(file_path)
    all_codes = sorted(bdd["Ensayo_norm"].dropna().unique())
    target = assays if assays is not None else all_codes

    results = {}
    frames = []
    for code in target:
        wide = extract_assay(bdd, code)
        if wide.empty:
            continue
        df_res = process_one_assay(wide, code)
        results[code] = df_res
        frames.append(df_res)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    
    return results, combined


# ---------- Helper robusto para columnas opcionales ----------
def _col_as_float_vector(df: pd.DataFrame, col: str, n: int) -> np.ndarray:
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        return s.to_numpy(dtype=float)
    else:
        return np.full(n, np.nan, dtype=float)


# ---------- Carga e interpolación de temperatura ----------
def _load_temperature_table_for_assay(assay_code: str) -> Optional[pd.DataFrame]:
    """Lee 'Data <ID>.xlsx' / hoja 'Manual Temperaturas' y devuelve DF con columnas:
       'ts_temp' (datetime64), 'temp_C' (float), y 'time_h_rel' (horas desde t0 temp).
    """
    ens_id = SB2ID.get(assay_code)
    if ens_id is None:
        return None

    fname = f"Data {ens_id}.xlsx"
    fpath = os.path.join(TEMPS_DIR, fname)
    if not os.path.exists(fpath):
        return None

    try:
        dfT = pd.read_excel(fpath, sheet_name=TEMP_SHEET)
    except Exception:
        return None

    if TEMP_DATE_COL not in dfT.columns or TEMP_VALUE_COL not in dfT.columns:
        return None

    ts = pd.to_datetime(dfT[TEMP_DATE_COL], errors="coerce")
    temp = pd.to_numeric(dfT[TEMP_VALUE_COL], errors="coerce")
    ok = ts.notna() & temp.notna()
    dfT = pd.DataFrame({"ts_temp": ts[ok], "temp_C": temp[ok]}).dropna().sort_values("ts_temp")
    if dfT.empty:
        return None

    t0 = dfT["ts_temp"].iloc[0]
    dfT["time_h_rel"] = (dfT["ts_temp"] - t0).dt.total_seconds() / 3600.0
    return dfT


def _interp_temperature_for_assay(df_assay: pd.DataFrame, assay_code: str) -> np.ndarray:
    """Devuelve vector de temperatura (°C) alineado a df_assay, usando timestamp si existe;
       si no, interpola por tiempo relativo (time_hours) contra 'time_h_rel' del archivo de temperaturas.
    """
    dfT = _load_temperature_table_for_assay(assay_code)
    n = len(df_assay)
    if dfT is None or n == 0:
        return np.full(n, np.nan, dtype=float)

    # Preferimos alinear por timestamp si el ensayo lo tiene
    if "timestamp" in df_assay.columns and df_assay["timestamp"].notna().any():
        ts_samp = pd.to_datetime(df_assay["timestamp"], errors="coerce")
        if ts_samp.notna().any():
            # construir eje relativo con base en el primer timestamp de temperatura
            t0T = dfT["ts_temp"].iloc[0]
            t_rel_samp = (ts_samp - t0T).dt.total_seconds() / 3600.0
            # interp tipo 1D (extrapolación por extremos)
            x = dfT["time_h_rel"].to_numpy(dtype=float)
            y = dfT["temp_C"].to_numpy(dtype=float)
            xr = np.clip(t_rel_samp.to_numpy(dtype=float), x.min(), x.max())
            return np.interp(xr, x, y)

    # Fallback: alinear por tiempo relativo (ensayo) vs tiempo relativo (temperatura)
    t_h = pd.to_numeric(df_assay.get("time_hours", np.arange(n)*1.0), errors="coerce").to_numpy(dtype=float) # type: ignore
    x = dfT["time_h_rel"].to_numpy(dtype=float)
    y = dfT["temp_C"].to_numpy(dtype=float)
    xr = np.clip(t_h, x.min(), x.max())
    return np.interp(xr, x, y)


def attach_temperature_to_results(results: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Añade columna 'Temperature_C' a cada DF de results con la temperatura interpolada."""
    out = {}
    for code, df in results.items():
        df2 = df.copy()
        df2["Temperature_C"] = _interp_temperature_for_assay(df2, code)
        out[code] = df2
    return out


# ---------- MATRICES PARA CALIBRACIÓN (tiempo en horas) ----------
def build_calibration_matrices(results: dict,
                               use_smoothed_biomass: bool = False,
                               cols_order: List[str] = None) -> Dict[str, pd.DataFrame]: # type: ignore
    """
    Construye una matriz por ensayo para calibración:
    columnas = [time_h, BiomasaViable_gL, BiomasaMuerta_gL, YAN, AMMONIA, PAN, Fructose, Glucose, Glycerol, Ethanol, Temperature_C]
    - time_h en HORAS desde t0 del ensayo (float)
    - biomasa: usa *_adj por defecto, o *_ma3 si use_smoothed_biomass=True
    Devuelve: dict { ensayo -> DataFrame }
    """
    default_cols = [
        "time_h",
        "biomass_viable_gL", "biomass_dead_gL",
        "YAN", "AMMONIA", "PAN",
        "Fructose", "Glucose", "Glycerol", "Ethanol",
        "Temperature_C",
    ]
    if cols_order is None:
        cols_order = default_cols

    out = {}
    for code, df in results.items():
        # tiempo en horas
        if "time_hours" in df.columns:
            t_h = df["time_hours"].to_numpy(dtype=float)
        elif "time_days" in df.columns:
            t_h = (df["time_days"] * 24.0).to_numpy(dtype=float)
        else:
            t_h = df["idx"].astype(float).to_numpy()

        n = len(t_h)

        # biomasa
        if use_smoothed_biomass:
            viable = df.get("viable_gL_ma3", df["viable_gL_adj"]).to_numpy(dtype=float)
            dead   = df.get("dead_gL_ma3",   df["dead_gL_adj"]).to_numpy(dtype=float)
        else:
            viable = df["viable_gL_adj"].to_numpy(dtype=float)
            dead   = df["dead_gL_adj"].to_numpy(dtype=float)

        # analitos adicionales
        yan  = _col_as_float_vector(df, "YAN",      n)
        amm  = _col_as_float_vector(df, "AMMONIA",  n)
        pan  = _col_as_float_vector(df, "PAN",      n)
        fru  = _col_as_float_vector(df, "Fructose", n)
        glu  = _col_as_float_vector(df, "Glucose",  n)
        glyc = _col_as_float_vector(df, "Glycerol", n)

        # Etanol en % v/v -> g/L
        etoh_pct = _col_as_float_vector(df, "Ethanol", n)  # % v/v
        etoh_gL  = np.clip(etoh_pct * ETHANOL_DENSITY_G_ML * 10.0, 0.0, None)

        # temperatura (si no existe ya en df, la interpolamos on-the-fly)
        if "Temperature_C" in df.columns:
            tempC = _col_as_float_vector(df, "Temperature_C", n)
        else:
            tempC = _interp_temperature_for_assay(df, code)

        mat = pd.DataFrame({
            "time_h": t_h,
            "biomass_viable_gL": viable,
            "biomass_dead_gL": dead,
            "YAN": yan,
            "AMMONIA": amm,
            "PAN": pan,
            "Fructose": fru,
            "Glucose": glu,
            "Glycerol": glyc,
            "Ethanol": etoh_gL,           # <- ahora en g/L
            "Temperature_C": tempC,
        })

        mat = mat.reindex(columns=cols_order)
        mat = mat.sort_values("time_h").reset_index(drop=True)
        out[code] = mat

    return out


# ---------- Gráficos (eje X en días) ----------
def plot_panel(results: dict, ncols=3, smooth=True):
    """Panel de subplots con Viable/Muerta/TOTAL (g/L) y YAN (YAN sin suavizar).
       Eje X = tiempo en días desde la primera muestra de cada ensayo.
    """
    assays = list(results.keys())
    n = len(assays)
    if n == 0:
        print("No hay ensayos para graficar.")
        return
    nrows = ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4.8*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])

    COL_VIABLE="#1f77b4"; COL_DEAD="#d62728"; COL_TOTAL="#2ca02c"; COL_YAN="#ff7f0e"

    for ax, code in zip(axes.flat, assays):
        df = results[code]
        x = df["time_days"].values if "time_days" in df.columns else df["idx"].astype(float).values

        # puntos biomasa ajustada
        ax.plot(x, df["viable_gL_adj"], 'o', color=COL_VIABLE, label="Viable (g/L)", alpha=0.85, ms=4)
        ax.plot(x, df["dead_gL_adj"],   'v', color=COL_DEAD,   label="Muerta (g/L)", alpha=0.85, ms=4)
        ax.plot(x, df["total_gL_adj"],  's', color=COL_TOTAL,  label="TOTAL (g/L)",  alpha=0.85, ms=4)

        # líneas suavizadas SOLO biomasa
        if smooth:
            ax.plot(x, df["viable_gL_ma3"], '-', color=COL_VIABLE, lw=2)
            ax.plot(x, df["dead_gL_ma3"],   '-', color=COL_DEAD,   lw=2)
            ax.plot(x, df["total_gL_ma3"],  '-', color=COL_TOTAL,  lw=2)

        ax.set_title(f"Ensayo {code}")
        ax.set_xlabel("Tiempo [días]")
        ax.set_ylabel("Biomasa [g/L]")
        ax.grid(True, alpha=0.3)

        # YAN (solo crudo, sin suavizado)
        ax2 = ax.twinx()
        if "YAN" in df.columns:
            ax2.plot(x, df["YAN"].values, 'd--', color=COL_YAN, label="YAN", lw=1.2, ms=4, alpha=0.9)
        ax2.set_ylabel("YAN [u.a.]")

        if code == assays[0]:
            l1, lab1 = ax.get_legend_handles_labels()
            l2, lab2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lab1 + lab2, loc="upper left", fontsize=9)

    for i in range(len(assays), nrows*ncols):
        axes.flat[i].set_visible(False)

    plt.suptitle(f"Biomasa viable, muerta y TOTAL (g/L) — MA({SMOOTH_WINDOW}) solo biomasa + YAN crudo\nEje X = tiempo desde t0 (días)", fontsize=13)
    plt.tight_layout(rect=[0,0,1,0.94]) # type: ignore
    plt.show()


# ===================== EJEMPLO DE USO =====================
if __name__ == "__main__":
    # 0) (opcional) verificar carpeta de temperaturas
    if not os.path.isdir(TEMPS_DIR):
        print(f"[ADVERTENCIA] Carpeta de temperaturas no encontrada: {TEMPS_DIR}\n"
              f"Se crearán matrices con Temperature_C = NaN donde no haya archivo.")
    # 1) Procesar todo lo disponible en la BDD
    results_dict, combined_df = process_all(FILE_PATH, assays=None)

    # 1.1) Adjuntar la temperatura interpolada a cada ensayo (no es estrictamente necesario,
    #      porque build_calibration_matrices también interpola si falta)
    results_with_T = attach_temperature_to_results(results_dict)

    # 2) Graficar (3 columnas por defecto) con eje X en días
    plot_panel(results_with_T, ncols=3, smooth=True)

    # 3) Construir matrices para calibración (tiempo en horas) con biomasa AJUSTADA (no suavizada)
    mats = build_calibration_matrices(results_with_T, use_smoothed_biomass=True)

    # 4) Inspección rápida: mostrar tamaño y primeras filas de 2 ensayos (si existen)
    keys = list(mats.keys())
    print("\n=== Resumen matrices de calibración (con Temperature_C) ===")
    for code in keys[:2]:
        print(f"\nEnsayo {code}  -> matriz shape: {mats[code].shape}")
        cols_show = ["time_h","biomass_viable_gL","biomass_dead_gL","YAN","Glucose","Fructose","Ethanol","Temperature_C"]
        cols_show = [c for c in cols_show if c in mats[code].columns]
        print(mats[code][cols_show].head(8).to_string(index=False))

    # 5) (Opcional) Exportar todas las matrices a CSV
    # os.makedirs("calib_matrices", exist_ok=True)
    # for code, dfm in mats.items():
    #     dfm.to_csv(os.path.join("calib_matrices", f"calib_matrix_{code}.csv"), index=False)
