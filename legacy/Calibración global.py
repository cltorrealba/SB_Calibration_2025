# -*- coding: utf-8 -*-
"""
Calibración global (stiff + °C->K + multistart con early stopping)

Cambios clave:
- Conversión explícita de temperatura °C -> K (T_K = T_C + 273.15)
- Integración stiff con solve_ivp(method='Radau') y tolerancias por estado
- Pulsos de N como saltos instantáneos en N: se integra por tramos y se aplica ΔN
- Multistart con early stopping si no hay mejoras del SSE

Requiere: numpy, pandas, matplotlib, scipy
Usa: modelo_dinamico_sim.py y Calibration_data_preprocess.py
"""

import os  # <- mover antes de uso en constantes
import sys, time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution, minimize
from scipy.integrate import solve_ivp

# === Tu simulador / loader (asegúrate de que esté importable) ===
from modelo_dinamico_sim import (
    zenteno_model,           # dxdt = f(t, x, u=[T_K, _], p)
    load_parameters_from_excel,
    DEFAULT_X0
    )

# ================== CONFIGURACIÓN ==================
# Ruta al Excel de parámetros iniciales
PARAM_XLSX  = "zenteno_parameters.xlsx"   # <-- EDITA si es necesario
PARAM_SHEET = "Hoja1"
PARAM_SET   = 4

# Pesos base (antes de normalizar por std)
WEIGHTS = {"X": 1.0, "N": 1.0, "G": 1.0, "F": 1.0, "E": 1.0, "S": 1.0}  # S: azúcar total (G+F vs densidad)

# Límites "reales" (positivos) de parámetros (14)
P_BOUNDS_REAL = [
    (1e-2, 10),   # mu0
    (1e-2, 10),   # betaG0
    (1e-2, 10),   # betaF0
    (1e-3, 1.0),   # Kn0
    (1e-1, 100),   # Kg0
    (1e-1, 100),   # Kf0
    (1e-1, 100),   # Kig0
    (1e-1, 100),   # Kie0
    (1e-5, 1e-3),   # Kd0
    (1e-1, 10),   # Yxn
    (1e-1, 10),   # Yxg
    (1e-1, 10),   # Yxf
    (1e-1, 10),   # Yeg
    (1e-1, 10),   # Yef
]

N_SCALE = 1e-3  # mg/L -> g/L

TF_HOURS   = 14*24.0

DEFAULT_TC = 20.0  # °C para fallback, si no hay columna temperatura
EXCLUDE_ASSAYS = {"SB001","SB002","SB012"}  # puedes añadir más

# Pulsos default (si no pasas por ensayo). Lista de (t_h, ΔN_gL).
PULSOS_N_DEF = [(40.0, 0.045)]

# Integración stiff
USE_STIFF  = True
STIFF_METH = "Radau"  # "Radau" o "BDF"
RTOL       = 1e-6
ATOL_VEC   = np.array([1e-3, 1e-2, 1e-2, 1e-2, 1e-3], dtype=float)  # [X,N,G,F,E]

# Multistart con early stopping
MODE            = "de"
N_STARTS        = 10
LOCAL_MAXITER   = 300
LOCAL_FTOL      = 1e-9
ES_PATIENCE     = 5       # nº de starts sin mejora tras los cuales se corta
ES_REL_IMPROVE  = 1e-3    # mejora relativa mínima para resetear paciencia
EPS = 1e-9

# --- NUEVOS PARÁMETROS DE CONTROL ---
RANDOM_SEED          = 12345
SSE_BALANCE_MODE     = "per_point"   # "none" | "per_point" | "per_assay"
SSE_RESAMPLE_DT_H    = 6.0           # None o valor (ej: 6.0) para submuestreo temporal en SSE
SAVE_ARTIFACTS_DIR   = "mats"        # carpeta salida artefactos Step 3
PERSIST_ARTIFACTS    = True          # activar persistencia
PLOT_MAX_DAYS       = 20.0          # límite superior fijo eje X (días)

# === NUEVA CONFIG INTEGRACIÓN 2024/2025 ===
METADATA_CSV          = "metadata_2024_2025.csv"
SPLITS_DIR            = "splits"
TRAIN_IDS_CSV         = os.path.join(SPLITS_DIR, "train_ids.csv")
VALID_IDS_CSV         = os.path.join(SPLITS_DIR, "valid_ids.csv")
DATA_DIR_2024         = "Datos Experimentales"        # donde están Data <code>.xlsx año 2024
DATA_DIR_2025_FILE    = "Procesos_I+D_2025_3.xlsx"
SUGAR_MODEL_TXT       = os.path.join("sugar_density_out", "sugar_density_model_coeffs.txt")
YAN_OFFSET_CORRECTION = 10.0        # ajuste offset YAN (antes 15.0) — se resta a datos exp
YAN_MIN_MG_L          = 0.0
USE_DENSITY_MODEL_2024 = True       # activar modelo densidad→azúcar en 2024
FILL_MISSING_G_F_SPLIT = 0.5        # reparto por defecto si faltan ambos azúcares

# ===================================================


# ---------- Helpers ----------
def build_temp_profile_from_df(df: pd.DataFrame) -> List[Tuple[float, float]]:
    """
    Devuelve segmentos [(t_h, T_C)] en °C. El modelo recibirá Kelvin
    más adelante (C + 273.15). Columnas aceptadas: Temperature_C / temperature /
    temp_c / temperatura.
    """
    cand = None
    for c in df.columns:
        l = str(c).strip().lower()
        if l in ("temperature_c", "temperature", "temp_c", "temperatura"):
            cand = c
            break
    if cand is None:
        return [(0.0, float(DEFAULT_TC))]

    t = pd.to_numeric(df["time_h"], errors="coerce")
    Tc = pd.to_numeric(df[cand], errors="coerce")
    m = ~(t.isna() | Tc.isna())
    t, Tc = t[m].to_numpy(float), Tc[m].to_numpy(float)
    if t.size == 0:
        return [(0.0, float(DEFAULT_TC))]

    order = np.argsort(t)
    t, Tc = t[order], Tc[order]
    segs = [(float(t[0]), float(Tc[0]))]
    for i in range(1, len(t)):
        if not np.isclose(Tc[i], Tc[i-1], atol=1e-6):
            segs.append((float(t[i]), float(Tc[i])))
    if segs[0][0] > 0.0:
        segs.insert(0, (0.0, segs[0][1]))
    return segs

def segments_C_to_profile_K(tf: float, n_pts: int, segs_C: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convierte segmentos [(t, T_C)] a perfil T_K(t) sobre malla uniforme de n_pts+1 nodos en [0, tf].
    Devuelve (t_u, T_K) con len = n_pts+1.
    """
    t_u = np.linspace(0.0, tf, n_pts+1)
    # perfil por tramos constantes
    T_c = np.empty_like(t_u)
    j = 0
    for k, tk in enumerate(t_u):
        while j+1 < len(segs_C) and tk >= segs_C[j+1][0] - 1e-12:
            j += 1
        T_c[k] = segs_C[j][1]
    T_k = T_c + 273.15
    return t_u, T_k

# --- Jacobiano: sparsity conocida del modelo ---
J_SPARSE = np.array([
    [1,1,0,0,1],
    [1,1,0,0,0],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
], dtype=bool)

def make_jacobian_num(zenteno_model, u_of_t, p, h_c=1e-20, h_fd=1e-6):
    """
    Devuelve jac(t,x) que usa complex-step (si es posible) o forward-diff.
    - zenteno_model: f(t, x, u, p)
    - u_of_t: callable t -> u vector (p.ej. [T_K, 0])
    - p: parámetros reales del modelo
    - h_c: paso complex-step (muy pequeño, p.ej. 1e-20)
    - h_fd: paso finito en fallback
    """
    n = 5

    def f_real(t, x):
        return np.asarray(zenteno_model(t, x, u_of_t(t), p), dtype=float)

    def try_complex(t, x):
        # intenta complex-step; si falla, lanza excepción para fallback
        fx = zenteno_model(t, x, u_of_t(t), p)
        _ = np.asarray(fx, dtype=complex)  # fuerza camino complejo
        J = np.zeros((n, n), dtype=float)
        for j in range(n):
            # si la columna j es todo cero en sparsity, sáltala rápido
            if not J_SPARSE[:, j].any():
                continue
            xh = x.astype(complex) + 0j
            xh[j] += 1j*h_c
            fj = zenteno_model(t, xh, u_of_t(t), p)
            fj = np.asarray(fj, dtype=complex)
            # derivada = Im(f)/h_c (solo para filas marcadas)
            fj_complex = np.asarray(fj, dtype=complex)
            J[J_SPARSE[:, j], j] = (fj_complex[J_SPARSE[:, j]].imag) / h_c # type: ignore
        return J

    def forward_diff(t, x):
        fx = f_real(t, x)
        J = np.zeros((n, n), dtype=float)
        for j in range(n):
            if not J_SPARSE[:, j].any():
                continue
            xh = x.copy()
            step = h_fd * max(1.0, abs(xh[j]))
            xh[j] += step
            fj = f_real(t, xh)
            J[J_SPARSE[:, j], j] = (fj[J_SPARSE[:, j]] - fx[J_SPARSE[:, j]]) / step
        return J

    def jac(t, x):
        # intenta complex-step una vez; si falla, usa forward diff
        try:
            return try_complex(t, x)
        except Exception:
            return forward_diff(t, x)

    return jac


def simulate_stiff_with_pulses(p, t_meas_h, temp_segments_C, pulses, x0):
    tf = float(np.nanmax(t_meas_h)) if np.isfinite(np.nanmax(t_meas_h)) and np.nanmax(t_meas_h) > 0 else TF_HOURS

    # Malla de control (°C->K)
    n_u = max(int(np.ceil(tf/0.25)), 200)
    t_u, T_K = segments_C_to_profile_K(tf, n_u, temp_segments_C)

    def u_of_t(t):
        T = np.interp(t, t_u, T_K)
        return np.array([T, 0.0], dtype=float)

    # f envoltorio
    def f_ivp(t, x):
        return np.asarray(zenteno_model(t, x, u_of_t(t), p), dtype=float)

    # << NUEVO: Jacobiano numérico con sparsity >>
    jac = make_jacobian_num(zenteno_model, u_of_t, p, h_c=1e-20, h_fd=1e-6)

    # Pulsos ordenados
    pulses = [(float(max(0.0, min(tf, t))), float(dN)) for (t, dN) in (pulses or [])]
    pulses = sorted(list({(t, dN) for (t, dN) in pulses}), key=lambda z: z[0])

    breakpoints = [0.0] + [t for (t, _) in pulses if 0.0 < t < tf] + [tf]
    x_curr = np.asarray(x0, dtype=float).copy()
    t_all = [breakpoints[0]]
    X_all = [x_curr.copy()]

    for i in range(len(breakpoints) - 1):
        ta, tb = breakpoints[i], breakpoints[i+1]
        if tb - ta >= 1e-9:
            sol = solve_ivp(
                f_ivp, (ta, tb), x_curr,
                method=STIFF_METH, rtol=RTOL, atol=ATOL_VEC, dense_output=True,
                jac=jac,                        # << usa Jacobiano numérico
                jac_sparsity=J_SPARSE           # << y su sparsity
            )
            if not sol.success:
                # relajar tolerancias si falló
                sol = solve_ivp(
                    f_ivp, (ta, tb), x_curr,
                    method=STIFF_METH, rtol=max(RTOL*10, 1e-5), atol=np.maximum(ATOL_VEC*10, 1e-2),
                    dense_output=True, jac=jac, jac_sparsity=J_SPARSE
                )
            t_seg = sol.t; X_seg = sol.y.T
            if len(t_seg) > 0:
                if np.isclose(t_seg[0], t_all[-1]):
                    t_seg = t_seg[1:]; X_seg = X_seg[1:]
                t_all.extend(t_seg.tolist()); X_all.extend(X_seg.tolist())
                x_curr = X_seg[-1].copy()

        # aplicar pulso en tb si corresponde
        for (tp, dN) in pulses:
            if np.isclose(tp, tb, atol=1e-12):
                x_curr = x_curr.copy()
                x_curr[1] = max(0.0, x_curr[1] + dN)
                t_all.append(tb); X_all.append(x_curr.copy())

    t_all = np.asarray(t_all, dtype=float)
    X_all = np.asarray(X_all, dtype=float)
    if t_all[-1] < tf:
        t_all = np.append(t_all, tf); X_all = np.vstack([X_all, X_all[-1]])
    return t_all, X_all

def simulate_on_grid(p_real, time_h, temp_segments, pulses, x0):
    """
    Versión que SÍ aplica pulsos instantáneos (ΔN):
    delega en simulate_stiff_with_pulses(.) que integra por tramos y
    aplica ΔN al final de cada tramo exactamente en el pulso.
    También asegura que el perfil térmico entre al modelo en Kelvin.
    """
    if x0 is None:
        x0 = DEFAULT_X0.copy()

    # temp_segments viene en °C; simulate_stiff_with_pulses hace C->K internamente
    t_sim, Xsim = simulate_stiff_with_pulses(
        p=p_real,
        t_meas_h=np.asarray(time_h, dtype=float),
        temp_segments_C=temp_segments,
        pulses=pulses,
        x0=x0
    )
    return t_sim, Xsim

# ---------- Vista previa ----------
def preview_data(mats: Dict[str, pd.DataFrame], max_print: int = 6, plot_temp: bool = True):
    kept = [k for k in mats.keys() if k not in EXCLUDE_ASSAYS]
    print("\n=== Ensayos incluidos (excluyendo SB001/SB002) ===")
    print(", ".join(kept))
    print("\n=== Resumen por ensayo ===")
    for code in kept[:max_print]:
        df = mats[code]
        t = pd.to_numeric(df["time_h"], errors="coerce")
        Tmin, Tmax = np.nanmin(t), np.nanmax(t)
        Tc = None
        for c in ("Temperature_C","temperature","temp_c","temperatura"):
            if c in df.columns:
                Tc = pd.to_numeric(df[c], errors="coerce"); break
        tmsg = f"{Tmin:.2f}h → {Tmax:.2f}h" if np.isfinite(Tmin) and np.isfinite(Tmax) else "NA"
        if Tc is not None:
            print(f"{code}: n={len(df)}  t:{tmsg}  Temp[°C] min/mean/max = {np.nanmin(Tc):.2f}/{np.nanmean(Tc):.2f}/{np.nanmax(Tc):.2f}")
        else:
            print(f"{code}: n={len(df)}  t:{tmsg}  (sin columna de temperatura)")
    if plot_temp and kept:
        keys = kept
        n = len(keys); ncols=2; nrows=int(np.ceil(n/ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 3.5*nrows))
        axes = np.atleast_2d(axes)
        for i, code in enumerate(keys):
            ax = axes.flat[i]
            df = mats[code]
            t = pd.to_numeric(df["time_h"], errors="coerce").to_numpy(dtype=float)/24.0
            Tc = None
            for c in ("Temperature_C","temperature","temp_c","temperatura"):
                if c in df.columns:
                    Tc = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float); break
            if Tc is not None:
                ax.plot(t, Tc, 'o-', ms=3)
                ax.set_title(f"{code} · Temp vs días")
                ax.set_xlabel("t [d]"); ax.set_ylabel("T [°C]"); ax.grid(True, alpha=0.3)
            else:
                ax.set_visible(False)
        plt.tight_layout(); plt.show()

# ---------- Normalización (std global por variable) ----------
def compute_global_stds(mats: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    vals = {"X": [], "N": [], "G": [], "F": [], "E": [], "S": []}
    for code, df in mats.items():
        if code in EXCLUDE_ASSAYS: continue
        if "biomass_viable_gL" in df: vals["X"].extend(pd.to_numeric(df["biomass_viable_gL"], errors="coerce").dropna().values.tolist())
        if "YAN"               in df:vals["N"].extend( (pd.to_numeric(df["YAN"], errors="coerce").dropna().values.astype(float) * N_SCALE).tolist() )        
        if "Glucose"           in df: vals["G"].extend(pd.to_numeric(df["Glucose"], errors="coerce").dropna().values.tolist())
        if "Fructose"          in df: vals["F"].extend(pd.to_numeric(df["Fructose"], errors="coerce").dropna().values.tolist())
        if "Ethanol"           in df: vals["E"].extend(pd.to_numeric(df["Ethanol"], errors="coerce").dropna().values.tolist())
        if "SugarTotal_exp" in df:
            vals["S"].extend(pd.to_numeric(df["SugarTotal_exp"], errors="coerce").dropna().values.tolist())
    stds = {}
    for k, arr in vals.items():
        if len(arr) == 0:
            stds[k] = 1.0
        else:
            s = float(np.nanstd(np.asarray(arr, dtype=float)))
            stds[k] = s if s > 0 else 1.0
    return stds

# --- NUEVO: cálculo de rangos estándar para ejes (evitar auto escalado) ---
def compute_standard_axis_ranges(mats: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Determina límites globales para:
      - time_max_days : eje X (0 .. ceil(max(time_h)/24))
      - y_left_max    : eje Y izquierdo (N mg/L, G, F, E, S(G+F))
      - y_right_max   : eje Y derecho (X biomasa viable)
    Usa solo datos experimentales (robusto a NaN). Aplica margen +5%.
    """
    tmax_d = 0.0
    left_vals = []
    right_vals = []
    for code, df in mats.items():
        if "time_h" in df.columns:
            tmax_d = max(tmax_d, np.nanmax(pd.to_numeric(df["time_h"], errors="coerce"))/24.0)
        # N (YAN en mg/L ya)
        if "YAN" in df.columns:
            left_vals.extend(pd.to_numeric(df["YAN"], errors="coerce").dropna().tolist())
        # G, F, E
        for c in ("Glucose","Fructose","Ethanol"):
            if c in df.columns:
                left_vals.extend(pd.to_numeric(df[c], errors="coerce").dropna().tolist())
        # Azúcar total experimental (o derivada)
        if "SugarTotal_exp" in df.columns and not df["SugarTotal_exp"].isna().all():
            left_vals.extend(pd.to_numeric(df["SugarTotal_exp"], errors="coerce").dropna().tolist())
        else:
            if "Glucose" in df.columns and "Fructose" in df.columns:
                g = pd.to_numeric(df["Glucose"], errors="coerce")
                f = pd.to_numeric(df["Fructose"], errors="coerce")
                ssum = (g + f).dropna()
                left_vals.extend(ssum.tolist())
        # Biomasa viable
        if "biomass_viable_gL" in df.columns:
            right_vals.extend(pd.to_numeric(df["biomass_viable_gL"], errors="coerce").dropna().tolist())

    if tmax_d <= 0 or not np.isfinite(tmax_d):
        tmax_d = 1.0
    tmax_d = float(np.ceil(tmax_d))

    def _max_or_default(vals, dflt):
        if not vals:
            return dflt
        m = float(np.nanmax(np.asarray(vals, dtype=float)))
        if not np.isfinite(m):
            return dflt
        return m

    y_left_max  = _max_or_default(left_vals, 1.0)
    y_right_max = _max_or_default(right_vals, 1.0)

    # Margen 5%
    y_left_max  *= 1.05
    y_right_max *= 1.05

    # Evitar límites cero
    if y_left_max <= 0: y_left_max = 1.0
    if y_right_max <= 0: y_right_max = 1.0

    # Forzar límite máximo global en días
    tmax_d = min(tmax_d, PLOT_MAX_DAYS)
    return {
        "time_max_days": tmax_d,
        "y_left_max": y_left_max,
        "y_right_max": y_right_max
    }

# ---------- Costo (SSE) normalizado ----------
def sse_for_experiments_real(p_real: np.ndarray,
                             mats: Dict[str, pd.DataFrame],
                             pulses_by_assay: Optional[Dict[str, List[Tuple[float, float]]]] = None,
                             x0_by_assay: Optional[Dict[str, np.ndarray]] = None,
                             weights: Optional[Dict[str, float]] = None,
                             stds: Optional[Dict[str, float]] = None,
                             verbose: bool = False,
                             balance: str = "per_point",
                             resample_dt_h: Optional[float] = None) -> float:
    if weights is None: weights = WEIGHTS
    if stds is None: stds = {k:1.0 for k in ["X","N","G","F","E","S"]}
    total = 0.0
    def _maybe_resample(t, y):
        if resample_dt_h is None or len(t) <= 1:
            return t, y
        t = np.asarray(t, float); y = np.asarray(y, float)
        t0 = np.nanmin(t)
        grid = np.arange(t0, np.nanmax(t)+1e-9, resample_dt_h)
        idx = np.searchsorted(t, grid, side="left")
        idx[idx == len(t)] = len(t)-1
        return t[idx], y[idx]
    def _series_loss(sim, y, std, w):
        m = ~np.isnan(y)
        if not m.any():
            return None
        err2 = ((sim[m] - y[m]) / std)**2
        # MSE por serie si balance en ("per_point","per_assay"); suma cruda solo en "none"
        if balance in ("per_point", "per_assay"):
            return float(np.nanmean(err2) * w)
        else:  # "none"
            return float(np.nansum(err2) * w)
    for code, df in mats.items():
        if code in EXCLUDE_ASSAYS:
            continue
        if "time_h" not in df.columns:
            continue
        t_meas = pd.to_numeric(df["time_h"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(t_meas).any():
            continue
        temp_segs_C = build_temp_profile_from_df(df)
        pulses = (pulses_by_assay or {}).get(code, PULSOS_N_DEF)
        x0 = (x0_by_assay or {}).get(code, DEFAULT_X0.copy())
        t_sim, Xsim = simulate_on_grid(p_real, t_meas, temp_segs_C, pulses, x0=x0)
        t_sim = np.asarray(t_sim, dtype=float)
        X_interp = np.vstack([
            np.interp(t_meas, t_sim, Xsim[:,0]),
            np.interp(t_meas, t_sim, Xsim[:,1]),
            np.interp(t_meas, t_sim, Xsim[:,2]),
            np.interp(t_meas, t_sim, Xsim[:,3]),
            np.interp(t_meas, t_sim, Xsim[:,4]),
        ]).T
        per_var_losses = []
        # X
        if "biomass_viable_gL" in df.columns:
            y = pd.to_numeric(df["biomass_viable_gL"], errors="coerce").to_numpy(float)
            if resample_dt_h:
                t_rs, y_rs = _maybe_resample(t_meas, y)
                sim_rs = np.interp(t_rs, t_sim, Xsim[:,0])
                per = _series_loss(sim_rs, y_rs, stds["X"], weights.get("X",1.0))
            else:
                per = _series_loss(X_interp[:,0], y, stds["X"], weights.get("X",1.0))
            if per is not None: per_var_losses.append(per)
        # N
        if "YAN" in df.columns:
            y = pd.to_numeric(df["YAN"], errors="coerce").to_numpy(float) * N_SCALE
            if resample_dt_h:
                t_rs, y_rs = _maybe_resample(t_meas, y)
                sim_rs = np.interp(t_rs, t_sim, Xsim[:,1])
                per = _series_loss(sim_rs, y_rs, stds["N"], weights.get("N",0.5))
            else:
                per = _series_loss(X_interp[:,1], y, stds["N"], weights.get("N",0.5))
            if per is not None: per_var_losses.append(per)
        # G
        if "Glucose" in df.columns:
            y = pd.to_numeric(df["Glucose"], errors="coerce").to_numpy(float)
            if resample_dt_h:
                t_rs, y_rs = _maybe_resample(t_meas, y)
                sim_rs = np.interp(t_rs, t_sim, Xsim[:,2])
                per = _series_loss(sim_rs, y_rs, stds["G"], weights.get("G",1.0))
            else:
                per = _series_loss(X_interp[:,2], y, stds["G"], weights.get("G",1.0))
            if per is not None: per_var_losses.append(per)
        # F
        if "Fructose" in df.columns:
            y = pd.to_numeric(df["Fructose"], errors="coerce").to_numpy(float)
            if resample_dt_h:
                t_rs, y_rs = _maybe_resample(t_meas, y)
                sim_rs = np.interp(t_rs, t_sim, Xsim[:,3])
                per = _series_loss(sim_rs, y_rs, stds["F"], weights.get("F",1.0))
            else:
                per = _series_loss(X_interp[:,3], y, stds["F"], weights.get("F",1.0))
            if per is not None: per_var_losses.append(per)
        # E
        if "Ethanol" in df.columns:
            y = pd.to_numeric(df["Ethanol"], errors="coerce").to_numpy(float)
            if resample_dt_h:
                t_rs, y_rs = _maybe_resample(t_meas, y)
                sim_rs = np.interp(t_rs, t_sim, Xsim[:,4])
                per = _series_loss(sim_rs, y_rs, stds["E"], weights.get("E",0.5))
            else:
                per = _series_loss(X_interp[:,4], y, stds["E"], weights.get("E",0.5))
            if per is not None: per_var_losses.append(per)
        # S (azúcar total)
        if "SugarTotal_exp" in df.columns:
            y = pd.to_numeric(df["SugarTotal_exp"], errors="coerce").to_numpy(float)
            if np.isfinite(y).any():
                sim_total = X_interp[:,2] + X_interp[:,3]
                if resample_dt_h:
                    t_rs, y_rs = _maybe_resample(t_meas, y)
                    sim_total = np.interp(t_rs, t_sim, Xsim[:,2] + Xsim[:,3])
                    y = y_rs
                per = _series_loss(sim_total, y, stds.get("S",1.0), weights.get("S",1.0))
                if per is not None: per_var_losses.append(per)
        if not per_var_losses:
            continue
        if balance == "per_assay":
            total += float(np.mean(per_var_losses))
        else:
            total += float(np.sum(per_var_losses))
    if verbose:
        print(f"SSE={total:.4e} (balance={balance})")
    return float(total)

# ---------- Reparam log-escala ----------
def make_internal_transform(p0_real: np.ndarray, bounds_real: List[Tuple[float, float]]):
    p0 = np.asarray(p0_real, dtype=float)
    s  = np.maximum(p0, 1e-6)
    lb = np.array([lo for (lo,hi) in bounds_real], dtype=float)
    ub = np.array([hi for (lo,hi) in bounds_real], dtype=float)
    z_lb = np.log(np.maximum(lb/s, 1e-12))
    z_ub = np.log(ub/s)
    def real_from_z(z): return s * np.exp(np.asarray(z, dtype=float))
    def z_from_real(p): return np.log(np.asarray(p, dtype=float)/s)
    return real_from_z, z_from_real, list(zip(z_lb, z_ub)), s

# ---------- Progreso ----------
class Progress:
    def __init__(self, name="OPT"):
        self.name=name; self.t0=time.time(); self.eval_count=0
        self.best_sse=float("inf")
    def mark_eval(self, sse, every=50):   # <-- antes 200
        self.eval_count += 1
        if sse < self.best_sse: self.best_sse = float(sse)
        if (self.eval_count % every)==0:
            dt=time.time()-self.t0
            print(f"[{self.name}] eval={self.eval_count:6d}  best_SSE={self.best_sse:.4e}  t={dt:6.1f}s")
            sys.stdout.flush()

# ---------- Calibración ----------
def calibrate_global_internal(mats: Dict[str, pd.DataFrame],
                              p0_real: np.ndarray,
                              bounds_real: List[Tuple[float, float]],
                              mode: str = "de",
                              pulses_by_assay: Optional[Dict[str, List[Tuple[float, float]]]] = None,
                              x0_by_assay: Optional[Dict[str, np.ndarray]] = None,
                              n_starts: int = 20,
                              verbose: bool = True,
                              patience_starts: int = 6,
                              patience_evals: int = 5000,
                              min_improvement_rel: float = 1e-3):
    """
    Optimiza en z-space (p = s*exp(z)). Early-stop:
      - 'patience_starts': # de arranques sin mejora antes de cortar
      - 'patience_evals' : # de evaluaciones sin mejora global antes de cortar
      - 'min_improvement_rel': mejora relativa mínima para contabilizar como mejora
    """
    real_from_z, z_from_real, z_bounds, s = make_internal_transform(p0_real, bounds_real)
    z0 = np.clip(z_from_real(p0_real), [b[0] for b in z_bounds], [b[1] for b in z_bounds])

    # Precompute escalas (std) para normalizar la SSE
    stds = compute_global_stds(mats)

    prog = Progress(name=f"OPT-{mode.upper()}")
    best_sse_seen = np.inf
    last_improve_eval = 0

    # --- BEST CHECKPOINT ---
    best_ckpt = {"sse": np.inf, "z": None}

    def _save_ckpt(z, sse):
        os.makedirs(SAVE_ARTIFACTS_DIR, exist_ok=True)
        p = real_from_z(z)
        np.savez(os.path.join(SAVE_ARTIFACTS_DIR, "pbest_checkpoint.npz"),
                 z=z, p=p, sse=np.array([sse]))
        print(f"[CKPT] guardado best SSE={sse:.4e}")

    def obj_z(z):
        nonlocal best_sse_seen, last_improve_eval
        p = real_from_z(z)
        sse = sse_for_experiments_real(
            p, mats, pulses_by_assay, x0_by_assay, WEIGHTS, stds,
            verbose=False,
            balance=SSE_BALANCE_MODE,
            resample_dt_h=SSE_RESAMPLE_DT_H
        )
        prog.mark_eval(sse)
        # --- actualizar y guardar best ---
        if sse < best_ckpt["sse"]:
            best_ckpt["sse"] = float(sse)
            best_ckpt["z"] = np.array(z, dtype=float)
            _save_ckpt(best_ckpt["z"], best_ckpt["sse"])
        if sse < (1.0 - min_improvement_rel) * best_sse_seen:
            best_sse_seen = sse
            last_improve_eval = prog.eval_count
        # early-stop por evaluaciones
        if (prog.eval_count - last_improve_eval) >= patience_evals:
            raise RuntimeError("EARLY_STOP_EVALS")
        return sse

    result = {}
    if mode == "multistart":
        # candidatos Sobol / uniforme
        try:
            from scipy.stats.qmc import Sobol
            qmc = Sobol(d=len(z_bounds), scramble=True)
            U = qmc.random_base2(int(np.ceil(np.log2(n_starts))))
            U = U[:n_starts]
        except Exception:
            U = np.random.default_rng(123).uniform(size=(n_starts, len(z_bounds)))
        z_lo = np.array([b[0] for b in z_bounds]); z_hi = np.array([b[1] for b in z_bounds])
        Z = z_lo + U * (z_hi - z_lo)
        Z[0, :] = z0

        z_best = None; sse_best = np.inf; local_runs = []
        no_improve_starts = 0

        for i, zi in enumerate(Z, start=1):
            print(f"[MS] start {i}/{len(Z)}: lanzando L-BFGS-B...")
            try:
                loc = minimize(obj_z, zi, method="L-BFGS-B", bounds=z_bounds,
                               options=dict(maxiter=400, ftol=1e-9))
            except RuntimeError as e:
                if "EARLY_STOP_EVALS" in str(e):
                    print("[MS] :: parada temprana por paciencia en evaluaciones ::")
                    break
                else:
                    raise
            local_runs.append(loc)
            print(f"[MS]  done  {i}/{len(Z)}: success={loc.success}  SSE={loc.fun:.4e}")

            improved = loc.success and (loc.fun < (1.0 - min_improvement_rel) * sse_best)
            if improved:
                sse_best = float(loc.fun)
                z_best = loc.x.copy()
                no_improve_starts = 0
                print(f"[MS]  >>> nuevo BEST SSE={sse_best:.4e}")
            else:
                no_improve_starts += 1
                if no_improve_starts >= patience_starts:
                    print(f"[MS] :: parada temprana por {no_improve_starts} starts sin mejora ::")
                    break
        result = {"multistart": local_runs}
        if z_best is None:
            # en caso extremo, toma el mejor de los intentos igualmente
            j = int(np.argmin([r.fun for r in local_runs]))
            z_best = local_runs[j].x.copy()
            sse_best = float(local_runs[j].fun)

    elif mode == "de":

        def obj_z_de(z):
            p = real_from_z(z)
            sse = sse_for_experiments_real(
                p, mats, pulses_by_assay, x0_by_assay, WEIGHTS, stds,
                verbose=False,
                balance=SSE_BALANCE_MODE,
                resample_dt_h=SSE_RESAMPLE_DT_H
            )
            prog.mark_eval(sse)
            # --- actualizar y guardar best ---
            if sse < best_ckpt["sse"]:
                best_ckpt["sse"] = float(sse)
                best_ckpt["z"] = np.array(z, dtype=float)
                _save_ckpt(best_ckpt["z"], best_ckpt["sse"])
            return sse

        def cb_de(xk, convergence):
            # imprime una línea por iter
            dt = time.time() - prog.t0
            print(f"[DE] conv={convergence:.3e}  best_SSE={prog.best_sse:.4e}  t={dt:6.1f}s")
            sys.stdout.flush()
            return False

        de_res = differential_evolution(
            obj_z_de,
            bounds=z_bounds,
            maxiter=60,
            popsize=12,
            mutation=(0.5, 1.0),
            recombination=0.7,
            tol=1e-6,
            polish=False,
            updating='deferred',
            workers=1,
            disp=False,
            callback=cb_de,
            seed=RANDOM_SEED
        )

        # pulido local protegido contra EARLY_STOP_EVALS
        try:
            loc = minimize(obj_z, de_res.x, method="L-BFGS-B", bounds=z_bounds,
                           options=dict(maxiter=LOCAL_MAXITER, ftol=LOCAL_FTOL))
            print(f"[LOCAL] success={loc.success}  SSE={loc.fun:.4e}")
            z_best = (loc.x if (loc.success and loc.fun < de_res.fun) else de_res.x).copy()
            sse_best = float(min(loc.fun, de_res.fun))
            result = {"de": de_res, "local": loc}
        except RuntimeError as e:
            if "EARLY_STOP_EVALS" in str(e):
                print("[LOCAL] :: parada temprana por paciencia en evaluaciones ::")
                z_best = de_res.x.copy()
                sse_best = float(de_res.fun)
                result = {"de": de_res, "local": None}
            else:
                raise

    else:
        raise ValueError("mode debe ser 'de' o 'multistart'")

    p_best_real = real_from_z(z_best)
    return p_best_real, float(sse_best), result

# ---------- Gráfica ----------
def plot_fit_per_assay(p_real: np.ndarray,
                       mats: Dict[str, pd.DataFrame],
                       pulses_by_assay: Optional[Dict[str, List[Tuple[float, float]]]] = None,
                       x0_by_assay: Optional[Dict[str, np.ndarray]] = None,
                       axis_ranges: Optional[Dict[str,float]] = None):
    import matplotlib as mpl
    kept = [k for k in mats.keys() if k not in EXCLUDE_ASSAYS]
    if not kept:
        print("No hay ensayos para graficar."); return
    n = len(kept); ncols = 2; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 3.9*nrows))
    axes = np.atleast_2d(axes)
    palette = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    cN, cG, cF, cE, cX = palette[:5]; cS = "#8c564b"

    # Límites estándar
    tmax_d  = axis_ranges.get("time_max_days") if axis_ranges else None
    yL_max  = axis_ranges.get("y_left_max")    if axis_ranges else None
    yR_max  = axis_ranges.get("y_right_max")   if axis_ranges else None

    def _scatter_if(ax, df, col, label, xvals, color, transform=None, marker='o'):
        if col in df.columns:
            y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            if transform is not None: y = transform(y)
            ax.scatter(xvals, y, s=22, alpha=0.9, label=label, color=color,
                       marker=marker, zorder=3, linewidths=0.0)

    for i, code in enumerate(kept):
        ax1 = axes.flat[i]
        df = mats[code]
        t_meas_h = pd.to_numeric(df["time_h"], errors="coerce").to_numpy(dtype=float)
        t_meas_d = t_meas_h / 24.0
        temp_segs_C = build_temp_profile_from_df(df)
        pulses = (pulses_by_assay or {}).get(code, PULSOS_N_DEF)
        x0 = (x0_by_assay or {}).get(code, DEFAULT_X0.copy())
        t_sim, Xsim = simulate_on_grid(p_real, t_meas_h, temp_segs_C, pulses, x0)
        td = np.asarray(t_sim, dtype=float) / 24.0

        ax1.plot(td, Xsim[:,1]/N_SCALE, '-', label="N sim", color=cN, linewidth=1.7)
        ax1.plot(td, Xsim[:,2], '-', label="G sim", color=cG, linewidth=1.7)
        ax1.plot(td, Xsim[:,3], '-', label="F sim", color=cF, linewidth=1.7)
        ax1.plot(td, Xsim[:,4], '-', label="E sim", color=cE, linewidth=1.7)
        ax1.plot(td, (Xsim[:,2]+Xsim[:,3]), '--', label="S sim (G+F)", color=cS, linewidth=1.4)
        ax1.grid(alpha=0.25)

        ax2 = ax1.twinx()
        ax2.plot(td, Xsim[:,0], '-', label="X sim", color=cX, linewidth=1.7)

        _scatter_if(ax2, df, "biomass_viable_gL", "X exp", t_meas_d, color=cX, marker='o')
        _scatter_if(ax1, df, "YAN",      "N exp", t_meas_d, color=cN, marker='^')
        _scatter_if(ax1, df, "Glucose",  "G exp", t_meas_d, color=cG, marker='s')
        _scatter_if(ax1, df, "Fructose", "F exp", t_meas_d, color=cF, marker='D')
        _scatter_if(ax1, df, "Ethanol",  "E exp", t_meas_d, color=cE, marker='P')
        if "SugarTotal_exp" in df.columns:
            yS = pd.to_numeric(df["SugarTotal_exp"], errors="coerce").to_numpy(dtype=float)
            ax1.scatter(t_meas_d, yS, s=24, marker='X', color=cS, label="S exp", alpha=0.9)

        ax1.set_title(f"Ensayo {code}")
        ax1.set_xlabel("Tiempo (días)")
        ax1.set_ylabel("N (mg/L), G/F/E/S (g/L)")
        ax2.set_ylabel("X (g/L)")

        # Fijar rangos si se entregaron
        if tmax_d is not None:
            # Reforzar límite duro (por seguridad si axis_ranges viene de fuera)
            lim = min(tmax_d, PLOT_MAX_DAYS)
            ax1.set_xlim(0.0, lim)
            ax2.set_xlim(0.0, lim)
        if yL_max is not None:
            ax1.set_ylim(0.0, yL_max)
        if yR_max is not None:
            ax2.set_ylim(0.0, yR_max)

        h1,l1 = ax1.get_legend_handles_labels()
        h2,l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.14))

    total_axes = nrows * ncols
    for k in range(n, total_axes):
        axes.flat[k].set_visible(False)
    plt.tight_layout()
    plt.show()

def plot_initial_multi(p_real: np.ndarray,
                       mats: Dict[str, pd.DataFrame],
                       pulses_by_assay: Optional[Dict[str, List[Tuple[float,float]]]] = None,
                       x0_by_assay: Optional[Dict[str, np.ndarray]] = None,
                       ncols: int = 3,
                       axis_ranges: Optional[Dict[str,float]] = None):
    """
    Pre-visualización: simulación con parámetros iniciales (p_real = P0) para TODOS los ensayos.
    YAN se grafica en mg/L (simulación: g/L * 1000). Glucosa, Fructosa, Etanol en g/L.
    Biomasa viable en eje secundario.
    """
    kept = [k for k in mats.keys() if k not in EXCLUDE_ASSAYS]
    if not kept:
        print("[PREVIEW] No hay ensayos para graficar.")
        return
    n = len(kept)
    if n <= 4: ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.6*ncols, 3.4*nrows))
    axes = np.atleast_2d(axes)
    base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cN, cG, cF, cE, cX = base_colors[:5]; cS = "#8c564b"

    tmax_d  = axis_ranges.get("time_max_days") if axis_ranges else None
    yL_max  = axis_ranges.get("y_left_max")    if axis_ranges else None
    yR_max  = axis_ranges.get("y_right_max")   if axis_ranges else None

    for i, code in enumerate(kept):
        ax = axes.flat[i]
        df = mats[code]
        t_meas_h = pd.to_numeric(df["time_h"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(t_meas_h).any():
            ax.set_title(f"{code} (sin tiempo)"); ax.axis("off"); continue
        temp_segs_C = build_temp_profile_from_df(df)
        pulses = (pulses_by_assay or {}).get(code, PULSOS_N_DEF)
        x0 = (x0_by_assay or {}).get(code, DEFAULT_X0.copy())
        try:
            t_sim, Xsim = simulate_on_grid(p_real, t_meas_h, temp_segs_C, pulses, x0)
        except Exception as e:
            ax.set_title(f"{code} (ERROR sim)")
            ax.text(0.1,0.5,str(e)[:50], fontsize=8); ax.axis("off"); continue
        td_sim_d = np.asarray(t_sim)/24.0
        td_meas_d = t_meas_h/24.0

        ax.plot(td_sim_d, Xsim[:,1]*1000.0, '-', color=cN, lw=1.4, label="YAN sim (mg/L)")
        ax.plot(td_sim_d, Xsim[:,2], '-', color=cG, lw=1.2, label="G sim")
        ax.plot(td_sim_d, Xsim[:,3], '-', color=cF, lw=1.2, label="F sim")
        ax.plot(td_sim_d, Xsim[:,4], '-', color=cE, lw=1.2, label="E sim")
        ax.plot(td_sim_d, (Xsim[:,2]+Xsim[:,3]), '--', color=cS, lw=1.2, label="S sim (G+F)")

        if "YAN" in df.columns:
            ax.scatter(td_meas_d, pd.to_numeric(df["YAN"], errors="coerce"), s=20, c=cN, marker='^', label="YAN exp")
        if "Glucose" in df.columns:
            ax.scatter(td_meas_d, pd.to_numeric(df["Glucose"], errors="coerce"), s=18, c=cG, marker='o', label="G exp")
        if "Fructose" in df.columns:
            ax.scatter(td_meas_d, pd.to_numeric(df["Fructose"], errors="coerce"), s=18, c=cF, marker='s', label="F exp")
        if "Ethanol" in df.columns:
            ax.scatter(td_meas_d, pd.to_numeric(df["Ethanol"], errors="coerce"), s=18, c=cE, marker='D', label="E exp")
        if "SugarTotal_exp" in df.columns:
            ax.scatter(td_meas_d, pd.to_numeric(df["SugarTotal_exp"], errors="coerce"), s=26, c=cS, marker='X', label="S exp")

        ax.set_xlabel("t (d)")
        ax.set_ylabel("N (mg/L) / G,F,E,S (g/L)")
        ax.grid(alpha=0.25)

        ax2 = ax.twinx()
        if "biomass_viable_gL" in df.columns:
            ax2.plot(td_sim_d, Xsim[:,0], '-', color=cX, lw=1.3, label="X sim")
            ax2.scatter(td_meas_d, pd.to_numeric(df["biomass_viable_gL"], errors="coerce"), s=20,
                        c=cX, marker='P', label="X exp")
        ax2.set_ylabel("X (g/L)")
        ax.set_title(code)

        # Aplicar límites estándar
        if tmax_d is not None:
            lim = min(tmax_d, PLOT_MAX_DAYS)
            ax.set_xlim(0.0, lim)
            ax2.set_xlim(0.0, lim)
        if yL_max is not None:
            ax.set_ylim(0.0, yL_max)
        if yR_max is not None:
            ax2.set_ylim(0.0, yR_max)

        if i == 0:
            h1,l1 = ax.get_legend_handles_labels()
            h2,l2 = ax2.get_legend_handles_labels()
            ax.legend(h1+h2, l1+l2, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5,1.18))

    total_axes = nrows*ncols
    for j in range(n, total_axes):
        axes.flat[j].set_visible(False)
    fig.suptitle("Previsualización inicial (P0) · Sim vs Exp (YAN en mg/L)", fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.97])
    plt.show()

 # ===================== MAIN (ajustado) =====================
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    # 0) Cargar parámetros base
    P0 = load_parameters_from_excel(PARAM_XLSX, sheet_name=PARAM_SHEET, param_set=PARAM_SET)
    if P0 is None:
        raise RuntimeError("No se pudieron cargar P0 desde Excel.")
    print("P0 (reales):", P0)

    # 1) Cargar metadata + splits
    def _load_splits_and_meta():
        meta = pd.read_csv(METADATA_CSV)
        train_ids = pd.read_csv(TRAIN_IDS_CSV, header=None)[0].astype(str).tolist()
        valid_ids = pd.read_csv(VALID_IDS_CSV, header=None)[0].astype(str).tolist()
        return meta, train_ids, valid_ids
    meta, train_ids, valid_ids = _load_splits_and_meta()
    print(f"[SPLITS] Train={len(train_ids)}  Valid={len(valid_ids)}")

    # Modelo densidad→azúcar
    def _sugar_density_model_from_file(path: str):
        """
        Parser robusto del archivo generado por sugar_density.py:
          Scaler mean: [m] var: [v]
          Coefficients: [c1, c2, c3, ...]
          Intercept: b
        Reconstruye: y = b + Σ c_i * ( ( (d - m)/std )^(i) )  (i=1..deg)
        (PolynomialFeatures(include_bias=False) sobre z=(d-m)/std)
        """
        if not os.path.exists(path):
            print(f"[SUGAR-MODEL] No encontrado: {path}")
            return lambda d: np.full_like(np.asarray(d, dtype=float), np.nan)
        mean = var = intercept = None
        coefs = None
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if line.startswith("Scaler mean:"):
                    try:
                        # Scaler mean: [m] var: [v]
                        seg = line.split("Scaler mean:")[1].strip()
                        parts = seg.split("var:")
                        mean = float(parts[0].strip("[] ,"))
                        var  = float(parts[1].strip("[] ,"))
                    except Exception:
                        pass
                elif line.startswith("Coefficients:"):
                    inside = line.split("Coefficients:")[1].strip().strip("[]")
                    try:
                        coefs = [float(x) for x in inside.split(",")]
                    except Exception:
                        pass
                elif line.startswith("Intercept:"):
                    try:
                        intercept = float(line.split("Intercept:")[1].strip())
                    except Exception:
                        pass
        if None in (mean, var, coefs, intercept):
            print("[SUGAR-MODEL] Parsing incompleto; fallback NaN.")
            return lambda d: np.full_like(np.asarray(d, dtype=float), np.nan)
        std = var**0.5
        coefs = np.asarray(coefs, dtype=float)
        deg = len(coefs)
        print(f"[SUGAR-MODEL] Cargado grado={deg} mean={mean:.3f} std={std:.3f}")
        def _predict(density):
            z = (np.asarray(density, dtype=float) - mean)/std
            feats = [z**(i+1) for i in range(deg)]  # z, z^2, ...
            return intercept + np.sum(coefs[:, None] * np.vstack(feats), axis=0)
        return _predict

    sugar_model = _sugar_density_model_from_file(SUGAR_MODEL_TXT)

    # ---------------- INTEGRACIÓN REAL 2024 + 2025 ----------------
    def build_unified_mats_and_pulses(meta: pd.DataFrame,
                                      train_ids: List[str],
                                      sugar_model):
        """
        Construye:
          mats  : dict assay -> DataFrame con columnas usadas por el costo
          X0S   : dict assay -> np.array([X0,N0,G0,F0,E0])
          PULSOS: dict assay -> list[(t_h, dN_gL)]
        Año 2025: usa Calibration_data_preprocess (build_calibration_matrices).
        Año 2024: usa SW_Preprocess_data.process_multiple y normaliza columnas.
        """
        from Calibration_data_preprocess import process_all as proc25, attach_temperature_to_results, build_calibration_matrices
        from SW_Preprocess_data import process_multiple as proc24

        def _year_of(a: str) -> int:
            r = meta.loc[meta["assay"].astype(str) == str(a)]
            if r.empty:
                raise RuntimeError(f"Assay {a} no está en metadata.")
            return int(r["year"].iloc[0])

        def _apply_density_to_sugars_2024(df: pd.DataFrame):
            if not USE_DENSITY_MODEL_2024:
                return df
            if "Densidad" not in df.columns:
                return df
            dens = pd.to_numeric(df["Densidad"], errors="coerce")
            total = sugar_model(dens)
            if np.isnan(total).all():
                print("[SUGAR-MODEL] Total azúcar NaN (modelo densidad no disponible); se conservan valores G/F originales.")
            if "Glucosa" not in df.columns:
                df["Glucosa"] = np.nan
            if "Fructosa" not in df.columns:
                df["Fructosa"] = np.nan
            G = pd.to_numeric(df["Glucosa"], errors="coerce")
            F = pd.to_numeric(df["Fructosa"], errors="coerce")
            both = G.isna() & F.isna()
            onlyG = G.notna() & F.isna()
            onlyF = F.notna() & G.isna()
            if both.any():
                df.loc[both, "Glucosa"]  = total[both] * FILL_MISSING_G_F_SPLIT
                df.loc[both, "Fructosa"] = total[both] * (1.0 - FILL_MISSING_G_F_SPLIT)
            if onlyG.any():
                df.loc[onlyG, "Fructosa"] = np.maximum(total[onlyG] - G[onlyG], 0.0)
            if onlyF.any():
                df.loc[onlyF, "Glucosa"] = np.maximum(total[onlyF] - F[onlyF], 0.0)
            return df

        def _correct_yan(series: pd.Series):
            y = pd.to_numeric(series, errors="coerce") - YAN_OFFSET_CORRECTION
            y[y < YAN_MIN_MG_L] = 0.0
            return y

        def _pulses_from_chem_df(chem_df: Optional[pd.DataFrame]):
            if chem_df is None or chem_df.empty:
                return []
            if not {"time_h", "valor"}.issubset(chem_df.columns):
                return []
            rows = []
            for _, r in chem_df.iterrows():
                t = pd.to_numeric(r["time_h"], errors="coerce")
                v = pd.to_numeric(r["valor"], errors="coerce")
                if np.isfinite(t) and np.isfinite(v) and v > 0:
                    rows.append((float(t), float(v) / 1000.0))
            if not rows:
                return []
            dfp = pd.DataFrame(rows, columns=["t", "dN"])
            dfp = dfp.groupby("t", as_index=False).agg({"dN": "sum"}).sort_values("t")
            return list(dfp.itertuples(index=False, name=None))

        # --- Filtrar train por año ---
        train_2025 = [a for a in train_ids if _year_of(a) == 2025]
        train_2024 = [a for a in train_ids if _year_of(a) == 2024]

        mats = {}
        X0S = {}
        PULSOS = {}

        # ================= 2025 =================
        if train_2025:
            res25, chem25 = proc25(DATA_DIR_2025_FILE, assays=None)
            res25 = {k: v for k, v in res25.items() if k in train_2025}
            res25 = attach_temperature_to_results(res25)
            mats25 = build_calibration_matrices(res25, use_smoothed_biomass=True)

            # --- NUEVO: Pulsos por ensayo (restaurado del enfoque old) ---
            def _col_like(df, *cands):
                for c in df.columns:
                    l = str(c).strip().lower()
                    for cand in cands:
                        if l == cand.lower():
                            return c
                for c in df.columns:  # fallback contains
                    l = str(c).strip().lower()
                    for cand in cands:
                        if cand.lower() in l:
                            return c
                return None

            def _build_pulses_from_chem_per_assay(chem_df: Optional[pd.DataFrame]) -> Dict[str, List[Tuple[float,float]]]:
                if chem_df is None or chem_df.empty:
                    return {}
                dfc = chem_df.copy()
                col_code = _col_like(dfc, "Código", "codigo", "sample_id", "muestra")
                col_yan  = _col_like(dfc, "YAN", "yan")
                col_time = _col_like(dfc, "time_h", "tiempo_h", "horas", "t_h")
                if col_code is None or col_yan is None:
                    return {}
                # derivar time_h si no existe
                if col_time is None:
                    col_ts = _col_like(dfc, "timestamp", "fecha", "datetime")
                    if col_ts is not None:
                        ts = pd.to_datetime(dfc[col_ts], errors="coerce")
                        t0 = ts.min()
                        dfc["__time_h__"] = (ts - t0).dt.total_seconds()/3600.0
                        col_time = "__time_h__"
                    else:
                        dfc["__time_h__"] = 0.0
                        col_time = "__time_h__"
                dfc[col_yan]  = pd.to_numeric(dfc[col_yan], errors="coerce")
                dfc[col_time] = pd.to_numeric(dfc[col_time], errors="coerce")

                # Extraer código ensayo (SBxxx) de la columna de código de muestra
                import re
                def _extract_assay(s):
                    m = re.search(r"(SB\d{3})", str(s).upper())
                    return m.group(1) if m else None
                dfc["__assay__"] = dfc[col_code].apply(_extract_assay)
                pulses: Dict[str, List[Tuple[float,float]]] = {}
                # Orden por tiempo dentro de cada ensayo
                for ass, grp in dfc.groupby("__assay__"):
                    if ass is None or ass not in train_2025:
                        continue
                    g = grp.sort_values(col_time)
                    y = g[col_yan].to_numpy(dtype=float)
                    t = g[col_time].to_numpy(dtype=float)
                    if len(y) < 2:
                        continue
                    dy = np.diff(y)  # mg/L
                    tm = 0.5*(t[1:]+t[:-1])
                    lst = []
                    for dyi, ti in zip(dy, tm):
                        if np.isfinite(dyi) and dyi > 0 and np.isfinite(ti):
                            lst.append((float(ti), float(dyi)/1000.0))  # mg/L -> g/L
                    if lst:
                        # combinar tiempos repetidos
                        dfp = pd.DataFrame(lst, columns=["t","dN"]).groupby("t", as_index=False).agg({"dN":"sum"}).sort_values("t")
                        pulses[ass] = list(dfp.itertuples(index=False, name=None))
                if pulses:
                    print("[PULSOS][2025] Generados por ensayo:",
                          "; ".join(f"{k}:{[(round(t,1),round(d,4)) for t,d in v]}" for k,v in pulses.items()))
                else:
                    print("[PULSOS][2025] Sin pulsos detectados en planilla química; se usarán por defecto.")
                return pulses

            pulses_2025_by_assay = _build_pulses_from_chem_per_assay(chem25)

            # === NUEVO: reducir a UN solo pulso (inyección principal) por ensayo 2025 ===
            # Ventana típica intermedia (horas) donde ocurre la adición rutinaria
            MID_WIN_LO, MID_WIN_HI = 24.0, 72.0
            for _assay, lst in list(pulses_2025_by_assay.items()):
                if not lst or len(lst) == 1:
                    continue
                # Pulsos en ventana intermedia
                mid = [p for p in lst if MID_WIN_LO <= p[0] <= MID_WIN_HI]
                if mid:
                    main = max(mid, key=lambda z: z[1])  # mayor ΔN dentro de ventana
                    reason = "maxΔN en ventana 24-72h"
                else:
                    main = max(lst, key=lambda z: z[1])  # mayor ΔN global
                    reason = "maxΔN global (sin pulsos en ventana)"
                if main[1] <= 0:
                    # fallback (no positivo real): tomar el pulso con mayor tiempo dentro ventana o el último
                    main = max(lst, key=lambda z: z[0])
                    reason = "fallback último por no ΔN>0"
                if len(lst) > 1:
                    print(f"[PULSOS][2025][REDUCE] {_assay}: {len(lst)}→1  kept={tuple(round(x,2) for x in main)}  motivo={reason}  descartados={[(round(t,2),round(d,4)) for (t,d) in lst if (t,d)!=main]}")
                pulses_2025_by_assay[_assay] = [main]

            for code, m in mats25.items():
                # ...existing code (YAN correction, X0, asignación PULSOS)...
                if "YAN" in m.columns:
                    m["YAN"] = _correct_yan(m["YAN"])
                mats[code] = m
                r0 = m.iloc[0]
                X0S[code] = np.array([
                    float(r0.get("biomass_viable_gL", DEFAULT_X0[0])),
                    float(r0.get("YAN", DEFAULT_X0[1])) * N_SCALE,
                    float(r0.get("Glucose", DEFAULT_X0[2])),
                    float(r0.get("Fructose", DEFAULT_X0[3])),
                    float(r0.get("Ethanol", DEFAULT_X0[4])),
                ], dtype=float)
                PULSOS[code] = pulses_2025_by_assay.get(code, PULSOS_N_DEF.copy())

        # ================= 2024 =================
        if train_2024:
            bundles24 = proc24(train_2024, DATA_DIR_2024)
            for raw_code, bundle in bundles24.items():
                assay = str(raw_code)
                if assay not in train_2024:
                    continue
                df = bundle.get("data")
                chem_df = bundle.get("chem_df")
                if df is None or df.empty:
                    continue
                # Normalizar nombres clave
                rename_map = {
                    "Biomasa viable": "biomass_viable_gL",
                    "Biomasa Viable": "biomass_viable_gL",
                    "Biomasa Total": "biomass_total_gL",
                    "Glucosa": "Glucose",
                    "Fructosa": "Fructose",
                    "Alcohol": "Ethanol",
                    "Temperatura": "Temperature_C",
                }
                for k, v in rename_map.items():
                    if k in df.columns and v not in df.columns:
                        df[v] = df[k]
                # time_h columna
                if "time_h" in df.columns:
                    df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
                elif "time_hours" in df.columns:
                    df["time_h"] = pd.to_numeric(df["time_hours"], errors="coerce")
                elif "time_days" in df.columns:
                    df["time_h"] = pd.to_numeric(df["time_days"], errors="coerce") * 24.0
                else:
                    df["time_h"] = np.arange(len(df), dtype=float)
                # Azúcar total desde densidad si faltan
                df = _apply_density_to_sugars_2024(df)
                # YAN corrección
                if "YAN" in df.columns:
                    df["YAN"] = _correct_yan(df["YAN"])
                # Subset columnas esperadas
                needed = ["time_h", "biomass_viable_gL", "YAN", "Glucose", "Fructose", "Ethanol", "Temperature_C", "Densidad", "SugarTotal_exp"]
                for c in needed:
                    if c not in df.columns:
                        df[c] = np.nan
                df_m = df[needed].sort_values("time_h").reset_index(drop=True)
                # Generar SugarTotal_exp desde densidad si no existe
                if df_m["SugarTotal_exp"].isna().all() and "Densidad" in df_m.columns:
                    dens_vec = pd.to_numeric(df_m["Densidad"], errors="coerce")
                    df_m["SugarTotal_exp"] = sugar_model(dens_vec)
                # Conversión Alcohol 2024 (% v/v -> g/L) heurística
                if "Ethanol" in df_m.columns:
                    Et = pd.to_numeric(df_m["Ethanol"], errors="coerce")
                    if Et.max(skipna=True) < 30:  # probablemente %
                        df_m["Ethanol"] = Et * 0.78924 * 10.0
                mats[assay] = df_m
                # Pulsos
                PULSOS[assay] = _pulses_from_chem_df(chem_df)
                # X0 (añadir corrección azúcar total 50/50)
                r0 = df_m.iloc[0]
                sugar_total0 = np.nan
                if "SugarTotal_exp" in df_m.columns and not df_m["SugarTotal_exp"].isna().all():
                    sugar_total0 = float(df_m["SugarTotal_exp"].dropna().iloc[0])
                else:
                    g0 = float(r0.get("Glucose", np.nan))
                    f0 = float(r0.get("Fructose", np.nan))
                    sugar_total0 = g0 + f0
                if not np.isfinite(sugar_total0) or sugar_total0 <= 0:
                    sugar_total0 = float(r0.get("Glucose", DEFAULT_X0[2])) + float(r0.get("Fructose", DEFAULT_X0[3]))
                # Reparto 50/50 siempre que exista un total estimado (>0)
                g_init = f_init = np.nan
                idx0 = df_m["time_h"].idxmin()
                if np.isfinite(sugar_total0) and sugar_total0 > 0:
                    g_init = 0.5 * sugar_total0
                    f_init = 0.5 * sugar_total0
                    # Asegurar columnas (en inglés) consistentes con 'needed'
                    if "Glucose" not in df_m.columns:
                        df_m["Glucose"] = np.nan
                    if "Fructose" not in df_m.columns:
                        df_m["Fructose"] = np.nan
                    # Sobrescribir solo si difiere (>5%) o está NaN
                    try:
                        g0 = df_m.at[idx0, "Glucose"]
                        f0 = df_m.at[idx0, "Fructose"]
                    except Exception:
                        g0 = f0 = np.nan
                    need_override = (
                        (not np.isfinite(g0)) or (not np.isfinite(f0)) or
                        (abs(( (np.nan_to_num(g0)+np.nan_to_num(f0)) - sugar_total0)/sugar_total0) > 0.05)
                    )
                    if need_override:
                        df_m.at[idx0, "Glucose"]  = g_init
                        df_m.at[idx0, "Fructose"] = f_init
                else:
                    # Fallback si no hay total válido
                    g_init = float(r0.get("Glucose", DEFAULT_X0[2]))
                    f_init = float(r0.get("Fructose", DEFAULT_X0[3]))
                # Actualizar en mats (por si se modificó df_m)
                mats[assay] = df_m
                X0S[assay] = np.array([
                    float(r0.get("biomass_viable_gL", DEFAULT_X0[0])),
                    float(r0.get("YAN", DEFAULT_X0[1])) * N_SCALE,
                    g_init,
                    f_init,
                    float(r0.get("Ethanol", DEFAULT_X0[4])),
                ], dtype=float)
        # (Reversión manejo YAN / N):
        # - No se absorben pulsos iniciales en 2025 dentro de X0 (se mantienen para la simulación).
        # - No se modifica el primer valor experimental de YAN en 2024.
        # - Se conserva el ajuste de Etanol inicial a 0 para 2024.
        for code in train_2024:
            if code in mats and code in X0S:
                df = mats[code]
                if "Ethanol" in df.columns and "time_h" in df.columns:
                    t0 = df["time_h"].min()
                    idx0 = df.index[df["time_h"] == t0]
                    if len(idx0):
                        changed = False
                        if np.any(df.loc[idx0, "Ethanol"] != 0):
                            df.loc[idx0, "Ethanol"] = 0.0
                            mats[code] = df
                            changed = True
                        if X0S[code].shape[0] >= 5 and X0S[code][4] != 0.0:
                            X0S[code][4] = 0.0
                            changed = True
                        if changed:
                            print(f"[INIT-EtOH][2024] {code}: Etanol inicial forzado a 0 g/L")
        # --- Saneo inicial X0 (igual) ---
        for k, x in list(X0S.items()):
            # Salvaguarda: corregir cualquier X0 mal formado (ej: longitud 7 previa)
            if x.shape[0] !=  5:
                if x.shape[0] == 7:
                    # patrón antiguo: [X,N,G,F,g_init,f_init,E] -> [X,N,g_init,f_init,E]
                    x = np.array([x[0], x[1], x[4], x[5], x[6]], dtype=float)
                    X0S[k] = x
                else:
                    # fallback duro
                    X0S[k] = np.array(DEFAULT_X0, dtype=float)
            bad = ~np.isfinite(x)
            if bad.any():
                x_fixed = x.copy()
                x_fixed[bad] = np.array(DEFAULT_X0)[bad]

                X0S[k] = x_fixed
        print(f"[INTEGRACIÓN] 2025(train)={len(train_2025)}  2024(train)={len(train_2024)}  mats_total={len(mats)}")
        return mats, X0S, PULSOS

    # Construir mats/pulsos (ahora correctamente dentro de main)
    mats, X0S, PULSOS = build_unified_mats_and_pulses(meta, train_ids, sugar_model)
    if not mats:
        raise RuntimeError("mats vacío tras integración 2024+2025.")

    # --------- NUEVO: saneo profundo de matrices y X0 antes de calibrar ---------
    def _first_valid_value(series: pd.Series):
        s = pd.to_numeric(series, errors="coerce")
        s = s[s.notna()]
        return float(s.iloc[0]) if len(s) else np.nan
    
    def sanitize_x0(code: str, df: pd.DataFrame, x0_current: np.ndarray) -> np.ndarray:
        """
        Intenta derivar X0=[X,N,G,F,E] desde primeros valores válidos;
        si falta alguno usa DEFAULT_X0. Asegura no-negatividad y finito.
        """
        comp_cols = ["biomass_viable_gL","YAN","Glucose","Fructose","Ethanol"]
        derived = []
        for i,col in enumerate(comp_cols):
            if col in df.columns:
                v = _first_valid_value(df[col])
            else:
                v = np.nan
            if not np.isfinite(v):
                v = DEFAULT_X0[i]
            if col == "YAN":
                v = max(v, 0.0) * N_SCALE  # convertir a g/L interno
            else:
                v = max(v, 0.0)
            derived.append(v)
        x0_new = np.array(derived, dtype=float)
        if not np.isfinite(x0_new).all():
            print(f"[SANITIZE][WARN] X0 no finito tras derivar -> usando DEFAULT para {code}")
            x0_new = np.array([
                DEFAULT_X0[0],
                DEFAULT_X0[1],
                DEFAULT_X0[2],
                DEFAULT_X0[3],
                DEFAULT_X0[4]
            ], dtype=float)
        return x0_new
    
    def normalize_mats_for_calibration(mats: Dict[str, pd.DataFrame],
                                       X0S: Dict[str, np.ndarray]) -> Tuple[Dict[str,pd.DataFrame], Dict[str,np.ndarray]]:
        required_cols = ["time_h","biomass_viable_gL","YAN","Glucose","Fructose","Ethanol","Temperature_C"]
        drop = []
        for code, df in mats.items():
            # Asegurar columnas
            for c in required_cols:
                if c not in df.columns:
                    df[c] = np.nan
            # time_h válido
            df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
            df = df[~df["time_h"].isna()].sort_values("time_h").reset_index(drop=True)
            # Reemplazar negativos en variables químicas
            for c in ["biomass_viable_gL","YAN","Glucose","Fructose","Ethanol"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                df.loc[df[c] < 0, c] = 0.0
            # Derivar/ajustar X0
            x0_old = X0S.get(code, np.array(DEFAULT_X0, dtype=float))
            x0_new = sanitize_x0(code, df, x0_old)
            if not np.isfinite(x0_new).all():
                print(f"[SANITIZE][DROP] Eliminando {code} por X0 inválido.")
                drop.append(code); continue
            if df.shape[0] < 2:
                print(f"[SANITIZE][DROP] {code} con <2 puntos válidos.")
                drop.append(code); continue
            mats[code] = df
            X0S[code] = x0_new
        for d in drop:
            mats.pop(d, None)
            X0S.pop(d, None)
            PULSOS.pop(d, None)
        if drop:
            print(f"[SANITIZE] Ensayos descartados: {drop}")
        print(f"[SANITIZE] Ensayos finales: {len(mats)}")
        return mats, X0S

    mats, X0S = normalize_mats_for_calibration(mats, X0S)
    # Reforzar nuevamente para 2024 (por si sanitize_x0 se ejecutó antes de que se actualizara Glucose/Fructose)
    for code in list(mats.keys()):
        row_meta = meta.loc[meta["assay"].astype(str)==code]
        if not row_meta.empty and int(row_meta["year"].iloc[0]) == 2024:
            dfc = mats[code]
            if "SugarTotal_exp" in dfc.columns and dfc["SugarTotal_exp"].notna().any():
                idx0 = dfc["time_h"].idxmin()
                st0 = float(dfc.loc[idx0, "SugarTotal_exp"]) if np.isfinite(dfc.loc[idx0, "SugarTotal_exp"]) else np.nan
                if np.isfinite(st0) and st0 > 0:
                    half = 0.5 * st0
                    # Sólo ajustar si la suma inicial difiere >5% del total estimado
                    g0 = float(dfc.loc[idx0, "Glucose"]) if "Glucose" in dfc.columns else np.nan
                    f0 = float(dfc.loc[idx0, "Fructose"]) if "Fructose" in dfc.columns else np.nan
                    if np.isfinite(g0) and np.isfinite(f0):
                        if abs((g0+f0) - st0) / st0 > 0.05:
                            dfc.loc[idx0, "Glucose"]  = half
                            dfc.loc[idx0, "Fructose"] = half
                            mats[code] = dfc
                            # Actualizar X0 si ya derivado
                            if code in X0S and X0S[code].shape[0] == 5:
                                X0S[code][2] = half
                                X0S[code][3] = half

    # Corrección final YAN (solo asegurar numérico y clamp)
    def _year_of_code(a: str) -> Optional[int]:
         r = meta.loc[meta["assay"].astype(str) == str(a)]
         if r.empty:
            return None
         return int(r["year"].iloc[0])

    for code, df in mats.items():
         if "YAN" not in df.columns:
             continue
         yr = _year_of_code(code)
         # Ya corregido por rama; sólo asegurar tipo y clamp
         vals = pd.to_numeric(df["YAN"], errors="coerce")
         vals[vals < 0.0] = 0.0
         df["YAN"] = vals
         mats[code] = df

    # Sanity (primer ensayo)
    first_code = next(iter(mats.keys()))
    df0 = mats[first_code]
    if "time_h" not in df0.columns:
        raise RuntimeError(f"{first_code} sin time_h")
    t_meas = pd.to_numeric(df0["time_h"], errors="coerce").to_numpy(float)
    temp_segs_C = build_temp_profile_from_df(df0)
    x0 = X0S[first_code]; pulses0 = PULSOS.get(first_code, [])
    print(f"[Sanity] {first_code}  n={len(df0)}  pulses={len(pulses0)}  x0={np.round(x0,3)}")
    t_sim0,Xsim0 = simulate_on_grid(P0, t_meas, temp_segs_C, pulses0, x0)
    if not np.isfinite(Xsim0).all(): raise AssertionError("NaN en simulación sanity")
    print("[Sanity] OK")

    # --- NUEVO: previsualización completa con parámetros iniciales ---
    print("[PREVIEW] Generando simulaciones iniciales (P0) para todos los ensayos...")
    # Calcular rangos de ejes estándar (tiempo y concentraciones) antes de graficar
    AXIS_RANGES = compute_standard_axis_ranges(mats)
    print(f"[AXES] train ranges: {AXIS_RANGES}")
    # --- previsualización inicial con ejes fijos ---
    plot_initial_multi(P0, mats, pulses_by_assay=PULSOS, x0_by_assay=X0S, axis_ranges=AXIS_RANGES)

    # Calibración
    print(f"\nCalibrando sobre {len(mats)} ensayos train...")
    try:
        p_best, sse_best, result = calibrate_global_internal(
            mats=mats,
            p0_real=P0,
            bounds_real=P_BOUNDS_REAL,
            mode=MODE,
            pulses_by_assay=PULSOS,
            x0_by_assay=X0S,
            n_starts=N_STARTS,
            verbose=True
        )
    except KeyboardInterrupt:
        print("\n[STOP] Interrumpido por el usuario. Intentando recuperar checkpoint...")
        ckpt_path = os.path.join(SAVE_ARTIFACTS_DIR, "pbest_checkpoint.npz")
        if os.path.exists(ckpt_path):
            data = np.load(ckpt_path, allow_pickle=True)
            p_best = data["p"]
            sse_best = float(data["sse"][0])
            result = {"interrupted": True}
            print(f"[STOP] Recuperado best SSE={sse_best:.4e} desde {ckpt_path}")
        else:
            raise

    print("\n=== RESULTADOS ===")
    print(f"SSE final (o best hasta el corte): {sse_best:.4e}")
    print("Parámetros estimados:", p_best)

    plot_fit_per_assay(p_best, mats, pulses_by_assay=PULSOS, x0_by_assay=X0S, axis_ranges=AXIS_RANGES)

    # ===== VALIDACIÓN (opcional) =====
    if valid_ids:
        print(f"\n[VALID] Construyendo matrices de validación ({len(valid_ids)})...")
        mats_val, X0S_val, PULSOS_val = build_unified_mats_and_pulses(meta, valid_ids, sugar_model)
        mats_val, X0S_val = normalize_mats_for_calibration(mats_val, X0S_val)
        # YAN ya corregido por rama dentro de build_unified_mats_and_pulses; sólo asegurar numérico y clamp
        for code, dfv in mats_val.items():
            if "YAN" in dfv.columns:
                vals = pd.to_numeric(dfv["YAN"], errors="coerce")
                vals[vals < 0] = 0
                dfv["YAN"] = vals
        AXIS_RANGES_VAL = compute_standard_axis_ranges(mats_val)
        print(f"[VALID] SSE valid (balance={SSE_BALANCE_MODE}):",
              sse_for_experiments_real(p_best, mats_val, PULSOS_val, X0S_val,
                                       WEIGHTS, compute_global_stds(mats_val),
                                       balance=SSE_BALANCE_MODE,
                                       resample_dt_h=SSE_RESAMPLE_DT_H))
        plot_fit_per_assay(p_best, mats_val, pulses_by_assay=PULSOS_val,
                           x0_by_assay=X0S_val, axis_ranges=AXIS_RANGES_VAL)

    # ===== PERSISTENCIA DE ARTEFACTOS =====
    if PERSIST_ARTIFACTS:
        os.makedirs(SAVE_ARTIFACTS_DIR, exist_ok=True)
        index_rows = []
        for code, dfm in mats.items():
            out_path = os.path.join(SAVE_ARTIFACTS_DIR, f"assay={code}.parquet")
            try:
                dfm.to_parquet(out_path, index=False)
            except Exception:
                # fallback a csv si parquet no disponible
                dfm.to_csv(out_path.replace(".parquet",".csv"), index=False)
            index_rows.append({"assay": code,
                               "n_rows": len(dfm),
                               "t_min_h": float(dfm["time_h"].min()),
                               "t_max_h": float(dfm["time_h"].max())})
        pd.DataFrame(index_rows).to_csv(os.path.join(SAVE_ARTIFACTS_DIR, "mats_index.csv"), index=False)
        # X0
        x0_rows = []
        for code, x0v in X0S.items():
            x0_rows.append({
                "assay": code,
                "X0_biomass": x0v[0],
                "N0_gL": x0v[1],
                "G0": x0v[2],
                "F0": x0v[3],
                "E0": x0v[4],
            })
        pd.DataFrame(x0_rows).to_csv(os.path.join(SAVE_ARTIFACTS_DIR, "initial_conditions.csv"), index=False)
        # Pulsos
        pulse_rows = []
        for code, lst in PULSOS.items():
            for t_p, dN in lst:
                pulse_rows.append({"assay": code, "time_h": t_p, "dN_gL": dN})
        pd.DataFrame(pulse_rows).to_csv(os.path.join(SAVE_ARTIFACTS_DIR, "pulses_YAN.csv"), index=False)
        print(f"[ARTIFACTS] Guardados en {SAVE_ARTIFACTS_DIR}")
