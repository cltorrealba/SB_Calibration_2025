# -*- coding: utf-8 -*-
"""
Calibración global de parámetros (con perfiles térmicos por ensayo e
internalización log-escala para positividad y escalas homogéneas).

- Global: SciPy Differential Evolution ("de")  ó  Multistart + L-BFGS-B ("multistart")
- Pulido local opcional: L-BFGS-B en el espacio interno
- Requiere: numpy, pandas, matplotlib, scipy
- Usa el simulador provisto en modelo_dinamico_sim.py
"""

import os, sys, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution, minimize

# === Tu simulador / loader ===
from modelo_dinamico_sim import (
    zenteno_model,
    RK4_method,
    build_profiles,
    simulate_process_time,  # si lo usas en pruebas
    load_parameters_from_excel,
    DEFAULT_X0,
)

# ---------------- CONFIG ----------------
PARAM_XLSX  = "zenteno_parameters.xlsx"   # <-- EDITA si es necesario
PARAM_SHEET = "Hoja1"
PARAM_SET   = 3

WEIGHTS = {
    "X": 1.0,   # Biomasa viable (modelo) vs biomass_viable_gL (exp)
    "N": 0.5,   # YAN (g/L) tras tu pipeline
    "G": 1.0,   # Glucose
    "F": 1.0,   # Fructose
    "E": 0.5,   # Ethanol
}

# Límites (reales, positivos) – 14 parámetros
P_BOUNDS_REAL = [
    (1e-5, 2.0),   # mu0    1/h
    (1e-5, 5.0),   # betaG0
    (1e-5, 5.0),   # betaF0
    (1e-6, 5.0),   # Kn0
    (1e-6, 5.0),   # Kg0
    (1e-6, 5.0),   # Kf0
    (1e-6, 5.0),   # Kig0
    (1e-6, 5.0),   # Kie0
    (1e-6, 1.0),   # Kd0
    (1e-4, 5.0),   # Yxn
    (1e-4, 5.0),   # Yxg
    (1e-4, 5.0),   # Yxf
    (1e-4, 5.0),   # Yeg
    (1e-4, 5.0),   # Yef
]

TF_HOURS = 14 * 24.0   # si no hay tiempos en datos
N_STEPS  = None        # None -> dt ~ 1h en RK4_method

DEFAULT_TEMP_C = 20.0

# Pulsos por defecto (si no hay por-ensayo ni columnas en matriz)
PULSOS_N = [(43.0, 0.045)]   # [(t_h, cantidad_gL), ...]

# Opcional: dict manual de pulsos/x0 por ensayo (sobre-escriben lo inferido)
PULSOS_BY_ASSAY: Dict[str, List[Tuple[float, float]]] = {
    # "SB003": [(24.0, 0.03), (48.0, 0.02)],
}
X0_BY_ASSAY: Dict[str, np.ndarray] = {
    # "SB003": np.array([0.3, 0.5, 80.0, 70.0, 0.0], dtype=float)
}

# Carpeta de matrices CSV (opcional)
RUTA_MATRICES = None

MODE = "multistart"  # "de" o "multistart"

# ---- Normalización de la SSE ----
NORMALIZE_MODE = "std"   # "std" | "range" | "max"
GLOBAL_NORMALIZATION = True  # True: una escala global por variable usando todos los ensayos; False: por-ensayo
MIN_SCALE = 1e-6         # evita divisiones por ~0


# Nota: tú comentaste sacar SB001 y SB002; aquí también excluyo SB011/SB012 (ajusta si quieres)
EXCLUDE_ASSAYS = {"SB001", "SB002", "SB012"}

EPS = 1e-9
BIG = 1e6  # límite de saturación

# ---- Early stopping / límites globales de evaluación ----
MAX_EVALS_TOTAL        = 10000    # tope duro de evaluaciones de la función
NO_IMPROVE_EVALS       = 5_000     # si no mejora en este # de evals, paramos
STALE_STARTS_PATIENCE  = 5         # # de starts seguidos sin mejora -> paramos


# -------------- FIN CONFIG --------------

def safe_div(a, b, eps=EPS):
    return a / (b + eps)

def safe_exp(x, lo=-50.0, hi=50.0):
    return np.exp(np.clip(x, lo, hi))

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def _nanstd_safe(a):  # desviación estándar robusta a todo-NaN
    a = np.asarray(a, dtype=float)
    if np.all(np.isnan(a)):
        return np.nan
    return np.nanstd(a)

def _range_safe(a):   # max-min robusto a NaN
    a = np.asarray(a, dtype=float)
    if np.all(np.isnan(a)):
        return np.nan
    return np.nanmax(a) - np.nanmin(a)

def _max_safe(a):
    a = np.asarray(a, dtype=float)
    if np.all(np.isnan(a)):
        return np.nan
    return np.nanmax(a)

def compute_variable_scales(
    mats: Dict[str, pd.DataFrame],
    mode: str = "std",
    global_normalization: bool = True,
    min_scale: float = 1e-6
) -> Dict[str, float] | Dict[str, Dict[str, float]]:
    """
    Devuelve escalas por variable para normalizar la SSE.
    - Si global_normalization=True: {'X': sX, 'N': sN, 'G': sG, 'F': sF, 'E': sE}
    - Si False: {'SB003': {'X': sX, ...}, 'SB004': {...}, ...}

    mode: "std" | "range" | "max"
    """
    def agg(arr):
        if mode == "std":
            return _nanstd_safe(arr)
        elif mode == "range":
            return _range_safe(arr)
        elif mode == "max":
            return _max_safe(arr)
        else:
            raise ValueError("NORMALIZE_MODE debe ser 'std', 'range' o 'max'.")

    # recolecta vectores por variable
    vars_keys = {"X":"biomass_viable_gL", "N":"YAN", "G":"Glucose", "F":"Fructose", "E":"Ethanol"}

    if global_normalization:
        pools = {k: [] for k in vars_keys.keys()}
        for code, df in mats.items():
            if code in EXCLUDE_ASSAYS: 
                continue
            for key, col in vars_keys.items():
                if col in df.columns:
                    pools[key].append(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float))
        scales = {}
        for key, chunks in pools.items():
            if len(chunks):
                arr = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
                s = agg(arr)
                if not np.isfinite(s) or s < min_scale:
                    s = max(np.nanmean(np.abs(arr)), min_scale)  # fallback
                scales[key] = float(max(s, min_scale))
            else:
                scales[key] = 1.0  # si no hay datos, no escalar
        return scales
    else:
        scales = {}
        for code, df in mats.items():
            if code in EXCLUDE_ASSAYS:
                continue
            per = {}
            for key, col in vars_keys.items():
                if col in df.columns:
                    arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                    s = agg(arr)
                    if not np.isfinite(s) or s < min_scale:
                        s = max(np.nanmean(np.abs(arr)), min_scale)  # fallback
                    per[key] = float(max(s, min_scale))
                else:
                    per[key] = 1.0
            scales[code] = per
        return scales


# ---------- Perfiles de temperatura desde la matriz ----------
def build_temp_profile_from_df(df: pd.DataFrame) -> List[Tuple[float, float]]:
    """
    Construye un perfil piecewise-constant a partir de la columna de temperatura
    en la MATRIZ (columna en °C) y lo devuelve en KELVIN para el modelo.
    Devuelve lista [(t_change_h, T_K), ...].
    """
    # Detecta columna de T en °C
    cand = None
    for c in df.columns:
        l = str(c).strip().lower()
        if l in ("temperature_c", "temperature", "temp_c", "temperatura"):
            cand = c
            break

    # Si no hay columna, usa 20°C -> Kelvin
    if cand is None:
        return [(0.0, float(20.0 + 273.15))]

    t = pd.to_numeric(df["time_h"], errors="coerce")
    Tc = pd.to_numeric(df[cand], errors="coerce")
    mask = ~(t.isna() | Tc.isna())
    t = t[mask].to_numpy(dtype=float)
    Tc = Tc[mask].to_numpy(dtype=float)

    if t.size == 0:
        return [(0.0, float(20.0 + 273.15))]

    # Ordenar y compactar por tramos constantes
    order = np.argsort(t)
    t = t[order]; Tc = Tc[order]
    # Convertir a Kelvin aquí
    Tk = Tc + 273.15

    segs = [(float(t[0]), float(Tk[0]))]
    for i in range(1, len(t)):
        if not np.isclose(Tk[i], Tk[i-1], atol=1e-6):
            segs.append((float(t[i]), float(Tk[i])))

    if segs[0][0] > 0.0:
        segs.insert(0, (0.0, segs[0][1]))
    return segs

# ---------- Carga opcional de matrices desde carpeta ----------
def load_mats_from_folder(folder: str) -> Dict[str, pd.DataFrame]:
    d = {}
    for p in Path(folder).glob("calib_matrix_*.csv"):
        code = p.stem.replace("calib_matrix_", "")
        df = pd.read_csv(p)
        d[code] = df
    return d

# ---------- Inferencia de x0/pulsos desde matrices ----------
def infer_x0_from_df(df: pd.DataFrame) -> np.ndarray:
    """
    Inicializa x0 por ensayo. Prioridad:
     1) columnas explícitas X0_X, X0_N, X0_G, X0_F, X0_E si existen
     2) primera medición disponible (t mínimo) de biomass_viable_gL, YAN, Glucose, Fructose, Ethanol
     3) fallback DEFAULT_X0
    """
    # Opción 1: columnas X0_*
    cols = {"X": "X0_X", "N": "X0_N", "G": "X0_G", "F": "X0_F", "E": "X0_E"}
    if all(c in df.columns for c in cols.values()):
        try:
            x0 = np.array([df[cols["X"]].iloc[0],
                           df[cols["N"]].iloc[0],
                           df[cols["G"]].iloc[0],
                           df[cols["F"]].iloc[0],
                           df[cols["E"]].iloc[0]], dtype=float)
            if np.all(np.isfinite(x0)):
                return x0
        except Exception:
            pass

    # Opción 2: primera fila por tiempo
    t = pd.to_numeric(df["time_h"], errors="coerce").to_numpy(dtype=float)
    idx0 = np.nanargmin(t) if np.isfinite(np.nanmin(t)) else 0
    def get(col, default):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(v[idx0]):
                return float(v[idx0])
        return float(default)
    X0 = get("biomass_viable_gL", DEFAULT_X0[0])
    N0 = get("YAN",               DEFAULT_X0[1])
    G0 = get("Glucose",           DEFAULT_X0[2])
    F0 = get("Fructose",          DEFAULT_X0[3])
    E0 = get("Ethanol",           DEFAULT_X0[4])
    x0 = np.array([X0, N0, G0, F0, E0], dtype=float)
    # Limpieza
    x0 = np.where(np.isfinite(x0), x0, DEFAULT_X0)
    x0 = np.maximum(x0, 0.0)
    return x0

def infer_pulses_from_df(df: pd.DataFrame) -> Optional[List[Tuple[float, float]]]:
    """
    Si la matriz trae columnas 'pulse_time_h' y 'pulse_amount_gL', arma la lista.
    Devuelve None si no existen (para que apliquen defaults).
    """
    c_t = None; c_a = None
    for c in df.columns:
        l = str(c).strip().lower()
        if l == "pulse_time_h":
            c_t = c
        elif l in ("pulse_amount_gl", "pulse_amount_g_l", "pulse_gl"):
            c_a = c
    if c_t is None or c_a is None:
        return None
    tt = pd.to_numeric(df[c_t], errors="coerce").to_numpy(dtype=float)
    aa = pd.to_numeric(df[c_a], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(tt) & np.isfinite(aa)
    T = tt[mask]; A = aa[mask]
    if T.size == 0:
        return None
    order = np.argsort(T)
    out = [(float(T[i]), float(A[i])) for i in order]
    return out

def build_x0_and_pulses(mats: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Tuple[float,float]]]]:
    x0_by = {}
    pulses_by = {}
    for code, df in mats.items():
        if code in EXCLUDE_ASSAYS:
            continue
        # x0: manual > inferido > DEFAULT_X0
        x0_by[code] = X0_BY_ASSAY.get(code, infer_x0_from_df(df))
        # pulsos: manual > columnas > PULSOS_N (global)
        p_cols = infer_pulses_from_df(df)
        if code in PULSOS_BY_ASSAY:
            pulses_by[code] = list(PULSOS_BY_ASSAY[code])
        elif p_cols is not None:
            pulses_by[code] = p_cols
        else:
            pulses_by[code] = list(PULSOS_N)
    return x0_by, pulses_by

# ---------- Simulación alineada a mediciones ----------
def simulate_on_grid(p_real, time_h, temp_segments, pulses, x0=None):
    if x0 is None:
        x0 = DEFAULT_X0.copy()

    tf = float(np.nanmax(time_h)) if np.nanmax(time_h) > 0 else TF_HOURS
    n_min = int(np.ceil(tf / 0.5))
    n = max(n_min, int(tf)) if N_STEPS is None else int(N_STEPS)

    # temp_segments viene ahora en Kelvin, pero por seguridad validamos:
    T_profile, Nadd_profile = build_profiles(tf, n, temps_c=None, injections=pulses, temp_segments=temp_segments)

    # --- Parche defensivo: si accidentalmente vinieran en °C, convierto a K ---
    if np.nanmedian(T_profile) < 200.0:   # umbral claro para detectar °C
        T_profile = T_profile + 273.15
    # -------------------------------------------------------------------------

    u = np.vstack([T_profile, Nadd_profile]).T
    t_sim, Xsim = RK4_method(zenteno_model, tf, x0, n, u, p_real)
    return t_sim, Xsim

# ---------- Resumen de datos (para verificar) ----------
def preview_data(mats: Dict[str, pd.DataFrame], max_print: int = 6, plot_temp: bool = True):
    kept = [k for k in mats.keys() if k not in EXCLUDE_ASSAYS]
    print("\n=== Ensayos incluidos (excluyendo SB001/SB002) ===")
    print(", ".join(kept))
    print("\n=== Resumen por ensayo ===")
    for code in kept[:max_print]:
        df = mats[code]
        t = pd.to_numeric(df["time_h"], errors="coerce")
        Tmin = np.nanmin(t); Tmax = np.nanmax(t)
        Tc = None
        for c in ("Temperature_C","temperature","temp_c","temperatura"):
            if c in df.columns:
                Tc = pd.to_numeric(df[c], errors="coerce"); break
        tmsg = f"{Tmin:.2f}h → {Tmax:.2f}h" if np.isfinite(Tmin) and np.isfinite(Tmax) else "NA"
        if Tc is not None:
            print(f"{code}: n={len(df)}  t:{tmsg}  Temp[°C] min/mean/max = {np.nanmin(Tc):.2f}/{np.nanmean(Tc):.2f}/{np.nanmax(Tc):.2f}")
        else:
            print(f"{code}: n={len(df)}  t:{tmsg}  (sin columna de temperatura)")
    if plot_temp:
        keys = kept[:4]
        if keys:
            n = len(keys)
            ncols = 2
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 3.5*nrows))
            axes = np.atleast_2d(axes)
            for i, code in enumerate(keys):
                ax = axes.flat[i]
                df = mats[code]
                t = pd.to_numeric(df["time_h"], errors="coerce").to_numpy(dtype=float)/24.0
                Tc = None
                for c in ("Temperature_C","temperature","temp_c","temperatura"):
                    if c in df.columns:
                        Tc = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
                        break
                if Tc is not None:
                    ax.plot(t, Tc, 'o-', ms=3)
                    ax.set_title(f"{code} · Temp vs días")
                    ax.set_xlabel("t [d]")
                    ax.set_ylabel("T [°C]")
                    ax.grid(True, alpha=0.3)
                else:
                    ax.set_visible(False)
            plt.tight_layout()
            plt.show()

# ---------- Costo (SSE) en espacio REAL ----------
def sse_for_experiments_real(p_real: np.ndarray,
                             mats: Dict[str, pd.DataFrame],
                             pulses_by_assay: Dict[str, List[Tuple[float, float]]] = None,
                             x0_by_assay: Dict[str, np.ndarray] = None,
                             weights: Dict[str, float] = None,
                             scales=None,
                             verbose: bool = False) -> float:
    if weights is None:
        weights = WEIGHTS
    if pulses_by_assay is None:
        pulses_by_assay = {}
    if x0_by_assay is None:
        x0_by_assay = {}

    total = 0.0
    
    # Resolver de escalas por variable (global o por ensayo)
    def get_scale(code: str, var_key: str) -> float:
        if scales is None:
            return 1.0
        if isinstance(scales, dict) and all(isinstance(v, (int, float)) for v in scales.values()):
            return float(scales.get(var_key, 1.0))  # global
        elif isinstance(scales, dict) and code in scales:
            return float(scales[code].get(var_key, 1.0))  # por-ensayo
        return 1.0
    
    for code, df in mats.items():
        if code in EXCLUDE_ASSAYS:
            continue
        df = df.copy()
        t_meas = pd.to_numeric(df["time_h"], errors="coerce").to_numpy(dtype=float)

        temp_segs = build_temp_profile_from_df(df)
        pulses = pulses_by_assay.get(code, PULSOS_N)
        x0 = x0_by_assay.get(code, DEFAULT_X0.copy())

        # Simula
        t_sim, Xsim = simulate_on_grid(p_real, t_meas, temp_segs, pulses, x0=x0)

        # Interpola simulación en tiempos medidos
        t_sim = np.asarray(t_sim, dtype=float)
        X_interp = np.vstack([
            np.interp(t_meas, t_sim, Xsim[:, 0]),  # X
            np.interp(t_meas, t_sim, Xsim[:, 1]),  # N
            np.interp(t_meas, t_sim, Xsim[:, 2]),  # G
            np.interp(t_meas, t_sim, Xsim[:, 3]),  # F
            np.interp(t_meas, t_sim, Xsim[:, 4]),  # E
        ]).T

        sse = 0.0
        # X
        if "biomass_viable_gL" in df.columns:
            y = pd.to_numeric(df["biomass_viable_gL"], errors="coerce").to_numpy(dtype=float)
            m = ~np.isnan(y)
            if m.any():
                sX = max(get_scale(code, "X"), MIN_SCALE)
                r = (X_interp[m,0] - y[m]) / sX
                sse += weights.get("X", 1.0) * np.nansum(r*r)
        # N ~ YAN
        if "YAN" in df.columns:
            y = pd.to_numeric(df["YAN"], errors="coerce").to_numpy(dtype=float)
            m = ~np.isnan(y)
            if m.any():
                sN = max(get_scale(code, "N"), MIN_SCALE)
                r = (X_interp[m,1] - y[m]) / sN
                sse += weights.get("N", 0.5) * np.nansum(r*r)
        # Glu
        if "Glucose" in df.columns:
            y = pd.to_numeric(df["Glucose"], errors="coerce").to_numpy(dtype=float)
            m = ~np.isnan(y)
            if m.any():
                sG = max(get_scale(code, "G"), MIN_SCALE)
                r = (X_interp[m,2] - y[m]) / sG
                sse += weights.get("G", 1.0) * np.nansum(r*r)
        # Fru
        if "Fructose" in df.columns:
            y = pd.to_numeric(df["Fructose"], errors="coerce").to_numpy(dtype=float)
            m = ~np.isnan(y)
            if m.any():
                sF = max(get_scale(code, "F"), MIN_SCALE)
                r = (X_interp[m,3] - y[m]) / sF
                sse += weights.get("F", 1.0) * np.nansum(r*r)
        # EtOH
        if "Ethanol" in df.columns:
            y = pd.to_numeric(df["Ethanol"], errors="coerce").to_numpy(dtype=float)
            m = ~np.isnan(y)
            if m.any():
                sE = max(get_scale(code, "E"), MIN_SCALE)
                r = (X_interp[m,4] - y[m]) / sE
                sse += weights.get("E", 0.5) * np.nansum(r*r)

        total += sse

    if verbose:
        print(f"SSE={total:.4e}")
    return float(total)

# ---------- Reparametrización log-escala ----------
def make_internal_transform(p0_real: np.ndarray,
                            bounds_real: List[Tuple[float, float]]):
    """
    Devuelve closures:
      - real_from_z(z) = s * exp(z)
      - z_from_real(p) = log(p/s)
    con escala s = max(p0, 1e-6) elemento a elemento, y bounds en z-space.
    """
    p0 = np.asarray(p0_real, dtype=float)
    s  = np.maximum(p0, 1e-6)

    lb = np.array([lo for (lo, hi) in bounds_real], dtype=float)
    ub = np.array([hi for (lo, hi) in bounds_real], dtype=float)
    z_lb = np.log(np.maximum(lb / s, 1e-12))
    z_ub = np.log(ub / s)

    def real_from_z(z: np.ndarray) -> np.ndarray:
        return s * np.exp(np.asarray(z, dtype=float))

    def z_from_real(p: np.ndarray) -> np.ndarray:
        return np.log(np.asarray(p, dtype=float) / s)

    return real_from_z, z_from_real, list(zip(z_lb, z_ub)), s

# ======= PROGRESO / LOGGING =======
class EarlyStop(Exception):
    pass

class Progress:
    def __init__(self, name="OPT"):
        self.name = name
        self.t0 = time.time()
        self.eval_count = 0
        self.best_sse = float("inf")
        self.best_z = None
        self.best_eval = 0
        self.best_time = self.t0
        self.iter_de = 0

    def mark_eval(self, sse, z=None, every=50):
        self.eval_count += 1
        if sse < self.best_sse:
            self.best_sse = float(sse)
            self.best_eval = self.eval_count
            self.best_time = time.time()
            if z is not None:
                self.best_z = np.asarray(z, dtype=float).copy()

        if (self.eval_count % every) == 0:
            dt = time.time() - self.t0
            print(f"[{self.name}] eval={self.eval_count:6d}  best_SSE={self.best_sse:.4e}  t={dt:6.1f}s")
            sys.stdout.flush()

        # Early stop por evals totales o por no-mejora
        if self.eval_count >= MAX_EVALS_TOTAL:
            raise EarlyStop(f"MAX_EVALS_TOTAL alcanzado: {self.eval_count}")
        if (self.eval_count - self.best_eval) >= NO_IMPROVE_EVALS:
            raise EarlyStop(f"Sin mejora en {NO_IMPROVE_EVALS} evaluaciones consecutivas")

    def mark_de_iter(self, xk, convergence):
        self.iter_de += 1
        dt = time.time() - self.t0
        print(f"[DE] iter={self.iter_de:4d}  conv={convergence:.3e}  best_SSE={self.best_sse:.4e}  t={dt:6.1f}s")
        sys.stdout.flush()

# ---------- Calibración global en espacio interno ----------
def calibrate_global_internal(mats: Dict[str, pd.DataFrame],
                              p0_real: np.ndarray,
                              bounds_real: List[Tuple[float, float]],
                              mode: str = "de",
                              pulses_by_assay: Dict[str, List[Tuple[float, float]]] = None,
                              x0_by_assay: Dict[str, np.ndarray] = None,
                              maxiter_de: int = 80,
                              n_starts: int = 20,
                              verbose: bool = True):
    """
    Optimiza en z-space con p = s*exp(z). Devuelve (p_best_real, sse_best, result_obj).
    mode = "de"          -> Differential Evolution + pulido local
    mode = "multistart"  -> Múltiples L-BFGS-B con Sobol init + mejor solución (con early stopping)
    """
    real_from_z, z_from_real, z_bounds, s = make_internal_transform(p0_real, bounds_real)
    z0 = np.clip(z_from_real(p0_real), [b[0] for b in z_bounds], [b[1] for b in z_bounds])

    prog = Progress(name=f"OPT-{mode.upper()}")

    def obj_z(z):
        p = real_from_z(z)
        sse = sse_for_experiments_real(p, mats, pulses_by_assay, x0_by_assay, WEIGHTS, verbose=False)
        prog.mark_eval(sse, z=z, every=50)
        return sse

    result = {}

    if mode == "de":
        def cb_de(xk, convergence):
            prog.mark_de_iter(xk, convergence)
            return False

        try:
            de_res = differential_evolution(
                obj_z,
                bounds=z_bounds,
                maxiter=maxiter_de,
                popsize=15,
                mutation=(0.5, 1.0),
                recombination=0.7,
                tol=1e-6,
                polish=False,
                updating='immediate',
                workers=1,
                disp=False,
                callback=cb_de
            )
        except EarlyStop as e:
            print(f"[DE] EarlyStop: {e}")
            # Tomar lo mejor visto por el progreso (si lo hay)
            if prog.best_z is None:
                raise
            de_res = type("DEres", (), {})()
            de_res.x = prog.best_z
            de_res.fun = prog.best_sse

        z_best = de_res.x.copy()
        sse_best = de_res.fun

        # Pulido local
        try:
            loc = minimize(obj_z, z_best, method="L-BFGS-B", bounds=z_bounds,
                           options=dict(maxiter=300, ftol=1e-9, maxfun=10_000))
            print(f"[LOCAL] success={loc.success}  SSE={loc.fun:.4e}")
            if loc.success and loc.fun < sse_best:
                z_best = loc.x.copy()
                sse_best = loc.fun
            result = {"de": de_res, "local": loc}
        except EarlyStop as e:
            print(f"[LOCAL] EarlyStop: {e}")
            result = {"de": de_res}

    elif mode == "multistart":
        # Muestreo inicial Sobol/uniforme
        try:
            from scipy.stats.qmc import Sobol
            qmc = Sobol(d=len(z_bounds), scramble=True, seed=123)
            U = qmc.random_base2(int(np.ceil(np.log2(n_starts))))
            U = U[:n_starts]
        except Exception:
            U = np.random.default_rng(123).uniform(size=(n_starts, len(z_bounds)))

        z_lo = np.array([b[0] for b in z_bounds]); z_hi = np.array([b[1] for b in z_bounds])
        z_candidates = z_lo + U * (z_hi - z_lo)
        z_candidates[0, :] = z0  # incluye p0 como start

        z_best = None; sse_best = np.inf; local_runs = []
        stale_starts = 0  # paciencia en starts sin mejora

        for i, zi in enumerate(z_candidates, start=1):
            print(f"[MS] start {i}/{len(z_candidates)}: lanzando L-BFGS-B...")
            try:
                loc = minimize(
                    obj_z, zi, method="L-BFGS-B", bounds=z_bounds,
                    options=dict(maxiter=250, ftol=1e-8, maxfun=6_000)
                )
            except EarlyStop as e:
                # Parada global: si tenemos algo mejor, lo usamos y salimos
                print(f"[MS] EarlyStop durante start {i}: {e}")
                if z_best is None and prog.best_z is not None:
                    z_best = prog.best_z.copy()
                    sse_best = prog.best_sse
                break

            local_runs.append(loc)
            print(f"[MS]  done  {i}/{len(z_candidates)}: success={loc.success}  SSE={loc.fun:.4e}")

            # ¿Mejoró global?
            if loc.success and loc.fun < sse_best - 1e-10:
                sse_best = float(loc.fun)
                z_best = loc.x.copy()
                stale_starts = 0
                print(f"[MS]  >>> nuevo BEST SSE={sse_best:.4e}")
            else:
                stale_starts += 1
                if stale_starts >= STALE_STARTS_PATIENCE:
                    print(f"[MS]  >>> Early stop por starts estancados (paciencia={STALE_STARTS_PATIENCE}).")
                    break

        # Si por early-stop nunca seteamos z_best, usa lo mejor visto por el progreso
        if z_best is None and prog.best_z is not None:
            z_best = prog.best_z.copy()
            sse_best = prog.best_sse

        if z_best is None:
            # Nada válido: relanza con z0 al menos
            z_best = z0.copy()
            sse_best = obj_z(z_best)

        result = {"multistart": local_runs}
    else:
        raise ValueError("mode debe ser 'de' o 'multistart'")

    p_best_real = real_from_z(z_best)
    return p_best_real, float(sse_best), result

# ---------- Gráfica de ajuste ----------
def plot_fit_per_assay(p_real: np.ndarray,
                       mats: Dict[str, pd.DataFrame],
                       pulses_by_assay: Dict[str, List[Tuple[float, float]]] = None,
                       x0_by_assay: Dict[str, np.ndarray] = None):
    kept = [k for k in mats.keys() if k not in EXCLUDE_ASSAYS]
    n = len(kept)
    if n == 0:
        print("No hay ensayos para graficar (tras excluir).")
        return
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 3.8*nrows))
    axes = np.atleast_2d(axes)

    for i, code in enumerate(kept):
        r = i // ncols; c = i % ncols; ax = axes[r, c]
        df = mats[code]

        t_meas_h = pd.to_numeric(df["time_h"], errors="coerce").to_numpy(dtype=float)
        t_meas_d = t_meas_h / 24.0

        temp_segs = build_temp_profile_from_df(df)
        pulses = (pulses_by_assay or {}).get(code, PULSOS_N)
        x0 = (x0_by_assay or {}).get(code, DEFAULT_X0.copy())

        t_sim, Xsim = simulate_on_grid(p_real, t_meas_h, temp_segs, pulses, x0=x0)
        td = np.asarray(t_sim, dtype=float) / 24.0

        ax.plot(td, Xsim[:, 0], '-', label="X sim (g/L)")
        ax.plot(td, Xsim[:, 1], '-', label="N sim (g/L)")
        ax.plot(td, Xsim[:, 2], '-', label="G sim (g/L)")
        ax.plot(td, Xsim[:, 3], '-', label="F sim (g/L)")
        ax.plot(td, Xsim[:, 4], '-', label="E sim (g/L)")

        if "biomass_viable_gL" in df.columns:
            ax.plot(t_meas_d, df["biomass_viable_gL"], 'o', label="X exp")
        if "YAN" in df.columns:
            ax.plot(t_meas_d, df["YAN"], 'o', label="N exp")
        if "Glucose" in df.columns:
            ax.plot(t_meas_d, df["Glucose"], 'o', label="G exp")
        if "Fructose" in df.columns:
            ax.plot(t_meas_d, df["Fructose"], 'o', label="F exp")
        if "Ethanol" in df.columns:
            ax.plot(t_meas_d, df["Ethanol"], 'o', label="E exp")

        ax.set_title(f"Ensayo {code}")
        ax.set_xlabel("Tiempo (días)")
        ax.set_ylabel("Concentración")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=3)

    total_axes = nrows * ncols
    for k in range(n, total_axes):
        axes.flat[k].set_visible(False)
    plt.tight_layout()
    plt.show()

# ===================== MAIN =====================
if __name__ == "__main__":
    # 0) Parámetros iniciales (reales) desde Excel
    P0 = load_parameters_from_excel(PARAM_XLSX, sheet_name=PARAM_SHEET, param_set=PARAM_SET)
    print("P0 (reales):", P0)

    # 1) Cargar matrices:
    mats = None
    if RUTA_MATRICES:
        mats = load_mats_from_folder(RUTA_MATRICES)

    if mats is None:
        from Calibration_data_preprocess import process_all, build_calibration_matrices, attach_temperature_to_results
        FILE_PATH = r"C:/Users/ctorrealba/OneDrive - Viña Concha y Toro S.A/documentos/proyectos i+d/PI-4497/resultados/2025/dFBA/calibración modelo piloto 24_25/calibración data 2025/Procesos_I+D_2025_3.xlsx"   # <-- EDITA
        results_dict, _ = process_all(FILE_PATH, assays=None)
        results_with_T = attach_temperature_to_results(results_dict)
        mats = build_calibration_matrices(results_with_T, use_smoothed_biomass=False)

    # 1.1) Excluir ensayos problemáticos
    for bad in list(EXCLUDE_ASSAYS):
        if bad in mats:
            del mats[bad]

    if not mats:
        raise RuntimeError("No se encontraron matrices válidas para calibración.")

    # 1.2) Vista previa
    preview_data(mats, max_print=10, plot_temp=True)

    # 1.3) Construir x0 y pulsos por ensayo (de datos + overrides)
    X0S, PULSOS = build_x0_and_pulses(mats)
    n_x0 = sum(1 for _ in X0S)
    n_pu = sum(1 for _ in PULSOS)
    print(f"\n>>> x0 por ensayo: {n_x0} entradas  |  Pulsos por ensayo: {n_pu} entradas")
    # Muestra 3 ejemplos
    for k in list(X0S.keys())[:3]:
        print(f"  {k}: x0={np.array2string(X0S[k], precision=3)}  pulses={PULSOS.get(k, [])}")

    # 2) Calibración
    if MODE not in ("de", "multistart"):
        MODE = "de"

    print(f"\n>>> MODO de optimización: {MODE}  (parámetros en z-space con p = s*exp(z))")
    p_best, sse_best, result = calibrate_global_internal(
        mats=mats,
        p0_real=P0,
        bounds_real=P_BOUNDS_REAL,
        mode=MODE,
        pulses_by_assay=PULSOS,   # <<<<<<<<<<<<<< AHORA SE PASAN
        x0_by_assay=X0S,          # <<<<<<<<<<<<<< AHORA SE PASAN
        maxiter_de=80,
        n_starts=24,
        verbose=True
    )

    print("\n=== RESULTADOS ===")
    print("SSE (final):", sse_best)
    print("Parámetros (reales) estimados:\n", p_best)

    # 3) Gráfico de ajuste por ensayo (con x0 y pulsos)
    plot_fit_per_assay(p_best, mats, pulses_by_assay=PULSOS, x0_by_assay=X0S)
