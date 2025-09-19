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
WEIGHTS = {"X": 1.0, "N": 1.0, "G": 1.0, "F": 1.0, "E": 1.0}

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
N_STEPS  = None             # None -> el código usa ~0.5 h por paso automáticamente
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
MODE            = "de"  # "de" o "multistart"
N_STARTS        = 10
LOCAL_MAXITER   = 300
LOCAL_FTOL      = 1e-9
ES_PATIENCE     = 5       # nº de starts sin mejora tras los cuales se corta
ES_REL_IMPROVE  = 1e-3    # mejora relativa mínima para resetear paciencia
EPS = 1e-9

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

def simulate_on_grid(p_real, time_h, temp_segments, pulses, x0, method="Radau"):
    """
    Versión que SÍ aplica pulsos instantáneos (ΔN):
    delega en simulate_stiff_with_pulses(.) que integra por tramos y
    aplica ΔN al final de cada tramo exactamente en el pulso.
    También asegura que el perfil térmico entre al modelo en Kelvin.
    """
    if x0 is None:
        print('AAAA')
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
    vals = {"X": [], "N": [], "G": [], "F": [], "E": []}
    for code, df in mats.items():
        if code in EXCLUDE_ASSAYS: continue
        if "biomass_viable_gL" in df: vals["X"].extend(pd.to_numeric(df["biomass_viable_gL"], errors="coerce").dropna().values.tolist())
        if "YAN"               in df:vals["N"].extend( (pd.to_numeric(df["YAN"], errors="coerce").dropna().values.astype(float) * N_SCALE).tolist() )        
        if "Glucose"           in df: vals["G"].extend(pd.to_numeric(df["Glucose"],           errors="coerce").dropna().values.tolist())
        if "Fructose"          in df: vals["F"].extend(pd.to_numeric(df["Fructose"],          errors="coerce").dropna().values.tolist())
        if "Ethanol"           in df: vals["E"].extend(pd.to_numeric(df["Ethanol"],           errors="coerce").dropna().values.tolist())
    stds = {}
    for k, arr in vals.items():
        if len(arr) == 0:
            stds[k] = 1.0
        else:
            s = float(np.nanstd(np.asarray(arr, dtype=float)))
            stds[k] = s if s > 0 else 1.0
    return stds

# ---------- Costo (SSE) normalizado ----------
def sse_for_experiments_real(p_real: np.ndarray,
                             mats: Dict[str, pd.DataFrame],
                             pulses_by_assay: 'Optional[Dict[str, List[Tuple[float, float]]]]' = None,
                             x0_by_assay: Optional[Dict[str, np.ndarray]] = None,
                             weights: Optional[Dict[str, float]] = None,
                             stds: Optional[Dict[str, float]] = None,
                             verbose: bool = False) -> float:
    if weights is None: weights = WEIGHTS
    if stds is None: stds = {k:1.0 for k in ["X","N","G","F","E"]}
    total = 0.0
    for code, df in mats.items():
        if code in EXCLUDE_ASSAYS: continue
        t_meas = pd.to_numeric(df["time_h"], errors="coerce").to_numpy(dtype=float)
        temp_segs_C = build_temp_profile_from_df(df)
        pulses = (pulses_by_assay or {}).get(code, PULSOS_N_DEF)
        x0 = (x0_by_assay or {}).get(code, DEFAULT_X0.copy())

        # simula
        t_sim, Xsim = simulate_on_grid(p_real, t_meas, temp_segs_C, pulses, x0=x0)
        t_sim = np.asarray(t_sim, dtype=float)

        X_interp = np.vstack([
            np.interp(t_meas, t_sim, Xsim[:,0]),
            np.interp(t_meas, t_sim, Xsim[:,1]),
            np.interp(t_meas, t_sim, Xsim[:,2]),
            np.interp(t_meas, t_sim, Xsim[:,3]),
            np.interp(t_meas, t_sim, Xsim[:,4]),
        ]).T

        sse = 0.0
        # X
        if "biomass_viable_gL" in df.columns:
            y = pd.to_numeric(df["biomass_viable_gL"], errors="coerce").to_numpy(dtype=float)
            m = ~np.isnan(y)
            if m.any():
                sse += weights.get("X",1.0) * np.nansum(((X_interp[m,0] - y[m]) / stds["X"])**2)
        # N
        if "YAN" in df.columns:
            y = pd.to_numeric(df["YAN"], errors="coerce").to_numpy(dtype=float) * N_SCALE
            m = ~np.isnan(y)
            if m.any():
                sse += weights.get("N",0.5) * np.nansum(((X_interp[m,1] - y[m]) / stds["N"])**2)
        # G
        if "Glucose" in df.columns:
            y = pd.to_numeric(df["Glucose"], errors="coerce").to_numpy(dtype=float)
            m = ~np.isnan(y)
            if m.any():
                sse += weights.get("G",1.0) * np.nansum(((X_interp[m,2] - y[m]) / stds["G"])**2)
        # F
        if "Fructose" in df.columns:
            y = pd.to_numeric(df["Fructose"], errors="coerce").to_numpy(dtype=float)
            m = ~np.isnan(y)
            if m.any():
                sse += weights.get("F",1.0) * np.nansum(((X_interp[m,3] - y[m]) / stds["F"])**2)
        # E
        if "Ethanol" in df.columns:
            y = pd.to_numeric(df["Ethanol"], errors="coerce").to_numpy(dtype=float)
            m = ~np.isnan(y)
            if m.any():
                sse += weights.get("E",0.5) * np.nansum(((X_interp[m,4] - y[m]) / stds["E"])**2)

        total += sse
    if verbose:
        print(f"SSE={total:.4e}")
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
                              # --- nuevos controles de parada temprana:
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

    def obj_z(z):
        nonlocal best_sse_seen, last_improve_eval
        p = real_from_z(z)
        sse = sse_for_experiments_real(p, mats, pulses_by_assay, x0_by_assay, WEIGHTS, stds, verbose=False)
        prog.mark_eval(sse)
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
            # misma función que obj_z, pero sin early-stop por evaluaciones
            p = real_from_z(z)
            sse = sse_for_experiments_real(p, mats, pulses_by_assay, x0_by_assay, WEIGHTS, stds, verbose=False)
            prog.mark_eval(sse)
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
            callback=cb_de
        )

        # pulido local
        loc = minimize(obj_z, de_res.x, method="L-BFGS-B", bounds=z_bounds,
                       options=dict(maxiter=LOCAL_MAXITER, ftol=LOCAL_FTOL))
        print(f"[LOCAL] success={loc.success}  SSE={loc.fun:.4e}")
        z_best = (loc.x if (loc.success and loc.fun < de_res.fun) else de_res.x).copy()
        sse_best = float(min(loc.fun, de_res.fun))
        result = {"de": de_res, "local": loc}

    else:
        raise ValueError("mode debe ser 'de' o 'multistart'")

    p_best_real = real_from_z(z_best)
    return p_best_real, float(sse_best), result

# ---------- Gráfica ----------
def plot_fit_per_assay(p_real: np.ndarray,
                       mats: Dict[str, pd.DataFrame],
                       pulses_by_assay: Optional[Dict[str, List[Tuple[float, float]]]] = None,
                       x0_by_assay: Optional[Dict[str, np.ndarray]] = None):
    import matplotlib as mpl

    kept = [k for k in mats.keys() if k not in EXCLUDE_ASSAYS]
    if not kept:
        print("No hay ensayos para graficar."); return

    n = len(kept); ncols = 2; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 3.9*nrows))
    axes = np.atleast_2d(axes)

    # Paleta (colores fijos por variable)
    palette = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    cN, cG, cF, cE, cX = palette[:5]

    def _scatter_if(ax, df, col, label, xvals, color, transform=None, marker='o'):
        if col in df.columns:
            y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            if transform is not None:
                y = transform(y)
            ax.scatter(
                xvals, y, s=22, alpha=0.9, label=label, color=color,
                marker=marker, zorder=3, linewidths=0.0
            )

    for i, code in enumerate(kept):
        ax1 = axes.flat[i]
        df = mats[code]

        # Tiempo medición (días)
        t_meas_h = pd.to_numeric(df["time_h"], errors="coerce").to_numpy(dtype=float)
        t_meas_d = t_meas_h / 24.0

        # Entradas de simulación
        temp_segs_C = build_temp_profile_from_df(df)
        pulses = (pulses_by_assay or {}).get(code, PULSOS_N_DEF)
        x0 = (x0_by_assay or {}).get(code, DEFAULT_X0.copy())

        # Simulación
        t_sim, Xsim = simulate_on_grid(p_real, t_meas_h, temp_segs_C, pulses, x0)
        td = np.asarray(t_sim, dtype=float) / 24.0

        # --- Eje principal (N, G, F, E) ---
        ax1.plot(td, Xsim[:, 1]/N_SCALE,   '-', label="N sim", color=cN, linewidth=1.7)
        ax1.plot(td, Xsim[:, 2],           '-', label="G sim", color=cG, linewidth=1.7)
        ax1.plot(td, Xsim[:, 3],           '-', label="F sim", color=cF, linewidth=1.7)
        ax1.plot(td, Xsim[:, 4],           '-', label="E sim", color=cE, linewidth=1.7)
        ax1.grid(alpha=0.25)

        # --- Eje secundario (X) ---
        ax2 = ax1.twinx()
        ax2.plot(td, Xsim[:, 0], '-', label="X sim", color=cX, linewidth=1.7)

        # --- Scatter EXP ---
        # X EXP al eje Y2
        _scatter_if(ax2, df, "biomass_viable_gL", "X exp", t_meas_d, color=cX, marker='o')
        # Resto EXP al eje principal (YAN en misma escala que N sim)
        _scatter_if(ax1, df, "YAN",      "N exp", t_meas_d, color=cN, marker='^')
        _scatter_if(ax1, df, "Glucose",  "G exp", t_meas_d, color=cG, marker='s')
        _scatter_if(ax1, df, "Fructose", "F exp", t_meas_d, color=cF, marker='D')
        _scatter_if(ax1, df, "Ethanol",  "E exp", t_meas_d, color=cE, marker='P')

        # Etiquetas y título
        ax1.set_title(f"Ensayo {code}")
        ax1.set_xlabel("Tiempo (días)")
        ax1.set_ylabel("N, G, F, E")
        ax2.set_ylabel("X")

        # Leyenda combinada (sim + exp, ambos ejes)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, ncol=3, fontsize=8,
                   loc="upper center", bbox_to_anchor=(0.5, 1.14))

    # Ocultar ejes sobrantes
    total_axes = nrows * ncols
    for k in range(n, total_axes):
        axes.flat[k].set_visible(False)

    plt.tight_layout()
    plt.show()

# ===================== MAIN =====================
if __name__ == "__main__":
    # 0) Parámetros iniciales (reales)
    P0 = load_parameters_from_excel(PARAM_XLSX, sheet_name=PARAM_SHEET, param_set=PARAM_SET)
    if P0 is None:
        raise RuntimeError("No se pudieron cargar P0 desde el Excel.")
    print("P0 (reales):", P0)

    # 1) Cargar matrices desde tu preprocesamiento
    from Calibration_data_preprocess import process_all, build_calibration_matrices, attach_temperature_to_results
    FILE_PATH = r"C:/Users/ctorrealba/OneDrive - Viña Concha y Toro S.A/Documentos/Proyectos I+D/PI-4497/Resultados/2025/SB_Calibration_2025/Procesos_I+D_2025_3.xlsx"   # <-- EDITA si cambia
    results_dict, chem_df = process_all(FILE_PATH, assays=None)
    results_with_T = attach_temperature_to_results(results_dict)
    
    # ------- NUEVO: construir pulsos desde planilla química -------
    def _col_like(df, *cands):
        for c in df.columns:
            l = str(c).strip().lower()
            for cand in cands:
                if l == cand.lower():
                    return c
        # fallback por contiene
        for c in df.columns:
            l = str(c).strip().lower()
            for cand in cands:
                if cand.lower() in l:
                    return c
        return None
    
    def build_pulses_from_chem(chem_df):
        """
        Devuelve dict {assay: [(t_h, dN_gL), ...]} usando códigos de muestras entregados.
        Busca por coincidencia en columna 'Código' (o similar). Si no encuentra exacto,
        intenta buscar por el número entre paréntesis o por 'contains'.
        """
        if chem_df is None or len(chem_df) == 0:
            print("[PULSOS] Advertencia: chem_df vacío; usaré pulsos por defecto.")
            return {}
    
        col_code = _col_like(chem_df, "Código", "codigo", "sample_id", "muestra")
        col_time = _col_like(chem_df, "time_h", "tiempo_h", "horas", "t_h")
        col_yan  = _col_like(chem_df, "YAN", "yan", "nitrogeno", "n_asm")
    
        if col_code is None or col_yan is None:
            print("[PULSOS] No se hallaron columnas de Código/YAN en chem_df; usaré pulsos por defecto.")
            return {}
    
        # Si no hay time_h, intentamos derivarlo desde timestamps
        if col_time is None:
            col_ts = _col_like(chem_df, "timestamp", "fecha", "datetime", "fecha_muestra")
            if col_ts is not None:
                ts = pd.to_datetime(chem_df[col_ts], errors="coerce")
                t0 = ts.min()
                chem_df["__time_h__"] = (ts - t0).dt.total_seconds() / 3600.0
                col_time = "__time_h__"
            else:
                print("[PULSOS] No hay 'time_h' ni timestamp; usaré 0.0 h para los pulsos encontrados.")
                chem_df["__time_h__"] = 0.0
                col_time = "__time_h__"
    
        # Normalizamos tipos
        dfc = chem_df.copy()
        dfc[col_yan]  = pd.to_numeric(dfc[col_yan], errors="coerce")
        dfc[col_time] = pd.to_numeric(dfc[col_time], errors="coerce")
    
        # Mapa de códigos objetivo -> etiqueta de ensayo
        target_codes = [
            "ING25-SB003-3 (25026)",
            "ING25-SB004-3  (25027)",
            "ING25-SB005-3  (25028)",
            "ING25-SB006-3  (25029)",
            "ING25-SB007-6 (25085)",
            "ING25-SB008-6 (25086)",
            "ING25-SB009-4  Post Nutri",
            "ING25-SB010-4  Post Nutri",
            "ING25-SB011-5 (25170)",
            "ING25-SB012-5 (25171)"
        ]
    
        def extract_assay(s):
            # busca patrón SBxxx en el string
            import re
            m = re.search(r"(SB\d{3})", s)
            return m.group(1) if m else None
    
        pulses = {}  # {assay: [(t_h, dN_gL)]}
        for cod in target_codes:
            assay = extract_assay(cod) or "UNKNOWN"
            # búsqueda flexible
            m = dfc[col_code].astype(str).str.contains(cod.strip(), case=False, regex=False)
            # si no hay match, intenta por el número entre paréntesis
            if not m.any():
                import re
                paren = re.search(r"\((\d+)\)", cod)
                if paren:
                    num = paren.group(1)
                    m = dfc[col_code].astype(str).str.contains(num, case=False, regex=False)
            # si aún no, intenta por "ING25-SBxxx-idx" sin paréntesis ni extra
            if not m.any():
                base = cod.split("(")[0].strip()
                m = dfc[col_code].astype(str).str.contains(base, case=False, regex=False)
    
            if not m.any():
                print(f"[PULSOS] Aviso: no se encontró '{cod}' en planilla química.")
                continue
    
            idxs = np.where(m.values)[0]
            idx = int(idxs[0])  # primer match
            yan_cur = float(dfc.iloc[idx][col_yan]) if pd.notna(dfc.iloc[idx][col_yan]) else np.nan
            t_cur   = float(dfc.iloc[idx][col_time]) if pd.notna(dfc.iloc[idx][col_time]) else 0.0
    
            # muestra anterior (en la planilla) → delta YAN
            if idx > 0:
                yan_prev = float(dfc.iloc[idx-1][col_yan]) if pd.notna(dfc.iloc[idx-1][col_yan]) else np.nan
            else:
                yan_prev = np.nan
    
            if np.isfinite(yan_cur) and np.isfinite(yan_prev):
                dYAN_mgL = max(yan_cur - yan_prev, 0.0)
            else:
                dYAN_mgL = 0.0
    
            dN_gL = dYAN_mgL / 1000.0  # mg/L → g/L
            pulses.setdefault(assay, []).append((t_cur, dN_gL))
    
        # ordenar por tiempo y acumular por ensayo (por si hubiera múltiples)
        for k in list(pulses.keys()):
            pulses[k] = sorted(pulses[k], key=lambda z: z[0])
    
        # Mensaje
        if pulses:
            print("[PULSOS] Construidos desde planilla química:")
            for a, lst in pulses.items():
                print(f"  {a}: {[(round(t,2), round(d,5)) for (t,d) in lst]}")
        else:
            print("[PULSOS] No se generaron entradas; se usarán pulsos por defecto donde aplique.")
    
        return pulses
    
    PULSOS_DESDE_QUIMICA = build_pulses_from_chem(chem_df)
    
    mats = build_calibration_matrices(results_with_T, use_smoothed_biomass=True)
    
    # --- Ajuste YAN en mats: restar 15, truncar <5 a 0 y negativos a 0 ---
    for code, df in mats.items():
        if "YAN" in df.columns:
            vals = pd.to_numeric(df["YAN"], errors="coerce")
            vals = vals - 15.0
            vals[vals < 5.0] = 0.0
            vals[vals < 0.0] = 0.0
            df["YAN"] = vals
            mats[code] = df
        
    # 1.1) excluir ensayos malos
    for bad in list(EXCLUDE_ASSAYS):
        if bad in mats: del mats[bad]

    if not mats:
        raise RuntimeError("No se encontraron matrices válidas para calibración.")

    # 1.2) vista previa
    preview_data(mats, max_print=10, plot_temp=True)

    # 1.3) x0 y pulsos por ensayo (ejemplo: inicializar desde primera fila)
    
    X0S = {}
    PULSOS = {}
    
    for code, df in mats.items():
        if code in EXCLUDE_ASSAYS: continue
        row0 = df.iloc[0]
        x0 = np.array([
            float(row0.get("biomass_viable_gL", DEFAULT_X0[0])),
            float(row0.get("YAN",               DEFAULT_X0[1])) * N_SCALE,  # ← escala a g/L
            float(row0.get("Glucose",           DEFAULT_X0[2])),
            float(row0.get("Fructose",          DEFAULT_X0[3])),
            float(row0.get("Ethanol",           DEFAULT_X0[4])),
        ], dtype=float)
        X0S[code] = x0
        # si quieres pulsos diferentes por ensayo, ajusta aquí
        PULSOS[code] = PULSOS_DESDE_QUIMICA[code]
    
    # --- 1.4) Chequeo rápido de simulación con P0 (sanity check) ---
    print("\n[Sanity] Simulando con P0 (pipeline idéntico al de calibración)...")
    
    # 1) Toma un ensayo representativo (el primero) y valida que exista
    if not mats:
        raise RuntimeError("[Sanity] No hay ensayos en 'mats'. Verifica el preprocesamiento.")
    first_code = next(iter(mats.keys()))
    df0 = mats[first_code]
    
    # 2) Prepara entradas
    t_meas = pd.to_numeric(df0["time_h"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(np.nanmax(t_meas)):
        raise RuntimeError(f"[Sanity] 'time_h' inválido en {first_code}.")
    
    temp_segs_C = build_temp_profile_from_df(df0)
    x0       = X0S.get(first_code)
    pulses0  = PULSOS.get(first_code, PULSOS_N_DEF)
    
    # 3) Simulación
    try:
        t_sim0, Xsim0 = simulate_on_grid(P0, t_meas, temp_segs_C, pulses0, x0)
    except Exception as e:
        raise RuntimeError(f"[Sanity] Falló simulate_on_grid con P0 en {first_code}: {e}")
    
    # 4) Validaciones básicas
    if not np.all(np.isfinite(Xsim0)):
        bad = np.where(~np.isfinite(Xsim0))[0][:5]
        raise AssertionError(f"[Sanity] NaN/Inf en Xsim con P0 (ej. filas {bad}).")
    if t_sim0[-1] < np.nanmax(t_meas) - 1e-6:
        raise AssertionError("[Sanity] t_sim no cubre el rango de mediciones.")
    
    # 5) Resumen rápido de órdenes de magnitud
    names = ["X","N","G","F","E"]
    mins  = np.nanmin(Xsim0, axis=0)
    maxs  = np.nanmax(Xsim0, axis=0)
    print("[Sanity] Rango simulado (min..max): " + ", ".join(f"{n}={mn:.3g}..{mx:.3g}" for n,mn,mx in zip(names, mins, maxs)))
    
    # 6) Plot comparativo (sim vs exp) para todas las variables disponibles
    import matplotlib as mpl
    
    plt.figure(figsize=(8.5, 5))
    td = np.asarray(t_sim0) / 24.0
    
    # Paleta consistente
    palette = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    cN, cG, cF, cE, cX = palette[:5]
    
    # --- Eje principal (N, G, F, E) ---
    ax1 = plt.gca()
    ax1.plot(td, Xsim0[:,1] / N_SCALE, '-', label="N sim", color=cN, linewidth=1.8)
    ax1.plot(td, Xsim0[:,2],           '-', label="G sim", color=cG, linewidth=1.8)
    ax1.plot(td, Xsim0[:,3],           '-', label="F sim", color=cF, linewidth=1.8)
    ax1.plot(td, Xsim0[:,4],           '-', label="E sim", color=cE, linewidth=1.8)
    ax1.set_xlabel("Tiempo (días)")
    ax1.set_ylabel("N, G, F, E")
    ax1.grid(alpha=0.25)
    
    # --- Eje secundario (X) ---
    ax2 = ax1.twinx()
    ax2.plot(td, Xsim0[:,0], '-', label="X sim", color=cX, linewidth=1.8)
    ax2.set_ylabel("X")
    
    # --- Scatter helper (usa color explícito para evitar reciclar) ---
    def _scatter_if(ax, col, label, xvals, color, transform=None, marker='o'):
        if col in df0.columns:
            y = pd.to_numeric(df0[col], errors="coerce").to_numpy(dtype=float)
            if transform is not None:
                y = transform(y)
            ax.scatter(
                xvals, y, s=28, alpha=0.9, label=label, color=color,
                marker=marker, zorder=3, linewidths=0.0
            )
    
    t_exp = np.asarray(t_meas) / 24.0
    
    # X EXP al eje Y2 (mismo color que X sim)
    _scatter_if(ax2, "biomass_viable_gL", "X exp", t_exp, color=cX)
    
    # Resto EXP al eje principal (mismos colores que sim; YAN escalado)
    _scatter_if(ax1, "YAN",      "N exp", t_exp, color=cN)
    _scatter_if(ax1, "Glucose",  "G exp", t_exp, color=cG)
    _scatter_if(ax1, "Fructose", "F exp", t_exp, color=cF)
    _scatter_if(ax1, "Ethanol",  "E exp", t_exp, color=cE)
    
    # --- Leyenda combinada ---
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    
    plt.title(f"Sanity check con P0 · {first_code}")
    plt.tight_layout()
    plt.show()

    
    print("[Sanity] OK: simulación con P0 completada.")


    print(f"\nx0 por ensayo: {len(X0S)} entradas  |  Pulsos por ensayo: {len(PULSOS)} entradas")
    for i,(k,v) in enumerate(X0S.items()):
        if i>=3: break
        print(f"  {k}: x0={np.round(v,3)}  pulses={PULSOS[k]}")

    # 2) Calibración
    print(f"\nMODO de optimización: {MODE}  (parámetros en z-space con p = s*exp(z))")
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

    print("\n=== RESULTADOS ===")
    print("SSE (final):", sse_best)
    print("Parámetros (reales) estimados:\n", p_best)

    # 3) Gráfico de ajuste
    plot_fit_per_assay(P0, mats, pulses_by_assay=PULSOS, x0_by_assay=X0S)
