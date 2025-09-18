# -*- coding: utf-8 -*-
"""
Calibración de parámetros (Zenteno) con Pyomo (RK4) + opciones de búsqueda:
  - ipopt_single
  - ipopt_multistart
  - de_global (DE externo + pulido opcional con Ipopt)
Requiere: pyomo, numpy, pandas, scipy
"""

import math
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# --- Pyomo ---
from pyomo.environ import (ConcreteModel, Var, Param, Set, NonNegativeReals,
                           Reals, Objective, Constraint, minimize, value,
                           SolverFactory, exp)
from pyomo.opt import TerminationCondition

# --- SciPy para DE global (opcional) ---
from scipy.optimize import differential_evolution

# --- Tu loader de parámetros iniciales (del archivo que compartiste) ---
from modelo_dinamico_sim import load_parameters_from_excel

# ===================== CONFIG =====================
# Pesos en el SSE (ajústalos si hace falta)
WEIGHTS = {"X": 1.0, "N": 0.2, "G": 1.0, "F": 1.0, "E": 0.5}

# Límites realistas (no escalados) para los 14 parámetros (ajusta si procede)
P_BOUNDS_REAL = [
    (1e-5, 2.0),   # mu0
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

# Por si falta temperatura en matrices:
DEFAULT_TEMP_C = 20.0

# IPOPT options (puedes ajustar tolerancias aquí)
IPOPT_OPTIONS = {
    "tol": 1e-8,
    "max_iter": 2000,
    "print_level": 5
}

# ==================================================

# ---------- Utilidades ----------
def kelvin(T_c: float) -> float:
    return float(T_c) + 273.15

def build_temperature_series(df: pd.DataFrame) -> np.ndarray:
    """Devuelve un vector T[K] para cada punto de tiempo (time_h)."""
    if "Temp_C" in df.columns and not df["Temp_C"].isna().all():
        T = pd.to_numeric(df["Temp_C"], errors="coerce").fillna(DEFAULT_TEMP_C).to_numpy(dtype=float)
    elif "Temperature" in df.columns and not df["Temperature"].isna().all():
        T = pd.to_numeric(df["Temperature"], errors="coerce").fillna(DEFAULT_TEMP_C).to_numpy(dtype=float)
    elif "Temperatura" in df.columns and not df["Temperatura"].isna().all():
        T = pd.to_numeric(df["Temperatura"], errors="coerce").fillna(DEFAULT_TEMP_C).to_numpy(dtype=float)
    else:
        T = np.full(len(df), DEFAULT_TEMP_C, dtype=float)
    return kelvin(T)

def finite_differences(t_h: np.ndarray) -> np.ndarray:
    """dt[k] = t[k+1] - t[k] (h), tamaño n-1."""
    t = np.asarray(t_h, dtype=float)
    return np.diff(t)

# --- RHS (Zenteno) expresado en términos Pyomo (usa exp de Pyomo) ---
def rhs_zenteno_pyomo(X, N, G, F, E, T, Nadd, p):
    """
    Devuelve diccionario con derivadas (expresiones Pyomo):
    dXdt, dNdt, dGdt, dFdt, dEdt
    p = [mu0, betaG0, betaF0, Kn0, Kg0, Kf0, Kig0, Kie0, Kd0, Yxn, Yxg, Yxf, Yeg, Yef]
    """
    mu0, betaG0, betaF0, Kn0, Kg0, Kf0, Kig0, Kie0, Kd0, Yxn, Yxg, Yxf, Yeg, Yef = p

    # Constantes (mismo paper)
    Cde  = 0.0415
    Etd  = 130000.0
    R    = 8.314
    Eac  = 59453.0
    Eafe = 11000.0
    EaKn = 46055.0
    EaKg = 46055.0
    EaKf = 46055.0
    EaKig= 46055.0
    EaKie= 46055.0
    Eam  = 37681.0
    m0   = 0.01

    # Ecuaciones constitutivas (usar exp de Pyomo)
    mu_max    = mu0 * exp(Eac*(T-300.0)/(300.0*R*T))
    betaG_max = betaG0 * exp(Eafe*(T-296.15)/(296.15*R*T))
    betaF_max = betaF0 * exp(Eafe*(T-296.15)/(296.15*R*T))
    Kn        = Kn0  * exp(EaKn*(T-293.15)/(293.15*R*T))
    Kg        = Kg0  * exp(EaKg*(T-293.15)/(293.15*R*T))
    Kf        = Kf0  * exp(EaKf*(T-293.15)/(293.15*R*T))
    Kig       = Kig0 * exp(EaKig*(T-293.15)/(293.15*R*T))
    Kie       = Kie0 * exp(EaKie*(T-293.15)/(293.15*R*T))
    m         = m0   * exp(Eam *(T-293.3) /(293.3 *R*T))

    # Velocidades
    mu     = mu_max * (N/(N+Kn))
    beta_G = betaG_max * (G/(G+Kg)) * (Kie/(E+Kie))
    beta_F = betaF_max * (F/(F+Kf)) * (Kig/(G+Kig)) * (Kie/(E+Kie))

    # Umbral de muerte térmica
    Td = (-0.0001*(E**3)) + (0.0049*(E**2)) - (0.1279*E) + 315.89  # K

    # Tasa de muerte específica
    # (En Pyomo, usa condicional con 'if' a nivel de regla no es diferenciable,
    #  pero Ipopt normalmente tolera esta piecewise simple; si no, podemos suavizar)
    Kd = Kd0 * exp((Cde*E) + (Etd*(T-305.65))/(305.65*R*T)) if (T >= Td) else 0.0

    dXdt = mu*X - Kd*X
    dNdt = -mu*(X/Yxn) + Nadd
    # evitar división por cero en (G+F) con un epsilon pequeño:
    eps = 1e-9
    dGdt = -((mu/Yxg) + (beta_G/Yeg) + m*(G/(G + F + eps)))*X
    dFdt = -((mu/Yxf) + (beta_F/Yef) + m*(F/(G + F + eps)))*X
    dEdt = (beta_G + beta_F)*X

    return {"X": dXdt, "N": dNdt, "G": dGdt, "F": dFdt, "E": dEdt}

def sanitize_mats_for_pyomo(mats: dict) -> dict:
    clean = {}
    for code, df in mats.items():
        if df is None or df.empty or "time_h" not in df.columns:
            continue
        dfx = df.copy().sort_values("time_h")
        t = pd.to_numeric(dfx["time_h"], errors="coerce")
        t = t.dropna().to_numpy(dtype=float)
        if t.size < 2:
            # sin transiciones -> sin RK4
            continue

        # ¿hay al menos una observación no-NaN en alguna variable?
        cols_obs = ["biomass_viable_gL", "YAN", "Glucose", "Fructose", "Ethanol"]
        has_any = False
        for c in cols_obs:
            if c in dfx.columns and pd.to_numeric(dfx[c], errors="coerce").notna().any():
                has_any = True
                break
        if not has_any:
            continue

        # Reanclar t0=0 para ese ensayo (por si no lo estaba)
        t0 = t[0]
        dfx["time_h"] = pd.to_numeric(dfx["time_h"], errors="coerce") - t0
        clean[code] = dfx
    return clean

# ---------- MODELO PYOMO (RK4 sobre malla de datos) ----------
def build_pyomo_model(mats: Dict[str, pd.DataFrame],
                      p0_real: np.ndarray,
                      bounds_real: List[Tuple[float, float]] = None,
                      weights: Dict[str, float] = None,
                      multistart_seed: Optional[int] = None):
    """
    Modelo Pyomo con:
      - Parámetros escalados ps[j] (p[j] = scale[j]*ps[j]) con init clipeado a bounds.
      - Estados X,N,G,F,E indexados por pares (a,k) en m.AK.
      - Dinámica RK4 en m.AK_TRANS (pares donde existe el siguiente nodo k+1).
      - SSE ponderado frente a observaciones disponibles.
    """
    if bounds_real is None:
        bounds_real = P_BOUNDS_REAL
    if weights is None:
        weights = WEIGHTS

    m = ConcreteModel()

    # --- Dimensión parámetros y escalado ---
    npar = len(p0_real)
    assert npar == len(bounds_real), "p0_real y bounds_real deben tener mismo largo"

    scale = np.maximum(np.abs(p0_real), 1e-3)
    lb_s = np.array([lo/s for (lo, _), s in zip(bounds_real, scale)], dtype=float)
    ub_s = np.array([hi/s for (_, hi), s in zip(bounds_real, scale)], dtype=float)

    m.J = Set(initialize=list(range(npar)))
    m.ps = Var(m.J, domain=Reals)

    # Inicialización clipeada a los bounds escalados
    for j in range(npar):
        base = float(p0_real[j] / scale[j])
        ini = min(max(base, lb_s[j]), ub_s[j])
        m.ps[j].setlb(float(lb_s[j]))
        m.ps[j].setub(float(ub_s[j]))
        m.ps[j].value = ini

    m.scale = Param(m.J, initialize={j: float(scale[j]) for j in range(npar)}, mutable=False)

    # --- Preparación de mallas por ensayo ---
    m.ASSAY = Set(initialize=list(mats.keys()))

    tgrid, Tgrid, dtgrid, obs, last_k = {}, {}, {}, {}, {}
    for code, df in mats.items():
        df = df.copy().sort_values("time_h")
        t = pd.to_numeric(df["time_h"], errors="coerce").to_numpy(dtype=float)
        if len(t) == 0:
            t = np.array([0.0], dtype=float)
        t = t - t[0]  # anclar t0=0
        T = build_temperature_series(df)  # Kelvin
        dt = finite_differences(t)

        tgrid[code] = t
        Tgrid[code] = T
        dtgrid[code] = dt
        last_k[code] = len(t) - 1

        obs[code] = {
            "X": pd.to_numeric(df.get("biomass_viable_gL", np.nan), errors="coerce").to_numpy(dtype=float),
            "N": pd.to_numeric(df.get("YAN",               np.nan), errors="coerce").to_numpy(dtype=float),
            "G": pd.to_numeric(df.get("Glucose",           np.nan), errors="coerce").to_numpy(dtype=float),
            "F": pd.to_numeric(df.get("Fructose",          np.nan), errors="coerce").to_numpy(dtype=float),
            "E": pd.to_numeric(df.get("Ethanol",           np.nan), errors="coerce").to_numpy(dtype=float),
        }

    # --- Set aplanado de índices (a,k) ---
    def _init_AK(model):
        pairs = []
        for a in m.ASSAY:
            for k in range(len(tgrid[a])):
                pairs.append((a, k))
        return pairs

    def _init_AK_TRANS(model):
        pairs = []
        for a in m.ASSAY:
            for k in range(last_k[a]):  # hasta penúltimo
                pairs.append((a, k))
        return pairs

    m.AK       = Set(dimen=2, initialize=_init_AK)
    m.AK_TRANS = Set(dimen=2, initialize=_init_AK_TRANS)

    # --- Variables de estado en (a,k) ---
    m.X = Var(m.AK, domain=NonNegativeReals, initialize=0.1)
    m.N = Var(m.AK, domain=NonNegativeReals, initialize=0.1)
    m.G = Var(m.AK, domain=NonNegativeReals, initialize=1.0)
    m.F = Var(m.AK, domain=NonNegativeReals, initialize=1.0)
    m.E = Var(m.AK, domain=NonNegativeReals, initialize=0.0)

    # --- Helper para recuperar p real ---
    def get_p_real(model):
        return [model.ps[j]*model.scale[j] for j in model.J]

    # --- Restricciones RK4 en pares de transición (a,k) ---
    def rk4_rule_X(model, a, k):
        h   = float(dtgrid[a][k])
        Tk  = float(Tgrid[a][k])
        Xk, Nk, Gk, Fk, Ek = model.X[a, k], model.N[a, k], model.G[a, k], model.F[a, k], model.E[a, k]
        p = get_p_real(model)
        Nadd = 0.0

        f1 = rhs_zenteno_pyomo(Xk,            Nk,            Gk,            Fk,            Ek,            Tk, Nadd, p)
        f2 = rhs_zenteno_pyomo(Xk+0.5*h*f1["X"], Nk+0.5*h*f1["N"], Gk+0.5*h*f1["G"], Fk+0.5*h*f1["F"], Ek+0.5*h*f1["E"], Tk, Nadd, p)
        f3 = rhs_zenteno_pyomo(Xk+0.5*h*f2["X"], Nk+0.5*h*f2["N"], Gk+0.5*h*f2["G"], Fk+0.5*h*f2["F"], Ek+0.5*h*f2["E"], Tk, Nadd, p)
        f4 = rhs_zenteno_pyomo(Xk+    h*f3["X"], Nk+    h*f3["N"], Gk+    h*f3["G"], Fk+    h*f3["F"], Ek+    h*f3["E"], Tk, Nadd, p)

        X_next = Xk + (h/6.0)*(f1["X"] + 2*f2["X"] + 2*f3["X"] + f4["X"])
        return model.X[a, k+1] == X_next

    def rk4_rule_N(model, a, k):
        h   = float(dtgrid[a][k]); Tk = float(Tgrid[a][k])
        Xk, Nk, Gk, Fk, Ek = model.X[a, k], model.N[a, k], model.G[a, k], model.F[a, k], model.E[a, k]
        p = get_p_real(model); Nadd = 0.0
        f1 = rhs_zenteno_pyomo(Xk,            Nk,            Gk,            Fk,            Ek,            Tk, Nadd, p)
        f2 = rhs_zenteno_pyomo(Xk+0.5*h*f1["X"], Nk+0.5*h*f1["N"], Gk+0.5*h*f1["G"], Fk+0.5*h*f1["F"], Ek+0.5*h*f1["E"], Tk, Nadd, p)
        f3 = rhs_zenteno_pyomo(Xk+0.5*h*f2["X"], Nk+0.5*h*f2["N"], Gk+0.5*h*f2["G"], Fk+0.5*h*f2["F"], Ek+0.5*h*f2["E"], Tk, Nadd, p)
        f4 = rhs_zenteno_pyomo(Xk+    h*f3["X"], Nk+    h*f3["N"], Gk+    h*f3["G"], Fk+    h*f3["F"], Ek+    h*f3["E"], Tk, Nadd, p)
        N_next = Nk + (h/6.0)*(f1["N"] + 2*f2["N"] + 2*f3["N"] + f4["N"])
        return model.N[a, k+1] == N_next

    def rk4_rule_G(model, a, k):
        h   = float(dtgrid[a][k]); Tk = float(Tgrid[a][k])
        Xk, Nk, Gk, Fk, Ek = model.X[a, k], model.N[a, k], model.G[a, k], model.F[a, k], model.E[a, k]
        p = get_p_real(model); Nadd = 0.0
        f1 = rhs_zenteno_pyomo(Xk,            Nk,            Gk,            Fk,            Ek,            Tk, Nadd, p)
        f2 = rhs_zenteno_pyomo(Xk+0.5*h*f1["X"], Nk+0.5*h*f1["N"], Gk+0.5*h*f1["G"], Fk+0.5*h*f1["F"], Ek+0.5*h*f1["E"], Tk, Nadd, p)
        f3 = rhs_zenteno_pyomo(Xk+0.5*h*f2["X"], Nk+0.5*h*f2["N"], Gk+0.5*h*f2["G"], Fk+0.5*h*f2["F"], Ek+0.5*h*f2["E"], Tk, Nadd, p)
        f4 = rhs_zenteno_pyomo(Xk+    h*f3["X"], Nk+    h*f3["N"], Gk+    h*f3["G"], Fk+    h*f3["F"], Ek+    h*f3["E"], Tk, Nadd, p)
        G_next = Gk + (h/6.0)*(f1["G"] + 2*f2["G"] + 2*f3["G"] + f4["G"])
        return model.G[a, k+1] == G_next

    def rk4_rule_F(model, a, k):
        h   = float(dtgrid[a][k]); Tk = float(Tgrid[a][k])
        Xk, Nk, Gk, Fk, Ek = model.X[a, k], model.N[a, k], model.G[a, k], model.F[a, k], model.E[a, k]
        p = get_p_real(model); Nadd = 0.0
        f1 = rhs_zenteno_pyomo(Xk,            Nk,            Gk,            Fk,            Ek,            Tk, Nadd, p)
        f2 = rhs_zenteno_pyomo(Xk+0.5*h*f1["X"], Nk+0.5*h*f1["N"], Gk+0.5*h*f1["G"], Fk+0.5*h*f1["F"], Ek+0.5*h*f1["E"], Tk, Nadd, p)
        f3 = rhs_zenteno_pyomo(Xk+0.5*h*f2["X"], Nk+0.5*h*f2["N"], Gk+0.5*h*f2["G"], Fk+0.5*h*f2["F"], Ek+0.5*h*f2["E"], Tk, Nadd, p)
        f4 = rhs_zenteno_pyomo(Xk+    h*f3["X"], Nk+    h*f3["N"], Gk+    h*f3["G"], Fk+    h*f3["F"], Ek+    h*f3["E"], Tk, Nadd, p)
        F_next = Fk + (h/6.0)*(f1["F"] + 2*f2["F"] + 2*f3["F"] + f4["F"])
        return model.F[a, k+1] == F_next

    def rk4_rule_E(model, a, k):
        h   = float(dtgrid[a][k]); Tk = float(Tgrid[a][k])
        Xk, Nk, Gk, Fk, Ek = model.X[a, k], model.N[a, k], model.G[a, k], model.F[a, k], model.E[a, k]
        p = get_p_real(model); Nadd = 0.0
        f1 = rhs_zenteno_pyomo(Xk,            Nk,            Gk,            Fk,            Ek,            Tk, Nadd, p)
        f2 = rhs_zenteno_pyomo(Xk+0.5*h*f1["X"], Nk+0.5*h*f1["N"], Gk+0.5*h*f1["G"], Fk+0.5*h*f1["F"], Ek+0.5*h*f1["E"], Tk, Nadd, p)
        f3 = rhs_zenteno_pyomo(Xk+0.5*h*f2["X"], Nk+0.5*h*f2["N"], Gk+0.5*h*f2["G"], Fk+0.5*h*f2["F"], Ek+0.5*h*f2["E"], Tk, Nadd, p)
        f4 = rhs_zenteno_pyomo(Xk+    h*f3["X"], Nk+    h*f3["N"], Gk+    h*f3["G"], Fk+    h*f3["F"], Ek+    h*f3["E"], Tk, Nadd, p)
        E_next = Ek + (h/6.0)*(f1["E"] + 2*f2["E"] + 2*f3["E"] + f4["E"])
        return model.E[a, k+1] == E_next

    m.rk4_X = Constraint(m.AK_TRANS, rule=rk4_rule_X)
    m.rk4_N = Constraint(m.AK_TRANS, rule=rk4_rule_N)
    m.rk4_G = Constraint(m.AK_TRANS, rule=rk4_rule_G)
    m.rk4_F = Constraint(m.AK_TRANS, rule=rk4_rule_F)
    m.rk4_E = Constraint(m.AK_TRANS, rule=rk4_rule_E)

    # --- Condiciones iniciales (k=0) si hay medición ---
    def ic_rule_X(model, a):
        y0 = obs[a]["X"][0] if len(obs[a]["X"])>0 else np.nan
        return Constraint.Skip if np.isnan(y0) else (model.X[a,0] == float(y0))
    def ic_rule_N(model, a):
        y0 = obs[a]["N"][0] if len(obs[a]["N"])>0 else np.nan
        return Constraint.Skip if np.isnan(y0) else (model.N[a,0] == float(y0))
    def ic_rule_G(model, a):
        y0 = obs[a]["G"][0] if len(obs[a]["G"])>0 else np.nan
        return Constraint.Skip if np.isnan(y0) else (model.G[a,0] == float(y0))
    def ic_rule_F(model, a):
        y0 = obs[a]["F"][0] if len(obs[a]["F"])>0 else np.nan
        return Constraint.Skip if np.isnan(y0) else (model.F[a,0] == float(y0))
    def ic_rule_E(model, a):
        y0 = obs[a]["E"][0] if len(obs[a]["E"])>0 else np.nan
        return Constraint.Skip if np.isnan(y0) else (model.E[a,0] == float(y0))

    m.ic_X = Constraint(m.ASSAY, rule=ic_rule_X)
    m.ic_N = Constraint(m.ASSAY, rule=ic_rule_N)
    m.ic_G = Constraint(m.ASSAY, rule=ic_rule_G)
    m.ic_F = Constraint(m.ASSAY, rule=ic_rule_F)
    m.ic_E = Constraint(m.ASSAY, rule=ic_rule_E)

    # --- Objetivo SSE ponderado en todos los nodos (a,k) ---
    def sse_expr(model):
        sse = 0.0
        for (a,k) in model.AK:
            if not np.isnan(obs[a]["X"][k]):
                sse += weights["X"] * (model.X[a,k] - float(obs[a]["X"][k]))**2
            if not np.isnan(obs[a]["N"][k]):
                sse += weights["N"] * (model.N[a,k] - float(obs[a]["N"][k]))**2
            if not np.isnan(obs[a]["G"][k]):
                sse += weights["G"] * (model.G[a,k] - float(obs[a]["G"][k]))**2
            if not np.isnan(obs[a]["F"][k]):
                sse += weights["F"] * (model.F[a,k] - float(obs[a]["F"][k]))**2
            if not np.isnan(obs[a]["E"][k]):
                sse += weights["E"] * (model.E[a,k] - float(obs[a]["E"][k]))**2

        # Regularizador muy pequeño para evitar objetivo constante
        reg = 1e-12 * sum(model.X[a,k]**2 + model.N[a,k]**2 + model.G[a,k]**2
                          + model.F[a,k]**2 + model.E[a,k]**2 for (a,k) in model.AK)
        return sse + reg

    m.OBJ = Objective(rule=sse_expr, sense=minimize)

    # Guarda auxiliares por si quieres inspeccionar luego
    m._scale  = scale
    m._obs    = obs
    m._tgrid  = tgrid
    m._Tgrid  = Tgrid
    m._dtgrid = dtgrid

    return m

# ---------- Solución local con Ipopt ----------
def solve_ipopt(model: ConcreteModel, tee: bool = True):
    solver = SolverFactory("ipopt")
    for k, v in IPOPT_OPTIONS.items():
        solver.options[k] = v
    res = solver.solve(model, tee=tee)
    tc = res.solver.termination_condition
    if tc not in (TerminationCondition.optimal, TerminationCondition.locallyOptimal):
        print(f"[AVISO] Ipopt terminó con estado: {tc}")
    # recuperar parámetros reales
    p_est = np.array([value(model.ps[j])*value(model.scale[j]) for j in model.J], dtype=float)
    sse = value(model.OBJ)
    return p_est, sse, res

# ---------- Multistart con Ipopt ----------
def multistart_ipopt(mats, p0_real, n_starts=10, rng_seed=123, spread=0.5):
    rng = np.random.default_rng(rng_seed)
    best = (None, np.inf, None)
    for i in range(n_starts):
        m = build_pyomo_model(mats, p0_real)
        # perturbar ps (escala) alrededor de p0
        for j in m.J:
            # valor base = p0_real/scale; ruido uniforme ±spread
            base = float(p0_real[j]/value(m.scale[j]))
            m.ps[j].value = base * (1.0 + rng.uniform(-spread, spread))
        p_est, sse, res = solve_ipopt(m, tee=False)
        if sse < best[1]:
            best = (p_est, sse, res)
        print(f"[multistart {i+1}/{n_starts}] SSE={sse:.4e}")
    return best

# ---------- DE global (externo) con integración directa (rápida) ----------
def simulate_direct(t_h: np.ndarray, T_K: np.ndarray, x0: np.ndarray, p: np.ndarray):
    """Integra con RK4 'directo' fuera de Pyomo. Devuelve matriz (n,5) en tiempos dados."""
    # Igual que en Pyomo: Nadd=0, T por tramo constante entre puntos
    def rhs_num(x, T):
        X, N, G, F, E = x
        mu0, betaG0, betaF0, Kn0, Kg0, Kf0, Kig0, Kie0, Kd0, Yxn, Yxg, Yxf, Yeg, Yef = p
        # Constantes
        Cde, Etd, R = 0.0415, 130000.0, 8.314
        Eac, Eafe = 59453.0, 11000.0
        EaKn = EaKg = EaKf = EaKig = EaKie = 46055.0
        Eam, m0 = 37681.0, 0.01
        # constitutivas
        mu_max    = mu0 * math.exp(Eac*(T-300.0)/(300.0*R*T))
        betaG_max = betaG0 * math.exp(Eafe*(T-296.15)/(296.15*R*T))
        betaF_max = betaF0 * math.exp(Eafe*(T-296.15)/(296.15*R*T))
        Kn        = Kn0  * math.exp(EaKn*(T-293.15)/(293.15*R*T))
        Kg        = Kg0  * math.exp(EaKg*(T-293.15)/(293.15*R*T))
        Kf        = Kf0  * math.exp(EaKf*(T-293.15)/(293.15*R*T))
        Kig       = Kig0 * math.exp(EaKig*(T-293.15)/(293.15*R*T))
        Kie       = Kie0 * math.exp(EaKie*(T-293.15)/(293.15*R*T))
        m         = m0   * math.exp(Eam *(T-293.3) /(293.3 *R*T))
        mu     = mu_max * (N/(N+Kn))
        beta_G = betaG_max * (G/(G+Kg)) * (Kie/(E+Kie))
        beta_F = betaF_max * (F/(F+Kf)) * (Kig/(G+Kig)) * (Kie/(E+Kie))
        Td = (-0.0001*(E**3)) + (0.0049*(E**2)) - (0.1279*E) + 315.89
        if T >= Td:
            Kd = Kd0 * math.exp((Cde*E) + (Etd*(T-305.65))/(305.65*R*T))
        else:
            Kd = 0.0
        eps = 1e-9
        dX = mu*X - Kd*X
        dN = -mu*(X/Yxn) + 0.0
        dG = -((mu/Yxg) + (beta_G/Yeg) + m*(G/(G+F+eps)))*X
        dF = -((mu/Yxf) + (beta_F/Yef) + m*(F/(G+F+eps)))*X
        dE = (beta_G + beta_F)*X
        return np.array([dX, dN, dG, dF, dE], dtype=float)

    x = np.array(x0, dtype=float)
    out = [x.copy()]
    for k in range(len(t_h)-1):
        h = float(t_h[k+1] - t_h[k])
        T = float(T_K[k])
        k1 = rhs_num(x, T)
        k2 = rhs_num(x + 0.5*h*k1, T)
        k3 = rhs_num(x + 0.5*h*k2, T)
        k4 = rhs_num(x + h*k3, T)
        x  = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        x  = np.maximum(x, 0.0)
        out.append(x.copy())
    return np.vstack(out)

def sse_for_mats_direct(p_real: np.ndarray, mats: Dict[str, pd.DataFrame], weights: Dict[str, float]):
    total = 0.0
    for code, df in mats.items():
        df = df.sort_values("time_h")
        t = pd.to_numeric(df["time_h"], errors="coerce").to_numpy(dtype=float)
        t = t - t[0]
        T = build_temperature_series(df)
        # x0: si hay mediciones iniciales, usa esas
        X0 = float(pd.to_numeric(df.get("biomass_viable_gL", pd.Series([np.nan]))).iloc[0])
        N0 = float(pd.to_numeric(df.get("YAN", pd.Series([np.nan]))).iloc[0])
        G0 = float(pd.to_numeric(df.get("Glucose", pd.Series([np.nan]))).iloc[0])
        F0 = float(pd.to_numeric(df.get("Fructose", pd.Series([np.nan]))).iloc[0])
        E0 = float(pd.to_numeric(df.get("Ethanol", pd.Series([np.nan]))).iloc[0])
        x0 = np.array([X0 if not np.isnan(X0) else 0.5,
                       N0 if not np.isnan(N0) else 0.1,
                       G0 if not np.isnan(G0) else 100.0,
                       F0 if not np.isnan(F0) else 100.0,
                       E0 if not np.isnan(E0) else 0.0], dtype=float)
        Xsim = simulate_direct(t, T, x0, p_real)
        # SSE
        def add_sse(col, idx):
            y = pd.to_numeric(df.get(col, np.nan), errors="coerce").to_numpy(dtype=float)
            msk = ~np.isnan(y)
            return np.nansum((Xsim[msk, idx] - y[msk])**2)
        total += weights["X"] * add_sse("biomass_viable_gL", 0)
        total += weights["N"] * add_sse("YAN", 1)
        total += weights["G"] * add_sse("Glucose", 2)
        total += weights["F"] * add_sse("Fructose", 3)
        total += weights["E"] * add_sse("Ethanol", 4)
    return float(total)

def de_global_opt(mats, p0_real, bounds_real, weights, maxiter=80, polish_with_ipopt=True):
    # DE en espacio real (no escalado). Si quieres en escalado, divide/multiplica como arriba.
    def obj(pvec):
        return sse_for_mats_direct(np.asarray(pvec, dtype=float), mats, weights)
    result = differential_evolution(
        obj, bounds=bounds_real, maxiter=maxiter, popsize=15,
        mutation=(0.5, 1.0), recombination=0.7, tol=1e-6, disp=True
    )
    p_best = result.x.copy(); sse = result.fun
    if polish_with_ipopt:
        # pulido con Pyomo/Ipopt desde p_best
        m = build_pyomo_model(mats, p_best, bounds_real, weights)
        # inicializar ps con p_best escalado
        for j in m.J:
            m.ps[j].value = float(p_best[j]/value(m.scale[j]))
        p_best2, sse2, _ = solve_ipopt(m, tee=False)
        if sse2 < sse:
            p_best, sse = p_best2, sse2
    return p_best, sse, result

# ===================== EJEMPLO DE USO =====================
if __name__ == "__main__":
    # 1) Cargar matrices para calibración:
    #    - Opción A: desde carpeta (CSV exportados como calib_matrix_*.csv)
    mats = None
    # if RUTA_MATRICES:
    #     mats = load_mats_from_folder(RUTA_MATRICES)

    #    - Opción B: si ya las tienes en memoria como dict { ensayo: df }, pásalas aquí:
    from Calibration_data_preprocess import process_all, build_calibration_matrices
    FILE_PATH = r"C:/Users/ctorrealba/OneDrive - Viña Concha y Toro S.A/Documentos/Proyectos I+D/PI-4497/resultados/2025/calibración data 2025/Procesos_I+D_2025_3.xlsx"   # <-- EDITA (ruta a tu Excel)
    results_dict, _ = process_all(FILE_PATH, assays=None)
    mats = build_calibration_matrices(results_dict, use_smoothed_biomass=False)

    raise_if_empty = False
    MATRICES: Dict[str, pd.DataFrame] = {}  # <-- Asigna tus matrices aquí
    if not MATRICES and raise_if_empty:
        raise RuntimeError("Asigna tus matrices de calibración a la variable MATRICES.")

    # 2) Parámetros iniciales desde Excel (ajusta ruta/hoja/set)
    P0 = load_parameters_from_excel("zenteno_parameters.xlsx", sheet_name="Hoja1", param_set=3)

    # 3) Modo de optimización: 'ipopt_single' | 'ipopt_multistart' | 'de_global'
    MODE = "ipopt_single"   # cambia aquí según el experimento

    if MODE == "ipopt_single":
        # model = build_pyomo_model(MATRICES, P0, P_BOUNDS_REAL, WEIGHTS)
        MATS_CLEAN = sanitize_mats_for_pyomo(MATRICES)
        if not MATS_CLEAN:
            raise RuntimeError("No hay ensayos válidos: revisa que MATRICES no esté vacío y que cada ensayo tenga ≥2 tiempos y al menos una variable observada.")

        model = build_pyomo_model(MATS_CLEAN, P0, P_BOUNDS_REAL, WEIGHTS)

        p_est, sse, _res = solve_ipopt(model, tee=True)
        print("\n[Ipopt] SSE:", sse)
        print("[Ipopt] Parámetros:", p_est)

    elif MODE == "ipopt_multistart":
        p_best, sse_best, _res = multistart_ipopt(MATRICES, P0, n_starts=10, rng_seed=123, spread=0.5)
        print("\n[Multistart] SSE:", sse_best)
        print("[Multistart] Parámetros:", p_best)

    elif MODE == "de_global":
        p_best, sse, _ = de_global_opt(MATRICES, P0, P_BOUNDS_REAL, WEIGHTS, maxiter=80, polish_with_ipopt=True)
        print("\n[DE global] SSE:", sse)
        print("[DE global] Parámetros:", p_best)

    else:
        raise ValueError("MODE no reconocido.")
