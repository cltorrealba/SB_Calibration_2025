# -*- coding: utf-8 -*-
"""
Extracción de modelo dinámico y simulación (Zenteno et al.)
- Sin ComplexWarning (estados pasan por _real_pos)
- Jacobiano analítico consistente con zenteno_model
- Una sola build_profiles para RK4 (build_profiles_rk4)
- Integrador stiff con jacobiano analítico + jac_sparsity
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------- Utilidades numéricas seguras ----------------
EPS = 1e-9
BIG = 1e6  # techo para estados/tasas/derivadas

def safe_div(a, b, eps=EPS):
    return a / (b + eps)

def safe_exp(x, lo=-50.0, hi=50.0):
    """exp con saturación del exponente para evitar overflow/underflow extremo."""
    return np.exp(np.clip(x, lo, hi))

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def _real_pos(z):
    """Parte real y clamp a ≥ 0 (evita ComplexWarning y negativos numéricos)."""
    r = float(np.real(z))
    return r if r > 0.0 else 0.0


# ---------------- Carga de parámetros ----------------
def load_parameters_from_excel(xlsx_path="zenteno_parameters.xlsx", sheet_name="Hoja1", param_set=3):
    """
    Carga el vector de parámetros desde un Excel.
    Espera una columna 'set' para filtrar y el resto de columnas en el orden del modelo.
    Retorna: numpy.ndarray con 14 valores (o los que existan en la fila).
    """
    xp = Path(xlsx_path)
    if not xp.exists():
        try:
            base = Path(__file__).resolve().parent
        except Exception:
            base = Path.cwd()
        xp = (base / xlsx_path)
    if not xp.exists():
        raise FileNotFoundError(f"No se encontró el archivo de parámetros: {xlsx_path} (ruta resuelta: {xp})")

    df = pd.read_excel(xp, sheet_name=sheet_name)

    # normalizar nombre de columna 'set'
    set_col = None
    for c in df.columns:
        if str(c).strip().lower() == "set":
            set_col = c
            break
    if set_col is None:
        raise KeyError("La hoja no contiene una columna 'set' para seleccionar el conjunto de parámetros.")

    row = df.loc[df[set_col] == param_set]
    if row.empty:
        disponibles = sorted(df[set_col].unique().tolist())
        raise ValueError(f"No existe set={param_set}. Disponibles: {disponibles}")

    row = row.drop(columns=[set_col]).iloc[0]
    p = row.to_numpy(dtype=float)
    if p.size != 14:
        print(f"[AVISO] Se esperaban 14 parámetros, se encontraron {p.size}. Continuando con el tamaño encontrado.")
    return p


def list_available_sets(xlsx_path="zenteno_parameters.xlsx", sheet_name="Hoja1"):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    set_col = None
    for c in df.columns:
        if str(c).strip().lower() == "set":
            set_col = c
            break
    if set_col is None:
        return []
    return sorted(df[set_col].unique().tolist())


# ---------------- Modelo dinámico robusto ----------------
def zenteno_model(t, x, u, p):
    """
    Modelo de cinética de fermentación (versión robusta).
    Estados: x = [X, N, G, F, E] (g/L)
    Entradas: u = [T (K), Nadd (g/L/h)]  (Nadd se usa fuera como pulso; aquí no se consume directamente)
    Parámetros: p[0..13] positivos
    """
    # Entradas
    T = float(u[0])     # Kelvin
    Nadd = float(u[1])  # g/L/h (no entra directo a f; se suma fuera como pulso)

    # Estados (no-negatividad y parte real)
    X = _real_pos(x[0])
    N = _real_pos(x[1])
    G = _real_pos(x[2])
    F = _real_pos(x[3])
    E = _real_pos(x[4])

    # Limitar T a un rango físico razonable (0–60 °C)
    T = clamp(T, 273.15, 333.15)

    # Parámetros (positivos)
    vals = [max(float(pi), EPS) for pi in p]
    (mu0, betaG0, betaF0, Kn0, Kg0, Kf0, Kig0, Kie0, Kd0,
     Yxn, Yxg, Yxf, Yeg, Yef) = vals

    # Constantes
    Cde     = 0.0415      # m3/kg E (Salmon 2003)
    Etd     = 130000.0    # kJ/kmol
    R       = 8.314       # kJ/kmol/K (Boulton 1979)
    Eac     = 59453.0     # kJ/kmol (Boulton 1979)
    Eafe    = 11000.0     # kJ/kmol (Zenteno 2010)
    EaKn    = 46055.0     # kJ/kmol (Boulton 1979)
    EaKg    = 46055.0     # kJ/kmol (Boulton 1979)
    EaKf    = 46055.0     # kJ/kmol (Boulton 1979)
    EaKig   = 46055.0     # kJ/kmol (Boulton 1979)
    EaKie   = 46055.0     # kJ/kmol (Boulton 1979)
    Eam     = 37681.0     # kJ/kmol (Boulton 1979)
    m0      = 0.01        # kgS/kg bio/h

    # Arrhenius cappeado (evita desbordes)
    mu_max    = mu0   * safe_exp(Eac *(T-300.00)/(300.00*R*T))
    betaG_max = betaG0* safe_exp(Eafe*(T-296.15)/(296.15*R*T))
    betaF_max = betaF0* safe_exp(Eafe*(T-296.15)/(296.15*R*T))
    Kn        = Kn0   * safe_exp(EaKn*(T-293.15)/(293.15*R*T))
    Kg        = Kg0   * safe_exp(EaKg*(T-293.15)/(293.15*R*T))
    Kf        = Kf0   * safe_exp(EaKf*(T-293.15)/(293.15*R*T))
    Kig       = Kig0  * safe_exp(EaKig*(T-293.15)/(293.15*R*T))
    Kie       = Kie0  * safe_exp(EaKie*(T-293.15)/(293.15*R*T))
    m         = m0    * safe_exp(Eam *(T-293.30)/(293.30*R*T))

    # Tasas con divisiones seguras
    mu      = mu_max * safe_div(N, N + Kn)
    beta_G  = betaG_max* safe_div(G, G + Kg) * safe_div(Kie, E + Kie)
    beta_F  = betaF_max* safe_div(F, F + Kf) * safe_div(Kig, G + Kig) * safe_div(Kie, E + Kie)

    # Temperatura de muerte térmica con E acotado (evita E**3 enorme)
    E_cap = clamp(E, 0.0, 200.0)
    Td = -0.0001*(E_cap**3) + 0.0049*(E_cap**2) - 0.1279*E_cap + 315.89
    Td = clamp(Td, 273.15, 333.15)

    # Tasa de muerte específica con exponencial segura
    if T >= Td:
        exponent = (Cde*E_cap) + safe_div(Etd*(T-305.65), (305.65*R*T))
        Kd = Kd0 * safe_exp(exponent, lo=-50.0, hi=50.0)
    else:
        Kd = 0.0

    # Mezcla para mantenimiento (evita 0/0)
    GpF = G + F + EPS
    mG = G / GpF
    mF = F / GpF

    # EDOs con clamps de tasa
    dX = (mu - Kd) * X
    dN = -(mu / max(Yxn, EPS)) * X + 0.0  # Nadd se aplica fuera como pulso instantáneo
    dG = -((mu / max(Yxg, EPS)) + (beta_G / max(Yeg, EPS)) + m*mG) * X
    dF = -((mu / max(Yxf, EPS)) + (beta_F / max(Yef, EPS)) + m*mF) * X
    dE = (beta_G + beta_F) * X

    dX = float(clamp(dX, -BIG, BIG))
    dN = float(clamp(dN, -BIG, BIG))
    dG = float(clamp(dG, -BIG, BIG))
    dF = float(clamp(dF, -BIG, BIG))
    dE = float(clamp(dE, -BIG, BIG))
    return np.array([dX, dN, dG, dF, dE], dtype=float)


# ---------------- Jacobiano analítico (∂f/∂x) ----------------
def zenteno_jacobian(t, x, u, p):
    """
    Jacobiano analítico de zenteno_model respecto de x = [X, N, G, F, E].
    - u: [T_K, N_add]; T_K en Kelvin (N_add no afecta ∂f/∂x).
    - p: parámetros (mismo orden que zenteno_model).
    """
    # Estados (real+≥0)
    X = _real_pos(x[0]); N = _real_pos(x[1]); G = _real_pos(x[2]); F = _real_pos(x[3]); E = _real_pos(x[4])

    # Entrada
    T = float(u[0])
    T = clamp(T, 273.15, 333.15)

    # Parámetros
    (mu0, betaG0, betaF0, Kn0, Kg0, Kf0, Kig0, Kie0, Kd0,
     Yxn, Yxg, Yxf, Yeg, Yef) = [max(float(pi), EPS) for pi in p]

    # Constantes
    Cde     = 0.0415
    Etd     = 130000.0
    R       = 8.314
    Eac     = 59453.0
    Eafe    = 11000.0
    EaKn    = 46055.0
    EaKg    = 46055.0
    EaKf    = 46055.0
    EaKig   = 46055.0
    EaKie   = 46055.0
    Eam     = 37681.0
    m0      = 0.01

    # Arrhenius (dependen de T, no de estados)
    mu_max    = mu0   * safe_exp(Eac *(T-300.00)/(300.00*R*T))
    betaG_max = betaG0* safe_exp(Eafe*(T-296.15)/(296.15*R*T))
    betaF_max = betaF0* safe_exp(Eafe*(T-296.15)/(296.15*R*T))
    Kn        = Kn0   * safe_exp(EaKn*(T-293.15)/(293.15*R*T))
    Kg        = Kg0   * safe_exp(EaKg*(T-293.15)/(293.15*R*T))
    Kf        = Kf0   * safe_exp(EaKf*(T-293.15)/(293.15*R*T))
    Kig       = Kig0  * safe_exp(EaKig*(T-293.15)/(293.15*R*T))
    Kie       = Kie0  * safe_exp(EaKie*(T-293.15)/(293.15*R*T))
    m         = m0    * safe_exp(Eam *(T-293.30)/(293.30*R*T))

    # Tasas y derivadas parciales wrt estados
    mu     = mu_max * safe_div(N, N + Kn)
    dmu_dN = mu_max * (Kn / (N + Kn)**2)

    beta_G = betaG_max * safe_div(G, G + Kg) * safe_div(Kie, E + Kie)
    dbG_dG = betaG_max * (Kg / (G + Kg)**2) * safe_div(Kie, E + Kie)
    dbG_dE = betaG_max * safe_div(G, G + Kg) * (-Kie / (E + Kie)**2)

    beta_F = betaF_max * safe_div(F, F + Kf) * safe_div(Kig, G + Kig) * safe_div(Kie, E + Kie)
    dbF_dF = betaF_max * (Kf / (F + Kf)**2) * safe_div(Kig, G + Kig) * safe_div(Kie, E + Kie)
    dbF_dG = betaF_max * safe_div(F, F + Kf) * (-Kig / (G + Kig)**2) * safe_div(Kie, E + Kie)
    dbF_dE = betaF_max * safe_div(F, F + Kf) * safe_div(Kig, G + Kig) * (-Kie / (E + Kie)**2)

    # Td(E) y Kd(E, T)
    E_cap = clamp(E, 0.0, 200.0)
    Td = -0.0001*(E_cap**3) + 0.0049*(E_cap**2) - 0.1279*E_cap + 315.89
    Td = clamp(Td, 273.15, 333.15)

    if T >= Td:
        Kd     = Kd0 * safe_exp( Cde*E_cap + Etd*(T-305.65)/(305.65*R*T) )
        dKd_dE = Cde * Kd
    else:
        Kd     = 0.0
        dKd_dE = 0.0

    # Partición mantenimiento
    sumGF  = G + F + EPS
    phi_G  = G / sumGF
    phi_F  = F / sumGF
    dphiG_dG =  F / (sumGF**2)
    dphiG_dF = -G / (sumGF**2)
    dphiF_dF =  G / (sumGF**2)
    dphiF_dG = -F / (sumGF**2)

    # Jacobiano
    J = np.zeros((5,5), dtype=float)

    # f0 = (mu - Kd)*X
    J[0,0] = (mu - Kd)
    J[0,1] = dmu_dN * X
    J[0,2] = 0.0
    J[0,3] = 0.0
    J[0,4] = -(dKd_dE) * X

    # f1 = -(mu/Yxn)*X
    J[1,0] = -(mu / Yxn)
    J[1,1] = -(dmu_dN / Yxn) * X
    # (resto 0)

    # f2 = -[(mu/Yxg) + (beta_G/Yeg) + m*phi_G]*X
    term_G = (mu / Yxg) + (beta_G / Yeg) + m*phi_G
    J[2,0] = -term_G
    J[2,1] = -((dmu_dN / Yxg) * X)
    J[2,2] = -(((dbG_dG / Yeg) + m * dphiG_dG) * X)
    J[2,3] = -((m * dphiG_dF) * X)
    J[2,4] = -(((dbG_dE / Yeg) * X))

    # f3 = -[(mu/Yxf) + (beta_F/Yef) + m*phi_F]*X
    term_F = (mu / Yxf) + (beta_F / Yef) + m*phi_F
    J[3,0] = -term_F
    J[3,1] = -((dmu_dN / Yxf) * X)
    J[3,2] = -((dbF_dG * X))
    J[3,3] = -(((dbF_dF / Yef) + m * dphiF_dF) * X)
    J[3,4] = -(((dbF_dE / Yef) * X))

    # f4 = (beta_G + beta_F)*X
    J[4,0] = (beta_G + beta_F)
    J[4,1] = 0.0
    J[4,2] = (dbG_dG + dbF_dG) * X
    J[4,3] = (dbF_dF) * X
    J[4,4] = (dbG_dE + dbF_dE) * X

    return J


# ---------------- Integrador stiff (Radau/BDF) ----------------
# Sparsidad (conservadora) del Jacobiano wrt estados (True = posible no-cero)
J_SPARSE = np.array([
    [1,1,0,0,1],
    [1,1,0,0,0],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,0,1,1,1],
], dtype=bool)

def simulate_process_time_stiff(tf, x0, n_steps, u_fun, p, method="Radau",
                                atol=1e-8, rtol=1e-6):
    """
    Integración con solve_ivp y Jacobiano analítico.
    - u_fun(t) debe devolver [T_K, N_add] en Kelvin para el tiempo t (h).
      (N_add no se usa dentro de f; los pulsos se aplican fuera entre tramos)
    - p: vector de parámetros reales
    """
    t0 = 0.0
    if n_steps is None or n_steps <= 1:
        n_steps = max(int(np.ceil(tf)), 2)
    t_eval = np.linspace(t0, tf, n_steps)

    def f_ode(t, y):
        u = u_fun(t)  # [T_K, N_add]
        return zenteno_model(t, y, u, p)

    def j_ode(t, y):
        u = u_fun(t)
        return zenteno_jacobian(t, y, u, p)

    sol = solve_ivp(
        fun=f_ode,
        t_span=(t0, tf),
        y0=np.asarray(x0, dtype=float),
        method=method,              # "Radau" o "BDF"
        jac=j_ode,
        jac_sparsity=J_SPARSE,      # ayuda de rendimiento
        t_eval=t_eval,
        atol=atol,
        rtol=rtol,
        vectorized=False
    )
    if not sol.success:
        # fallback corto: reintenta con tolerancias más laxas
        sol = solve_ivp(
            fun=f_ode,
            t_span=(t0, tf),
            y0=np.asarray(x0, dtype=float),
            method=method,
            jac=j_ode,
            jac_sparsity=J_SPARSE,
            t_eval=t_eval,
            atol=max(atol*10, 1e-6),
            rtol=max(rtol*10, 1e-4),
            vectorized=False
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp falló: {sol.message}")

    return sol.t, sol.y.T


# ---------------- Integrador RK4 (para ejemplos/pruebas) ----------------
def RK4_method(f, tf, x0, n, u, p):
    """
    RK4 con:
      - no-negatividad tras cada paso
      - techo BIG de seguridad
      - n automático si viene None (dt ≲ 1 h)
    """
    tf = float(tf)
    if n is None or n <= 0:
        n = max(int(np.ceil(tf / 1.0)), 10)  # ~1 h por paso
    h = tf / n

    t = np.zeros(n+1, dtype=float)
    X = np.zeros((n+1, len(x0)), dtype=float)
    x = np.array(x0, dtype=float).copy()
    X[0, :] = np.maximum(np.minimum(x, BIG), 0.0)
    t[0] = 0.0

    for k in range(n):
        uk = u[k] if k < len(u) else u[-1]
        k1 = f(t[k],          x,              uk, p)
        k2 = f(t[k] + h/2.0,  x + h*k1/2.0,   uk, p)
        k3 = f(t[k] + h/2.0,  x + h*k2/2.0,   uk, p)
        k4 = f(t[k] + h,      x + h*k3,       uk, p)
        x  = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        # clamps a estados
        x = np.maximum(x, 0.0)
        x = np.minimum(x, BIG)

        X[k+1, :] = x
        t[k+1] = t[k] + h

    return t, X


# ---------------- Perfiles de T y adición de N (para RK4) ----------------
def build_profiles(tf, n, temps_c=None, injections=None, dt=None, temp_segments=None):
    """
    Construye perfiles discretos de temperatura (K) y adición de N (g/L/h) para integración RK4.
    - Soporta temp_segments como (t_start, t_end, T_C) o (t_change, T_C)
    - Clampa T a [273.15, 333.15] K
    """
    if n is None or n <= 0:
        n = max(int(np.ceil(tf / 1.0)), 10)
    h = (tf / n) if dt is None else dt
    T_profile = np.empty(n+1, dtype=float)

    if temp_segments is not None and len(temp_segments) > 0:
        if len(temp_segments[0]) == 3:
            # Formato (t_start, t_end, T_C)
            T_profile[:] = (temp_segments[0][2] + 273.15)
            for seg in temp_segments:
                t_start, t_end, Tc = seg
                a = max(0, min(n, int(round(t_start / h))))
                b = max(0, min(n, int(round(t_end   / h))))
                if b < a:
                    a, b = b, a
                T_profile[a:b] = Tc + 273.15
            if len(temp_segments) > 0:
                T_profile[-1] = (temp_segments[-1][2] + 273.15)
        else:
            # Formato (t_change, T_C)
            segs = sorted(temp_segments, key=lambda s: s[0])
            first_t, first_Tc = segs[0]
            T_profile[:] = first_Tc + 273.15
            for i, (tc, Tc) in enumerate(segs):
                a = max(0, min(n, int(round(tc / h))))
                b = n+1 if i == len(segs)-1 else max(0, min(n+1, int(round(segs[i+1][0] / h))))
                T_profile[a:b] = Tc + 273.15
    else:
        # 3 tramos iguales con temps_c
        if temps_c is None or len(temps_c) != 3:
            raise ValueError("Si no usas temp_segments, debes entregar temps_c con 3 valores (°C).")
        seg = n // 3
        idxs = [0, seg, 2*seg, n]
        temps_k = [Tc + 273.15 for Tc in temps_c]
        for s in range(3):
            a, b = idxs[s], idxs[s+1]
            T_profile[a:b+1] = temps_k[s]

    # Clamp físico
    T_profile = np.clip(T_profile, 273.15, 333.15)

    # Perfil de N: impulsos como tasa en el paso (para RK4)
    Nadd_profile = np.zeros(n+1, dtype=float)
    for t_pulse, amount in (injections or []):
        k = int(round(t_pulse / h))
        k = max(0, min(n, k))
        Nadd_profile[k] += amount / h  # g/L/h en ese paso

    return T_profile, Nadd_profile


def simulate_process_time(p, x0, temps_c, injections, tf=14*24.0, n=None, threshold=5.0, temp_segments=None):
    """
    Simula la fermentación con RK4 y retorna tiempo (h) cuando G+F <= threshold.
    (Solo para pruebas/ejemplos; el flujo de calibración usa la versión stiff.)
    """
    if n is None:
        n = max(int(tf), 10)
    T_profile, Nadd_profile = build_profiles(tf, n, temps_c=temps_c, injections=injections, temp_segments=temp_segments)
    u = np.vstack([T_profile, Nadd_profile]).T
    t, x = RK4_method(zenteno_model, tf, x0, n, u, p)
    G = x[:, 2]
    F = x[:, 3]
    total_sugar = G + F

    # primer índice que cumple condición
    idx = np.argmax(total_sugar <= threshold)
    if total_sugar[0] <= threshold:
        t_proc = 0.0
    elif total_sugar[idx] <= threshold:
        t_proc = t[idx]
    else:
        t_proc = tf
    return t_proc, t, x, T_profile, Nadd_profile


# Valores iniciales por defecto
DEFAULT_X0 = np.array([0.5, 0.140, 110.0, 110.0, 0.0])


# # ---------------- Ejemplo mínimo ----------------
# if __name__ == "__main__":
#     p = load_parameters_from_excel("zenteno_parameters.xlsx", "Hoja1", param_set=3)

#     # Segmentos de temperatura (ejemplo para RK4)
#     temp_segments = [(0.0, 25.0), (48.0, 20.0), (224.0, 17.0)]  # (t_change_h, T_C)
#     pulsos = [(0.0, 0.05), (48.0, 0.10)]

#     tf = 14*24.0
#     n = None  # paso ~ 1 h

#     # Construcción de perfiles y simulación (RK4 demo)
#     T_profile, Nadd_profile = build_profiles(tf, n if n else int(tf),
#                                                  temps_c=None, injections=pulsos, temp_segments=temp_segments)
#     u = np.vstack([T_profile, Nadd_profile]).T
#     t, x = RK4_method(zenteno_model, tf, DEFAULT_X0, n, u, p)

#     # Gráficos rápidos (RK4)
#     td = np.asarray(t, dtype=float) / 24.0
#     plt.figure()
#     plt.plot(td, x[:, 0], label="X (g/L)")
#     plt.plot(td, x[:, 1], label="N (g/L)")
#     plt.plot(td, x[:, 2], label="G (g/L)")
#     plt.plot(td, x[:, 3], label="F (g/L)")
#     plt.plot(td, x[:, 4], label="E (g/L)")
#     plt.xlabel("Tiempo (días)")
#     plt.ylabel("Concentración (g/L)")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     plt.figure()
#     plt.step(td, T_profile - 273.15, where="post", label="Temperatura (°C)")
#     plt.xlabel("Tiempo (días)")
#     plt.ylabel("Temperatura (°C)")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     plt.figure()
#     plt.step(td, Nadd_profile, where="post", label="Nadd (g/L·h)")
#     plt.xlabel("Tiempo (días)")
#     plt.ylabel("Nadd (g/L·h)")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
