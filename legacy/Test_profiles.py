# -*- coding: utf-8 -*-
"""
Extracción de modelo dinámico y simulación (Zenteno et al.).
Generado automáticamente desde modelo_hibrido.ipynb.
Requisitos: numpy, pandas, matplotlib
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def load_parameters_from_excel(xlsx_path="zenteno_parameters.xlsx", sheet_name="Hoja1", param_set=4):
    """
    Carga el vector de parámetros desde un Excel.
    Espera una columna 'set' para filtrar y el resto de columnas en el orden del modelo.
    Retorna: numpy.ndarray con 14 valores.
    """
    # Resolver ruta relativa al CWD o al script, lo que exista primero
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
        # advertir pero permitir
        print(f"[AVISO] Se esperaban 14 parámetros, se encontraron {p.size}. Continuando con el tamaño encontrado.")
    return p


def list_available_sets(xlsx_path="zenteno_parameters.xlsx", sheet_name="Hoja1"):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    # detectar columna 'set'
    set_col = None
    for c in df.columns:
        if str(c).strip().lower() == "set":
            set_col = c
            break
    if set_col is None:
        return []
    return sorted(df[set_col].unique().tolist())


def zenteno_model(t, x, u, p):
    """ Modelo de cinética de fermentación propuesto por Zenteno et al. 2010

    Inputs
    ----------
    t: array
        vector de tiempo
    x: array
        vector de estados
    u: array
        vector de variables de entrada
    p: array
        vector de parámetros del modelo
    Returns
    -------
    dxdt: array
        vector derivadas (ODEs)
    """
    # control variables
    T = u[0]    # Temperature profile K
    Nadd = u[1] # Nitrogen additions  g/L

    # states variables
    X = x[0]    # Biomass (g/L)
    N = x[1]    # Nitrogen (g/L)
    G = x[2]    # Glucose (g/L)
    F = x[3]    # Fructose (g/L)
    E = x[4]    # Ethanol (g/L)

    # parameters (mantengo tu mapeo)
    mu0    = p[0]   # 1/h
    betaG0 = p[1]   # kg E/kg bio/h
    betaF0 = p[2]   # kg E/kg bio/h
    Kn0    = p[3]   # kg N/m3
    Kg0    = p[4]   # kg G/m3
    Kf0    = p[5]   # kg F/m3
    Kig0   = p[6]   # kg G/m3
    Kie0   = p[7]   # kg E/m3
    Kd0    = p[8]   # 1/h
    Yxn    = p[9]   # kg Bio/kg N, biomass/nitrogen yield coeff
    Yxg    = p[10]  # kg bio/kg G, biomass/glucose yield coeff
    Yxf    = p[11]  # kg bio/kg F, biomass/fructose yield coeff
    Yeg    = p[12]  # kg E/ kg G, ethanol/glucose yield coeff
    Yef    = p[13]  # kg E/ kg F, ethanol/fructose yield coeff

    # Fixed Parameters
    Cde     = 0.0415      # m3/kg E (Salmon 2003)
    Etd     = 130000.0    # kj/kmol
    R       = 8.314       # kJ/kmol/K (Boulton 1979)
    Eac     = 59453.0     # kJ/kmol (Boulton 1979)
    Eafe    = 11000.0     # kJ/kmol (Zenteno 2010, trial and error)
    EaKn    = 46055.0     # kJ/kmol (Boulton 1979)
    EaKg    = 46055.0     # kJ/kmol (Boulton 1979)
    EaKf    = 46055.0     # kJ/kmol (Boulton 1979)
    EaKig   = 46055.0     # kJ/kmol (Boulton 1979)
    EaKie   = 46055.0     # kJ/kmol (Boulton 1979)
    Eam     = 37681.0     # kJ/kmol (Boulton 1979)
    m0      = 0.01        # kgS/kg bio/h

    # constitutive equations
    mu_max    = mu0*np.exp(Eac*(T-300)/(300*R*T))                    # 1/h
    betaG_max = betaG0*np.exp(Eafe*(T-296.15)/(296.15*R*T))          # kg E/kg bio/h
    betaF_max = betaF0*np.exp(Eafe*(T-296.15)/(296.15*R*T))          # kg E/kg bio/h
    Kn        = Kn0*np.exp(EaKn*(T-293.15)/(293.15*R*T))             # kg N/m3
    Kg        = Kg0*np.exp(EaKg*(T-293.15)/(293.15*R*T))             # kg G/m3
    Kf        = Kf0*np.exp(EaKf*(T-293.15)/(293.15*R*T))             # kg F/m3
    Kig       = Kig0*np.exp(EaKig*(T-293.15)/(293.15*R*T))           # kg G/m3
    Kie       = Kie0*np.exp(EaKie*(T-293.15)/(293.15*R*T))           # kg E/m3
    m         = m0*np.exp(Eam*(T-293.3)/(293.3*R*T))                 # kg S/kg bio/h

    mu      = mu_max*(N/(N+Kn))                                      # growth
    beta_G  = betaG_max*(G/(G+Kg))*(Kie/(E+Kie))                      # ethanol from G
    beta_F  = betaF_max*(F/(F+Kf))*(Kig/(G+Kig))*(Kie/(E+Kie))        # ethanol from F

    # threshold temperature of thermal death
    Td        = -0.0001*E**3 + 0.0049*E**2 - 0.1279*E + 315.89       # K

    # specific death rate
    if T >= Td:
        Kd = Kd0*np.exp((Cde*E) + (Etd*(T-305.65))/(305.65*R*T))
    else:
        Kd = 0.0

    # Differential Equations
    dxdt = np.zeros(len(x), dtype='object')
    dxdt[0] = mu*X - Kd*X                          # Biomass
    dxdt[1] = -mu*(X/Yxn) + Nadd                   # Nitrogen
    dxdt[2] = -((mu/Yxg)+(beta_G/Yeg)+m*(G/(G+F)))*X   # Glucose
    dxdt[3] = -((mu/Yxf)+(beta_F/Yef)+m*(F/(G+F)))*X   # Fructose
    dxdt[4] = (beta_G + beta_F)*X                  # Ethanol
    return dxdt


def RK4_method(f, tf, x0, n, u, p):
    """
    Método Runge-Kutta explicito de 4to orden para integrar ODE.
    """
    h = tf/n
    t = np.zeros(n+1)
    x = np.array((n+1)*[x0])
    t[0] = 0.0
    x[0] = x0

    for k in range(n):
        k1 = f(t[k],         x[k],           u[k], p)
        k2 = f(t[k]+h/2,     x[k]+(h/2)*k1,  u[k], p)
        k3 = f(t[k]+h/2,     x[k]+(h/2)*k2,  u[k], p)
        k4 = f(t[k]+h,       x[k]+h*k3,      u[k], p)
        x[k+1] = x[k] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t[k+1] = t[k] + h

    return t, x


def build_profiles(tf, n, temps_c, injections, dt=None, temp_segments=None):
    """
    Construye perfiles discretos de temperatura (K) y adición de N (g/L/h) para la integración.

    Parámetros
    ----------
    tf : float
        Tiempo total de simulación (h).
    n : int
        Número de pasos (elementos finitos). Paso h = tf/n si dt es None.
    temps_c : list[float] de largo 3
        Tres temperaturas en °C para aplicar en tercios del horizonte (se ignora si se da temp_segments).
    injections : list[tuple]
        Lista de pulsos de nitrógeno [(t_h, cantidad_gL), ...].
        Cada pulso se aproxima como una tasa Nadd = cantidad / h en el paso donde cae.
    dt : float | None
        Paso de discretización (h). Si None, usa tf/n.
    temp_segments : list[tuple] | None
        Si se provee, define el perfil T(t) por tramos. Dos formatos soportados:
        - [(t_start, t_end, T_C), ...] en horas y °C
        - [(t_change, T_C), ...] cambios de T a partir de t_change hasta el siguiente cambio (o tf).
          Si el primer t_change > 0, se asume que antes de ese tiempo también rige su T_C.

    Retorna
    -------
    T_profile : ndarray (n+1,)
        Temperatura en Kelvin por paso (piecewise constante).
    Nadd_profile : ndarray (n+1,)
        Tasa de adición de nitrógeno en g/L/h por paso.
    """
    h = (tf / n) if dt is None else dt
    T_profile = np.empty(n+1, dtype=float)

    if temp_segments is not None and len(temp_segments) > 0:
        # Detectar formato de segmentos
        if len(temp_segments[0]) == 3:
            # Formato (t_start, t_end, T_C)
            T_profile[:] = (temp_segments[0][2] + 273.15)  # valor por defecto inicial
            for seg in temp_segments:
                t_start, t_end, Tc = seg
                a = max(0, min(n, int(round(t_start / h))))
                b = max(0, min(n, int(round(t_end   / h))))
                if b < a:
                    a, b = b, a
                T_profile[a:b] = Tc + 273.15
            # Asegurar último punto
            if len(temp_segments) > 0:
                T_profile[-1] = (temp_segments[-1][2] + 273.15)
        else:
            # Formato (t_change, T_C)
            segs = sorted(temp_segments, key=lambda s: s[0])
            # Llenado inicial
            first_t, first_Tc = segs[0]
            T_profile[:] = first_Tc + 273.15
            for i, (tc, Tc) in enumerate(segs):
                a = max(0, min(n, int(round(tc / h))))
                b = n+1 if i == len(segs)-1 else max(0, min(n+1, int(round(segs[i+1][0] / h))))
                T_profile[a:b] = Tc + 273.15
    else:
        # Comportamiento por defecto: 3 tramos iguales con temps_c
        if temps_c is None or len(temps_c) != 3:
            raise ValueError("Si no usas temp_segments, debes entregar temps_c con 3 valores (°C).")
        seg = n // 3
        idxs = [0, seg, 2*seg, n]
        temps_k = [Tc + 273.15 for Tc in temps_c]
        for s in range(3):
            a, b = idxs[s], idxs[s+1]
            T_profile[a:b+1] = temps_k[s]

    # Perfil de N: impulsos como tasa en el paso
    Nadd_profile = np.zeros(n+1, dtype=float)
    for t_pulse, amount in injections or []:
        k = int(round(t_pulse / h))
        k = max(0, min(n, k))
        Nadd_profile[k] += amount / h  # g/L/h en ese paso

    return T_profile, Nadd_profile


def simulate_process_time(p, x0, temps_c, injections, tf=14*24.0, n=None, threshold=5.0, temp_segments=None):
    """
    Simula la fermentación con 3 temperaturas (o segmentos explícitos) y pulsos de N y retorna
    el tiempo (h) cuando G+F <= threshold (g/L). Si no se alcanza, retorna tf.
    """
    if n is None:
        n = int(tf)
    T_profile, Nadd_profile = build_profiles(tf, n, temps_c, injections, temp_segments=temp_segments)
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


# Valores iniciales por defecto (desde el cuaderno)
DEFAULT_X0 = np.array([0.5, 0.140, 110.0, 110.0, 0.0])


if __name__ == "__main__":
    p = load_parameters_from_excel("zenteno_parameters.xlsx", "Hoja1", param_set=4)

    # Opción A (legacy): tres tramos iguales
    temps = [25.0, 20.0, 15.0]  # °C

    # Opción B (nueva): segmentos con tiempos de cambio (en horas)
    # Formato 1: (t_start, t_end, T_C)
    
    temp_segments = [
        (00.0,  24,    21.0),   # 0-24 h a 25°C - MODIFICAR AQUI
        (24.0,  21*24, 23.0),   # 24-96 h a 20°C - MODIFICAR AQUI
        # (224.0, 21*24, 17.0),   # 96-168 h a 15°C - MODIFICAR AQUI
    ]
    # Alternativamente, puedes usar el Formato 2: (t_change, T_C), p. ej.:
    # temp_segments = [(0.0, 25.0), (24.0, 20.0), (96.0, 15.0)]
    T_NUT = 48
    pulsos = [(0.0, 0.02), (T_NUT, 0.066)]    # (hora, g/L) - MODIFICAR AQUI

    t_proc, t, x, T_profile, Nadd_profile = simulate_process_time(
        p, DEFAULT_X0, temps, pulsos, tf=21*24.0, n=None, threshold=2.0, temp_segments=temp_segments
    ) #t_proc ES LA VARIABLE CON EL TIEMPO TOTAL DE PROCESO HASTA EL SECADO
    
    print("Tiempo total (G+F <= 2 g/L):", t_proc, "h")

    # === Gráficos ===
    td = np.asarray(t, dtype=float) / 24.0  # tiempo en días

    # Figura 1: estados (N en mg/L si quieres destacarlo)
    plt.figure()
    plt.plot(td, x[:, 0]*10, label="X (g/L)")
    plt.plot(td, x[:, 1]*1000, label="N (mg/L)")
    plt.plot(td, x[:, 2], label="G (g/L)")
    plt.plot(td, x[:, 3], label="F (g/L)")
    plt.plot(td, x[:, 4], label="E (g/L)")
    plt.axvline(x=t_proc/24, color="k", linestyle="--")
    plt.xlabel("Tiempo (días)")
    plt.ylabel("Concentración")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Figura 2: temperatura (°C) con cambios
    T_c = np.asarray(T_profile, dtype=float) - 273.15
    plt.figure()
    plt.step(td, T_c, where="post", label="Temperatura (°C)")
    plt.xlabel("Tiempo (días)")
    plt.ylabel("Temperatura (°C)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Figura 3: perfil de adición de nitrógeno (tasa g/L/h)
    plt.figure()
    plt.step(td, Nadd_profile, where="post", label="Nadd (g/L·h)")
    plt.xlabel("Tiempo (días)")
    plt.ylabel("Nadd (g/L·h)")
    plt.legend()
    plt.tight_layout()
    plt.show()
