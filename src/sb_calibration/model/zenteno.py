"""
Zenteno model adapter (moved from top-level). Proporciona funciones de simulación y utilidades.
"""
from pathlib import Path
import numpy as np
from typing import Callable, Tuple, Optional

EPS = 1e-9
BIG = 1e6

def safe_div(a, b, eps=EPS):
    return a / (b + eps)

def safe_exp(x, lo=-50.0, hi=50.0):
    return np.exp(np.clip(x, lo, hi))

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def _real_pos(z):
    r = float(np.real(z))
    return r if r > 0.0 else 0.0


def load_parameters_from_excel(xlsx_path: str = "zenteno_parameters.xlsx", sheet_name: str = "Hoja1", param_set: int = 3) -> np.ndarray:
    xp = Path(xlsx_path)
    if not xp.exists():
        try:
            base = Path(__file__).resolve().parent.parent.parent
        except Exception:
            base = Path.cwd()
        xp = (base / xlsx_path)
    if not xp.exists():
        raise FileNotFoundError(f"No se encontró el archivo de parámetros: {xlsx_path} (ruta resuelta: {xp})")
    import pandas as pd
    df = pd.read_excel(xp, sheet_name=sheet_name)
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
    return p


def zenteno_model(t: float, x: np.ndarray, u: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = float(u[0]); Nadd = float(u[1])
    X = _real_pos(x[0]); N = _real_pos(x[1]); G = _real_pos(x[2]); F = _real_pos(x[3]); E = _real_pos(x[4])
    T = clamp(T, 273.15, 333.15)
    vals = [max(float(pi), EPS) for pi in p]
    (mu0, betaG0, betaF0, Kn0, Kg0, Kf0, Kig0, Kie0, Kd0,
     Yxn, Yxg, Yxf, Yeg, Yef) = vals
    Cde = 0.0415; Etd = 130000.0; R = 8.314; Eac = 59453.0; Eafe = 11000.0
    EaKn = 46055.0; EaKg = 46055.0; EaKf = 46055.0; EaKig = 46055.0; EaKie = 46055.0; Eam = 37681.0; m0 = 0.01
    mu_max = mu0 * safe_exp(Eac *(T-300.00)/(300.00*R*T))
    betaG_max = betaG0 * safe_exp(Eafe*(T-296.15)/(296.15*R*T))
    betaF_max = betaF0 * safe_exp(Eafe*(T-296.15)/(296.15*R*T))
    Kn = Kn0 * safe_exp(EaKn*(T-293.15)/(293.15*R*T))
    Kg = Kg0 * safe_exp(EaKg*(T-293.15)/(293.15*R*T))
    Kf = Kf0 * safe_exp(EaKf*(T-293.15)/(293.15*R*T))
    Kig = Kig0 * safe_exp(EaKig*(T-293.15)/(293.15*R*T))
    Kie = Kie0 * safe_exp(EaKie*(T-293.15)/(293.15*R*T))
    m = m0 * safe_exp(Eam *(T-293.30)/(293.30*R*T))
    mu = mu_max * safe_div(N, N + Kn)
    beta_G = betaG_max * safe_div(G, G + Kg) * safe_div(Kie, E + Kie)
    beta_F = betaF_max * safe_div(F, F + Kf) * safe_div(Kig, G + Kig) * safe_div(Kie, E + Kie)
    E_cap = clamp(E, 0.0, 200.0)
    Td = -0.0001*(E_cap**3) + 0.0049*(E_cap**2) - 0.1279*E_cap + 315.89
    Td = clamp(Td, 273.15, 333.15)
    if T >= Td:
        exponent = (Cde*E_cap) + safe_div(Etd*(T-305.65), (305.65*R*T))
        Kd = Kd0 * safe_exp(exponent, lo=-50.0, hi=50.0)
    else:
        Kd = 0.0
    GpF = G + F + EPS
    mG = G / GpF; mF = F / GpF
    dX = (mu - Kd) * X
    dN = -(mu / max(Yxn, EPS)) * X
    dG = -((mu / max(Yxg, EPS)) + (beta_G / max(Yeg, EPS)) + m*mG) * X
    dF = -((mu / max(Yxf, EPS)) + (beta_F / max(Yef, EPS)) + m*mF) * X
    dE = (beta_G + beta_F) * X
    dX = float(clamp(dX, -BIG, BIG))
    dN = float(clamp(dN, -BIG, BIG))
    dG = float(clamp(dG, -BIG, BIG))
    dF = float(clamp(dF, -BIG, BIG))
    dE = float(clamp(dE, -BIG, BIG))
    return np.array([dX, dN, dG, dF, dE], dtype=float)


def zenteno_jacobian(t: float, x: np.ndarray, u: np.ndarray, p: np.ndarray) -> np.ndarray:
    X = _real_pos(x[0]); N = _real_pos(x[1]); G = _real_pos(x[2]); F = _real_pos(x[3]); E = _real_pos(x[4])
    T = float(u[0]); T = clamp(T, 273.15, 333.15)
    (mu0, betaG0, betaF0, Kn0, Kg0, Kf0, Kig0, Kie0, Kd0,
     Yxn, Yxg, Yxf, Yeg, Yef) = [max(float(pi), EPS) for pi in p]
    Cde = 0.0415; Etd = 130000.0; R = 8.314; Eac = 59453.0; Eafe = 11000.0; EaKn = 46055.0; EaKg = 46055.0
    EaKf = 46055.0; EaKig = 46055.0; EaKie = 46055.0; Eam = 37681.0; m0 = 0.01
    mu_max = mu0 * safe_exp(Eac *(T-300.00)/(300.00*R*T))
    betaG_max = betaG0 * safe_exp(Eafe*(T-296.15)/(296.15*R*T))
    betaF_max = betaF0 * safe_exp(Eafe*(T-296.15)/(296.15*R*T))
    Kn = Kn0 * safe_exp(EaKn*(T-293.15)/(293.15*R*T))
    Kg = Kg0 * safe_exp(EaKg*(T-293.15)/(293.15*R*T))
    Kf = Kf0 * safe_exp(EaKf*(T-293.15)/(293.15*R*T))
    Kig = Kig0 * safe_exp(EaKig*(T-293.15)/(293.15*R*T))
    Kie = Kie0 * safe_exp(EaKie*(T-293.15)/(293.15*R*T))
    m = m0 * safe_exp(Eam *(T-293.30)/(293.30*R*T))
    mu = mu_max * safe_div(N, N + Kn)
    dmu_dN = mu_max * (Kn / (N + Kn)**2)
    beta_G = betaG_max * safe_div(G, G + Kg) * safe_div(Kie, E + Kie)
    dbG_dG = betaG_max * (Kg / (G + Kg)**2) * safe_div(Kie, E + Kie)
    dbG_dE = betaG_max * safe_div(G, G + Kg) * (-Kie / (E + Kie)**2)
    beta_F = betaF_max * safe_div(F, F + Kf) * safe_div(Kig, G + Kig) * safe_div(Kie, E + Kie)
    dbF_dF = betaF_max * (Kf / (F + Kf)**2) * safe_div(Kig, G + Kig) * safe_div(Kie, E + Kie)
    dbF_dG = betaF_max * safe_div(F, F + Kf) * (-Kig / (G + Kig)**2) * safe_div(Kie, E + Kie)
    dbF_dE = betaF_max * safe_div(F, F + Kf) * safe_div(Kig, G + Kig) * (-Kie / (E + Kie)**2)
    E_cap = clamp(E, 0.0, 200.0)
    Td = -0.0001*(E_cap**3) + 0.0049*(E_cap**2) - 0.1279*E_cap + 315.89
    Td = clamp(Td, 273.15, 333.15)
    if T >= Td:
        Kd = Kd0 * safe_exp( Cde*E_cap + Etd*(T-305.65)/(305.65*R*T) )
        dKd_dE = Cde * Kd
    else:
        Kd = 0.0
        dKd_dE = 0.0
    sumGF = G + F + EPS
    phi_G = G / sumGF; phi_F = F / sumGF
    dphiG_dG = F / (sumGF**2); dphiG_dF = -G / (sumGF**2)
    dphiF_dF = G / (sumGF**2); dphiF_dG = -F / (sumGF**2)
    J = np.zeros((5,5), dtype=float)
    J[0,0] = (mu - Kd); J[0,1] = dmu_dN * X; J[0,2] = 0.0; J[0,3] = 0.0; J[0,4] = -(dKd_dE) * X
    J[1,0] = -(mu / Yxn); J[1,1] = -(dmu_dN / Yxn) * X
    term_G = (mu / Yxg) + (beta_G / Yeg) + m*phi_G
    J[2,0] = -term_G; J[2,1] = -((dmu_dN / Yxg) * X)
    J[2,2] = -(((dbG_dG / Yeg) + m * dphiG_dG) * X)
    J[2,3] = -((m * dphiG_dF) * X); J[2,4] = -(((dbG_dE / Yeg) * X))
    term_F = (mu / Yxf) + (beta_F / Yef) + m*phi_F
    J[3,0] = -term_F; J[3,1] = -((dmu_dN / Yxf) * X)
    J[3,2] = -((dbF_dG * X)); J[3,3] = -(((dbF_dF / Yef) + m * dphiF_dF) * X)
    J[3,4] = -(((dbF_dE / Yef) * X))
    J[4,0] = (beta_G + beta_F); J[4,1] = 0.0
    J[4,2] = (dbG_dG + dbF_dG) * X; J[4,3] = (dbF_dF) * X; J[4,4] = (dbG_dE + dbF_dE) * X
    return J


# Sparsidad conservadora
J_SPARSE = np.array([
    [1,1,0,0,1],
    [1,1,0,0,0],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,0,1,1,1],
], dtype=bool)


def simulate_process_time_stiff(tf: float, x0: np.ndarray, n_steps: Optional[int], u_fun: Callable[[float], np.ndarray], p: np.ndarray, method: str = "Radau", atol: float = 1e-8, rtol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.integrate import solve_ivp
    t0 = 0.0
    if n_steps is None or n_steps <= 1:
        n_steps = max(int(np.ceil(tf)), 2)
    t_eval = np.linspace(t0, tf, n_steps)
    def f_ode(t, y):
        u = u_fun(t)
        return zenteno_model(t, y, u, p)
    def j_ode(t, y):
        u = u_fun(t)
        return zenteno_jacobian(t, y, u, p)
    sol = solve_ivp(fun=f_ode, t_span=(t0, tf), y0=np.asarray(x0, dtype=float), method=method, jac=j_ode, jac_sparsity=J_SPARSE, t_eval=t_eval, atol=atol, rtol=rtol, vectorized=False)
    if not sol.success:
        sol = solve_ivp(fun=f_ode, t_span=(t0, tf), y0=np.asarray(x0, dtype=float), method=method, jac=j_ode, jac_sparsity=J_SPARSE, t_eval=t_eval, atol=max(atol*10, 1e-6), rtol=max(rtol*10, 1e-4), vectorized=False)
        if not sol.success:
            raise RuntimeError(f"solve_ivp falló: {sol.message}")
    return sol.t, sol.y.T


def RK4_method(f, tf, x0, n, u, p):
    tf = float(tf)
    if n is None or n <= 0:
        n = max(int(np.ceil(tf / 1.0)), 10)
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
        x = np.maximum(x, 0.0)
        x = np.minimum(x, BIG)
        X[k+1, :] = x
        t[k+1] = t[k] + h
    return t, X


def build_profiles(tf, n, temps_c=None, injections=None, dt=None, temp_segments=None):
    if n is None or n <= 0:
        n = max(int(np.ceil(tf / 1.0)), 10)
    h = (tf / n) if dt is None else dt
    T_profile = np.empty(n+1, dtype=float)
    if temp_segments is not None and len(temp_segments) > 0:
        if len(temp_segments[0]) == 3:
            T_profile[:] = (temp_segments[0][2] + 273.15)
            for seg in temp_segments:
                t_start, t_end, Tc = seg
                a = max(0, min(n, int(round(t_start / h))))
                b = max(0, min(n, int(round(t_end   / h))))
                if b < a: a, b = b, a
                T_profile[a:b] = Tc + 273.15
            if len(temp_segments) > 0: T_profile[-1] = (temp_segments[-1][2] + 273.15)
        else:
            segs = sorted(temp_segments, key=lambda s: s[0])
            first_t, first_Tc = segs[0]
            T_profile[:] = first_Tc + 273.15
            for i, (tc, Tc) in enumerate(segs):
                a = max(0, min(n, int(round(tc / h))))
                b = n+1 if i == len(segs)-1 else max(0, min(n+1, int(round(segs[i+1][0] / h))))
                T_profile[a:b] = Tc + 273.15
    else:
        if temps_c is None or len(temps_c) != 3:
            raise ValueError("Si no usas temp_segments, debes entregar temps_c con 3 valores (°C).")
        seg = n // 3
        idxs = [0, seg, 2*seg, n]
        temps_k = [Tc + 273.15 for Tc in temps_c]
        for s in range(3):
            a, b = idxs[s], idxs[s+1]
            T_profile[a:b+1] = temps_k[s]
    T_profile = np.clip(T_profile, 273.15, 333.15)
    Nadd_profile = np.zeros(n+1, dtype=float)
    for t_pulse, amount in (injections or []):
        k = int(round(t_pulse / h)); k = max(0, min(n, k))
        Nadd_profile[k] += amount / h
    return T_profile, Nadd_profile


def simulate_process_time(p, x0, temps_c, injections, tf=14*24.0, n=None, threshold=5.0, temp_segments=None):
    if n is None: n = max(int(tf), 10)
    T_profile, Nadd_profile = build_profiles(tf, n, temps_c=temps_c, injections=injections, temp_segments=temp_segments)
    u = np.vstack([T_profile, Nadd_profile]).T
    t, x = RK4_method(zenteno_model, tf, x0, n, u, p)
    G = x[:, 2]; F = x[:, 3]; total_sugar = G + F
    idx = np.argmax(total_sugar <= threshold)
    if total_sugar[0] <= threshold: t_proc = 0.0
    elif total_sugar[idx] <= threshold: t_proc = t[idx]
    else: t_proc = tf
    return t_proc, t, x, T_profile, Nadd_profile


DEFAULT_X0 = np.array([0.5, 0.140, 110.0, 110.0, 0.0])


def build_temp_profile_from_df(df):
    """Create segments list [(t_h, T_C), ...] from a DataFrame similar to legacy helper.

    Accepts columns like 'time_h' and temperature columns ('Temperature_C','temperature','temp_c','temperatura').
    """
    if df is None:
        return [(0.0, 20.0)]
    cand = None
    for c in df.columns:
        l = str(c).strip().lower()
        if l in ("temperature_c", "temperature", "temp_c", "temperatura"):
            cand = c
            break
    if cand is None or "time_h" not in df.columns:
        return [(0.0, 20.0)]
    try:
        import pandas as pd
        t = pd.to_numeric(df["time_h"], errors="coerce").to_numpy()
        Tc = pd.to_numeric(df[cand], errors="coerce").to_numpy()
    except Exception:
        # fallback
        return [(0.0, 20.0)]
    mask = ~(np.isnan(t) | np.isnan(Tc))
    t, Tc = t[mask], Tc[mask]
    if len(t) == 0:
        return [(0.0, 20.0)]
    order = np.argsort(t)
    t, Tc = t[order], Tc[order]
    segs = [(float(t[0]), float(Tc[0]))]
    for i in range(1, len(t)):
        if not np.isclose(Tc[i], Tc[i-1], atol=1e-6):
            segs.append((float(t[i]), float(Tc[i])))
    if segs[0][0] > 0.0:
        segs.insert(0, (0.0, segs[0][1]))
    return segs


def simulate_on_grid(p_real, time_h, temp_segments, pulses, x0=None, method="Radau"):
    """Lightweight simulate_on_grid adapter used by the calibrator.

    - `temp_segments` may be a DataFrame (with time_h and temp column) or a list of (t_h, T_C).
    - `pulses` is a list of (t_h, dN_gL) or None. Pulses are applied approximately by adding dN at nearest time index.
    Returns (t_sim, Xsim) where t_sim is a 1D time array and Xsim is (n,5) states.
    """
    # resolve x0
    if x0 is None:
        x0 = DEFAULT_X0.copy()

    # build temp segments list if DataFrame provided
    segs = None
    if hasattr(temp_segments, "columns"):
        segs = build_temp_profile_from_df(temp_segments)
    else:
        segs = temp_segments

    tf = float(np.nanmax(time_h)) if np.isfinite(np.nanmax(time_h)) and np.nanmax(time_h) > 0 else 14 * 24.0
    # control grid size: prefer fine grid but bounded for tests
    n_u = max(int(np.ceil(tf / 0.25)), 200)
    # use existing build_profiles to get T_profile and Nadd (Nadd will be zero because injections param handled differently)
    try:
        T_profile, Nadd_profile = build_profiles(tf, n_u, temp_segments=segs)
    except Exception:
        # fallback to uniform 20°C
        T_profile = np.full(n_u + 1, 20.0 + 273.15)
        Nadd_profile = np.zeros(n_u + 1, dtype=float)

    u = np.vstack([T_profile, Nadd_profile]).T
    # run RK4 integration
    t_sim, Xsim = RK4_method(zenteno_model, tf, x0, n_u, u, p_real)

    # apply instantaneous pulses approximately: add dN to the N state at nearest index
    if pulses:
        for (tp, dN) in pulses:
            if tp < 0 or tp > t_sim[-1]:
                continue
            idx = int(np.argmin(np.abs(t_sim - tp)))
            Xsim[idx:, 1] = Xsim[idx:, 1] + dN

    return t_sim, Xsim
