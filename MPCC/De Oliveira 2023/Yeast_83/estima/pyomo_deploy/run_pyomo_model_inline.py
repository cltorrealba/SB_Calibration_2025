"""
============================================================
Pyomo MPCC/dFBA (Yeast_83) — inline runner
Structure aligned with Julia main.jl via titled sections.
============================================================
"""

import os
import sys
import math
import time
import argparse
import numpy as np
import pandas as pd
from pyomo.environ import (
    ConcreteModel, RangeSet, Set, Param, Var, Reals, Objective, Constraint,
    NonNegativeReals, value, minimize, Expression, summation, SolverFactory, exp as pyomo_exp
)
from pyomo.util.model_size import build_model_size_report
from pyomo.repn.standard_repn import generate_standard_repn


# =============================================
# 1) Paths and IO configuration
# =============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ESTIMA_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
YEAST83_DIR = os.path.abspath(os.path.join(ESTIMA_DIR, ".."))

# Files expected (match Julia's main.jl: files live in `estima/`)
S_CSV = os.path.join(ESTIMA_DIR, "S.csv")
LB_CSV = os.path.join(ESTIMA_DIR, "lb.csv")
UB_CSV = os.path.join(ESTIMA_DIR, "ub.csv")
DATA_LONG_CSV = os.path.join(BASE_DIR, "data_long.csv")  # long format (state,fe,cp,value)

# Results dir
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================
# 2) Helpers: load matrices and experimental data
# =============================================
def load_stoichiometry_and_bounds():
    if not os.path.exists(S_CSV):
        raise FileNotFoundError(f"Stoichiometric matrix not found: {S_CSV}")
    if not os.path.exists(LB_CSV):
        raise FileNotFoundError(f"Lower bounds not found: {LB_CSV}")
    if not os.path.exists(UB_CSV):
        raise FileNotFoundError(f"Upper bounds not found: {UB_CSV}")

    S = np.loadtxt(S_CSV, delimiter=",")
    vlb = np.loadtxt(LB_CSV, delimiter=",")
    vub = np.loadtxt(UB_CSV, delimiter=",")

    # Flatten if column vectors
    vlb = vlb.flatten()
    vub = vub.flatten()

    nm, nv = S.shape
    return S, vlb, vub, nm, nv


def load_data_long(nc: int, ph: int, ncp: int):
    """
    Load experimental data in long format CSV with columns:
    state,fe,cp,value (1-based indices for state/fe/cp).
    Returns a dictionary keyed by (state, fe, cp) -> value.
    """
    if not os.path.exists(DATA_LONG_CSV):
        msg = (
            f"Missing experimental data file: {DATA_LONG_CSV}\n"
            f"Please export data from ../data.jld2 using export_data_to_csv.jl first.\n"
        )
        raise FileNotFoundError(msg)

    df = pd.read_csv(DATA_LONG_CSV)
    required_cols = {"state", "fe", "cp", "value"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"{DATA_LONG_CSV} must contain columns: {required_cols}")

    # Filter to model ranges to allow reduced horizons for quick tests
    max_state, max_fe, max_cp = int(df["state"].max()), int(df["fe"].max()), int(df["cp"].max())
    if max_state > nc or max_fe > ph or max_cp > ncp:
        before = len(df)
        df = df[(df["state"] <= nc) & (df["fe"] <= ph) & (df["cp"] <= ncp)].copy()
        after = len(df)
        print(f"[WARN] Truncated data_long.csv to model ranges (nc={nc}, ph={ph}, ncp={ncp}). Rows: {before} -> {after}")

    data = {(int(r.state), int(r.fe), int(r.cp)): float(r.value) for r in df.itertuples(index=False)}
    return data


# =============================================
# 3) Build Pyomo model (sections mirror Julia main.jl)
# =============================================
def build_model(
    nfe: int = 12,
    th: float = 22.0,
    phi1_val: float = 1.0,
    phi2_val: float = 1.0,
    phi3_val: float = 1.0,
    fix_hv: bool = False,
    teta_start_mode: str = "julia",
    drop_fo_products: bool = False,
    vgvz_mode: str = "expr",
):
    t0 = time.time()
    print("[INFO] Loading S, lb, ub ...")
    # Read stoichiometry and bounds
    S, vlb, vub, nm, nv = load_stoichiometry_and_bounds()
    print(f"[INFO] Loaded S ({S.shape[0]} x {S.shape[1]}), lb/ub vectors. Took {time.time()-t0:.2f}s")

    # Precompute sparse S structure to avoid constructing a gigantic dense Param
    tS = time.time()
    rows, cols = np.nonzero(S)
    row_to_cols: dict[int, list[int]] = {}
    col_to_rows: dict[int, list[int]] = {}
    S_vals: dict[tuple[int, int], float] = {}
    for k in range(len(rows)):
        mm = int(rows[k]) + 1  # 1-based
        rr = int(cols[k]) + 1  # 1-based
        val = float(S[mm-1, rr-1])
        S_vals[(mm, rr)] = val
        row_to_cols.setdefault(mm, []).append(rr)
        col_to_rows.setdefault(rr, []).append(mm)
    print(f"[INFO] Built sparse S with {len(S_vals)} nonzeros. Took {time.time()-tS:.2f}s")

    # ---------------------------------------------
    # 3.1) Model constants and configuration (Julia parity)
    # ---------------------------------------------
    nc = 4
    ncp = 3
    ph = nfe  # prediction horizon equals number of finite elements
    h = th / nfe

    # ---------------------------------------------
    # 3.2) Initial conditions (states)
    # ---------------------------------------------
    x0 = 0.2
    g0 = 4.0
    z0 = 2.0
    e0 = 0.0
    c0 = np.array([x0, g0, z0, e0], dtype=float)

    # ---------------------------------------------
    # 3.3) Reaction indices (1-based, Julia mapping)
    # ---------------------------------------------
    eth = 2630
    obj = 3414
    glu = 2588
    o2 = 2816
    ATP = 3415
    xyl = 2592

    # ---------------------------------------------
    # 3.4) Parameter guesses/bounds (log scale)
    # ---------------------------------------------
    teta_opt = np.array([7.3, 1.03, 32.0, 14.85, 0.5])
    teta0 = np.log(np.array([7.5, 1.0, 35.0, 15.00, 0.5]))
    UB = np.log(np.array([8.0, 1.13, 33.0, 15.85, 0.6]))
    LB = np.log(np.array([7.0, 0.9, 31.0, 13.85, 0.4]))

    # ---------------------------------------------
    # 3.5) Flux bounds adjustments (anaerobic, ATP)
    # ---------------------------------------------
    vlb_adj = vlb.copy()
    vub_adj = vub.copy()
    # o2 fixed to 0
    if 1 <= o2 <= nv:
        vlb_adj[o2 - 1] = 0.0
        vub_adj[o2 - 1] = 0.0
    # ATP lower bound 0
    if 1 <= ATP <= nv:
        vlb_adj[ATP - 1] = 0.0

    # ---------------------------------------------
    # 3.6) Scaling vectors (states and fluxes)
    # ---------------------------------------------
    cs = np.ones(nc)
    vs = np.ones(nv)

    # ---------------------------------------------
    # 3.7) Radau collocation matrices
    # ---------------------------------------------
    colmat = np.array([
        [0.19681547722366, -0.06553542585020, 0.02377097434822],
        [0.39442431473909,  0.29207341166523, -0.04154875212600],
        [0.37640306270047,  0.51248582618842,  0.11111111111111],
    ])

    # ---------------------------------------------
    # 3.8) Weights and penalties (objective)
    # ---------------------------------------------
    w = 1e-20
    omega = 1e2
    phi1 = float(phi1_val)
    phi2 = float(phi2_val)
    phi3 = float(phi3_val)

    # ---------------------------------------------
    # 3.9) pFBA direction and uptake selectors (d, up, up2)
    # ---------------------------------------------
    d = np.zeros(nv)
    d[obj - 1] = -1.0

    # Uptake selector vectors for KKT stationarity
    up = np.zeros(nv)
    up[glu - 1] = 1.0
    up2 = np.zeros(nv)
    up2[xyl - 1] = 1.0
    n_up = 2

    # ---------------------------------------------
    # 3.10) Load experimental data (requires data_long.csv)
    # ---------------------------------------------
    t1 = time.time()
    print("[INFO] Loading experimental data (data_long.csv) ...")
    data = load_data_long(nc=nc, ph=ph, ncp=ncp)
    print(f"[INFO] Loaded data entries: {len(data)}. Took {time.time()-t1:.2f}s")

    # ---------------------------------------------
    # 3.11) Build model container
    # ---------------------------------------------
    print("[INFO] Building Pyomo model components ...")
    m = ConcreteModel()

    # ---------------------------------------------
    # 3.12) Sets (1-based; Julia-style indexing)
    # ---------------------------------------------
    m.C = RangeSet(1, nc)          # states
    m.I = RangeSet(1, nfe)         # finite elements
    m.J = RangeSet(1, ncp)         # collocation points
    m.PH = RangeSet(1, ph)         # prediction horizon index (like I)
    m.R = RangeSet(1, nv)          # reactions
    m.M = RangeSet(1, nm)          # metabolites
    m.UP = RangeSet(1, n_up)       # uptake constraint indices (glu,xyl)
    m.P = RangeSet(1, 5)           # parameter indices

    # ---------------------------------------------
    # 3.13) Parameters (scalar h/th, collocation, S, bounds, scaling)
    # ---------------------------------------------
    m.h = Param(initialize=h)
    m.th = Param(initialize=th)
    m.col = Param(m.J, m.J, initialize=lambda _m, a, b: float(colmat[a-1, b-1]))

    # Attach sparse S structures as Python attributes (for fast rule evaluation)
    m._S_vals = S_vals
    m._row_to_cols = row_to_cols
    m._col_to_rows = col_to_rows
    m.vlb = Param(m.R, initialize=lambda _m, rr: float(vlb_adj[rr-1]))
    m.vub = Param(m.R, initialize=lambda _m, rr: float(vub_adj[rr-1]))

    m.cs = Param(m.C, initialize=lambda _m, cc: float(cs[cc-1]))
    m.vs = Param(m.R, initialize=lambda _m, rr: float(vs[rr-1]))

    m.d = Param(m.R, initialize=lambda _m, rr: float(d[rr-1]))
    m.up = Param(m.R, initialize=lambda _m, rr: float(up[rr-1]))
    m.up2 = Param(m.R, initialize=lambda _m, rr: float(up2[rr-1]))

    # ---------------------------------------------
    # 3.14) Initial conditions (scaled)
    # ---------------------------------------------
    c0_scaled = c0 / cs
    m.c0 = Param(m.C, initialize=lambda _m, cc: float(c0_scaled[cc-1]))

    # ---------------------------------------------
    # 3.15) Parameter LB/UB (explicit inequalities)
    # ---------------------------------------------
    m.LB = Param(m.P, initialize=lambda _m, pp: float(LB[pp-1]))
    m.UB = Param(m.P, initialize=lambda _m, pp: float(UB[pp-1]))

    # Data param (sparse dict)
    def data_init(_m, cc, ii, jj):
        return float(data.get((cc, ii, jj), 0.0))
    m.data = Param(m.C, m.PH, m.J, initialize=data_init, mutable=False)

    # ---------------------------------------------
    # 3.16) Variables
    # ---------------------------------------------
    m.c = Var(m.C, m.PH, m.J, domain=Reals, initialize=lambda _m, c, i, j: float(m.c0[c]))
    m.cdot = Var(m.C, m.PH, m.J, domain=Reals, initialize=0.0)
    m.v = Var(m.R, m.I, domain=Reals, initialize=0.0)
    m.lam = Var(m.M, m.I, domain=Reals, initialize=0.0)
    m.alpha_L = Var(m.R, m.I, domain=Reals, initialize=0.0)
    m.alpha_U = Var(m.R, m.I, domain=Reals, initialize=0.0)
    m.alpha_upt = Var(m.UP, m.I, domain=Reals, initialize=0.0)
    m.FO_L = Var(m.R, m.I, domain=Reals, initialize=0.0)
    m.FO_U = Var(m.R, m.I, domain=Reals, initialize=0.0)
    m.FO_upt = Var(m.UP, m.I, domain=Reals, initialize=0.0)
    # Finite-element durations (present in Julia, not used in collocation equations)
    m.hv = Var(m.I, domain=Reals, initialize=float(h))
    if fix_hv:
        for i in m.I:
            m.hv[i].fix(float(h))
    # FO scalar for SSE (as in Julia)
    m.FO = Var(domain=Reals, initialize=0.0)
    # Initialize teta: match Julia (default 0.0) or clamp to [LB,UB]
    if (teta_start_mode or "").lower() == "julia":
        m.teta = Var(m.P, domain=Reals, initialize=0.0)
    else:
        def teta_init(_m, p):
            lo = float(LB[p-1])
            hi = float(UB[p-1])
            val = float(teta0[p-1])
            return float(min(max(val, lo), hi))
        m.teta = Var(m.P, domain=Reals, initialize=teta_init)
    def teta_lb_rule(_m, p):
        return _m.teta[p] >= _m.LB[p]
    def teta_ub_rule(_m, p):
        return _m.teta[p] <= _m.UB[p]
    m.TETA_LB = Constraint(m.P, rule=teta_lb_rule)
    m.TETA_UB = Constraint(m.P, rule=teta_ub_rule)

    # ---------------------------------------------
    # 3.17) Uptake functions vg, vz
    # ---------------------------------------------
    vgvz_mode = (vgvz_mode or "expr").lower()
    if vgvz_mode == "expr":
        def _vg_rule(_m, ii):
            G = _m.c[2, ii, 3]
            return pyomo_exp(_m.teta[1]) * (G / (pyomo_exp(_m.teta[2]) + G))
        m.vg = Expression(m.PH, rule=_vg_rule)

        def _vz_rule(_m, ii):
            Z = _m.c[3, ii, 3]
            G = _m.c[2, ii, 3]
            return pyomo_exp(_m.teta[3]) * (Z / (pyomo_exp(_m.teta[4]) + Z)) * (1.0 / (1.0 + (G / pyomo_exp(_m.teta[5]))))
        m.vz = Expression(m.PH, rule=_vz_rule)
    elif vgvz_mode == "varc":
        # Rich-Hessian variant: vg, vz as Vars with NL definition constraints
        m.vg = Var(m.PH, domain=Reals, initialize=0.0)
        m.vz = Var(m.PH, domain=Reals, initialize=0.0)
        def _vg_def_rule(_m, ii):
            G = _m.c[2, ii, 3]
            return _m.vg[ii] == pyomo_exp(_m.teta[1]) * (G / (pyomo_exp(_m.teta[2]) + G))
        def _vz_def_rule(_m, ii):
            Z = _m.c[3, ii, 3]
            G = _m.c[2, ii, 3]
            return _m.vz[ii] == pyomo_exp(_m.teta[3]) * (Z / (pyomo_exp(_m.teta[4]) + Z)) * (1.0 / (1.0 + (G / pyomo_exp(_m.teta[5]))))
        m.VG_DEF = Constraint(m.PH, rule=_vg_def_rule)
        m.VZ_DEF = Constraint(m.PH, rule=_vz_def_rule)
    else:
        raise ValueError(f"Invalid vgvz_mode='{vgvz_mode}', expected 'expr' or 'varc'")

    # ---------------------------------------------
    # 3.18) Objective function (omega*FO + penalties)
    #          — placed here to match Julia ordering
    # ---------------------------------------------
    def obj_rule(_m):
        if drop_fo_products:
            term_pen = 0.0
        else:
            term_pen = sum(
                sum(-phi1 * _m.FO_L[rr, i] + -phi3 * _m.FO_U[rr, i] for rr in _m.R)
                + phi2 * _m.FO_upt[1, i] + phi2 * _m.FO_upt[2, i]
                for i in _m.I
            )
        return omega * _m.FO + term_pen
    m.OBJ = Objective(rule=obj_rule, sense=minimize)

    # ---------------------------------------------
    # 3.19) Linear/affine constraints (collocation, bounds, stoichiometry)
    # ---------------------------------------------
    # Collocation equations
    def coll_c_n_rule(_m, l, i, j):
        if i == 1:
            return Constraint.Skip
        # Use constant h as in JuMP
        return _m.c[l, i, j] == _m.c[l, i-1, ncp] + _m.h * sum(_m.col[j, k] * _m.cdot[l, i, k] for k in _m.J)
    m.coll_c_n = Constraint(m.C, m.PH, m.J, rule=coll_c_n_rule)

    def coll_c_0_rule(_m, l, i, j):
        if i != 1:
            return Constraint.Skip
        # First element uses constant h
        return _m.c[l, i, j] == _m.c0[l] + _m.h * sum(_m.col[j, k] * _m.cdot[l, i, k] for k in _m.J)
    m.coll_c_0 = Constraint(m.C, m.PH, m.J, rule=coll_c_0_rule)

    # Stoichiometry and bounds per FE
    def Sc_rule(_m, mm, ii):
        rr_list = _m._row_to_cols.get(mm, [])
        return sum(_m._S_vals[(mm, rr)] * _m.v[rr, ii] * _m.vs[rr] for rr in rr_list) == 0.0
    m.Sc = Constraint(m.M, m.I, rule=Sc_rule)

    def v_UB_rule(_m, rr, ii):
        return _m.v[rr, ii] * _m.vs[rr] - _m.vub[rr] <= 0.0
    m.v_UB = Constraint(m.R, m.I, rule=v_UB_rule)

    def v_LB_rule(_m, rr, ii):
        return -_m.v[rr, ii] * _m.vs[rr] + _m.vlb[rr] <= 0.0
    m.v_LB = Constraint(m.R, m.I, rule=v_LB_rule)

    # c non-negativity
    def c_LB_rule(_m, cc, ii, jj):
        return -_m.c[cc, ii, jj] <= 0.0
    m.c_LB = Constraint(m.C, m.PH, m.J, rule=c_LB_rule)

    # Finite-element duration constraints (match Julia), collocation still uses constant h
    if not fix_hv:
        def mfe1_rule(_m):
            return sum(_m.hv[i] for i in _m.I) == _m.th
        m.MFE1 = Constraint(rule=mfe1_rule)

        var_h = 1.0
        hm1 = h
        def mfe3_rule(_m, i):
            return _m.hv[i] >= 0.0
        def mfe4_rule(_m, i):
            return _m.hv[i] >= (1 - var_h) * hm1
        def mfe5_rule(_m, i):
            return _m.hv[i] <= (1 + var_h) * hm1
        m.MFE3 = Constraint(m.I, rule=mfe3_rule)
        m.MFE4 = Constraint(m.I, rule=mfe4_rule)
        m.MFE5 = Constraint(m.I, rule=mfe5_rule)

    # KKT stationarity for pFBA objective
    def lagr_rule(_m, rr, ii):
        kk_list = _m._col_to_rows.get(rr, [])
        term_S = sum(_m._S_vals[(kk, rr)] * _m.lam[kk, ii] for kk in kk_list)
        return (
            _m.d[rr] + w * _m.v[rr, ii] * _m.vs[rr]
            + _m.alpha_L[rr, ii] + _m.alpha_U[rr, ii]
            + _m.up[rr] * _m.alpha_upt[1, ii] + _m.up2[rr] * _m.alpha_upt[2, ii]
            + term_S == 0.0
        )
    m.Lagr = Constraint(m.R, m.I, rule=lagr_rule)

    # Alpha signs
    def alpha1_LB_rule(_m, rr, ii):
        return _m.alpha_L[rr, ii] <= 0.0
    def alpha4_LB_rule(_m, uu, ii):
        return _m.alpha_upt[uu, ii] <= 0.0
    def alpha1_UB_rule(_m, rr, ii):
        return _m.alpha_U[rr, ii] >= 0.0
    m.alpha1_LB = Constraint(m.R, m.I, rule=alpha1_LB_rule)
    m.alpha4_LB = Constraint(m.UP, m.I, rule=alpha4_LB_rule)
    m.alpha1_UB = Constraint(m.R, m.I, rule=alpha1_UB_rule)

    # ---------------------------------------------
    # 3.20) Nonlinear constraints (ODEs, uptake LBs)
    # ---------------------------------------------
    # ODEs (mass balances)
    def m1_rule(_m, ii, jj):
        return _m.cdot[1, ii, jj] == _m.v[obj, ii] * _m.c[1, ii, jj]
    def m2_rule(_m, ii, jj):
        return _m.cdot[2, ii, jj] == -0.180156 * _m.vg[ii] * _m.c[1, ii, jj]
    def m3_rule(_m, ii, jj):
        return _m.cdot[3, ii, jj] == -0.15013 * _m.vz[ii] * _m.c[1, ii, jj]
    def m4_rule(_m, ii, jj):
        return _m.cdot[4, ii, jj] == 0.04607 * _m.v[eth, ii] * _m.c[1, ii, jj]
    m.m1 = Constraint(m.PH, m.J, rule=m1_rule)
    m.m2 = Constraint(m.PH, m.J, rule=m2_rule)
    m.m3 = Constraint(m.PH, m.J, rule=m3_rule)
    m.m4 = Constraint(m.PH, m.J, rule=m4_rule)

    # Uptake lower bounds coupling to vg/vz
    def v_LB_g_rule(_m, ii):
        return -_m.v[glu, ii] * _m.vs[glu] - _m.vg[ii] <= 0.0
    def v_LB_z_rule(_m, ii):
        return -_m.v[xyl, ii] * _m.vs[xyl] - _m.vz[ii] <= 0.0
    m.v_LB_g = Constraint(m.I, rule=v_LB_g_rule)
    m.v_LB_z = Constraint(m.I, rule=v_LB_z_rule)

    # ---------------------------------------------
    # 3.21) Complementarity penalty products
    # ---------------------------------------------
    if not drop_fo_products:
        def FO1_rule(_m, rr, ii):
            return _m.FO_L[rr, ii] == (_m.v[rr, ii] * _m.vs[rr] - _m.vlb[rr]) * _m.alpha_L[rr, ii]
        def FO2_rule(_m, rr, ii):
            return _m.FO_U[rr, ii] == (_m.v[rr, ii] * _m.vs[rr] - _m.vub[rr]) * _m.alpha_U[rr, ii]
        def FO3_rule(_m, ii):
            return _m.FO_upt[1, ii] == (-_m.v[glu, ii] * _m.vs[glu] - _m.vg[ii]) * _m.alpha_upt[1, ii]
        def FO4_rule(_m, ii):
            return _m.FO_upt[2, ii] == (-_m.v[xyl, ii] * _m.vs[xyl] - _m.vz[ii]) * _m.alpha_upt[2, ii]
        m.FO1 = Constraint(m.R, m.I, rule=FO1_rule)
        m.FO2 = Constraint(m.R, m.I, rule=FO2_rule)
        m.FO3 = Constraint(m.I, rule=FO3_rule)
        m.FO4 = Constraint(m.I, rule=FO4_rule)

    # ---------------------------------------------
    # 3.22) SSE equality: FO == sum of squared errors (Julia parity)
    # ---------------------------------------------
    def sse_rule(_m):
        return _m.FO == sum((_m.data[c, i, j] - _m.c[c, i, j])**2 for c in _m.C for i in _m.PH for j in _m.J)
    m.SSE = Constraint(rule=sse_rule)

    print(f"[INFO] Model built. Build time: {time.time()-t0:.2f}s")
    return m


# ------------------------------
# Solve and export results
# ------------------------------
def solve_and_export(model: ConcreteModel, solver_path: str | None = None, max_iter: int = 5000, print_level: int = 5, use_lbfgs: bool = False, tol: float = 1e-4, acceptable_tol: float = 1e-2, timings: bool = False):
    # =============================================
    # 4) Solver options (Ipopt) and export routines
    # =============================================
    solver = SolverFactory("ipopt", executable=solver_path) if solver_path else SolverFactory("ipopt")

    # Options analogous to Julia, avoiding HSL-specific linear_solver
    solver.options["tol"] = float(tol)
    solver.options["acceptable_tol"] = float(acceptable_tol)
    solver.options["acceptable_iter"] = 5
    solver.options["print_level"] = int(print_level)
    solver.options["max_iter"] = int(max_iter)
    if use_lbfgs:
        solver.options["hessian_approximation"] = "limited-memory"
    solver.options["linear_solver"] = "mumps"
    solver.options["mu_strategy"] = "adaptive"
    solver.options["nlp_scaling_method"] = "gradient-based"
    if timings:
        solver.options["print_timing_statistics"] = "yes"
        solver.options["print_user_options"] = "yes"
    # Wall time can be added via CLI in the future if needed
    # Warm-start hint (Pyomo/Ipopt sometimes honors initial values by default)
    solver.options["warm_start_init_point"] = "yes"

    t0 = time.time()
    results = solver.solve(model, tee=True)
    print(f"[INFO] Solve finished in {time.time()-t0:.2f}s")

    # Collect results
    # c values
    c_rows = []
    for c in model.C:
        for i in model.PH:
            for j in model.J:
                c_rows.append({"state": int(c), "fe": int(i), "cp": int(j), "value": float(value(model.c[c, i, j]))})
    pd.DataFrame(c_rows).to_csv(os.path.join(RESULTS_DIR, "c_values.csv"), index=False)

    # v values
    v_rows = []
    for r in model.R:
        for i in model.I:
            v_rows.append({"rxn": int(r), "fe": int(i), "value": float(value(model.v[r, i]))})
    pd.DataFrame(v_rows).to_csv(os.path.join(RESULTS_DIR, "v_values.csv"), index=False)

    # Parameters in original scale
    t_rows = []
    for p in model.P:
        val = float(value(model.teta[p]))
        t_rows.append({"p": int(p), "teta_log": val, "teta": math.exp(val)})
    pd.DataFrame(t_rows).to_csv(os.path.join(RESULTS_DIR, "teta_values.csv"), index=False)

    # Residual (FO variable equals SSE)
    with open(os.path.join(RESULTS_DIR, "FO.txt"), "w", encoding="utf-8") as f:
        f.write(f"FO (SSE) = {float(value(model.FO))}\n")

    return results


# =============================================
# 5) Nonlinearity audit utility (optional)
# =============================================
def _audit_nonlinearity(model: ConcreteModel):
    """Report which constraints are linear vs nonlinear using standard repn."""
    print("[AUDIT] Nonlinearity audit (standard representation)")

    def _iter_con(comp):
        try:
            if comp.is_indexed():
                return list(comp.values())
        except Exception:
            pass
        return [comp]

    def _is_nonlinear_expr(expr):
        repn = generate_standard_repn(expr, compute_values=False)
        quad = getattr(repn, "quadratic_vars", [])
        nlexpr = getattr(repn, "nonlinear_expr", None)
        return bool(quad) or (nlexpr is not None)

    comps = []
    # Collocation
    comps.append(("coll_c_n", _iter_con(model.coll_c_n)))
    comps.append(("coll_c_0", _iter_con(model.coll_c_0)))
    # Linear groups
    comps.append(("Sc", _iter_con(model.Sc)))
    comps.append(("v_UB", _iter_con(model.v_UB)))
    comps.append(("v_LB", _iter_con(model.v_LB)))
    comps.append(("c_LB", _iter_con(model.c_LB)))
    comps.append(("Lagr", _iter_con(model.Lagr)))
    # Parameter bounds (explicit inequalities)
    if hasattr(model, "TETA_LB"):
        comps.append(("TETA_LB", _iter_con(model.TETA_LB)))
    if hasattr(model, "TETA_UB"):
        comps.append(("TETA_UB", _iter_con(model.TETA_UB)))
    if hasattr(model, "MFE1"):
        comps.append(("MFE1", [model.MFE1]))
    for nm in ("MFE3", "MFE4", "MFE5"):
        if hasattr(model, nm):
            comps.append((nm, _iter_con(getattr(model, nm))))
    # Alpha sign constraints
    comps.append(("alpha1_LB", _iter_con(model.alpha1_LB)))
    comps.append(("alpha1_UB", _iter_con(model.alpha1_UB)))
    comps.append(("alpha4_LB", _iter_con(model.alpha4_LB)))
    # Nonlinear groups
    comps.append(("m1", _iter_con(model.m1)))
    comps.append(("m2", _iter_con(model.m2)))
    comps.append(("m3", _iter_con(model.m3)))
    comps.append(("m4", _iter_con(model.m4)))
    comps.append(("v_LB_g", _iter_con(model.v_LB_g)))
    comps.append(("v_LB_z", _iter_con(model.v_LB_z)))
    if hasattr(model, "FO1"):
        comps.append(("FO1", _iter_con(model.FO1)))
    if hasattr(model, "FO2"):
        comps.append(("FO2", _iter_con(model.FO2)))
    if hasattr(model, "FO3"):
        comps.append(("FO3", _iter_con(model.FO3)))
    if hasattr(model, "FO4"):
        comps.append(("FO4", _iter_con(model.FO4)))
    # Include vg/vz definition constraints if in var+constraint mode
    if hasattr(model, "VG_DEF"):
        comps.append(("VG_DEF", _iter_con(model.VG_DEF)))
    if hasattr(model, "VZ_DEF"):
        comps.append(("VZ_DEF", _iter_con(model.VZ_DEF)))
    comps.append(("SSE", [model.SSE]))

    # Build expected linearity only for present groups
    present = {name for name, _ in comps}
    expect_nl = {}
    for nm_lin in ("coll_c_n","coll_c_0","Sc","v_UB","v_LB","c_LB","Lagr","MFE1","MFE3","MFE4","MFE5","alpha1_LB","alpha1_UB","alpha4_LB","TETA_LB","TETA_UB"):
        if nm_lin in present:
            expect_nl[nm_lin] = False
    for nm_nl in ("m1","m2","m3","m4","v_LB_g","v_LB_z","FO1","FO2","FO3","FO4","VG_DEF","VZ_DEF","SSE"):
        if nm_nl in present:
            expect_nl[nm_nl] = True

    summary = []
    mismatches = []
    for name, lst in comps:
        lin_count = 0
        nl_count = 0
        for c in lst:
            if c is None:
                continue
            try:
                body = c.body
            except Exception:
                continue
            try:
                is_nl = _is_nonlinear_expr(body)
            except Exception:
                # If repn fails, treat as nonlinear to be conservative
                is_nl = True
            if is_nl:
                nl_count += 1
            else:
                lin_count += 1
        summary.append((name, lin_count, nl_count))
        exp = expect_nl.get(name, None)
        if exp is not None:
            if exp and lin_count > 0:
                mismatches.append((name, "expected nonlinear", f"linear={lin_count}, nonlinear={nl_count}"))
            if (not exp) and nl_count > 0:
                mismatches.append((name, "expected linear", f"linear={lin_count}, nonlinear={nl_count}"))

    print("[AUDIT] Constraint linearity summary:")
    for name, lin, nl in summary:
        print(f"  - {name:9s}: linear={lin:7d}, nonlinear={nl:7d}")
    if mismatches:
        print("[AUDIT] Mismatches found:")
        for m in mismatches:
            print(f"   * {m[0]} -> {m[1]} | {m[2]}")
    else:
        print("[AUDIT] No mismatches vs expectations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pyomo MPCC model (Yeast_83)")
    parser.add_argument("--ipopt", dest="ipopt", default=None, help="Path to ipopt executable (optional if on PATH)")
    parser.add_argument("--nfe", dest="nfe", type=int, default=12, help="Number of finite elements (default 12)")
    parser.add_argument("--th", dest="th", type=float, default=22.0, help="Time horizon (default 22.0)")
    parser.add_argument("--max_iter", dest="max_iter", type=int, default=500, help="Ipopt max iterations (default 500)")
    parser.add_argument("--print_level", dest="print_level", type=int, default=3, help="Ipopt print_level (default 3)")
    parser.add_argument("--tol", dest="tol", type=float, default=1e-4, help="Ipopt tol (default 1e-4)")
    parser.add_argument("--acceptable_tol", dest="acceptable_tol", type=float, default=1e-2, help="Ipopt acceptable_tol (default 1e-2)")
    parser.add_argument("--timings", dest="timings", action="store_true", help="Enable Ipopt timing statistics and user options printout")
    parser.add_argument("--no_solve", dest="no_solve", action="store_true", help="Only build model; skip solve")
    parser.add_argument("--check_init", dest="check_init", action="store_true", help="Evaluate constraints at initial point and exit")
    parser.add_argument("--phi1", dest="phi1", type=float, default=1.0, help="Penalty weight phi1 (default 1.0)")
    parser.add_argument("--phi2", dest="phi2", type=float, default=1.0, help="Penalty weight phi2 (default 1.0)")
    parser.add_argument("--phi3", dest="phi3", type=float, default=1.0, help="Penalty weight phi3 (default 1.0)")
    parser.add_argument("--fix_hv", dest="fix_hv", action="store_true", help="Fix finite-element durations hv to h to simplify timing constraints")
    parser.add_argument("--lbfgs", dest="lbfgs", action="store_true", help="Use limited-memory Hessian approximation (default: exact Hessian)")
    parser.add_argument("--report", dest="report", action="store_true", help="Only build and print model size report (implies --no_solve)")
    parser.add_argument("--audit_nl", dest="audit_nl", action="store_true", help="Audit linear vs nonlinear constraints and exit")
    parser.add_argument("--teta_start_mode", dest="teta_start_mode", choices=["julia", "clamped"], default="julia", help="Initial values for teta: 'julia' sets 0.0 (Julia-like), 'clamped' uses teta0 clamped to [LB,UB]")
    parser.add_argument("--drop_fo_products", dest="drop_fo_products", action="store_true", help="Drop FO1..FO4 complementarity product constraints and remove penalty terms from objective")
    parser.add_argument("--vgvz_mode", dest="vgvz_mode", choices=["expr", "varc"], default="expr", help="Formulation for vg/vz: 'expr' as Expressions, 'varc' as Var+Constraint definitions")
    args = parser.parse_args()

    try:
        m = build_model(
            nfe=args.nfe,
            th=args.th,
            phi1_val=args.phi1,
            phi2_val=args.phi2,
            phi3_val=args.phi3,
            fix_hv=args.fix_hv,
            teta_start_mode=args.teta_start_mode,
            drop_fo_products=args.drop_fo_products,
            vgvz_mode=args.vgvz_mode,
        )
    except Exception as e:
        print("[ERROR] Building model failed:", e)
        sys.exit(1)

    if args.audit_nl:
        # Build-only audit and exit
        try:
            _audit_nonlinearity(m)
        except Exception as e:
            print(f"[ERROR] NL audit failed: {e}")
            sys.exit(2)
        sys.exit(0)

    if args.report:
        # Always skip solve when reporting
        try:
            from pyomo.util.model_size import build_model_size_report
            rep = build_model_size_report(m)
            text = str(rep)
            path = os.path.join(RESULTS_DIR, "model_report.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            print(text)
            print(f"[REPORT] Saved model_report.txt to {path}")
        except Exception as e:
            print(f"[WARN] Model size report failed: {e}")
        print("[INFO] Skipping solve due to --report.")
        sys.exit(0)

    if args.no_solve or args.check_init:
        # Diagnostics: evaluate constraints at initial point
        try:
            import math as _math
            from pyomo.environ import value as _value

            def _is_bad(val):
                if val is None:
                    return True
                try:
                    v = float(val)
                except Exception:
                    return True
                return not _math.isfinite(v)

            bad = []
            max_resid = 0.0

            comps = []
            # Include MFE constraints if present (when hv not fixed)
            if hasattr(m, "MFE1"):
                comps.append(("MFE1", [m.MFE1]))
            if hasattr(m, "MFE3"):
                comps.append(("MFE3", [m.MFE3[i] for i in m.I]))
            if hasattr(m, "MFE4"):
                comps.append(("MFE4", [m.MFE4[i] for i in m.I]))
            if hasattr(m, "MFE5"):
                comps.append(("MFE5", [m.MFE5[i] for i in m.I]))
            comps.extend([
                ("Sc", [m.Sc[k, i] for k in m.M for i in m.I]),
                ("v_UB", [m.v_UB[r, i] for r in m.R for i in m.I]),
                ("v_LB", [m.v_LB[r, i] for r in m.R for i in m.I]),
                ("c_LB", [m.c_LB[c, i, j] for c in m.C for i in m.I for j in m.J]),
                ("Lagr", [m.Lagr[r, i] for r in m.R for i in m.I]),
                ("m1", [m.m1[i, j] for i in m.PH for j in m.J]),
                ("m2", [m.m2[i, j] for i in m.PH for j in m.J]),
                ("m3", [m.m3[i, j] for i in m.PH for j in m.J]),
                ("m4", [m.m4[i, j] for i in m.PH for j in m.J]),
                ("v_LB_g", [m.v_LB_g[i] for i in m.I]),
                ("v_LB_z", [m.v_LB_z[i] for i in m.I]),
                ("FO1", [m.FO1[r, i] for r in m.R for i in m.I]),
                ("FO2", [m.FO2[r, i] for r in m.R for i in m.I]),
                ("FO3", [m.FO3[i] for i in m.I]),
                ("FO4", [m.FO4[i] for i in m.I]),
                ("SSE", [m.SSE]),
            ])

            def _residual(c):
                body = c.body
                try:
                    val = _value(body)
                except Exception:
                    return float('nan')
                if c.equality:
                    res = abs(val - _value(c.upper))
                else:
                    # Inequalities are of the form body <= 0 or >= 0
                    if c.has_ub():
                        res = max(0.0, val - _value(c.upper))
                    elif c.has_lb():
                        res = max(0.0, _value(c.lower) - val)
                    else:
                        res = 0.0
                return res

            for name, lst in comps:
                count = 0
                for c in lst:
                    r = _residual(c)
                    if _is_bad(r):
                        bad.append(name)
                        break
                    max_resid = max(max_resid, float(r))
                    count += 1
                print(f"[CHECK] {name}: {count} checked.")

            print(f"[CHECK] Max residual at initial point: {max_resid:.3e}")
            if bad:
                print(f"[CHECK] Non-finite residuals detected in: {sorted(set(bad))}")
        except Exception as e:
            print(f"[WARN] Initial-point diagnostics failed: {e}")

        if args.no_solve:
            print("[INFO] Skipping solve as requested (--no_solve).")
            sys.exit(0)

    try:
        solve_and_export(
            m,
            solver_path=args.ipopt,
            max_iter=args.max_iter,
            print_level=args.print_level,
            use_lbfgs=args.lbfgs,
            tol=args.tol,
            acceptable_tol=args.acceptable_tol,
            timings=args.timings,
        )
        print(f"Results written to: {RESULTS_DIR}")
    except Exception as e:
        print("[ERROR] Solve failed:", e)
        sys.exit(2)
