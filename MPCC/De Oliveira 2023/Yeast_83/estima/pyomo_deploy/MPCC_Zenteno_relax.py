#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zenteno ODE model + relaxed MPCC/dFBA coupling (Radau collocation),
mirroring pyomo_deploy/pyomo_dc_dfba_mpcc.py formulation:
- Complementarity handled via FO_L/FO_U (bounds) and FO_upt for sugar uptakes (inequalities)
- Lagrangian stationarity with tiny ridge (w) and pFBA-like d vector (only growth reaction weighted)
- No equality ties at FE-end; only uptake inequalities against Zenteno FE-end rates rG_fe, rF_fe
- FO (data-fit SSE) objective plus linear complementarity terms (same signs as pyomo_dc_dfba_mpcc.py)

This script is intended to serve as "MPCC_Zenteno_relax" for comparison with the tighter
MPCC_Zenteno implementation that uses FE-end equality ties.
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ESTIMA_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# IO helpers (reuse shapes/behavior)
# -----------------------------

def load_S_lb_ub(S_path=None, lb_path=None, ub_path=None):
    if S_path is None:
        S_path = os.path.join(ESTIMA_DIR, "S.csv")
    if lb_path is None:
        lb_path = os.path.join(ESTIMA_DIR, "lb.csv")
    if ub_path is None:
        ub_path = os.path.join(ESTIMA_DIR, "ub.csv")
    S = np.loadtxt(S_path, delimiter=",")
    lb = np.loadtxt(lb_path, delimiter=",")
    ub = np.loadtxt(ub_path, delimiter=",")
    if lb.ndim > 1:
        lb = lb[:, 0]
    if ub.ndim > 1:
        ub = ub[:, 0]
    return S, lb, ub


def load_data_3d(data_csv=None, nc=5, ph=12, ncp=3, mode: str = "stacked"):
    if data_csv is None:
        data_csv = os.path.join(BASE_DIR, "data_long.csv")
    if not os.path.exists(data_csv):
        return np.zeros((nc, ph, ncp), dtype=float)
    df = pd.read_csv(data_csv)
    cols = [c.lower() for c in df.columns]
    arr = np.zeros((nc, ph, ncp), dtype=float)
    if set(["l", "i", "j", "value"]).issubset(set(cols)):
        colmap = {c.lower(): c for c in df.columns}
        df_sorted = df.sort_values([colmap["l"], colmap["i"], colmap["j"]], kind="mergesort")
        for _, row in df_sorted.iterrows():
            l = int(row[colmap["l"]]) - 1
            i = int(row[colmap["i"]]) - 1
            j = int(row[colmap["j"]]) - 1
            if 0 <= l < nc and 0 <= i < ph and 0 <= j < ncp:
                arr[l, i, j] = float(row[colmap["value"]])
        return arr
    if set(["state", "fe", "cp", "value"]).issubset(set(cols)):
        colmap = {c.lower(): c for c in df.columns}
        if (mode or "").lower() == "indexed":
            df_sorted = df.sort_values([colmap["state"], colmap["fe"], colmap["cp"]], kind="mergesort")
            for _, row in df_sorted.iterrows():
                l = int(row[colmap["state"]]) - 1
                i = int(row[colmap["fe"]]) - 1
                j = int(row[colmap["cp"]]) - 1
                if 0 <= l < nc and 0 <= i < ph and 0 <= j < ncp:
                    arr[l, i, j] = float(row[colmap["value"]])
            return arr
        else:
            df_sorted = df.sort_values([colmap["state"], colmap["fe"], colmap["cp"]], kind="mergesort")
            for l1 in sorted(set(int(v) for v in df_sorted[colmap["state"]].values)):
                sub = df_sorted[df_sorted[colmap["state"]] == l1]
                vals = sub[colmap["value"]].astype(float).to_numpy()
                idx = 0
                for i in range(ph):
                    for j in range(ncp):
                        if idx < len(vals):
                            arr[l1 - 1, i, j] = float(vals[idx]); idx += 1
            return arr
    vals = df.to_numpy().astype(float).flatten()
    if vals.size == nc * ph * ncp:
        return vals.reshape((nc, ph, ncp), order="C")
    return np.zeros((nc, ph, ncp), dtype=float)


# -----------------------------
# Zenteno ODE core (no MPCC ties)
# -----------------------------

def build_zenteno_core(nfe=12, ncp=3, th=22.0, var_h=True, data3d=None, T_const=293.15, alpha_switch=0.5):
    nc = 5  # X,N,G,F,E
    ph = nfe

    colmat = np.array([
        [0.19681547722366,  -0.06553542585020,  0.02377097434822],
        [0.39442431473909,   0.29207341166523, -0.04154875212600],
        [0.37640306270047,   0.51248582618842,  0.11111111111111],
    ])
    h = th / nfe

    X0, N0, G0, F0, E0 = 0.5, 0.14, 110.0, 110.0, 0.0
    c0 = [X0, N0, G0, F0, E0]

    nominal = {
        'mu0': 0.141665,
        'betaG0': 1.41182,
        'betaF0': 8.49482,
        'Kn0': 0.226882,
        'Kg0': 3.1514,
        'Kf0': 2.97625,
        'Kig0': 29.5276,
        'Kie0': 2.99809,
        'Kd0': 0.0000311736,
        'Yxn': 9.80576,
        'Yxg': 0.394345,
        'Yxf': 0.18622,
        'Yeg': 0.14133,
        'Yef': 0.96932,
    }

    def _lb(name):
        return np.log(max(1e-12, 0.5 * nominal[name]))
    def _ub(name):
        return np.log(max(2e-12, 2.0 * nominal[name]))

    m = pyo.ConcreteModel()
    m.L = pyo.RangeSet(1, nc)
    m.I = pyo.RangeSet(1, ph)
    m.J = pyo.RangeSet(1, ncp)

    m.hv = pyo.Var(m.I, domain=pyo.Reals, initialize=h)
    m.MFE1 = pyo.Constraint(expr=sum(m.hv[i] for i in m.I) == th)
    m.MFE3 = pyo.Constraint(m.I, rule=lambda mdl, i: mdl.hv[i] >= 0.0)
    m.MFE4 = pyo.Constraint(m.I, rule=lambda mdl, i: mdl.hv[i] >= (1.0 - (1.0 if var_h else 0.0)) * h)
    m.MFE5 = pyo.Constraint(m.I, rule=lambda mdl, i: mdl.hv[i] <= (1.0 + (1.0 if var_h else 0.0)) * h)

    m.c = pyo.Var(m.L, m.I, m.J, domain=pyo.Reals, initialize=lambda mdl, l, i, j: c0[l-1])
    m.cdot = pyo.Var(m.L, m.I, m.J, domain=pyo.Reals, initialize=0.0)
    m.c_LB = pyo.Constraint(m.L, m.I, m.J, rule=lambda mdl, l, i, j: -mdl.c[l, i, j] <= 0.0)

    def _coll_c_n(mdl, l, i, j):
        if i == 1:
            return pyo.Constraint.Skip
        return mdl.c[l, i, j] == mdl.c[l, i-1, ncp] + mdl.hv[i] * sum(colmat[j-1, k-1] * mdl.cdot[l, i, k] for k in mdl.J)
    m.coll_c_n = pyo.Constraint(m.L, m.I, m.J, rule=_coll_c_n)

    def _coll_c_0(mdl, l, j):
        return mdl.c[l, 1, j] == c0[l-1] + mdl.hv[1] * sum(colmat[j-1, k-1] * mdl.cdot[l, 1, k] for k in mdl.J)
    m.coll_c_0 = pyo.Constraint(m.L, m.J, rule=_coll_c_0)

    m.PN = pyo.Set(initialize=list(nominal.keys()), ordered=True)
    m.theta = pyo.Var(m.PN, initialize=lambda mdl, n: np.log(nominal[n]),
                      bounds=lambda mdl, n: (_lb(n), _ub(n)))

    def P(name):
        return pyo.exp(m.theta[name])

    m.T = pyo.Param(initialize=float(T_const), mutable=False)
    R = 8.314
    eps = 1e-9

    def _X_i(mdl, i): return mdl.c[1, i, ncp]
    def _N_i(mdl, i): return mdl.c[2, i, ncp]
    def _G_i(mdl, i): return mdl.c[3, i, ncp]
    def _F_i(mdl, i): return mdl.c[4, i, ncp]
    def _E_i(mdl, i): return mdl.c[5, i, ncp]
    m.Xe = pyo.Expression(m.I, rule=_X_i)
    m.Ne = pyo.Expression(m.I, rule=_N_i)
    m.Ge = pyo.Expression(m.I, rule=_G_i)
    m.Fe = pyo.Expression(m.I, rule=_F_i)
    m.Ee = pyo.Expression(m.I, rule=_E_i)

    # FE-end fractions
    m.phiG_fe = pyo.Expression(m.I, rule=lambda mdl, i: mdl.Ge[i] / (mdl.Ge[i] + mdl.Fe[i] + eps))
    m.phiF_fe = pyo.Expression(m.I, rule=lambda mdl, i: mdl.Fe[i] / (mdl.Ge[i] + mdl.Fe[i] + eps))

    def _mu_T(mdl):
        T = mdl.T; return pyo.exp(59453 * (T - 300) / (300 * R * T))
    def _Kg_T(mdl):
        T = mdl.T; return pyo.exp(46055 * (T - 293.15) / (293.15 * R * T))
    def _b_T(mdl):
        T = mdl.T; return pyo.exp(11000 * (T - 296.15) / (296.15 * R * T))
    def _mr_T(mdl):
        T = mdl.T; return 0.01 * pyo.exp(37681 * (T - 293.30) / (293.30 * R * T))

    # Pointwise kinetics
    m.mu_j = pyo.Expression(m.I, m.J, rule=lambda mdl, i, j: P('mu0') * _mu_T(mdl) * ( mdl.c[2, i, j] / ( mdl.c[2, i, j] + P('Kn0') * _Kg_T(mdl) + eps ) ))
    m.betaG_j = pyo.Expression(m.I, m.J, rule=lambda mdl, i, j: P('betaG0') * _b_T(mdl) *
                                ( mdl.c[3, i, j] / ( mdl.c[3, i, j] + P('Kg0') * _Kg_T(mdl) + eps ) ) *
                                ( P('Kie0') * _Kg_T(mdl) / ( mdl.c[5, i, j] + P('Kie0') * _Kg_T(mdl) + eps ) ))
    m.betaF_j = pyo.Expression(m.I, m.J, rule=lambda mdl, i, j: P('betaF0') * _b_T(mdl) *
                                ( mdl.c[4, i, j] / ( mdl.c[4, i, j] + P('Kf0') * _Kg_T(mdl) + eps ) ) *
                                ( P('Kig0') * _Kg_T(mdl) / ( mdl.c[3, i, j] + P('Kig0') * _Kg_T(mdl) + eps ) ) *
                                ( P('Kie0') * _Kg_T(mdl) / ( mdl.c[5, i, j] + P('Kie0') * _Kg_T(mdl) + eps ) ))
    m.mrate = pyo.Expression(rule=lambda mdl: _mr_T(mdl))

    # FE-end kinetics for coupling (no equality ties)
    m.mu_fe = pyo.Expression(m.I, rule=lambda mdl, i: P('mu0') * _mu_T(mdl) * ( mdl.Ne[i] / ( mdl.Ne[i] + P('Kn0') * _Kg_T(mdl) + eps ) ))
    m.betaG_fe = pyo.Expression(m.I, rule=lambda mdl, i: P('betaG0') * _b_T(mdl) *
                                 ( mdl.Ge[i] / ( mdl.Ge[i] + P('Kg0') * _Kg_T(mdl) + eps ) ) *
                                 ( P('Kie0') * _Kg_T(mdl) / ( mdl.Ee[i] + P('Kie0') * _Kg_T(mdl) + eps ) ))
    m.betaF_fe = pyo.Expression(m.I, rule=lambda mdl, i: P('betaF0') * _b_T(mdl) *
                                 ( mdl.Fe[i] / ( mdl.Fe[i] + P('Kf0') * _Kg_T(mdl) + eps ) ) *
                                 ( P('Kig0') * _Kg_T(mdl) / ( mdl.Ge[i] + P('Kig0') * _Kg_T(mdl) + eps ) ) *
                                 ( P('Kie0') * _Kg_T(mdl) / ( mdl.Ee[i] + P('Kie0') * _Kg_T(mdl) + eps ) ))

    # Pointwise fractions for maintenance
    m.phiG_j = pyo.Expression(m.I, m.J, rule=lambda mdl, i, j: mdl.c[3, i, j] / (mdl.c[3, i, j] + mdl.c[4, i, j] + eps))
    m.phiF_j = pyo.Expression(m.I, m.J, rule=lambda mdl, i, j: mdl.c[4, i, j] / (mdl.c[3, i, j] + mdl.c[4, i, j] + eps))

    # ODEs as in Zenteno core
    def _dX(mdl, i, j):
        # net growth using pointwise kinetics (no equality tie)
        Kd_ij = _Kd_j(mdl, i, j)
        return mdl.cdot[1, i, j] == ( mdl.mu_j[i, j] - Kd_ij ) * mdl.c[1, i, j]
    def _dN(mdl, i, j):
        return mdl.cdot[2, i, j] == -( mdl.mu_j[i, j] / P('Yxn') ) * mdl.c[1, i, j]
    def _dG(mdl, i, j):
        return mdl.cdot[3, i, j] == -( ( mdl.mu_j[i, j] / P('Yxg') ) + ( mdl.betaG_j[i, j] / P('Yeg') ) + mdl.mrate * mdl.phiG_j[i, j] ) * mdl.c[1, i, j]
    def _dF(mdl, i, j):
        return mdl.cdot[4, i, j] == -( ( mdl.mu_j[i, j] / P('Yxf') ) + ( mdl.betaF_j[i, j] / P('Yef') ) + mdl.mrate * mdl.phiF_j[i, j] ) * mdl.c[1, i, j]
    def _dE(mdl, i, j):
        return mdl.cdot[5, i, j] == ( mdl.betaG_j[i, j] + mdl.betaF_j[i, j] ) * mdl.c[1, i, j]

    # Death function helpers (same as in MPCC_Zenteno)
    def _Td(mdl, i, j):
        E_ij = mdl.c[5, i, j]
        return -0.0001 * E_ij**3 + 0.0049 * E_ij**2 - 0.1279 * E_ij + 315.89
    def _Kd_j(mdl, i, j):
        T = mdl.T; Rloc = 8.314
        Td_ij = _Td(mdl, i, j)
        s = 0.5 * (1.0 + pyo.tanh(float(alpha_switch) * (T - Td_ij)))
        base = P('Kd0') * pyo.exp(0.0415 * mdl.c[5, i, j] + (130000.0 * (T - 305.65)) / (305.65 * Rloc * T))
        return base * s

    m.ode_X = pyo.Constraint(m.I, m.J, rule=_dX)
    m.ode_N = pyo.Constraint(m.I, m.J, rule=_dN)
    m.ode_G = pyo.Constraint(m.I, m.J, rule=_dG)
    m.ode_F = pyo.Constraint(m.I, m.J, rule=_dF)
    m.ode_E = pyo.Constraint(m.I, m.J, rule=_dE)

    # FO (data-fit) — keep as in Zenteno
    if data3d is None:
        data3d = np.zeros((nc, ph, ncp), dtype=float)
    m.FO = pyo.Var(domain=pyo.Reals, initialize=1.0)
    m.FO_def = pyo.Constraint(rule=lambda mdl: mdl.FO == sum( (float(data3d[l-1, i-1, j-1]) - mdl.c[l, i, j])**2 for l in mdl.L for i in mdl.I for j in mdl.J ))

    # FE-end sugar demands used in uptake inequalities
    def _rG(mdl, i):
        return ( mdl.mu_fe[i] / pyo.exp(mdl.theta['Yxg']) ) + ( mdl.betaG_fe[i] / pyo.exp(mdl.theta['Yeg']) ) + ( mdl.mrate * mdl.phiG_fe[i] )
    def _rF(mdl, i):
        return ( mdl.mu_fe[i] / pyo.exp(mdl.theta['Yxf']) ) + ( mdl.betaF_fe[i] / pyo.exp(mdl.theta['Yef']) ) + ( mdl.mrate * mdl.phiF_fe[i] )
    m.rG = pyo.Expression(m.I, rule=_rG)
    m.rF = pyo.Expression(m.I, rule=_rF)

    return m


# -----------------------------
# MPCC (relaxed) builder around Zenteno core
# -----------------------------

def build_model_relax(S, lb, ub, data3d, nfe=12, ncp=3, th=22.0, var_h=True, ridge_w: float = 1e-20):
    m = build_zenteno_core(nfe=nfe, ncp=ncp, th=th, var_h=var_h, data3d=data3d)

    # Indices of key reactions (1-based, consistent with existing scripts)
    eth = 2630
    obj = 3414
    glu = 2588
    fru = 2583

    nm, nv = S.shape

    # Small pFBA ridge and objective weights (mirror pyomo_dc_dfba_mpcc.py)
    # Allow overriding ridge via CLI to improve curvature if desired
    w = float(ridge_w)
    omega = 1e2
    phi1 = 1.0
    phi2 = 1.0
    phi3 = 1.0

    # Enforce O2 and ATP bounds as in relaxed script (keep lb/ub arrays local)
    o2 = 2816
    ATP = 3415
    lb = lb.copy(); ub = ub.copy()
    if 1 <= o2 <= nv:
        lb[o2-1] = 0.0; ub[o2-1] = 0.0
    if 1 <= ATP <= nv:
        lb[ATP-1] = 0.0

    # Sets (reusing core sets for time), add reaction/metabolite sets
    m.NV = pyo.RangeSet(1, nv)
    m.NM = pyo.RangeSet(1, nm)

    # Sparse S dicts
    rows_nz, cols_nz = np.nonzero(S)
    row_to_cols = {}
    col_to_rows = {}
    S_vals = {}
    for idx in range(len(rows_nz)):
        rr = int(rows_nz[idx]) + 1
        cc = int(cols_nz[idx]) + 1
        val = float(S[rr-1, cc-1])
        S_vals[(rr, cc)] = val
        row_to_cols.setdefault(rr, []).append(cc)
        col_to_rows.setdefault(cc, []).append(rr)

    # Params for bounds
    m.vlb = pyo.Param(m.NV, initialize=lambda mdl, k: float(lb[k-1]), mutable=False)
    m.vub = pyo.Param(m.NV, initialize=lambda mdl, k: float(ub[k-1]), mutable=False)

    # Variables for MPCC
    m.v = pyo.Var(m.NV, m.I, domain=pyo.Reals, initialize=0.0)
    m.lmbda = pyo.Var(m.NM, m.I, domain=pyo.Reals, initialize=0.0)
    m.alpha_U = pyo.Var(m.NV, m.I, domain=pyo.Reals, initialize=0.0)
    m.alpha_L = pyo.Var(m.NV, m.I, domain=pyo.Reals, initialize=0.0)
    m.alpha_upt = pyo.Var([1, 2], m.I, domain=pyo.Reals, initialize=0.0)  # [glu, fru]

    m.FO_U = pyo.Var(m.NV, m.I, domain=pyo.Reals, initialize=0.0)
    m.FO_L = pyo.Var(m.NV, m.I, domain=pyo.Reals, initialize=0.0)
    m.FO_upt = pyo.Var([1, 2], m.I, domain=pyo.Reals, initialize=0.0)

    # Signs for bound multipliers
    m.alphaL_sign = pyo.Constraint(m.NV, m.I, rule=lambda mdl, k, i: mdl.alpha_L[k, i] <= 0.0)
    m.alphaU_sign = pyo.Constraint(m.NV, m.I, rule=lambda mdl, k, i: mdl.alpha_U[k, i] >= 0.0)
    m.alphaUPT_sign = pyo.Constraint([1, 2], m.I, rule=lambda mdl, uu, i: mdl.alpha_upt[uu, i] <= 0.0)

    # Flux bound inequalities
    m.v_UB = pyo.Constraint(m.NV, m.I, rule=lambda mdl, k, i: mdl.v[k, i] - mdl.vub[k] <= 0.0)
    m.v_LB = pyo.Constraint(m.NV, m.I, rule=lambda mdl, k, i: -mdl.v[k, i] + mdl.vlb[k] <= 0.0)

    # Stoichiometric balances S v = 0
    def _Sc(mdl, r, i):
        rr = int(r); nz_cols = row_to_cols.get(rr, [])
        return sum(S_vals[(rr, k)] * mdl.v[k, i] for k in nz_cols) == 0.0
    m.Sc = pyo.Constraint(m.NM, m.I, rule=_Sc)

    # Uptake selector vectors
    up_glu = np.zeros(nv); up_glu[glu-1] = 1.0
    up_fru = np.zeros(nv); up_fru[fru-1] = 1.0

    # pFBA d-vector (encourage positive growth flux)
    d = np.zeros(nv); d[obj-1] = -1.0
    vs = np.ones(nv)

    # Lagrangian stationarity (relaxed style)
    def _Lagr(mdl, k, i):
        rr_list = col_to_rows.get(int(k), [])
        term_S = sum(S_vals[(r, int(k))] * mdl.lmbda[r, i] for r in rr_list)
        term = d[k-1] + w * mdl.v[k, i] * vs[k-1] + mdl.alpha_L[k, i] + mdl.alpha_U[k, i] \
               + up_glu[k-1] * mdl.alpha_upt[1, i] + up_fru[k-1] * mdl.alpha_upt[2, i] \
               + term_S
        return term == 0.0
    m.Lagr = pyo.Constraint(m.NV, m.I, rule=_Lagr)

    # Complementarity products (penalized in objective)
    m.FO_L_def = pyo.Constraint(m.NV, m.I, rule=lambda mdl, k, i: mdl.FO_L[k, i] == (mdl.v[k, i] - mdl.vlb[k]) * mdl.alpha_L[k, i])
    m.FO_U_def = pyo.Constraint(m.NV, m.I, rule=lambda mdl, k, i: mdl.FO_U[k, i] == (mdl.v[k, i] - mdl.vub[k]) * mdl.alpha_U[k, i])
    # Uptake inequalities using Zenteno FE-end rates rG_fe, rF_fe
    m.v_LB_g = pyo.Constraint(m.I, rule=lambda mdl, i: -mdl.v[glu, i] - mdl.rG[i] <= 0.0)
    m.v_LB_f = pyo.Constraint(m.I, rule=lambda mdl, i: -mdl.v[fru, i] - mdl.rF[i] <= 0.0)
    m.FO_upt1_def = pyo.Constraint(m.I, rule=lambda mdl, i: mdl.FO_upt[1, i] == ( -mdl.v[glu, i] - mdl.rG[i] ) * mdl.alpha_upt[1, i])
    m.FO_upt2_def = pyo.Constraint(m.I, rule=lambda mdl, i: mdl.FO_upt[2, i] == ( -mdl.v[fru, i] - mdl.rF[i] ) * mdl.alpha_upt[2, i])

    # Objective (same signs as relaxed reference)
    def _obj(mdl):
        return omega * mdl.FO + sum( sum( -phi1 * mdl.FO_L[k, i] - phi3 * mdl.FO_U[k, i] for k in mdl.NV )
                                     + phi2 * mdl.FO_upt[1, i] + phi2 * mdl.FO_upt[2, i]
                                     for i in mdl.I )
    # Replace any existing objective from core
    try:
        m.del_component('OBJ')
    except Exception:
        try:
            m.del_component(m.OBJ)
        except Exception:
            pass
    m.OBJ = pyo.Objective(rule=_obj, sense=pyo.minimize)

    # Expose indices (helpful for audits)
    m.idx_eth = eth; m.idx_obj = obj; m.idx_glu = glu; m.idx_fru = fru

    return m


# -----------------------------
# Solver helper (Ipopt options like relaxed reference)
# -----------------------------

def solve_model(m, ipopt_max_iter=None, print_solver=False, timings=False, lbfgs=False, max_wall_time=None, tol=None, acceptable_tol=None):
    solver = pyo.SolverFactory("ipopt")
    solver.options["warm_start_init_point"] = "yes"
    solver.options["print_level"] = 5 if print_solver else 3
    solver.options["tol"] = 1e-4
    solver.options["acceptable_iter"] = 5
    solver.options["acceptable_tol"] = 1e-2
    solver.options["linear_solver"] = "mumps"
    solver.options["mu_strategy"] = "adaptive"
    solver.options["nlp_scaling_method"] = "gradient-based"
    if lbfgs:
        solver.options["hessian_approximation"] = "limited-memory"
    # Optional overrides for polishing or custom runs
    if tol is not None:
        solver.options["tol"] = float(tol)
    if acceptable_tol is not None:
        solver.options["acceptable_tol"] = float(acceptable_tol)
    if ipopt_max_iter is not None:
        solver.options["max_iter"] = int(ipopt_max_iter)
    if max_wall_time is not None:
        solver.options["max_wall_time"] = float(max_wall_time)
    if timings:
        solver.options["print_timing_statistics"] = "yes"
        solver.options["print_user_options"] = "yes"

    t0 = time.time()
    res = solver.solve(m, tee=print_solver)
    wall = time.time() - t0
    return res, wall


# -----------------------------
# CLI
# -----------------------------

def _maybe_initialize_relax(m: pyo.ConcreteModel,
                            S: np.ndarray,
                            lb: np.ndarray,
                            ub: np.ndarray,
                            init_flux: str = "zero",
                            init_alpha: str = "none",
                            init_lambda: str = "none",
                            alpha0: float = 1e-2,
                            ridge_w: float = 1e-20):
    """Optional light-touch initialization for relaxed MPCC.
    - init_flux="uptake": set v_glu=-rG_fe, v_fru=-rF_fe at FE; others keep current (default 0)
    - init_alpha="activity": seed alpha_L/alpha_U if v at bounds; alpha_upt if uptake ineq tight
    - init_lambda="ls": least-squares for lambda from stationarity residual at each FE
    Defaults preserve original behavior.
    """
    # Determine sizes from sets
    try:
        nv = max(int(k) for k in m.NV)
    except Exception:
        nv = len([k for k in m.NV])
    try:
        nm = max(int(r) for r in m.NM)
    except Exception:
        nm = len([r for r in m.NM])
    glu = int(getattr(m, 'idx_glu'))
    fru = int(getattr(m, 'idx_fru'))
    obj = int(getattr(m, 'idx_obj'))

    # 1) Flux initialization
    if (init_flux or "").lower() == "uptake":
        for i in list(m.I):
            try:
                rG = float(pyo.value(m.rG[i]))
                rF = float(pyo.value(m.rF[i]))
            except Exception:
                rG, rF = 0.0, 0.0
            # Set sugar uptakes to match FE-end demands (inequalities will allow slack)
            m.v[glu, i].value = -rG
            m.v[fru, i].value = -rF
            # Clip to bounds
            for k in list(m.NV):
                vlb = float(pyo.value(m.vlb[k])); vub = float(pyo.value(m.vub[k]))
                v0 = float(pyo.value(m.v[k, i]))
                if v0 < vlb:
                    m.v[k, i].value = vlb
                elif v0 > vub:
                    m.v[k, i].value = vub

    # 2) Alpha initialization by activity
    if (init_alpha or "").lower() == "activity":
        tol = 1e-8
        for i in list(m.I):
            # Bound alphas
            for k in list(m.NV):
                vlb = float(pyo.value(m.vlb[k])); vub = float(pyo.value(m.vub[k]))
                vk = float(pyo.value(m.v[k, i]))
                aL = -float(alpha0) if abs(vk - vlb) <= tol else 0.0
                aU = +float(alpha0) if abs(vk - vub) <= tol else 0.0
                m.alpha_L[k, i].value = aL
                m.alpha_U[k, i].value = aU
            # Uptake alphas
            rG = float(pyo.value(m.rG[i])); rF = float(pyo.value(m.rF[i]))
            sG = -float(pyo.value(m.v[glu, i])) - rG
            sF = -float(pyo.value(m.v[fru, i])) - rF
            m.alpha_upt[1, i].value = (-float(alpha0) if abs(sG) <= tol else 0.0)
            m.alpha_upt[2, i].value = (-float(alpha0) if abs(sF) <= tol else 0.0)

    # 3) Lambda least-squares init
    if (init_lambda or "").lower() == "ls":
        # Prepare S^T once
        ST = S.T  # shape (nv, nm)
        # Stationarity gradient pieces
        d = np.zeros(nv); d[obj - 1] = -1.0
        vs = np.ones(nv)
        w = float(ridge_w)
        for i in list(m.I):
            # Build rhs per k: -(d + w*v + alpha_L + alpha_U + uptake terms)
            rhs = np.zeros(nv)
            a_upt1 = float(pyo.value(m.alpha_upt[1, i]))
            a_upt2 = float(pyo.value(m.alpha_upt[2, i]))
            for kk in range(1, nv + 1):
                vki = float(pyo.value(m.v[kk, i]))
                aL = float(pyo.value(m.alpha_L[kk, i]))
                aU = float(pyo.value(m.alpha_U[kk, i]))
                upt = (a_upt1 if kk == glu else 0.0) + (a_upt2 if kk == fru else 0.0)
                rhs[kk - 1] = -(d[kk - 1] + w * vki * vs[kk - 1] + aL + aU + upt)
            try:
                lam_i, *_ = np.linalg.lstsq(ST, rhs, rcond=1e-6)
                # Assign
                for r in list(m.NM):
                    m.lmbda[r, i].value = float(lam_i[int(r) - 1])
            except Exception:
                # Fallback: leave zeros
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", default=os.path.join(ESTIMA_DIR, "S.csv"))
    ap.add_argument("--lb", default=os.path.join(ESTIMA_DIR, "lb.csv"))
    ap.add_argument("--ub", default=os.path.join(ESTIMA_DIR, "ub.csv"))
    ap.add_argument("--data_csv", default=os.path.join(BASE_DIR, "data_long.csv"))
    ap.add_argument("--nfe", type=int, default=12)
    ap.add_argument("--ncp", type=int, default=3)
    ap.add_argument("--th", type=float, default=22.0)
    ap.add_argument("--no_var_h", action="store_true")
    ap.add_argument("--ipopt_max_iter", type=int, default=None)
    ap.add_argument("--print_solver", action="store_true")
    ap.add_argument("--timings", action="store_true")
    ap.add_argument("--lbfgs", action="store_true")
    ap.add_argument("--max_wall_time", type=float, default=None)
    # Initialization and ridge controls (optional; defaults preserve current behavior)
    ap.add_argument("--ridge_w", type=float, default=1e-20, help="Ridge weight on v in stationarity (w). Default 1e-20.")
    ap.add_argument("--init_flux", choices=["zero", "uptake"], default="zero", help="Initial v guess: zero or set sugar uptakes to -rG/-rF at FE.")
    ap.add_argument("--init_alpha", choices=["none", "activity"], default="none", help="Initialize alpha_L/alpha_U, alpha_upt by activity heuristics.")
    ap.add_argument("--init_lambda", choices=["none", "ls"], default="none", help="Initialize lambda via least-squares S^T lambda ≈ -(grad + alphas).")
    ap.add_argument("--alpha0", type=float, default=1e-2, help="Magnitude for initial multipliers in activity-based init.")
    # Optional polish stage (exact Hessian, short time) executed in the same process for warm-start
    ap.add_argument("--polish_seconds", type=float, default=0.0, help="If > 0, run a second exact-Hessian polish for this many seconds.")
    ap.add_argument("--polish_tol", type=float, default=None, help="Override Ipopt tol during polish (e.g., 1e-5).")
    ap.add_argument("--polish_acceptable_tol", type=float, default=None, help="Override Ipopt acceptable_tol during polish (e.g., 5e-3).")
    args = ap.parse_args()

    S, lb, ub = load_S_lb_ub(args.S, args.lb, args.ub)
    data3d = load_data_3d(args.data_csv, nc=5, ph=args.nfe, ncp=args.ncp, mode="stacked")

    m = build_model_relax(S, lb, ub, data3d, nfe=args.nfe, ncp=args.ncp, th=args.th, var_h=(not args.no_var_h), ridge_w=args.ridge_w)

    # Optional initialization to accelerate convergence without adding rigidity
    try:
        _maybe_initialize_relax(m, S, lb, ub,
                                init_flux=args.init_flux,
                                init_alpha=args.init_alpha,
                                init_lambda=args.init_lambda,
                                alpha0=args.alpha0,
                                ridge_w=args.ridge_w)
    except Exception as e:
        print(f"[WARN] Initialization skipped due to: {e}")
    res, wall = solve_model(
        m,
        ipopt_max_iter=args.ipopt_max_iter,
        print_solver=args.print_solver,
        timings=args.timings,
        lbfgs=args.lbfgs,
        max_wall_time=args.max_wall_time,
    )

    print("\n=== MPCC_Zenteno_relax solve summary ===")
    print(f"Solver status: {res.solver.status}, termination: {res.solver.termination_condition}")
    print(f"Wall time: {wall:.2f} s")
    try:
        FO_val = float(pyo.value(m.FO))
        print(f"FO (data-fit SSE): {FO_val:.6g}")
    except Exception:
        pass

    # Optional polish stage: warm-start from current values, switch to exact Hessian, tighten tolerances if provided
    try:
        if float(getattr(args, "polish_seconds", 0.0) or 0.0) > 0.0:
            res2, wall2 = solve_model(
                m,
                ipopt_max_iter=args.ipopt_max_iter,
                print_solver=args.print_solver,
                timings=args.timings,
                lbfgs=False,  # exact Hessian for polish
                max_wall_time=float(args.polish_seconds),
                tol=args.polish_tol,
                acceptable_tol=args.polish_acceptable_tol,
            )
            print("\n=== MPCC_Zenteno_relax polish summary ===")
            print(f"Solver status: {res2.solver.status}, termination: {res2.solver.termination_condition}")
            print(f"Wall time (polish): {wall2:.2f} s")
            try:
                FO_val2 = float(pyo.value(m.FO))
                print(f"FO (data-fit SSE) after polish: {FO_val2:.6g}")
            except Exception:
                pass
    except Exception as e:
        print(f"[WARN] Polish stage skipped due to: {e}")

    # Minimal plot: states vs data at FE ends
    try:
        nfe = args.nfe; ncp = args.ncp; th = args.th; h = th / nfe
        radau_nodes = np.array([0.15505102572168, 0.64494897427832, 1.0], dtype=float) if ncp == 3 else (np.array([1.0]) if ncp == 1 else np.linspace(1.0 / ncp, 1.0, ncp))
        state_names = {1: "X", 2: "N", 3: "G", 4: "F", 5: "E"}
        for l in range(1, 6):
            t_line, y_line, t_dat, y_dat = [], [], [], []
            for i in range(1, nfe + 1):
                for j in range(1, ncp + 1):
                    t_ij = (i - 1 + float(radau_nodes[j - 1])) * h
                    t_line.append(t_ij)
                    y_line.append(float(pyo.value(m.c[l, i, j])))
                t_fe = i * h
                val_d = float(data3d[l - 1, i - 1, ncp - 1])
                if val_d != 0.0:
                    t_dat.append(t_fe); y_dat.append(val_d)
            if l == 1:
                print("[PLOT] Legend: Sim=line, Data=x at FE ends")
            plt.figure(figsize=(7, 4))
            plt.plot(t_line, y_line, '-', linewidth=1.6, alpha=0.9, label="Sim", color="tab:blue")
            if len(t_dat) > 0:
                plt.scatter(t_dat, y_dat, s=28, alpha=0.9, label="Data", marker="x", color="tab:orange")
            plt.xlabel("Time"); plt.ylabel(f"State {state_names.get(l, l)}")
            plt.title(f"Data vs Sim — State {state_names.get(l, l)}")
            plt.legend(); plt.tight_layout()
            out_png = os.path.join(RESULTS_DIR, f"zenteno_relax_state_{l}.png")
            plt.savefig(out_png, dpi=150); plt.close()
            print(f"[PLOT] Saved {out_png}")
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")


if __name__ == "__main__":
    main()
