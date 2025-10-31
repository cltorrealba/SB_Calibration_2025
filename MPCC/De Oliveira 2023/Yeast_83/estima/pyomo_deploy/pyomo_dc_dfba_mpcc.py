#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pyomo translation of your JuMP MPCC/dFBA (Radau collocation) model.
- Mirrors indexing, sets, parameters, and Ipopt options from main.jl
- Encodes the same penalized complementarity terms (FO_L, FO_U, FO_upt)
- Uses Ipopt with MUMPS and warm-start, same tolerances

Inputs expected in the working directory (same names as your Julia run):
  S.csv, lb.csv, ub.csv, data.jld2 (or data.csv as fallback)

Run:
  python pyomo_dc_dfba_mpcc.py --nfe 12 --ncp 3 --th 22 \
      --ipopt_max_iter 2000 --print_solver

Outputs (stdout): FO value, exp(teta), solve walltime.
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import h5py  # optional, for reading JLD2 (HDF5)
    HAVE_H5PY = True
except Exception:
    HAVE_H5PY = False

import pyomo.environ as pyo

# Base paths (resolve files relative to this script and the parent `estima/` folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ESTIMA_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# IO helpers
# -----------------------------

def load_S_lb_ub(S_path=None, lb_path=None, ub_path=None):
    # defaults to files in the parent `estima/` folder
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


def load_data_3d(data_path_primary=None, data_path_fallback=None,
                  nc=4, ph=12, ncp=3, mode: str = "stacked", prefer: str = "csv"):
    """
    Returns a numpy array shaped (nc, ph, ncp), matching main.jl usage:
       FO == sum_{l=1..nc} sum_{i=1..ph} sum_{j=1..ncp} (data[l,i,j] - c[l,i,j])^2
    Priority: JLD2 (HDF5) dataset named "data". Fallback: CSV in long format
    with columns [l,i,j,value] or [state,fe,cp,value] (1-based indices) or a raw grid
    with nc*ph*ncp values. Defaults search to `estima/data.jld2` and `pyomo_deploy/data_long.csv`.
    """
    if data_path_primary is None:
        data_path_primary = os.path.join(ESTIMA_DIR, "data.jld2")
    if data_path_fallback is None:
        data_path_fallback = os.path.join(BASE_DIR, "data_long.csv")

    # Helper to read CSV long -> (nc, ph, ncp)
    def _from_long_csv(csv_path: str):
        df = pd.read_csv(csv_path)
        # Case A: long format with l,i,j,value (1-based)
        cols = [c.lower() for c in df.columns]
        if set(["l", "i", "j", "value"]).issubset(set(cols)):
            # map actual columns
            colmap = {c.lower(): c for c in df.columns}
            arr = np.zeros((nc, ph, ncp), dtype=float)
            # Ensure sorted order to fill deterministically
            df_sorted = df.sort_values([colmap["l"], colmap["i"], colmap["j"]], kind="mergesort")
            for _, row in df_sorted.iterrows():
                l = int(row[colmap["l"]]) - 1
                i = int(row[colmap["i"]]) - 1
                j = int(row[colmap["j"]]) - 1
                if 0 <= l < nc and 0 <= i < ph and 0 <= j < ncp:
                    arr[l, i, j] = float(row[colmap["value"]])
            print(f"[DATA-LOAD] data_long mode=indexed(l,i,j); sorted and filled by indices.")
            return arr
        # Case A2: long format with state,fe,cp,value (1-based)
        if set(["state", "fe", "cp", "value"]).issubset(set(cols)):
            colmap = {c.lower(): c for c in df.columns}
            arr = np.zeros((nc, ph, ncp), dtype=float)
            if (mode or "").lower() == "indexed":
                # Strict mapping by provided (state,fe,cp)
                df_sorted = df.sort_values([colmap["state"], colmap["fe"], colmap["cp"]], kind="mergesort")
                for _, row in df_sorted.iterrows():
                    l = int(row[colmap["state"]]) - 1
                    i = int(row[colmap["fe"]]) - 1
                    j = int(row[colmap["cp"]]) - 1
                    if 0 <= l < nc and 0 <= i < ph and 0 <= j < ncp:
                        arr[l, i, j] = float(row[colmap["value"]])
                print(f"[DATA-LOAD] data_long mode=indexed; filled by (state,fe,cp).")
                return arr
            else:
                # Stacked mode: ignora fe/cp y recorre por estado rellenando secuencialmente (i,j)
                # según el orden de aparición de filas por cada estado.
                # Esto sigue la descripción: para un estado, los valores avanzan en el tiempo, luego cambia state.
                df_sorted = df.sort_values([colmap["state"], colmap["fe"], colmap["cp"]], kind="mergesort")
                states_in_file = sorted(set(int(v) for v in df_sorted[colmap["state"]].values))
                for l1 in states_in_file:
                    sub = df_sorted[df_sorted[colmap["state"]] == l1]
                    vals = sub[colmap["value"]].astype(float).to_numpy()
                    idx = 0
                    for i in range(ph):
                        for j in range(ncp):
                            if idx < len(vals):
                                arr[l1 - 1, i, j] = float(vals[idx])
                                idx += 1
                            else:
                                break
                print(f"[DATA-LOAD] data_long mode=stacked; filled sequentially per state, ignoring (fe,cp).")
                return arr
        # Case B: raw grid: try reshape
        vals = df.to_numpy().astype(float)
        flat = vals.flatten()
        if flat.size == nc * ph * ncp:
            return flat.reshape((nc, ph, ncp), order="C")
        return None

    # Helper to read JLD2 -> (nc, ph, ncp)
    def _from_jld2(jld2_path: str):
        if HAVE_H5PY and os.path.exists(jld2_path):
            try:
                with h5py.File(jld2_path, "r") as f:
                    for key in ["data", "\u0000#refs#\u0000", "__data__"]:
                        if key in f:
                            dset = f[key]
                            arr = np.array(dset)
                            if arr.ndim == 3:
                                out = np.zeros((nc, ph, ncp), dtype=float)
                                a0, a1, a2 = arr.shape
                                out[:min(nc, a0), :min(ph, a1), :min(ncp, a2)] = arr[:min(nc, a0), :min(ph, a1), :min(ncp, a2)]
                                print("[DATA-LOAD] JLD2 detected and sliced to (nc,ph,ncp).")
                                return out
                            if arr.ndim == 1 and arr.size == nc * ph * ncp:
                                print("[DATA-LOAD] JLD2 1D reshaped to (nc,ph,ncp).")
                                return arr.reshape((nc, ph, ncp), order="C")
                    # heuristic: take the first 3D dataset found
                    for k, d in f.items():
                        if hasattr(d, "shape") and len(d.shape) == 3:
                            arr = np.array(d)
                            out = np.zeros((nc, ph, ncp), dtype=float)
                            a0, a1, a2 = arr.shape
                            out[:min(nc, a0), :min(ph, a1), :min(ncp, a2)] = arr[:min(nc, a0), :min(ph, a1), :min(ncp, a2)]
                            print("[DATA-LOAD] JLD2 dataset (first 3D) sliced to (nc,ph,ncp).")
                            return out
            except Exception:
                return None
        return None

    # Preferred source order
    prefer = (prefer or "csv").lower()
    if prefer == "csv":
        if os.path.exists(data_path_fallback):
            arr = _from_long_csv(data_path_fallback)
            if arr is not None:
                return arr
        # fallback to JLD2
        arr = _from_jld2(data_path_primary)
        if arr is not None:
            return arr
    else:  # prefer jld2
        arr = _from_jld2(data_path_primary)
        if arr is not None:
            return arr
        if os.path.exists(data_path_fallback):
            arr = _from_long_csv(data_path_fallback)
            if arr is not None:
                return arr
    # As a last resort, return zeros
    return np.zeros((nc, ph, ncp), dtype=float)


# -----------------------------
# Model builder
# -----------------------------

def build_model(S, lb, ub, data3d, nfe=12, ncp=3, th=22.0, var_h=True):
    # Dimensions (1-based sets to match JuMP indexing)
    nm, nv = S.shape
    nc = 4      # states: X, G, Z, E
    ph = nfe    # prediction horizon = number of finite elements in main.jl

    # Indices from your main.jl (1-based)
    eth = 2630
    obj = 3414
    glu = 2588
    o2  = 2816
    ATP = 3415
    xyl = 2592

    # Constants and weights
    w = 1e-20
    omega = 1e2
    phi1 = 1.0
    phi2 = 1.0
    phi3 = 1.0

    # Initial conditions
    x0, g0, z0, e0 = 0.2, 4.0, 2.0, 0.0
    c0 = [x0, g0, z0, e0]

    # Parameter bounds (log-space) — keep exactly as in main.jl
    teta0 = np.log(np.array([7.5, 1.0, 35.0, 15.00, 0.5]))
    LB    = np.log(np.array([7.0, 0.9, 31.0, 13.85, 0.4]))
    UB    = np.log(np.array([8.0, 1.13, 33.0, 15.85, 0.6]))

    # Radau (3-point) collocation matrix and nodes
    colmat = np.array([
        [0.19681547722366,  -0.06553542585020,  0.02377097434822],
        [0.39442431473909,   0.29207341166523, -0.04154875212600],
        [0.37640306270047,   0.51248582618842,  0.11111111111111],
    ])

    # Time grid
    h = th / nfe

    # Unit scalings from main.jl (all 1.0)
    vs = np.ones(nv)

    # Uptake selector vectors
    d = np.zeros(nv); d[obj-1] = -1.0
    up = np.zeros(nv); up[glu-1] = 1.0
    up2 = np.zeros(nv); up2[xyl-1] = 1.0

    # Enforce O2 and ATP bounds as in main.jl
    lb = lb.copy(); ub = ub.copy()
    lb[o2-1] = 0.0; ub[o2-1] = 0.0
    lb[ATP-1] = 0.0

    # ----------------- Pyomo model -----------------
    m = pyo.ConcreteModel()

    # Sets (1-based)
    m.L = pyo.RangeSet(1, nc)       # states
    m.I = pyo.RangeSet(1, ph)       # finite elements
    m.J = pyo.RangeSet(1, ncp)      # collocation points
    m.NV = pyo.RangeSet(1, nv)      # reactions
    m.NM = pyo.RangeSet(1, nm)      # metabolites (rows of S)

    # Sparse S representation to avoid building a gigantic dense Param
    # Build Python dicts of nonzeros for fast rule evaluation (much faster writing .nl)
    rows_nz, cols_nz = np.nonzero(S)
    row_to_cols = {}
    col_to_rows = {}
    S_vals = {}
    for idx in range(len(rows_nz)):
        rr = int(rows_nz[idx]) + 1  # 1-based metabolite row
        cc = int(cols_nz[idx]) + 1  # 1-based reaction col
        val = float(S[rr-1, cc-1])
        S_vals[(rr, cc)] = val
        row_to_cols.setdefault(rr, []).append(cc)
        col_to_rows.setdefault(cc, []).append(rr)
    m.vlb = pyo.Param(m.NV, initialize=lambda mdl, k: float(lb[k-1]), mutable=False)
    m.vub = pyo.Param(m.NV, initialize=lambda mdl, k: float(ub[k-1]), mutable=False)

    # Variables
    m.c = pyo.Var(m.L, m.I, m.J, domain=pyo.Reals, initialize=lambda mdl, l, i, j: c0[l-1])
    m.cdot = pyo.Var(m.L, m.I, m.J, domain=pyo.Reals, initialize=0.0)
    m.FO = pyo.Var(domain=pyo.Reals, initialize=1.0)

    m.teta = pyo.Var(range(1, 6), initialize=lambda mdl, p: float(teta0[p-1]), bounds=lambda mdl, p: (float(LB[p-1]), float(UB[p-1])))

    m.v = pyo.Var(m.NV, m.I, domain=pyo.Reals, initialize=0.0)
    m.lmbda = pyo.Var(m.NM, m.I, domain=pyo.Reals, initialize=0.0)
    m.alpha_U = pyo.Var(m.NV, m.I, domain=pyo.Reals, initialize=0.0)
    m.alpha_L = pyo.Var(m.NV, m.I, domain=pyo.Reals, initialize=0.0)
    m.alpha_upt = pyo.Var([1, 2], m.I, domain=pyo.Reals, initialize=0.0)  # [glu, xyl]

    m.FO_U = pyo.Var(m.NV, m.I, domain=pyo.Reals, initialize=0.0)
    m.FO_L = pyo.Var(m.NV, m.I, domain=pyo.Reals, initialize=0.0)
    m.FO_upt = pyo.Var([1, 2], m.I, domain=pyo.Reals, initialize=0.0)

    m.hv = pyo.Var(m.I, domain=pyo.Reals, initialize=h)

    # Alpha signs
    def _alpha_L_ub(mdl, k, i):
        return mdl.alpha_L[k, i] <= 0.0
    m.alphaL_sign = pyo.Constraint(m.NV, m.I, rule=_alpha_L_ub)

    def _alpha_upt_ub(mdl, uu, i):
        return mdl.alpha_upt[uu, i] <= 0.0
    m.alphaUPT_sign = pyo.Constraint([1, 2], m.I, rule=_alpha_upt_ub)

    def _alpha_U_lb(mdl, k, i):
        return mdl.alpha_U[k, i] >= 0.0
    m.alphaU_sign = pyo.Constraint(m.NV, m.I, rule=_alpha_U_lb)

    # Collocation equations
    def _coll_c_n(mdl, l, i, j):
        if i == 1:
            return pyo.Constraint.Skip
        return mdl.c[l, i, j] == mdl.c[l, i-1, ncp] + h * sum(colmat[j-1, k-1] * mdl.cdot[l, i, k] for k in mdl.J)
    m.coll_c_n = pyo.Constraint(m.L, m.I, m.J, rule=_coll_c_n)

    def _coll_c_0(mdl, l, j):
        return mdl.c[l, 1, j] == c0[l-1] + h * sum(colmat[j-1, k-1] * mdl.cdot[l, 1, k] for k in mdl.J)
    m.coll_c_0 = pyo.Constraint(m.L, m.J, rule=_coll_c_0)

    # Nonnegativity on c via -c <= 0
    def _c_LB(mdl, l, i, j):
        return -mdl.c[l, i, j] <= 0.0
    m.c_LB = pyo.Constraint(m.L, m.I, m.J, rule=_c_LB)

    # Time partition constraints
    m.MFE1 = pyo.Constraint(expr=sum(m.hv[i] for i in m.I) == th)
    m.MFE3 = pyo.Constraint(m.I, rule=lambda mdl, i: mdl.hv[i] >= 0.0)
    m.MFE4 = pyo.Constraint(m.I, rule=lambda mdl, i: mdl.hv[i] >= (1.0 - (1.0 if var_h else 0.0)) * h)
    m.MFE5 = pyo.Constraint(m.I, rule=lambda mdl, i: mdl.hv[i] <= (1.0 + (1.0 if var_h else 0.0)) * h)

    # Flux bounds as inequalities
    def _v_UB(mdl, k, i):
        return mdl.v[k, i] - mdl.vub[k] <= 0.0
    def _v_LB(mdl, k, i):
        return -mdl.v[k, i] + mdl.vlb[k] <= 0.0
    m.v_UB = pyo.Constraint(m.NV, m.I, rule=_v_UB)
    m.v_LB = pyo.Constraint(m.NV, m.I, rule=_v_LB)

    # Stoichiometric balances S v = 0
    def _Sc(mdl, r, i):
        # Sum only over nonzeros in row r using precomputed dictionaries
        rr = int(r)
        nz_cols = row_to_cols.get(rr, [])
        return sum(S_vals[(rr, k)] * mdl.v[k, i] for k in nz_cols) == 0.0
    m.Sc = pyo.Constraint(m.NM, m.I, rule=_Sc)

    # vg(i) and vz(i) expressions; note c[2,i,3] and c[3,i,3]
    def _vg_rule(mdl, i):
        return pyo.exp(mdl.teta[1]) * ( mdl.c[2, i, 3] / (pyo.exp(mdl.teta[2]) + mdl.c[2, i, 3]) )
    def _vz_rule(mdl, i):
        return pyo.exp(mdl.teta[3]) * ( mdl.c[3, i, 3] / (pyo.exp(mdl.teta[4]) + mdl.c[3, i, 3]) ) * ( 1.0 / (1.0 + (mdl.c[2, i, 3] / pyo.exp(mdl.teta[5]))) )
    m.vg = pyo.Expression(m.I, rule=_vg_rule)
    m.vz = pyo.Expression(m.I, rule=_vz_rule)

    # ODEs at each collocation point (matches m1..m4)
    def _m1(mdl, i, j):
        return mdl.cdot[1, i, j] == mdl.v[obj, i] * mdl.c[1, i, j]
    def _m2(mdl, i, j):
        return mdl.cdot[2, i, j] == -0.180156 * mdl.vg[i] * mdl.c[1, i, j]
    def _m3(mdl, i, j):
        return mdl.cdot[3, i, j] == -0.15013 * mdl.vz[i] * mdl.c[1, i, j]
    def _m4(mdl, i, j):
        return mdl.cdot[4, i, j] ==  0.04607 * mdl.v[eth, i] * mdl.c[1, i, j]
    m.m1 = pyo.Constraint(m.I, m.J, rule=_m1)
    m.m2 = pyo.Constraint(m.I, m.J, rule=_m2)
    m.m3 = pyo.Constraint(m.I, m.J, rule=_m3)
    m.m4 = pyo.Constraint(m.I, m.J, rule=_m4)

    # Uptake lower bounds: -v[glu] - vg <= 0; -v[xyl] - vz <= 0
    m.v_LB_g = pyo.Constraint(m.I, rule=lambda mdl, i: -mdl.v[glu, i] - mdl.vg[i] <= 0.0)
    m.v_LB_z = pyo.Constraint(m.I, rule=lambda mdl, i: -mdl.v[xyl, i] - mdl.vz[i] <= 0.0)

    # Lagrangian stationarity (pFBA KKT-like)
    def _Lagr(mdl, k, i):
        # Only sum over metabolite rows with nonzero in column k
        rr_list = col_to_rows.get(int(k), [])
        term_S = sum(S_vals[(r, int(k))] * mdl.lmbda[r, i] for r in rr_list)
        term = d[k-1] + w * mdl.v[k, i] * vs[k-1] + mdl.alpha_L[k, i] + mdl.alpha_U[k, i] \
               + up[k-1] * mdl.alpha_upt[1, i] + up2[k-1] * mdl.alpha_upt[2, i] \
               + term_S
        return term == 0.0
    m.Lagr = pyo.Constraint(m.NV, m.I, rule=_Lagr)

    # Complementarity products (penalized in the objective)
    def _FO_L_rule(mdl, k, i):
        return mdl.FO_L[k, i] == (mdl.v[k, i] - mdl.vlb[k]) * mdl.alpha_L[k, i]
    def _FO_U_rule(mdl, k, i):
        return mdl.FO_U[k, i] == (mdl.v[k, i] - mdl.vub[k]) * mdl.alpha_U[k, i]
    def _FO_upt1_rule(mdl, i):
        return mdl.FO_upt[1, i] == ( -mdl.v[glu, i] - mdl.vg[i] ) * mdl.alpha_upt[1, i]
    def _FO_upt2_rule(mdl, i):
        return mdl.FO_upt[2, i] == ( -mdl.v[xyl, i] - mdl.vz[i] ) * mdl.alpha_upt[2, i]
    m.FO_L_def = pyo.Constraint(m.NV, m.I, rule=_FO_L_rule)
    m.FO_U_def = pyo.Constraint(m.NV, m.I, rule=_FO_U_rule)
    m.FO_upt1_def = pyo.Constraint(m.I, rule=_FO_upt1_rule)
    m.FO_upt2_def = pyo.Constraint(m.I, rule=_FO_upt2_rule)

    # Data-fit FO equality
    # data3d[l-1,i-1,j-1]
    def _FO_eq(mdl):
        return mdl.FO == sum( (float(data3d[l-1, i-1, j-1]) - mdl.c[l, i, j])**2
                              for l in mdl.L for i in mdl.I for j in mdl.J )
    m.FO_def = pyo.Constraint(rule=_FO_eq)

    # Objective (same signs as in main.jl)
    def _obj(mdl):
        return omega * mdl.FO + sum( sum( -phi1 * mdl.FO_L[k, i] - phi3 * mdl.FO_U[k, i]
                                          for k in mdl.NV )
                                     + phi2 * mdl.FO_upt[1, i] + phi2 * mdl.FO_upt[2, i]
                                     for i in mdl.I )
    m.OBJ = pyo.Objective(rule=_obj, sense=pyo.minimize)

    # Expose key reaction indices on the model for audits/plots
    m.idx_eth = eth
    m.idx_obj = obj
    m.idx_glu = glu
    m.idx_xyl = xyl

    return m


# -----------------------------
# Solve helper
# -----------------------------

def solve_model(m, ipopt_max_iter=None, print_solver=False, timings=False, lbfgs=False, max_wall_time=None):
    solver = pyo.SolverFactory("ipopt")
    # Match Ipopt options from main.jl
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", default=os.path.join(ESTIMA_DIR, "S.csv"))
    ap.add_argument("--lb", default=os.path.join(ESTIMA_DIR, "lb.csv"))
    ap.add_argument("--ub", default=os.path.join(ESTIMA_DIR, "ub.csv"))
    ap.add_argument("--data", default=os.path.join(ESTIMA_DIR, "data.jld2"))
    ap.add_argument("--data_csv", default=os.path.join(BASE_DIR, "data_long.csv"))
    ap.add_argument("--nfe", type=int, default=12)
    ap.add_argument("--ncp", type=int, default=3)
    ap.add_argument("--th", type=float, default=22.0)
    ap.add_argument("--no_var_h", action="store_true")
    ap.add_argument("--data_long_mode", choices=["stacked", "indexed"], default="stacked", help="How to interpret long CSV: stacked per state (ignore fe/cp) or indexed by (state,fe,cp)")
    ap.add_argument("--prefer", choices=["csv", "jld2"], default="csv", help="Preferred data source when both are present")
    ap.add_argument("--ipopt_max_iter", type=int, default=None)
    ap.add_argument("--print_solver", action="store_true")
    ap.add_argument("--timings", action="store_true", help="Enable Ipopt timing statistics and print user options")
    ap.add_argument("--lbfgs", action="store_true", help="Use limited-memory Hessian approximation")
    ap.add_argument("--max_wall_time", type=float, default=None, help="Ipopt max wall time in seconds")
    # Audit options
    ap.add_argument("--audit", action="store_true", help="After solve, compute uptake activity fractions and export series CSV")
    ap.add_argument("--audit_tol", type=float, default=1e-6, help="Tolerance to deem uptake constraint active (|slack|<=tol)")
    args = ap.parse_args()

    # Load inputs
    S, lb, ub = load_S_lb_ub(args.S, args.lb, args.ub)
    data3d = load_data_3d(args.data, args.data_csv, nc=4, ph=args.nfe, ncp=args.ncp, mode=args.data_long_mode, prefer=args.prefer)

    # Build & solve
    m = build_model(S, lb, ub, data3d, nfe=args.nfe, ncp=args.ncp, th=args.th, var_h=(not args.no_var_h))
    res, wall = solve_model(
        m,
        ipopt_max_iter=args.ipopt_max_iter,
        print_solver=args.print_solver,
        timings=args.timings,
        lbfgs=args.lbfgs,
        max_wall_time=args.max_wall_time,
    )

    # Report
    teta_exp = [pyo.value(pyo.exp(m.teta[p])) for p in range(1, 6)]
    FO_val = pyo.value(m.FO)

    print("\n=== Pyomo MPCC/dFBA solve summary ===")
    print(f"Solver status: {res.solver.status}, termination: {res.solver.termination_condition}")
    print(f"Wall time: {wall:.2f} s")
    print(f"FO (data-fit SSE): {FO_val:.6g}")
    print("teta (exp):", ", ".join(f"{v:.6g}" for v in teta_exp))

    # ========= Plot y log: scatter por variable de estado (datos vs simulación) =========
    try:
        nfe = args.nfe
        ncp = args.ncp
        th = args.th
        h = th / nfe
        # Nodos de Radau-3 (si aplica). Si no es ncp=3, usamos nodos equiespaciados en (0,1].
        if ncp == 3:
            radau_nodes = np.array([0.15505102572168, 0.64494897427832, 1.0], dtype=float)
        elif ncp == 1:
            radau_nodes = np.array([1.0], dtype=float)
        else:
            # aproximación simple si cambian ncp: repartir en (0,1]
            radau_nodes = np.linspace(1.0 / ncp, 1.0, ncp)

        # Nombres de estados (opcional / para títulos)
        state_names = {1: "X", 2: "G", 3: "Z", 4: "E"}

        for l in range(1, 5):
            # Construir tiempo-valor para simulación como línea continua (todos los nodos)
            # y datos SOLO al final de cada elemento (j = ncp), que es donde hay valor singular por FE
            t_line, y_line = [], []
            t_dat, y_dat = [], []
            for i in range(1, nfe + 1):
                # nodos de colocation dentro del elemento i para la curva
                for j in range(1, ncp + 1):
                    t_ij = (i - 1 + float(radau_nodes[j - 1])) * h
                    t_line.append(t_ij)
                    y_line.append(float(pyo.value(m.c[l, i, j])))
                # punto de medición (fin del elemento)
                t_fe = i * h  # radau_nodes[-1] == 1.0
                val_d = float(data3d[l - 1, i - 1, ncp - 1])
                if val_d != 0.0:
                    t_dat.append(t_fe)
                    y_dat.append(val_d)

            # Log de consola para verificar qué se está graficando
            try:
                sim_min = min(y_line) if y_line else float('nan')
                sim_max = max(y_line) if y_line else float('nan')
                dat_min = min(y_dat) if y_dat else float('nan')
                dat_max = max(y_dat) if y_dat else float('nan')
                print(f"[DATA] State {state_names.get(l, l)} -> sim_points={len(t_line)}, data_points={len(t_dat)}")
                print(f"       Sim range: [{sim_min:.6g}, {sim_max:.6g}] | Data range: [{dat_min:.6g}, {dat_max:.6g}]")
                if len(t_dat) > 0:
                    kshow = min(10, len(t_dat))
                    print("       Sample data (t, y):" + ", ".join(f"({t_dat[k]:.4f}, {y_dat[k]:.6g})" for k in range(kshow)))
            except Exception:
                pass

            # Solo imprimir mapeo de leyenda una vez
            if l == 1:
                print("[PLOT] Legend mapping: Sim = tab:blue line, Data = tab:orange x at FE ends")

            plt.figure(figsize=(7, 4))
            # Línea continua para la simulación
            plt.plot(t_line, y_line, '-', linewidth=1.6, alpha=0.9, label="Sim", color="tab:blue")
            if len(t_dat) > 0:
                plt.scatter(t_dat, y_dat, s=28, alpha=0.9, label="Data", marker="x", color="tab:orange")
            plt.xlabel("Time")
            plt.ylabel(f"State {state_names.get(l, l)}")
            plt.title(f"Data vs Sim — State {state_names.get(l, l)}")
            plt.legend()
            plt.tight_layout()
            out_png = os.path.join(RESULTS_DIR, f"scatter_state_{l}.png")
            plt.savefig(out_png, dpi=150)
            plt.close()
            print(f"[PLOT] Saved {out_png}")
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")

    # ========= Post-solve audit: uptake activity, slacks, and series export =========
    try:
        if args.audit:
            nfe = args.nfe
            # FE end times: variable hv may vary if var_h; compute cumulative
            hv_vals = [float(pyo.value(m.hv[i])) for i in range(1, nfe + 1)]
            t_end = np.cumsum(hv_vals)

            idx_glu = int(m.idx_glu)
            idx_xyl = int(m.idx_xyl)
            idx_obj = int(m.idx_obj)
            idx_eth = int(m.idx_eth)

            rows = []
            active_g = 0
            active_z = 0
            abs_slack_g = []
            abs_slack_z = []
            for i in range(1, nfe + 1):
                t_i = float(t_end[i - 1])
                vg_i = float(pyo.value(m.vg[i]))
                vz_i = float(pyo.value(m.vz[i]))
                v_glu_i = float(pyo.value(m.v[idx_glu, i]))
                v_xyl_i = float(pyo.value(m.v[idx_xyl, i]))
                mu_i = float(pyo.value(m.v[idx_obj, i]))
                qeth_i = float(pyo.value(m.v[idx_eth, i]))
                # Slack definitions consistent with constraints: -v[glu]-vg <= 0, -v[xyl]-vz <= 0
                slack_g = -v_glu_i - vg_i
                slack_z = -v_xyl_i - vz_i
                abs_slack_g.append(abs(slack_g))
                abs_slack_z.append(abs(slack_z))
                if abs(slack_g) <= args.audit_tol:
                    active_g += 1
                if abs(slack_z) <= args.audit_tol:
                    active_z += 1
                # Complementarity product variables
                try:
                    fou_g = float(pyo.value(m.FO_upt[1, i]))
                    fou_z = float(pyo.value(m.FO_upt[2, i]))
                except Exception:
                    fou_g = float('nan')
                    fou_z = float('nan')

                rows.append({
                    "time": t_i,
                    "mu": mu_i,
                    "q_eth": qeth_i,
                    "vg": vg_i,
                    "vz": vz_i,
                    "v_glu": v_glu_i,
                    "v_xyl": v_xyl_i,
                    "slack_glu": slack_g,
                    "slack_xyl": slack_z,
                    "FO_upt_glu": fou_g,
                    "FO_upt_xyl": fou_z,
                })

            frac_g = 100.0 * active_g / nfe if nfe > 0 else float('nan')
            frac_z = 100.0 * active_z / nfe if nfe > 0 else float('nan')
            print(f"\n=== Uptake activity audit ===")
            print(f"Active uptake (glu): {active_g}/{nfe}  ({frac_g:.1f}%) with tol={args.audit_tol:g}")
            print(f"Active uptake (xyl): {active_z}/{nfe}  ({frac_z:.1f}%) with tol={args.audit_tol:g}")
            print(f"Mean |slack_glu|: {np.mean(abs_slack_g):.3e},  Median: {np.median(abs_slack_g):.3e}")
            print(f"Mean |slack_xyl|: {np.mean(abs_slack_z):.3e},  Median: {np.median(abs_slack_z):.3e}")

            df_audit = pd.DataFrame(rows)
            out_csv = os.path.join(RESULTS_DIR, "audit_series.csv")
            df_audit.to_csv(out_csv, index=False)
            print(f"[AUDIT] Saved time series to {out_csv}")
    except Exception as e:
        print(f"[WARN] Audit failed: {e}")

if __name__ == "__main__":
    main()
