#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zenteno fermentation model (macroscopic ODEs) in Pyomo with Radau collocation.
Start simple (static T, no N addition). Optional dFBA MPCC coupling with equality ties
for growth (mu), ethanol, glucose/fructose uptakes, and lumped YAN (NH4+ + AA).
"""

import argparse
import os
import json
from datetime import datetime
import time
from math import log
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
# IO helpers
# -----------------------------

def load_data_3d(data_path_primary=None, data_path_fallback=None,
                  nc=5, ph=12, ncp=3, mode: str = "stacked", prefer: str = "csv"):
    if data_path_primary is None:
        data_path_primary = os.path.join(ESTIMA_DIR, "data.jld2")
    if data_path_fallback is None:
        data_path_fallback = os.path.join(BASE_DIR, "data_long.csv")

    def _from_long_csv(csv_path: str):
        if not os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)
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
            print("[DATA-LOAD] long CSV mode=indexed(l,i,j)")
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
                print("[DATA-LOAD] long CSV mode=indexed(state,fe,cp)")
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
                                arr[l1 - 1, i, j] = float(vals[idx])
                                idx += 1
                print("[DATA-LOAD] long CSV mode=stacked (per state)")
                return arr
        vals = df.to_numpy().astype(float).flatten()
        if vals.size == nc * ph * ncp:
            return vals.reshape((nc, ph, ncp), order="C")
        return None

    prefer = (prefer or "csv").lower()
    if prefer == "csv":
        arr = _from_long_csv(data_path_fallback)
        if arr is not None:
            return arr
    print("[DATA-LOAD] No CSV found; using zeros array")
    return np.zeros((nc, ph, ncp), dtype=float)


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


# -----------------------------
# Params helpers (CSV/XLSX)
# -----------------------------

def _norm_key(s):
    return str(s).strip().lower().replace(" ", "").replace("-", "").replace("_", "")

def _try_parse_params_csv(path):
    try:
        dfp = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] No se pudo leer CSV '{path}': {e}")
        return None
    vals = {}
    cols_lower = [_norm_key(c) for c in dfp.columns]
    if {"name", "value"}.issubset(set(cols_lower)):
        # Buscar columnas originales respetando el mapeo
        colmap = { _norm_key(c): c for c in dfp.columns }
        for _, row in dfp.iterrows():
            vals[str(row[colmap["name"]]).strip()] = float(row[colmap["value"]])
        return vals
    # Alternativa: una única fila con columnas de parámetros
    if len(dfp) >= 1:
        row = dfp.iloc[0]
        for c in dfp.columns:
            try:
                vals[str(c).strip()] = float(row[c])
            except Exception:
                pass
        if len(vals) > 0:
            return vals
    return None

def _try_parse_params_excel(path, set_id=None, sheet=None):
    """Lee parámetros desde un Excel con múltiples formatos posibles.
    Estrategia flexible:
      1) Si set_id está definido, intentar coincidir con nombre de hoja ("set 4", "4").
      2) Si no, usar primera hoja.
      3) Parsear según casos comunes:
         - Columnas [name, value]
         - Columnas [name, setX, setY, ...] elegir la de set_id
         - Columnas [param, set, value] filtrar set==set_id
         - Hoja del set: dos columnas (name, value) o cualquier combinación reconocible
    """
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        print(f"[WARN] No se pudo abrir Excel '{path}': {e}")
        return None

    # Construir lista de hojas candidatas
    wanted = _norm_key(str(set_id)) if set_id is not None else None
    candidates = []
    if sheet is not None:
        candidates.append(sheet)
    elif wanted is not None:
        # hojas que parezcan set 4, etc.
        for sh in xls.sheet_names:
            nsh = _norm_key(sh)
            if nsh == wanted or nsh == f"set{wanted}" or nsh == f"set_{wanted}" or nsh == f"paramset{wanted}":
                candidates.append(sh)
    # añadir todas si aún vacío
    if not candidates:
        candidates.extend(xls.sheet_names)

    # Sinónimos de encabezados
    synonyms = {
        "name": {"name","param","parameter","parametro","parámetro","parametros","parámetros"},
        "value": {"value","valor","val","values","valores"},
        "set": {"set","conjunto","grupo","group","id","setid"},
    }
    def _norm_header(c):
        nc = _norm_key(c)
        for k, ss in synonyms.items():
            if nc in ss:
                return k
        return nc

    # Intentar por hoja y por modo de lectura (header=0 y header=None)
    for sheet_to_read in candidates:
        for header_mode in (0, None):
            try:
                df = pd.read_excel(xls, sheet_name=sheet_to_read, header=header_mode)
            except Exception:
                continue
            if df is None or df.empty:
                continue

            raw_cols = list(df.columns)
            cols_norm = [_norm_header(c) for c in raw_cols]
            colmap = { _norm_header(c): c for c in raw_cols }

            # Caso filas por set: primera columna 'set' y el resto parámetros
            if "set" in cols_norm and set_id is not None:
                set_c = colmap["set"]
                # Coincidencia flexible: por número o texto normalizado
                def _match_sid(v):
                    try:
                        sid = int(str(set_id))
                        return int(float(v)) == sid
                    except Exception:
                        return _norm_key(str(v)) in {wanted, f"set{wanted}", f"conjunto{wanted}", f"grupo{wanted}"}
                mask = df[set_c].map(_match_sid)
                sub = df[mask]
                if not sub.empty:
                    row = sub.iloc[0]
                    vals = {}
                    for c in df.columns:
                        if c == set_c:
                            continue
                        try:
                            vals[str(c).strip()] = float(row[c])
                        except Exception:
                            pass
                    if len(vals) > 0:
                        print(f"[PARAMS] Excel hoja '{sheet_to_read}' seleccionando fila set=={set_id} (header={header_mode})")
                        return vals

            # Caso directo: name/value
            if {"name", "value"}.issubset(set(cols_norm)):
                vals = {}
                for _, row in df.iterrows():
                    try:
                        vals[str(row[colmap["name"]]).strip()] = float(row[colmap["value"]])
                    except Exception:
                        pass
                if len(vals) > 0:
                    print(f"[PARAMS] Excel hoja '{sheet_to_read}' usando columnas name/value (header={header_mode})")
                    return vals

            # Caso columnas por set: [name, set1, set2, ...]
            if "name" in cols_norm:
                if set_id is not None:
                    target_keys = {wanted, f"set{wanted}", f"set_{wanted}", f"paramset{wanted}"}
                    pick = None
                    for k in cols_norm:
                        if k in target_keys:
                            pick = k; break
                    if pick is not None:
                        col_pick = colmap[pick]
                        name_col = colmap["name"]
                        vals = {}
                        for _, row in df.iterrows():
                            name = str(row[name_col]).strip()
                            try:
                                vals[name] = float(row[col_pick])
                            except Exception:
                                pass
                        if len(vals) > 0:
                            print(f"[PARAMS] Excel hoja '{sheet_to_read}' usando columna de set '{col_pick}' (header={header_mode})")
                            return vals
                    # Fallback por índice (set_id-1) entre columnas no-name
                    try:
                        sid = int(str(set_id))
                        non_name_cols = [c for c in df.columns if _norm_key(c) != "name"]
                        if 1 <= sid <= len(non_name_cols):
                            col_pick = non_name_cols[sid - 1]
                            name_col = colmap["name"]
                            vals = {}
                            for _, row in df.iterrows():
                                name = str(row[name_col]).strip()
                                try:
                                    vals[name] = float(row[col_pick])
                                except Exception:
                                    pass
                            if len(vals) > 0:
                                print(f"[PARAMS] Excel hoja '{sheet_to_read}' usando columna #{sid} ('{col_pick}') (header={header_mode})")
                                return vals
                    except Exception:
                        pass

            # Caso filas con [name, set, value]
            key_param = "name" if "name" in cols_norm else None
            if key_param is not None and "set" in cols_norm and "value" in cols_norm and set_id is not None:
                name_c = colmap[key_param]; set_c = colmap["set"]; val_c = colmap["value"]
                mask = df[set_c].astype(str).map(lambda s: _norm_key(s) == wanted or _norm_key(s) == f"set{wanted}")
                sub = df[mask]
                vals = {}
                for _, row in sub.iterrows():
                    try:
                        vals[str(row[name_c]).strip()] = float(row[val_c])
                    except Exception:
                        pass
                if len(vals) > 0:
                    print(f"[PARAMS] Excel hoja '{sheet_to_read}' filtrando filas set=={set_id} (header={header_mode})")
                    return vals

            # Caso de dos columnas (name,value) sin encabezados estándar
            if df.shape[1] == 2:
                vals = {}
                for _, row in df.iterrows():
                    try:
                        k = str(row.iloc[0]).strip(); v = float(row.iloc[1])
                        vals[k] = v
                    except Exception:
                        pass
                if len(vals) > 0:
                    print(f"[PARAMS] Excel hoja '{sheet_to_read}' parseada como dos columnas (header={header_mode})")
                    return vals

            # Fallback: detectar columna de nombres por coincidencia con parámetros conocidos
            try:
                known = {"mu0","betag0","betaf0","kn0","kg0","kf0","kig0","kie0","kd0","yxn","yxg","yxf","yeg","yef"}
                name_col_guess = None
                best_hits = 0
                for c in df.columns:
                    hits = 0
                    for v in df[c].astype(str).values:
                        if _norm_key(v) in known:
                            hits += 1
                    if hits > best_hits:
                        best_hits = hits; name_col_guess = c
                if name_col_guess is not None and best_hits >= 5:
                    non_name_cols = [c for c in df.columns if c != name_col_guess]
                    if len(non_name_cols) >= 1:
                        # Elegir por set_id si posible; sino, la primera columna de valores
                        col_pick = None
                        if set_id is not None:
                            # Intentar localizar encabezado que mencione el set (en las primeras filas)
                            try:
                                sid = int(str(set_id))
                            except Exception:
                                sid = None
                            if header_mode is None:
                                # revisar primeras filas por texto de set
                                for j, c in enumerate(df.columns):
                                    if c == name_col_guess:
                                        continue
                                    top_vals = [str(x) for x in df[c].head(3).values]
                                    for tv in top_vals:
                                        nt = _norm_key(tv)
                                        if nt in {f"set{wanted}", f"conjunto{wanted}", f"grupo{wanted}", wanted}:
                                            col_pick = c; break
                                    if col_pick is not None:
                                        break
                        if col_pick is None:
                            if set_id is not None and isinstance(set_id, (int, str)):
                                try:
                                    sid = int(str(set_id))
                                    if 1 <= sid <= len(non_name_cols):
                                        col_pick = non_name_cols[sid - 1]
                                except Exception:
                                    pass
                        if col_pick is None:
                            col_pick = non_name_cols[0]
                        vals = {}
                        for _, row in df.iterrows():
                            name = str(row[name_col_guess]).strip()
                            try:
                                vals[name] = float(row[col_pick])
                            except Exception:
                                pass
                        if len(vals) > 0:
                            print(f"[PARAMS] Excel hoja '{sheet_to_read}' detectada por nombres conocidos; usando columna '{col_pick}' (header={header_mode})")
                            return vals
            except Exception:
                pass

    print(f"[WARN] Formato de Excel no reconocido para '{path}'. Hojas disponibles: {', '.join(xls.sheet_names)}")
    return None


# -----------------------------
# Model builder (ODE-only core)
# -----------------------------

def build_model(nfe=12, ncp=3, th=22.0, var_h=True, data3d=None, T_const=288.15, alpha_switch=0.5):
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
        return log(max(1e-12, 0.5 * nominal[name]))
    def _ub(name):
        return log(max(2e-12, 2.0 * nominal[name]))

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
    m.theta = pyo.Var(m.PN, initialize=lambda mdl, n: log(nominal[n]),
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

    # FE-end fractions (for reporting / MPCC ties)
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

    # Pointwise kinetics at each collocation point (i,j)
    m.mu_j = pyo.Expression(m.I, m.J, rule=lambda mdl, i, j: P('mu0') * _mu_T(mdl) * ( mdl.c[2, i, j] / ( mdl.c[2, i, j] + P('Kn0') * _Kg_T(mdl) + eps ) ))
    m.betaG_j = pyo.Expression(m.I, m.J, rule=lambda mdl, i, j: P('betaG0') * _b_T(mdl) *
                                ( mdl.c[3, i, j] / ( mdl.c[3, i, j] + P('Kg0') * _Kg_T(mdl) + eps ) ) *
                                ( P('Kie0') * _Kg_T(mdl) / ( mdl.c[5, i, j] + P('Kie0') * _Kg_T(mdl) + eps ) ))
    m.betaF_j = pyo.Expression(m.I, m.J, rule=lambda mdl, i, j: P('betaF0') * _b_T(mdl) *
                                ( mdl.c[4, i, j] / ( mdl.c[4, i, j] + P('Kf0') * _Kg_T(mdl) + eps ) ) *
                                ( P('Kig0') * _Kg_T(mdl) / ( mdl.c[3, i, j] + P('Kig0') * _Kg_T(mdl) + eps ) ) *
                                ( P('Kie0') * _Kg_T(mdl) / ( mdl.c[5, i, j] + P('Kie0') * _Kg_T(mdl) + eps ) ))
    m.mrate = pyo.Expression(rule=lambda mdl: _mr_T(mdl))

    # FE-end kinetics for MPCC ties and reports
    m.mu_fe = pyo.Expression(m.I, rule=lambda mdl, i: P('mu0') * _mu_T(mdl) * ( mdl.Ne[i] / ( mdl.Ne[i] + P('Kn0') * _Kg_T(mdl) + eps ) ))
    m.betaG_fe = pyo.Expression(m.I, rule=lambda mdl, i: P('betaG0') * _b_T(mdl) *
                                 ( mdl.Ge[i] / ( mdl.Ge[i] + P('Kg0') * _Kg_T(mdl) + eps ) ) *
                                 ( P('Kie0') * _Kg_T(mdl) / ( mdl.Ee[i] + P('Kie0') * _Kg_T(mdl) + eps ) ))
    m.betaF_fe = pyo.Expression(m.I, rule=lambda mdl, i: P('betaF0') * _b_T(mdl) *
                                 ( mdl.Fe[i] / ( mdl.Fe[i] + P('Kf0') * _Kg_T(mdl) + eps ) ) *
                                 ( P('Kig0') * _Kg_T(mdl) / ( mdl.Ge[i] + P('Kig0') * _Kg_T(mdl) + eps ) ) *
                                 ( P('Kie0') * _Kg_T(mdl) / ( mdl.Ee[i] + P('Kie0') * _Kg_T(mdl) + eps ) ))

    # Smoothness parameter for death switch (tanh)
    alpha = float(alpha_switch)  # smoothness for tanh switch (avoid overflow)

    # FE-end death rate and net growth
    def _Kd_fe(mdl, i):
        Td_i = -0.0001 * mdl.Ee[i]**3 + 0.0049 * mdl.Ee[i]**2 - 0.1279 * mdl.Ee[i] + 315.89
        s = 0.5 * (1.0 + pyo.tanh(alpha * (mdl.T - Td_i)))
        base = P('Kd0') * pyo.exp(0.0415 * mdl.Ee[i] + (130000.0 * (mdl.T - 305.65)) / (305.65 * R * mdl.T))
        return base * s
    m.Kd_fe = pyo.Expression(m.I, rule=_Kd_fe)
    m.mu_net_fe = pyo.Expression(m.I, rule=lambda mdl, i: mdl.mu_fe[i] - mdl.Kd_fe[i])

    # Death kinetics always enabled (smooth switch via logistic on T - Td)
    # Td(E) = -0.0001*E^3 + 0.0049*E^2 - 0.1279*E + 315.89
    def _Td(mdl, i, j):
        E_ij = mdl.c[5, i, j]
        return -0.0001 * E_ij**3 + 0.0049 * E_ij**2 - 0.1279 * E_ij + 315.89
    def _sigmoid(x):
        return 0.5 * (1.0 + pyo.tanh(alpha * x))
    def _Kd_j(mdl, i, j):
        Td_ij = _Td(mdl, i, j)
        s = _sigmoid(mdl.T - Td_ij)
        base = P('Kd0') * pyo.exp(0.0415 * mdl.c[5, i, j] + (130000.0 * (mdl.T - 305.65)) / (305.65 * R * mdl.T))
        return base * s
    m.Kd_j = pyo.Expression(m.I, m.J, rule=_Kd_j)

    m.Nadd = pyo.Expression(m.I, rule=lambda mdl, i: 0.0)

    # Pointwise fractions for maintenance
    m.phiG_j = pyo.Expression(m.I, m.J, rule=lambda mdl, i, j: mdl.c[3, i, j] / (mdl.c[3, i, j] + mdl.c[4, i, j] + eps))
    m.phiF_j = pyo.Expression(m.I, m.J, rule=lambda mdl, i, j: mdl.c[4, i, j] / (mdl.c[3, i, j] + mdl.c[4, i, j] + eps))

    def _dX(mdl, i, j):
        return mdl.cdot[1, i, j] == ( mdl.mu_j[i, j] - mdl.Kd_j[i, j] ) * mdl.c[1, i, j]
    def _dN(mdl, i, j):
        return mdl.cdot[2, i, j] == -( mdl.mu_j[i, j] / P('Yxn') ) * mdl.c[1, i, j] + mdl.Nadd[i]
    def _dG(mdl, i, j):
        return mdl.cdot[3, i, j] == -( ( mdl.mu_j[i, j] / P('Yxg') ) + ( mdl.betaG_j[i, j] / P('Yeg') ) + mdl.mrate * mdl.phiG_j[i, j] ) * mdl.c[1, i, j]
    def _dF(mdl, i, j):
        return mdl.cdot[4, i, j] == -( ( mdl.mu_j[i, j] / P('Yxf') ) + ( mdl.betaF_j[i, j] / P('Yef') ) + mdl.mrate * mdl.phiF_j[i, j] ) * mdl.c[1, i, j]
    def _dE(mdl, i, j):
        return mdl.cdot[5, i, j] == ( mdl.betaG_j[i, j] + mdl.betaF_j[i, j] ) * mdl.c[1, i, j]

    m.ode_X = pyo.Constraint(m.I, m.J, rule=_dX)
    m.ode_N = pyo.Constraint(m.I, m.J, rule=_dN)
    m.ode_G = pyo.Constraint(m.I, m.J, rule=_dG)
    m.ode_F = pyo.Constraint(m.I, m.J, rule=_dF)
    m.ode_E = pyo.Constraint(m.I, m.J, rule=_dE)

    if data3d is None:
        data3d = np.zeros((nc, ph, ncp), dtype=float)
    m.FO = pyo.Var(domain=pyo.Reals, initialize=1.0)
    m.FO_def = pyo.Constraint(rule=lambda mdl: mdl.FO == sum( (float(data3d[l-1, i-1, j-1]) - mdl.c[l, i, j])**2 for l in mdl.L for i in mdl.I for j in mdl.J ))
    m.OBJ = pyo.Objective(expr=1e2 * m.FO, sense=pyo.minimize)

    return m


def solve_model(m, ipopt_max_iter=None, print_solver=False, timings=False, lbfgs=False, max_wall_time=None,
                tol=None, acceptable_tol=None, acceptable_iter=None):
    solver = pyo.SolverFactory("ipopt")
    solver.options["warm_start_init_point"] = "yes"
    solver.options["print_level"] = 5 if print_solver else 3
    solver.options["tol"] = float(tol) if tol is not None else 1e-4
    solver.options["acceptable_iter"] = int(acceptable_iter) if acceptable_iter is not None else 5
    solver.options["acceptable_tol"] = float(acceptable_tol) if acceptable_tol is not None else 1e-2
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default=os.path.join(BASE_DIR, "data_long.csv"))
    ap.add_argument("--nfe", type=int, default=12)
    ap.add_argument("--ncp", type=int, default=3)
    ap.add_argument("--th", type=float, default=22.0)
    ap.add_argument("--no_var_h", action="store_true")
    ap.add_argument("--data_long_mode", choices=["stacked", "indexed"], default="stacked")
    ap.add_argument("--prefer", choices=["csv", "jld2"], default="csv")
    ap.add_argument("--ipopt_max_iter", type=int, default=None)
    ap.add_argument("--print_solver", action="store_true")
    ap.add_argument("--timings", action="store_true")
    ap.add_argument("--lbfgs", action="store_true")
    ap.add_argument("--max_wall_time", type=float, default=None)
    # Death switch smoothness (for Kd tanh)
    ap.add_argument("--alpha_death_switch", type=float, default=0.5, help="Suavidad del switch de muerte (tanh) en Kd; típico 0.3-0.8")
    ap.add_argument("--T", type=float, default=293.15, help="Static temperature in Kelvin")
    ap.add_argument("--params_csv", default=None, help="CSV de parámetros (name,value) o fila con columnas nombradas; si apunta a .xlsx, se interpretará como Excel")
    ap.add_argument("--params_xlsx", default=None, help="Excel de parámetros (opcional si --params_csv ya apunta a .xlsx)")
    ap.add_argument("--param_set", default=None, help="Identificador del set de parámetros en Excel (número o nombre). Ej: 4 o 'Set 4'")
    ap.add_argument("--params_sheet", default=None, help="Nombre de hoja en Excel (opcional; si no se especifica, se intenta inferir por set o se usa la primera hoja)")
    ap.add_argument("--params_debug", action="store_true", help="Imprimir diagnóstico del Excel de parámetros (nombres de hojas, columnas, primeras filas)")
    ap.add_argument("--simulate", action="store_true", help="Modo simulación: fija parámetros nominales y resuelve las ODE sin ajuste (ignora --enable_mpcc)")
    ap.add_argument("--simulate_mpcc", action="store_true", help="Modo simulación con MPCC: fija parámetros y activa dFBA/MPCC con amarres y penalizaciones (sin SSE de datos)")
    ap.add_argument("--show", action="store_true", help="Mostrar ventana de gráficos al finalizar")
    ap.add_argument("--compl_w", type=float, default=1e3, help="Peso de la penalización de complementariedad en modo simulate_mpcc")
    ap.add_argument("--reg_w", type=float, default=1e-9, help="Peso de la regularización ||v||^2 en modo simulate_mpcc")
    ap.add_argument("--stat_w", type=float, default=1e-6, help="Peso de regularización en la ecuación de estacionariedad (w * v) para estabilizar KKT")
    # Warm-start and scheduling helpers
    ap.add_argument("--pre_solve_ties", action="store_true", help="Pre-resuelve una proyección de v (por FE) desde los amarres sobre S·v=0 con cotas para un mejor warm-start")
    ap.add_argument("--pre_solve_delta", type=float, default=1e-3, help="Ridge (delta) para la proyección de pre-solve; v = (v0 - S^T lambda)/(1+delta)")
    ap.add_argument("--compl_schedule", type=str, default=None, help="Cadena separada por comas con pesos de complementariedad para ejecutar en cadena, ej: '1e3,1e5,1e6'")
    ap.add_argument("--stat_schedule", type=str, default=None, help="Cadena separada por comas para w de estacionariedad por etapa; si no se especifica, se mantiene --stat_w")
    # MPCC coupling toggles and inputs
    ap.add_argument("--enable_mpcc", action="store_true", help="Enable dFBA MPCC coupling with equality ties for growth, sugars, ethanol and lumped nitrogen (YAN)")
    ap.add_argument("--S", default=os.path.join(ESTIMA_DIR, "S.csv"))
    ap.add_argument("--lb", default=os.path.join(ESTIMA_DIR, "lb.csv"))
    ap.add_argument("--ub", default=os.path.join(ESTIMA_DIR, "ub.csv"))
    # Polish stage options (optional last stage with stricter tolerances)
    ap.add_argument("--polish_last_stage", action="store_true", help="Añade una etapa final de pulido con tolerancias más estrictas")
    ap.add_argument("--polish_stat_w", type=float, default=0.1, help="stat_w de la etapa de pulido final")
    ap.add_argument("--polish_tol", type=float, default=1e-6, help="Ipopt tol para la etapa de pulido")
    ap.add_argument("--polish_acceptable_tol", type=float, default=1e-7, help="Ipopt acceptable_tol para la etapa de pulido")
    ap.add_argument("--polish_acceptable_iter", type=int, default=5, help="Ipopt acceptable_iter para la etapa de pulido")
    ap.add_argument("--polish_wall_time", type=float, default=600.0, help="Tiempo máximo (s) para la etapa de pulido")
    args = ap.parse_args()

    data3d = load_data_3d(None, args.data_csv, nc=5, ph=args.nfe, ncp=args.ncp,
                          mode=args.data_long_mode, prefer=args.prefer)

    m = build_model(nfe=args.nfe, ncp=args.ncp, th=args.th, var_h=(not args.no_var_h),
                    data3d=data3d, T_const=args.T, alpha_switch=args.alpha_death_switch)

    # Cargar parámetros desde CSV/XLSX si se entrega
    param_vals_loaded = False
    loaded_from = None
    vals = None
    if args.params_csv:
        lower = str(args.params_csv).lower()
        if lower.endswith(".xlsx") or lower.endswith(".xls"):
            vals = _try_parse_params_excel(args.params_csv, set_id=args.param_set, sheet=args.params_sheet)
            loaded_from = f"Excel ({args.params_csv})"
        else:
            vals = _try_parse_params_csv(args.params_csv)
            loaded_from = f"CSV ({args.params_csv})"
    if vals is None and args.params_xlsx:
        # Modo diagnóstico opcional
        if args.params_debug:
            try:
                xls = pd.ExcelFile(args.params_xlsx)
                print(f"[PARAMS/DEBUG] Hojas: {', '.join([str(s) for s in xls.sheet_names])}")
                for sh in xls.sheet_names:
                    try:
                        df0 = pd.read_excel(xls, sheet_name=sh, header=0)
                        print(f"[PARAMS/DEBUG] Hoja '{sh}' header=0 -> columnas: {[str(c) for c in df0.columns]} shape={df0.shape}")
                        print(df0.head(3).to_string(index=False))
                    except Exception as e:
                        print(f"[PARAMS/DEBUG] Hoja '{sh}' header=0 error: {e}")
                    try:
                        dfN = pd.read_excel(xls, sheet_name=sh, header=None)
                        print(f"[PARAMS/DEBUG] Hoja '{sh}' header=None -> columnas: {[str(c) for c in dfN.columns]} shape={dfN.shape}")
                        print(dfN.head(3).to_string(index=False))
                    except Exception as e:
                        print(f"[PARAMS/DEBUG] Hoja '{sh}' header=None error: {e}")
            except Exception as e:
                print(f"[PARAMS/DEBUG] Error abriendo Excel: {e}")
        vals = _try_parse_params_excel(args.params_xlsx, set_id=args.param_set, sheet=args.params_sheet)
        loaded_from = f"Excel ({args.params_xlsx})"
    if vals is not None:
        for n in m.PN:
            if n in vals:
                try:
                    m.theta[n].set_value(log(float(vals[n])))
                except Exception:
                    pass
        param_vals_loaded = True
        print(f"[PARAMS] Cargados desde {loaded_from}{' (set='+str(args.param_set)+')' if args.param_set else ''}")
    else:
        if args.params_csv or args.params_xlsx:
            print("[WARN] No se pudieron cargar parámetros desde el archivo proporcionado. Se usarán nominales si aplica.")

    # Simulate-only mode: fix parameters to nominal and solve feasibility (no fitting)
    if args.simulate:
        if args.enable_mpcc:
            print("[INFO] --simulate ignora --enable_mpcc; se ejecutará ODE-only (sin dFBA).")
        args.enable_mpcc = False
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
        # Si hay parámetros cargados (CSV/XLSX), fijarlos y relajar bounds; si no, fijar nominales
        if param_vals_loaded:
            for n in m.PN:
                try:
                    # Si tenemos valor cargado explícito, relajar bounds para permitir fijarlo fuera de rango
                    if vals is not None and (n in vals):
                        m.theta[n].setlb(None)
                        m.theta[n].setub(None)
                        m.theta[n].fix(log(float(vals[n])))
                    else:
                        m.theta[n].fix(pyo.value(m.theta[n]))
                except Exception:
                    # Fallback: mantener valor actual fijo
                    try:
                        m.theta[n].fix(pyo.value(m.theta[n]))
                    except Exception:
                        pass
        else:
            for n, v in nominal.items():
                m.theta[n].fix(log(v))
        # Replace objective with constant 0 to enforce feasibility-only solve
        try:
            m.del_component('OBJ')
        except Exception:
            try:
                m.del_component(m.OBJ)
            except Exception:
                pass
        m.OBJ = pyo.Objective(expr=0.0)

    # Modo simulate_mpcc: forzar MPCC con parámetros fijos y objetivo sin SSE
    if args.simulate_mpcc:
        args.enable_mpcc = True
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
        if param_vals_loaded:
            for n in m.PN:
                try:
                    if vals is not None and (n in vals):
                        m.theta[n].setlb(None)
                        m.theta[n].setub(None)
                        m.theta[n].fix(log(float(vals[n])))
                    else:
                        m.theta[n].fix(pyo.value(m.theta[n]))
                except Exception:
                    try:
                        m.theta[n].fix(pyo.value(m.theta[n]))
                    except Exception:
                        pass
        else:
            for n, v in nominal.items():
                m.theta[n].fix(log(v))
        # objetivo nulo por ahora; se reemplaza tras construir MPCC
        try:
            m.del_component('OBJ')
        except Exception:
            try:
                m.del_component(m.OBJ)
            except Exception:
                pass
        m.OBJ = pyo.Objective(expr=0.0)

    if args.enable_mpcc:
        S, lb, ub = load_S_lb_ub(args.S, args.lb, args.ub)
        print(f"[MPCC] Loaded S shape: {S.shape}, lb/ub length: {len(lb)}/{len(ub)}")
        nm, nv = S.shape
        rows_nz, cols_nz = np.nonzero(S)
        row_to_cols = {}
        col_to_rows = {}
        S_vals = {}
        for ii in range(len(rows_nz)):
            rr = int(rows_nz[ii]) + 1
            cc = int(cols_nz[ii]) + 1
            val = float(S[rr - 1, cc - 1])
            S_vals[(rr, cc)] = val
            row_to_cols.setdefault(rr, []).append(cc)
            col_to_rows.setdefault(cc, []).append(rr)

        # Indices
        idx_obj = 3414; idx_eth = 2630; idx_glu = 2588; idx_fru = 2583; idx_nh4 = 2536
        aa_yes = [2729, 2730, 2740, 2754, 2759, 2723, 2731, 2738, 2762, 2747, 2745, 2751, 2761, 2760, 2750, 2811, 2653, 2742, 2662]
        aa_no = [2752, 2748, 2733]
        lb = lb.copy(); ub = ub.copy()
        for ridx in aa_no:
            if 1 <= ridx <= nv:
                lb[ridx - 1] = 0.0; ub[ridx - 1] = 0.0

        # Enforce sign conventions on key exchange/biomass reactions
        # - Glucose, Fructose, YAN cannot be exported: ub = 0, allow uptake (lb <= 0)
        # - Ethanol cannot be consumed: lb = 0, allow production (ub >= 0)
        # - Biomass growth flux non-negative: lb = 0
        if 1 <= idx_glu <= nv:
            lb[idx_glu - 1] = min(lb[idx_glu - 1], 0.0)
            ub[idx_glu - 1] = 0.0
        if 1 <= idx_fru <= nv:
            lb[idx_fru - 1] = min(lb[idx_fru - 1], 0.0)
            ub[idx_fru - 1] = 0.0
        if 1 <= idx_nh4 <= nv:
            lb[idx_nh4 - 1] = min(lb[idx_nh4 - 1], 0.0)
            ub[idx_nh4 - 1] = 0.0
        if 1 <= idx_eth <= nv:
            lb[idx_eth - 1] = max(lb[idx_eth - 1], 0.0)
        if 1 <= idx_obj <= nv:
            lb[idx_obj - 1] = max(lb[idx_obj - 1], 0.0)

        m.NV = pyo.RangeSet(1, nv)
        m.NM = pyo.RangeSet(1, nm)
        m.vlb = pyo.Param(m.NV, initialize=lambda mdl, k: float(lb[k - 1]), mutable=False)
        m.vub = pyo.Param(m.NV, initialize=lambda mdl, k: float(ub[k - 1]), mutable=False)
        m.v = pyo.Var(m.NV, m.I, domain=pyo.Reals, initialize=0.0)
        m.lmbda = pyo.Var(m.NM, m.I, domain=pyo.Reals, initialize=0.0)
        m.alpha_L = pyo.Var(m.NV, m.I, domain=pyo.Reals, initialize=0.0)
        m.alpha_U = pyo.Var(m.NV, m.I, domain=pyo.Reals, initialize=0.0)
        m.alphaL_sign = pyo.Constraint(m.NV, m.I, rule=lambda mdl, k, i: mdl.alpha_L[k, i] <= 0.0)
        m.alphaU_sign = pyo.Constraint(m.NV, m.I, rule=lambda mdl, k, i: mdl.alpha_U[k, i] >= 0.0)
        m.v_UB = pyo.Constraint(m.NV, m.I, rule=lambda mdl, k, i: mdl.v[k, i] - mdl.vub[k] <= 0.0)
        m.v_LB = pyo.Constraint(m.NV, m.I, rule=lambda mdl, k, i: -mdl.v[k, i] + mdl.vlb[k] <= 0.0)
        def _Sc(mdl, r, i):
            rr = int(r); nz_cols = row_to_cols.get(rr, [])
            return sum(S_vals[(rr, k)] * mdl.v[k, i] for k in nz_cols) == 0.0
        m.Sc = pyo.Constraint(m.NM, m.I, rule=_Sc)
        # Stationarity ridge as a mutable Param so we can schedule it later
        m.stat_w = pyo.Param(initialize=float(args.stat_w), mutable=True)
        vs = np.ones(nv)
        def _Lagr(mdl, k, i):
            rr_list = col_to_rows.get(int(k), [])
            term_S = sum(S_vals[(r, int(k))] * mdl.lmbda[r, i] for r in rr_list)
            return mdl.stat_w * mdl.v[k, i] * vs[int(k) - 1] + mdl.alpha_L[k, i] + mdl.alpha_U[k, i] + term_S == 0.0
        m.Lagr = pyo.Constraint(m.NV, m.I, rule=_Lagr)
        m.FO_L = pyo.Var(m.NV, m.I, domain=pyo.Reals, initialize=0.0)
        m.FO_U = pyo.Var(m.NV, m.I, domain=pyo.Reals, initialize=0.0)
        m.FO_L_def = pyo.Constraint(m.NV, m.I, rule=lambda mdl, k, i: mdl.FO_L[k, i] == (mdl.v[k, i] - mdl.vlb[k]) * mdl.alpha_L[k, i])
        m.FO_U_def = pyo.Constraint(m.NV, m.I, rule=lambda mdl, k, i: mdl.FO_U[k, i] == (mdl.v[k, i] - mdl.vub[k]) * mdl.alpha_U[k, i])
        # Usar valores en fin de elemento (FE-end) para los amarres
        def _rG(mdl, i):
            return ( mdl.mu_fe[i] / pyo.exp(mdl.theta['Yxg']) ) + ( mdl.betaG_fe[i] / pyo.exp(mdl.theta['Yeg']) ) + ( mdl.mrate * mdl.phiG_fe[i] )
        def _rF(mdl, i):
            return ( mdl.mu_fe[i] / pyo.exp(mdl.theta['Yxf']) ) + ( mdl.betaF_fe[i] / pyo.exp(mdl.theta['Yef']) ) + ( mdl.mrate * mdl.phiF_fe[i] )
        m.rG = pyo.Expression(m.I, rule=_rG)
        m.rF = pyo.Expression(m.I, rule=_rF)
        m.tie_mu = pyo.Constraint(m.I, rule=lambda mdl, i: mdl.v[idx_obj, i] == mdl.mu_fe[i])
        # factor 0.04607 (kg/mol?) heredado; mantener coherencia con el modelo metabólico
        m.tie_eth = pyo.Constraint(m.I, rule=lambda mdl, i: mdl.v[idx_eth, i] == (mdl.betaG_fe[i] + mdl.betaF_fe[i]) / 0.04607)
        m.tie_glu = pyo.Constraint(m.I, rule=lambda mdl, i: -mdl.v[idx_glu, i] == mdl.rG[i])
        m.tie_fru = pyo.Constraint(m.I, rule=lambda mdl, i: -mdl.v[idx_fru, i] == mdl.rF[i])
        def _vYAN(mdl, i):
            total = mdl.v[idx_nh4, i]
            for ridx in aa_yes:
                if 1 <= ridx <= nv:
                    total = total + mdl.v[ridx, i]
            return -total
        m.vYAN = pyo.Expression(m.I, rule=_vYAN)
        m.tie_YAN = pyo.Constraint(m.I, rule=lambda mdl, i: mdl.vYAN[i] == mdl.mu_fe[i] / pyo.exp(mdl.theta['Yxn']))
        # Inicialización inteligente de v para acelerar convergencia (igualar RHS de amarres)
        nfe_local = int(pyo.value(m.I.last()))
        for i in range(1, nfe_local + 1):
            try:
                v_mu = float(pyo.value(m.mu_fe[i]))
                v_eth = float(pyo.value((m.betaG_fe[i] + m.betaF_fe[i]) / 0.04607))
                rG_i = float(pyo.value(m.rG[i]))
                rF_i = float(pyo.value(m.rF[i]))
                rhsyan = float(pyo.value(m.mu_fe[i] / pyo.exp(m.theta['Yxn'])))
                # Set tied fluxes
                if 1 <= idx_obj <= nv:
                    m.v[idx_obj, i].set_value(v_mu)
                if 1 <= idx_eth <= nv:
                    m.v[idx_eth, i].set_value(v_eth)
                if 1 <= idx_glu <= nv:
                    m.v[idx_glu, i].set_value(-rG_i)
                if 1 <= idx_fru <= nv:
                    m.v[idx_fru, i].set_value(-rF_i)
                # Initialize YAN via NH4 only (AAs yes at 0)
                if 1 <= idx_nh4 <= nv:
                    lb_n = float(pyo.value(m.vlb[idx_nh4])); ub_n = float(pyo.value(m.vub[idx_nh4]))
                    vnh4_init = -rhsyan
                    vnh4_init = max(lb_n, min(ub_n, vnh4_init))
                    m.v[idx_nh4, i].set_value(vnh4_init)
                for ridx in aa_yes:
                    if 1 <= ridx <= nv:
                        # mantener 0 dentro de cotas
                        lb_k = float(pyo.value(m.vlb[ridx])); ub_k = float(pyo.value(m.vub[ridx]))
                        val0 = 0.0
                        if val0 < lb_k: val0 = lb_k
                        if val0 > ub_k: val0 = ub_k
                        m.v[ridx, i].set_value(val0)
            except Exception:
                pass

        # Pre-solve projection onto S·v=0 with bounds (warm-start), optional
        if args.pre_solve_ties:
            try:
                t_ps0 = time.time()
                # Precompute products for projection (dense; memory heavy but one-time)
                A = S @ S.T  # (nm x nm)
                # Ridge for stability
                A_reg = A + (float(args.pre_solve_delta) * np.eye(A.shape[0]))
                ST = S.T
                for i in range(1, nfe_local + 1):
                    # v0 from ties
                    v0 = np.zeros(nv, dtype=float)
                    try:
                        v0[idx_obj - 1] = float(pyo.value(m.mu_fe[i]))
                        v0[idx_eth - 1] = float(pyo.value((m.betaG_fe[i] + m.betaF_fe[i]) / 0.04607))
                        v0[idx_glu - 1] = -float(pyo.value(m.rG[i]))
                        v0[idx_fru - 1] = -float(pyo.value(m.rF[i]))
                        v0[idx_nh4 - 1] = -float(pyo.value(m.mu_fe[i] / pyo.exp(m.theta['Yxn'])))
                    except Exception:
                        pass
                    # equality projection: solve (S S^T + delta I) lambda = S v0
                    b = S @ v0
                    try:
                        lam = np.linalg.solve(A_reg, b)
                    except Exception:
                        lam, *_ = np.linalg.lstsq(A_reg, b, rcond=None)
                    v_proj = (v0 - ST @ lam) / (1.0 + float(args.pre_solve_delta))
                    # Clip to bounds
                    v_proj = np.minimum(ub, np.maximum(lb, v_proj))
                    # Apply warm-start
                    for k in range(1, nv + 1):
                        try:
                            m.v[k, i].set_value(float(v_proj[k - 1]))
                        except Exception:
                            pass
                print(f"[PRE-SOLVE] Projection completed in {time.time()-t_ps0:.2f}s (dense)")
            except Exception as e:
                print(f"[WARN] Pre-solve projection failed or ran out of memory: {e}")

        # Initialize KKT multipliers (alpha_L/alpha_U for bounds, lambda for S·v=0) to help Ipopt
        try:
            bndtol = 1e-7
            alpha_seed = max(1e-6, float(pyo.value(m.stat_w)) * 1e-2)
            # Precompute transpose once for lambda least-squares
            ST = S.T  # (nv x nm)
            for i in range(1, nfe_local + 1):
                # Snapshot current v after warm-start/pre-solve
                v_i = np.zeros(nv, dtype=float)
                for k in range(1, nv + 1):
                    try:
                        v_i[k - 1] = float(pyo.value(m.v[k, i]))
                    except Exception:
                        v_i[k - 1] = 0.0
                # Initialize alpha based on bound activity
                alpha_L_i = np.zeros(nv, dtype=float)
                alpha_U_i = np.zeros(nv, dtype=float)
                for k in range(1, nv + 1):
                    lb_k = float(pyo.value(m.vlb[k])); ub_k = float(pyo.value(m.vub[k]))
                    vk = v_i[k - 1]
                    if vk - lb_k <= bndtol:
                        alpha_L_i[k - 1] = -alpha_seed  # alpha_L <= 0
                    if ub_k - vk <= bndtol:
                        alpha_U_i[k - 1] = +alpha_seed  # alpha_U >= 0
                    # Write initial values
                    try:
                        m.alpha_L[k, i].set_value(alpha_L_i[k - 1])
                        m.alpha_U[k, i].set_value(alpha_U_i[k - 1])
                    except Exception:
                        pass
                # Least-squares initialize lambda: S^T lambda ≈ -(stat_w*v + alpha_L + alpha_U)
                try:
                    stat_w_val = float(pyo.value(m.stat_w))
                except Exception:
                    stat_w_val = float(args.stat_w)
                b = -(stat_w_val * v_i + alpha_L_i + alpha_U_i)  # shape (nv,)
                try:
                    # Solve min || ST * lam - b ||_2
                    lam, *_ = np.linalg.lstsq(ST, b, rcond=None)
                except Exception:
                    # Fallback to zeros if linalg fails
                    lam = np.zeros(S.shape[0], dtype=float)
                # Apply to model
                for r in range(1, S.shape[0] + 1):
                    try:
                        m.lmbda[r, i].set_value(float(lam[r - 1]))
                    except Exception:
                        pass
            print("[INIT] Multipliers initialized: alpha (by activity) and lambda (LSQ)")
        except Exception as e:
            print(f"[WARN] Multiplier initialization failed: {e}")

        # Objetivo: si estamos en simulate_mpcc, penalizaciones con pesos configurables; si no, mantener SSE + penalizaciones
        # Eliminar OBJ previo para evitar el warning de reemplazo implícito
        try:
            m.del_component('OBJ')
        except Exception:
            try:
                m.del_component(m.OBJ)
            except Exception:
                pass
        if args.simulate_mpcc:
            reg = args.reg_w * sum( sum(m.v[k, i]**2 for k in m.NV) for i in m.I )
            compl = sum( sum( (m.FO_L[k, i])**2 + (m.FO_U[k, i])**2 for k in m.NV ) for i in m.I )
            m.OBJ = pyo.Objective(expr=reg + args.compl_w * compl, sense=pyo.minimize)
        else:
            compl = sum( sum( (m.FO_L[k, i])**2 + (m.FO_U[k, i])**2 for k in m.NV ) for i in m.I )
            m.OBJ = pyo.Objective(expr=1e2 * m.FO + compl, sense=pyo.minimize)

    # Optional penalty schedule for simulate_mpcc
    res = None; wall = None
    if args.simulate_mpcc and args.compl_schedule:
        def _parse_sched(s):
            try:
                parts = [p.strip() for p in str(s).split(',') if p.strip() != '']
                return [float(p) for p in parts]
            except Exception:
                return None
        sched_c = _parse_sched(args.compl_schedule)
        sched_w = _parse_sched(args.stat_schedule) if args.stat_schedule else None
        if sched_c is None or len(sched_c) == 0:
            print("[WARN] --compl_schedule inválido; se ejecutará una sola etapa con --compl_w")
            # Rebuild objective for single stage
            try:
                m.del_component('OBJ')
            except Exception:
                try:
                    m.del_component(m.OBJ)
                except Exception:
                    pass
            reg = args.reg_w * sum( sum(m.v[k, i]**2 for k in m.NV) for i in m.I )
            compl = sum( sum( (m.FO_L[k, i])**2 + (m.FO_U[k, i])**2 for k in m.NV ) for i in m.I )
            m.OBJ = pyo.Objective(expr=reg + args.compl_w * compl, sense=pyo.minimize)
            res, wall = solve_model(m, ipopt_max_iter=args.ipopt_max_iter, print_solver=args.print_solver,
                                    timings=args.timings, lbfgs=args.lbfgs, max_wall_time=args.max_wall_time)
        else:
            stage_summaries = []
            def _collect_stage_metrics(stage_label:str, cw:float, sw:float, wall_time:float, res_obj):
                # Complementarity residuals per FE
                nfe_local = int(args.nfe)
                comp_res = []
                for i in range(1, nfe_local + 1):
                    s = 0.0
                    for k in m.NV:
                        try:
                            s += float(pyo.value(m.FO_L[k, i])**2 + pyo.value(m.FO_U[k, i])**2)
                        except Exception:
                            pass
                    comp_res.append(s**0.5)
                comp_max = float(np.nanmax(comp_res)) if len(comp_res) else float('nan')
                comp_mean = float(np.nanmean(comp_res)) if len(comp_res) else float('nan')
                # Bound activity: mean across FE
                tol_act = 1e-6
                actLB = []; actUB = []
                for i in range(1, nfe_local + 1):
                    actL = 0; actU = 0
                    for k in m.NV:
                        try:
                            vki = float(pyo.value(m.v[k, i]))
                            if abs(vki - float(pyo.value(m.vlb[k]))) <= tol_act:
                                actL += 1
                            if abs(vki - float(pyo.value(m.vub[k]))) <= tol_act:
                                actU += 1
                        except Exception:
                            pass
                    actLB.append(actL); actUB.append(actU)
                lb_mean = float(np.nanmean(actLB)) if len(actLB) else float('nan')
                ub_mean = float(np.nanmean(actUB)) if len(actUB) else float('nan')
                # Tie gaps (abs and relative) — aggregate max values
                def _safe_val(x):
                    try:
                        return float(pyo.value(x))
                    except Exception:
                        return np.nan
                gaps = { 'mu':[], 'eth':[], 'glu':[], 'fru':[], 'yan':[] }
                gaps_rel = { 'mu':[], 'eth':[], 'glu':[], 'fru':[], 'yan':[] }
                _eps = 1e-9
                for i in range(1, nfe_local + 1):
                    mu_rhs = _safe_val(m.mu_fe[i])
                    eth_rhs = _safe_val((m.betaG_fe[i] + m.betaF_fe[i]) / 0.04607)
                    rG_i = _safe_val(m.rG[i]) if hasattr(m, 'rG') else np.nan
                    rF_i = _safe_val(m.rF[i]) if hasattr(m, 'rF') else np.nan
                    yan_rhs = _safe_val(m.mu_fe[i] / pyo.exp(m.theta['Yxn']))
                    vmu = _safe_val(m.v[3414, i]) if 1 <= 3414 <= int(pyo.value(m.NV.last())) else np.nan
                    veth = _safe_val(m.v[2630, i]) if 1 <= 2630 <= int(pyo.value(m.NV.last())) else np.nan
                    vglu = _safe_val(m.v[2588, i]) if 1 <= 2588 <= int(pyo.value(m.NV.last())) else np.nan
                    vfru = _safe_val(m.v[2583, i]) if 1 <= 2583 <= int(pyo.value(m.NV.last())) else np.nan
                    vyan = _safe_val(m.vYAN[i]) if hasattr(m, 'vYAN') else np.nan
                    g_mu = abs(vmu - mu_rhs); g_eth = abs(veth - eth_rhs)
                    g_glu = abs(-vglu - rG_i); g_fru = abs(-vfru - rF_i); g_yan = abs(vyan - yan_rhs)
                    gaps['mu'].append(g_mu); gaps['eth'].append(g_eth); gaps['glu'].append(g_glu); gaps['fru'].append(g_fru); gaps['yan'].append(g_yan)
                    gaps_rel['mu'].append(g_mu / (abs(mu_rhs) + _eps))
                    gaps_rel['eth'].append(g_eth / (abs(eth_rhs) + _eps))
                    gaps_rel['glu'].append(g_glu / (abs(rG_i) + _eps))
                    gaps_rel['fru'].append(g_fru / (abs(rF_i) + _eps))
                    gaps_rel['yan'].append(g_yan / (abs(yan_rhs) + _eps))
                # Objective split (evaluate numerically)
                reg_raw = 0.0; compl_raw = 0.0
                for i in range(1, nfe_local + 1):
                    for k in m.NV:
                        try:
                            vk = float(pyo.value(m.v[k, i])); reg_raw += vk * vk
                            fl = float(pyo.value(m.FO_L[k, i])); fu = float(pyo.value(m.FO_U[k, i])); compl_raw += fl*fl + fu*fu
                        except Exception:
                            pass
                summary = {
                    'stage': stage_label,
                    'compl_w': float(cw),
                    'stat_w': float(sw),
                    'wall_time_sec': float(wall_time),
                    'solver_status': str(res_obj.solver.status),
                    'termination': str(res_obj.solver.termination_condition),
                    'obj_reg_term': float(args.reg_w * reg_raw),
                    'obj_compl_term_raw': float(compl_raw),
                    'obj_total': float(args.reg_w * reg_raw + cw * compl_raw),
                    'comp_res_l2_max': comp_max,
                    'comp_res_l2_mean': comp_mean,
                    'act_bounds_lb_mean': lb_mean,
                    'act_bounds_ub_mean': ub_mean,
                    'gap_abs_max_mu': float(np.nanmax(gaps['mu'])) if gaps['mu'] else float('nan'),
                    'gap_abs_max_eth': float(np.nanmax(gaps['eth'])) if gaps['eth'] else float('nan'),
                    'gap_abs_max_glu': float(np.nanmax(gaps['glu'])) if gaps['glu'] else float('nan'),
                    'gap_abs_max_fru': float(np.nanmax(gaps['fru'])) if gaps['fru'] else float('nan'),
                    'gap_abs_max_yan': float(np.nanmax(gaps['yan'])) if gaps['yan'] else float('nan'),
                    'gap_rel_max_mu': float(np.nanmax(gaps_rel['mu'])) if gaps_rel['mu'] else float('nan'),
                    'gap_rel_max_eth': float(np.nanmax(gaps_rel['eth'])) if gaps_rel['eth'] else float('nan'),
                    'gap_rel_max_glu': float(np.nanmax(gaps_rel['glu'])) if gaps_rel['glu'] else float('nan'),
                    'gap_rel_max_fru': float(np.nanmax(gaps_rel['fru'])) if gaps_rel['fru'] else float('nan'),
                    'gap_rel_max_yan': float(np.nanmax(gaps_rel['yan'])) if gaps_rel['yan'] else float('nan'),
                }
                stage_summaries.append(summary)

            stages = len(sched_c)
            print(f"[SCHEDULE] Running {stages} stage(s) with compl_w={sched_c} and stat_w={(sched_w if sched_w else [args.stat_w]*stages)}")
            for t in range(stages):
                cw = float(sched_c[t])
                sw = float(sched_w[t]) if (sched_w and t < len(sched_w)) else float(args.stat_w)
                # Update stationarity ridge
                try:
                    m.stat_w.set_value(sw)
                except Exception:
                    pass
                # Rebuild objective
                try:
                    m.del_component('OBJ')
                except Exception:
                    try:
                        m.del_component(m.OBJ)
                    except Exception:
                        pass
                reg = args.reg_w * sum( sum(m.v[k, i]**2 for k in m.NV) for i in m.I )
                compl = sum( sum( (m.FO_L[k, i])**2 + (m.FO_U[k, i])**2 for k in m.NV ) for i in m.I )
                m.OBJ = pyo.Objective(expr=reg + cw * compl, sense=pyo.minimize)
                print(f"[SCHEDULE] Stage {t+1}/{stages}: compl_w={cw:g}, stat_w={sw:g}")
                res, wall = solve_model(m, ipopt_max_iter=args.ipopt_max_iter, print_solver=args.print_solver,
                                        timings=args.timings, lbfgs=args.lbfgs, max_wall_time=args.max_wall_time)
                print(f"[SCHEDULE] Stage {t+1} done: status={res.solver.status}, term={res.solver.termination_condition}")
                _collect_stage_metrics(stage_label=f"stage_{t+1}", cw=cw, sw=sw, wall_time=wall, res_obj=res)

            # Optional final polish stage with stricter tolerances
            if args.polish_last_stage:
                cw_last = float(sched_c[-1]) if (sched_c and len(sched_c) > 0) else float(args.compl_w)
                try:
                    m.stat_w.set_value(float(args.polish_stat_w))
                except Exception:
                    pass
                try:
                    m.del_component('OBJ')
                except Exception:
                    try:
                        m.del_component(m.OBJ)
                    except Exception:
                        pass
                reg = args.reg_w * sum( sum(m.v[k, i]**2 for k in m.NV) for i in m.I )
                compl = sum( sum( (m.FO_L[k, i])**2 + (m.FO_U[k, i])**2 for k in m.NV ) for i in m.I )
                m.OBJ = pyo.Objective(expr=reg + cw_last * compl, sense=pyo.minimize)
                print(f"[POLISH] Final stage: compl_w={cw_last:g}, stat_w={float(args.polish_stat_w):g}, tol={args.polish_tol:g}, acceptable_tol={args.polish_acceptable_tol:g}, acceptable_iter={int(args.polish_acceptable_iter)}")
                res, wall = solve_model(
                    m,
                    ipopt_max_iter=args.ipopt_max_iter,
                    print_solver=args.print_solver,
                    timings=args.timings,
                    lbfgs=False,
                    max_wall_time=float(args.polish_wall_time),
                    tol=float(args.polish_tol),
                    acceptable_tol=float(args.polish_acceptable_tol),
                    acceptable_iter=int(args.polish_acceptable_iter),
                )
                print(f"[POLISH] Done: status={res.solver.status}, term={res.solver.termination_condition}")
                _collect_stage_metrics(stage_label="polish", cw=cw_last, sw=float(args.polish_stat_w), wall_time=wall, res_obj=res)

            # Persist per-stage summaries (JSON and CSV)
            try:
                run_meta = {
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    'nfe': int(args.nfe), 'ncp': int(args.ncp), 'th': float(args.th),
                    'lbfgs_stages': bool(args.lbfgs), 'lbfgs_polish': False,
                    'compl_schedule': sched_c,
                    'stat_schedule': (sched_w if sched_w else [args.stat_w]*stages),
                    'polish_last_stage': bool(args.polish_last_stage),
                    'polish_stat_w': float(args.polish_stat_w) if args.polish_last_stage else None,
                    'max_wall_time_per_stage': float(args.max_wall_time) if args.max_wall_time is not None else None,
                    'polish_wall_time': float(args.polish_wall_time) if args.polish_last_stage else None,
                }
                summary_json = {
                    'run_meta': run_meta,
                    'stages': stage_summaries,
                }
                json_path = os.path.join(RESULTS_DIR, 'mpcc_stage_summary.json')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(summary_json, f, indent=2)
                # CSV flat export
                try:
                    import csv
                    csv_path = os.path.join(RESULTS_DIR, 'mpcc_stage_summary.csv')
                    if stage_summaries:
                        keys = list(stage_summaries[0].keys())
                        with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
                            wr = csv.DictWriter(cf, fieldnames=keys)
                            wr.writeheader()
                            for row in stage_summaries:
                                wr.writerow(row)
                except Exception as e:
                    print(f"[WARN] Failed to write mpcc_stage_summary.csv: {e}")
                print(f"[STAGE-SUMMARY] Saved {json_path}")
            except Exception as e:
                print(f"[WARN] Failed to persist stage summaries: {e}")
    else:
        res, wall = solve_model(m, ipopt_max_iter=args.ipopt_max_iter, print_solver=args.print_solver,
                                timings=args.timings, lbfgs=args.lbfgs, max_wall_time=args.max_wall_time)

    print("\n=== Zenteno model solve summary ===")
    print(f"Solver status: {res.solver.status}, termination: {res.solver.termination_condition}")
    print(f"Wall time: {wall:.2f} s")
    param_vals = {n: pyo.value(pyo.exp(m.theta[n])) for n in m.PN}
    print("Parameters (current values):")
    for k, v in param_vals.items():
        print(f"  {k} = {v:.6g}")
    if args.simulate_mpcc:
        # Report MPCC objective parts instead of SSE
        try:
            nfe_local = int(args.nfe)
            reg_raw = 0.0; compl_raw = 0.0
            for i in range(1, nfe_local + 1):
                for k in m.NV:
                    try:
                        vk = float(pyo.value(m.v[k, i])); reg_raw += vk * vk
                        fl = float(pyo.value(m.FO_L[k, i])); fu = float(pyo.value(m.FO_U[k, i])); compl_raw += fl*fl + fu*fu
                    except Exception:
                        pass
            # Use last compl weight if schedule provided
            cw_last = None
            if args.compl_schedule:
                try:
                    cw_last = [float(p.strip()) for p in str(args.compl_schedule).split(',') if p.strip()][-1]
                except Exception:
                    cw_last = None
            if cw_last is None:
                cw_last = float(args.compl_w)
            print(f"OBJ (MPCC): reg={args.reg_w*reg_raw:.6g}, compl_raw={compl_raw:.6g}, total={args.reg_w*reg_raw + cw_last*compl_raw:.6g}")
        except Exception:
            pass
    else:
        print(f"FO (data-fit SSE): {pyo.value(m.FO):.6g}")

    # MPCC diagnostics: complementarity and flux activity summaries
    if args.enable_mpcc:
        try:
            nfe = args.nfe
            # Complementarity residual per FE (L2 norm of FO_L/FO_U)
            comp_res = []
            for i in range(1, nfe + 1):
                s = 0.0
                for k in m.NV:
                    s += float(pyo.value(m.FO_L[k, i])**2 + pyo.value(m.FO_U[k, i])**2)
                comp_res.append(s**0.5)
            worst = max(( (comp_res[i-1], i) for i in range(1, nfe + 1) ), key=lambda t: t[0])
            print(f"[MPCC] Complementarity L2 residual per FE (first 5): {[round(x,3) for x in comp_res[:5]]} ... max={worst[0]:.3g} at FE {worst[1]}")

            # Bound activity counts per FE
            tol = 1e-6
            for i in range(1, nfe + 1):
                actL = 0; actU = 0
                for k in m.NV:
                    vki = float(pyo.value(m.v[k, i]))
                    if abs(vki - float(pyo.value(m.vlb[k]))) <= tol:
                        actL += 1
                    if abs(vki - float(pyo.value(m.vub[k]))) <= tol:
                        actU += 1
                if i <= 5 or i == nfe:
                    print(f"[MPCC] FE {i}: active LB={actL}, UB={actU}")

            # Key tied fluxes preview (first 5 FE)
            preview = min(5, nfe)
            if hasattr(m, 'NV'):
                idx_obj = 3414; idx_eth = 2630; idx_glu = 2588; idx_fru = 2583; idx_nh4 = 2536
                print("[MPCC] Preview tied fluxes (first FEs):")
                for i in range(1, preview + 1):
                    try:
                        v_obj = float(pyo.value(m.v[idx_obj, i]))
                        v_eth = float(pyo.value(m.v[idx_eth, i]))
                        v_glu = float(pyo.value(m.v[idx_glu, i]))
                        v_fru = float(pyo.value(m.v[idx_fru, i]))
                        print(f"  FE {i}: v_obj={v_obj:.4g}, v_eth={v_eth:.4g}, v_glu={v_glu:.4g}, v_fru={v_fru:.4g}")
                    except Exception:
                        break

            # Plots: fluxes, bound activity and complementarity residuals, and tie gaps
            try:
                ncp = args.ncp; th = args.th; h = th / nfe
                t_fe = [i * h for i in range(1, nfe + 1)]
                # Extract flux series safely
                def _safe_v(idx):
                    vals = []
                    for i in range(1, nfe + 1):
                        try:
                            vals.append(float(pyo.value(m.v[idx, i])))
                        except Exception:
                            vals.append(np.nan)
                    return vals
                v_obj = _safe_v(3414)
                v_eth = _safe_v(2630)
                v_glu = _safe_v(2588)
                v_fru = _safe_v(2583)
                # vYAN from expression
                v_yan = []
                for i in range(1, nfe + 1):
                    try:
                        v_yan.append(float(pyo.value(m.vYAN[i])))
                    except Exception:
                        v_yan.append(np.nan)

                # Plot with standard FBA sign convention (negative = uptake, positive = production)
                fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
                axes[0].plot(t_fe, v_obj, '-o', ms=3, lw=1.2); axes[0].set_ylabel('v_obj')
                axes[1].plot(t_fe, v_eth, '-o', ms=3, lw=1.2); axes[1].set_ylabel('v_eth')
                axes[2].plot(t_fe, v_glu, '-o', ms=3, lw=1.2); axes[2].set_ylabel('v_glu (uptake<0)')
                axes[3].plot(t_fe, v_fru, '-o', ms=3, lw=1.2); axes[3].set_ylabel('v_fru (uptake<0)')
                axes[4].plot(t_fe, v_yan, '-o', ms=3, lw=1.2); axes[4].set_ylabel('v_YAN (uptake<0)'); axes[4].set_xlabel('Time')
                for ax in axes: ax.grid(alpha=0.2)
                fig.suptitle('MPCC fluxes (standard signs: uptake negative)')
                fig.tight_layout(rect=[0, 0.03, 1, 0.96])
                flux_png = os.path.join(RESULTS_DIR, 'mpcc_fluxes.png')
                fig.savefig(flux_png, dpi=150); plt.close(fig)
                print(f"[PLOT] Saved {flux_png}")

                # Bound activity arrays
                actLB = []; actUB = []
                for i in range(1, nfe + 1):
                    actL = 0; actU = 0
                    for k in m.NV:
                        vki = float(pyo.value(m.v[k, i]))
                        if abs(vki - float(pyo.value(m.vlb[k]))) <= tol:
                            actL += 1
                        if abs(vki - float(pyo.value(m.vub[k]))) <= tol:
                            actU += 1
                    actLB.append(actL); actUB.append(actU)
                plt.figure(figsize=(7,4))
                plt.plot(t_fe, actLB, '-o', ms=3, lw=1.2, label='LB active')
                plt.plot(t_fe, actUB, '-o', ms=3, lw=1.2, label='UB active')
                plt.xlabel('Time'); plt.ylabel('#active bounds'); plt.legend(); plt.grid(alpha=0.2); plt.tight_layout()
                ba_png = os.path.join(RESULTS_DIR, 'mpcc_bound_activity.png')
                plt.savefig(ba_png, dpi=150); plt.close()
                print(f"[PLOT] Saved {ba_png}")

                # Complementarity residuals
                plt.figure(figsize=(7,4))
                plt.plot(t_fe, comp_res, '-o', ms=3, lw=1.2)
                plt.xlabel('Time'); plt.ylabel('||compl||_2 per FE'); plt.grid(alpha=0.2); plt.tight_layout()
                cr_png = os.path.join(RESULTS_DIR, 'mpcc_complementarity_residual.png')
                plt.savefig(cr_png, dpi=150); plt.close()
                print(f"[PLOT] Saved {cr_png}")

                # Tie gaps
                gaps_mu = []; gaps_eth = []; gaps_glu = []; gaps_fru = []; gaps_yan = []
                # Relative gaps (normalized by rhs magnitude)
                gaps_mu_rel = []; gaps_eth_rel = []; gaps_glu_rel = []; gaps_fru_rel = []; gaps_yan_rel = []
                _eps = 1e-9
                for i in range(1, nfe + 1):
                    try:
                        mu_rhs = float(pyo.value(m.mu_fe[i]))
                        eth_rhs = float(pyo.value((m.betaG_fe[i] + m.betaF_fe[i]) / 0.04607))
                        rG_i = float(pyo.value(m.rG[i]))
                        rF_i = float(pyo.value(m.rF[i]))
                        yan_rhs = float(pyo.value(m.mu_fe[i] / pyo.exp(m.theta['Yxn'])))
                        vmu = float(pyo.value(m.v[3414, i])) if 1 <= 3414 <= pyo.value(m.NV.last()) else np.nan
                        veth = float(pyo.value(m.v[2630, i])) if 1 <= 2630 <= pyo.value(m.NV.last()) else np.nan
                        vglu = float(pyo.value(m.v[2588, i])) if 1 <= 2588 <= pyo.value(m.NV.last()) else np.nan
                        vfru = float(pyo.value(m.v[2583, i])) if 1 <= 2583 <= pyo.value(m.NV.last()) else np.nan
                        vyan = float(pyo.value(m.vYAN[i]))
                        g_mu = abs(vmu - mu_rhs)
                        g_eth = abs(veth - eth_rhs)
                        g_glu = abs(-vglu - rG_i)
                        g_fru = abs(-vfru - rF_i)
                        g_yan = abs(vyan - yan_rhs)
                        gaps_mu.append(g_mu); gaps_eth.append(g_eth); gaps_glu.append(g_glu); gaps_fru.append(g_fru); gaps_yan.append(g_yan)
                        gaps_mu_rel.append(g_mu / (abs(mu_rhs) + _eps))
                        gaps_eth_rel.append(g_eth / (abs(eth_rhs) + _eps))
                        gaps_glu_rel.append(g_glu / (abs(rG_i) + _eps))
                        gaps_fru_rel.append(g_fru / (abs(rF_i) + _eps))
                        gaps_yan_rel.append(g_yan / (abs(yan_rhs) + _eps))
                    except Exception:
                        gaps_mu.append(np.nan); gaps_eth.append(np.nan); gaps_glu.append(np.nan); gaps_fru.append(np.nan); gaps_yan.append(np.nan)
                        gaps_mu_rel.append(np.nan); gaps_eth_rel.append(np.nan); gaps_glu_rel.append(np.nan); gaps_fru_rel.append(np.nan); gaps_yan_rel.append(np.nan)
                fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
                axes[0].plot(t_fe, gaps_mu, '-o', ms=3, lw=1.2); axes[0].set_ylabel('|gap_mu|')
                axes[1].plot(t_fe, gaps_eth, '-o', ms=3, lw=1.2); axes[1].set_ylabel('|gap_eth|')
                axes[2].plot(t_fe, gaps_glu, '-o', ms=3, lw=1.2); axes[2].set_ylabel('|gap_glu|')
                axes[3].plot(t_fe, gaps_fru, '-o', ms=3, lw=1.2); axes[3].set_ylabel('|gap_fru|')
                axes[4].plot(t_fe, gaps_yan, '-o', ms=3, lw=1.2); axes[4].set_ylabel('|gap_YAN|'); axes[4].set_xlabel('Time')
                for ax in axes: ax.grid(alpha=0.2)
                fig.suptitle('MPCC tie gaps')
                fig.tight_layout(rect=[0, 0.03, 1, 0.96])
                gaps_png = os.path.join(RESULTS_DIR, 'mpcc_tie_gaps.png')
                fig.savefig(gaps_png, dpi=150); plt.close(fig)
                print(f"[PLOT] Saved {gaps_png}")

                # Relative tie gaps figure
                fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
                axes[0].plot(t_fe, gaps_mu_rel, '-o', ms=3, lw=1.2); axes[0].set_ylabel('rel_gap_mu')
                axes[1].plot(t_fe, gaps_eth_rel, '-o', ms=3, lw=1.2); axes[1].set_ylabel('rel_gap_eth')
                axes[2].plot(t_fe, gaps_glu_rel, '-o', ms=3, lw=1.2); axes[2].set_ylabel('rel_gap_glu')
                axes[3].plot(t_fe, gaps_fru_rel, '-o', ms=3, lw=1.2); axes[3].set_ylabel('rel_gap_fru')
                axes[4].plot(t_fe, gaps_yan_rel, '-o', ms=3, lw=1.2); axes[4].set_ylabel('rel_gap_YAN'); axes[4].set_xlabel('Time')
                for ax in axes: ax.grid(alpha=0.2)
                fig.suptitle('MPCC tie gaps (relative)')
                fig.tight_layout(rect=[0, 0.03, 1, 0.96])
                gaps_rel_png = os.path.join(RESULTS_DIR, 'mpcc_tie_gaps_relative.png')
                fig.savefig(gaps_rel_png, dpi=150); plt.close(fig)
                print(f"[PLOT] Saved {gaps_rel_png}")
            except Exception as e:
                print(f"[WARN] MPCC plotting failed: {e}")
        except Exception as e:
            print(f"[WARN] MPCC diagnostics failed: {e}")

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
                print("[PLOT] Legend: Sim = tab:blue line, Data = tab:orange x (FE ends)")
            plt.figure(figsize=(7, 4))
            plt.plot(t_line, y_line, '-', linewidth=1.6, alpha=0.9, label="Sim", color="tab:blue")
            if len(t_dat) > 0:
                plt.scatter(t_dat, y_dat, s=28, alpha=0.9, label="Data", marker="x", color="tab:orange")
            plt.xlabel("Time"); plt.ylabel(f"State {state_names.get(l, l)}")
            plt.title(f"Data vs Sim — State {state_names.get(l, l)}")
            plt.legend(); plt.tight_layout()
            out_png = os.path.join(RESULTS_DIR, f"zenteno_state_{l}.png")
            plt.savefig(out_png, dpi=150); plt.close()
            print(f"[PLOT] Saved {out_png}")

        # Kinetics plots at FE-end: mu_fe, Kd_fe, mu_net_fe; betaG_fe/betaF_fe; rG/rF
        t_fe = [i * h for i in range(1, nfe + 1)]
        def _series_from_attr(attr_name):
            comp = getattr(m, attr_name, None)
            if comp is None:
                return [np.nan for _ in range(nfe)]
            vals = []
            for i in range(1, nfe + 1):
                try:
                    vals.append(float(pyo.value(comp[i])))
                except Exception:
                    vals.append(np.nan)
            return vals

        mu_series = _series_from_attr('mu_fe')
        kd_series = _series_from_attr('Kd_fe')
        mu_net_series = _series_from_attr('mu_net_fe')
        bG_series = _series_from_attr('betaG_fe')
        bF_series = _series_from_attr('betaF_fe')
        # rG/rF: use MPCC expressions if present; otherwise compute from FE-end kinetics
        rG_comp = getattr(m, 'rG', None); rF_comp = getattr(m, 'rF', None)
        if (rG_comp is not None) and (rF_comp is not None):
            rG_series = _series_from_attr('rG')
            rF_series = _series_from_attr('rF')
        else:
            mu_fe_comp = getattr(m, 'mu_fe', None)
            betaG_comp = getattr(m, 'betaG_fe', None)
            betaF_comp = getattr(m, 'betaF_fe', None)
            theta_comp = getattr(m, 'theta', None)
            mrate_comp = getattr(m, 'mrate', None)
            phiG_fe = getattr(m, 'phiG_fe', None)
            phiF_fe = getattr(m, 'phiF_fe', None)
            def _safe_expr(i, sugar):
                try:
                    if sugar == 'G':
                        return float(pyo.value(mu_fe_comp[i] / pyo.exp(theta_comp['Yxg']) + betaG_comp[i] / pyo.exp(theta_comp['Yeg']) + mrate_comp * phiG_fe[i]))
                    else:
                        return float(pyo.value(mu_fe_comp[i] / pyo.exp(theta_comp['Yxf']) + betaF_comp[i] / pyo.exp(theta_comp['Yef']) + mrate_comp * phiF_fe[i]))
                except Exception:
                    return np.nan
            rG_series = [_safe_expr(i, 'G') for i in range(1, nfe + 1)]
            rF_series = [_safe_expr(i, 'F') for i in range(1, nfe + 1)]

        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        # Panel 1: growth and death
        axes[0].plot(t_fe, mu_series, '-o', ms=3, lw=1.4, label='mu_fe', color='tab:blue')
        axes[0].plot(t_fe, kd_series, '-o', ms=3, lw=1.2, label='Kd_fe', color='tab:red')
        axes[0].plot(t_fe, mu_net_series, '-o', ms=3, lw=1.2, label='mu_fe - Kd_fe', color='tab:green')
        axes[0].set_ylabel('Rates [1/h]'); axes[0].legend(); axes[0].grid(alpha=0.2)
        # Panel 2: ethanol production kinetics
        axes[1].plot(t_fe, bG_series, '-o', ms=3, lw=1.4, label='betaG_fe', color='tab:purple')
        axes[1].plot(t_fe, bF_series, '-o', ms=3, lw=1.4, label='betaF_fe', color='tab:orange')
        axes[1].set_ylabel('beta [1/h]'); axes[1].legend(); axes[1].grid(alpha=0.2)
        # Panel 3: substrate uptake demands used in ties
        axes[2].plot(t_fe, rG_series, '-o', ms=3, lw=1.4, label='rG (glu)', color='tab:brown')
        axes[2].plot(t_fe, rF_series, '-o', ms=3, lw=1.4, label='rF (fru)', color='tab:cyan')
        axes[2].set_ylabel('r [1/h]'); axes[2].set_xlabel('Time'); axes[2].legend(); axes[2].grid(alpha=0.2)
        fig.suptitle('Kinetics @ FE-end')
        fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.96))
        kin_png = os.path.join(RESULTS_DIR, 'zenteno_kinetics.png')
        fig.savefig(kin_png, dpi=150)
        plt.close(fig)
        print(f"[PLOT] Saved {kin_png}")

        # CSV export for kinetics
        try:
            df_kin = pd.DataFrame({
                'time': t_fe,
                'mu_fe': mu_series,
                'Kd_fe': kd_series,
                'mu_net_fe': mu_net_series,
                'betaG_fe': bG_series,
                'betaF_fe': bF_series,
                'rG': rG_series,
                'rF': rF_series,
            })
            kin_csv = os.path.join(RESULTS_DIR, 'zenteno_kinetics.csv')
            df_kin.to_csv(kin_csv, index=False)
            print(f"[CSV] Saved {kin_csv}")
        except Exception as e:
            print(f"[WARN] Failed to write zenteno_kinetics.csv: {e}")

        # Extra: en modo simulate, generar un gráfico resumen con 5 subplots (valores a fin de FE)
        if args.simulate:
            t_fe = [i * h for i in range(1, nfe + 1)]
            fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
            c_comp = getattr(m, 'c', None)
            for l in range(1, 6):
                if c_comp is None:
                    y_fe = [np.nan for _ in range(nfe)]
                else:
                    y_fe = []
                    for i in range(1, nfe + 1):
                        try:
                            tmp_val = pyo.value(c_comp[l, i, ncp])
                            y_fe.append(float(tmp_val) if tmp_val is not None else np.nan)
                        except Exception:
                            y_fe.append(np.nan)
                ax = axes[l - 1]
                ax.plot(t_fe, y_fe, '-o', ms=3, lw=1.4, color='tab:blue')
                ax.set_ylabel(state_names.get(l, str(l)))
                ax.grid(alpha=0.2)
            axes[-1].set_xlabel('Time')
            fig.suptitle('Simulación (modo simulate) — Valores al final de cada FE')
            fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.97))
            overview_png = os.path.join(RESULTS_DIR, 'zenteno_simulate_overview.png')
            fig.savefig(overview_png, dpi=150)
            print(f"[PLOT] Saved {overview_png}")
            if args.show:
                plt.show()
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")

    if not args.enable_mpcc:
        print("\n[INFO] MPCC coupling is disabled. Run with --enable_mpcc to activate equality ties and dFBA constraints.")


if __name__ == "__main__":
    main()
