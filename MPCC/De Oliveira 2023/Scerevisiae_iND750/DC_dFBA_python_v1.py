# dc_dfba_ind750.py
# Pyomo + IPOPT implementation of DC dFBA (iND750, Fig. 8 setup)
# Requires: S.csv, lb.csv, ub.csv (column-aligned). Python 3.10+, pyomo, ipopt in PATH.

from pathlib import Path
import numpy as np
from pyomo.environ import (ConcreteModel, Var, Set, NonNegativeReals, Reals,
                           Constraint, Objective, minimize, SolverFactory, value)
# from pyomo.environ import inequality
# from pyomo.core.base.initializer import Initializer

# ----------------------------
# Data loading
# ----------------------------
HERE = Path(__file__).parent
S = np.loadtxt(HERE / "S.csv", delimiter=",")  # nm x nv
nm, nv = S.shape

# Provide lb/ub or dummy if missing
lb_path = HERE / "lb.csv"
ub_path = HERE / "ub.csv"
if lb_path.exists() and ub_path.exists():
    vlb = np.loadtxt(lb_path, delimiter=",").reshape(-1)
    vub = np.loadtxt(ub_path, delimiter=",").reshape(-1)
else:
    # Fallback: zero bounds (will NOT solve right) -> replace with real files
    vlb = np.zeros(nv)
    vub = np.zeros(nv)

# ----------------------------
# Indices (as in your Julia main.jl)
# ----------------------------
eth = 420-1   # 0-based
obj = 1266-1
glu = 428-1
o2  = 458-1

# ----------------------------
# Collocation & mesh
# ----------------------------
# Radau roots and collocation matrix (author's values)
radau = np.array([0.15505, 0.64495, 1.0])
colmat = np.array([
    [ 0.19681547722366, -0.06553542585020, 0.02377097434822],
    [ 0.39442431473909,  0.29207341166523,-0.04154875212600],
    [ 0.37640306270047,  0.51248582618842, 0.11111111111111]
])

nfe = 10   # Fig. 8 uses 10 elements
ncp = 3
th  = 3.0  # time horizon (h) — mismo que tu main.jl por ahora

# ----------------------------
# Parameters (from main.jl)
# ----------------------------
conv_eth = 0.04607    # mmol/g
conv_glu = 0.180156   # mmol/g

Gf  = 50/conv_glu     # mmol/L feed glucosa
Kie = 10/conv_eth     # mmol/L inhibición por etanol
Kg  = 0.5/conv_glu    # mmol/L saturación glucosa
Ko  = 3.00e-6         # mol/L (DO half-sat)
Kig = 10/conv_glu     # mmol/L término inhibición (glucosa^2)
vg_max = 20.0         # mmol/gDW h (max uptake glucosa)
vo_max = 8.0          # mmol/gDW h (max uptake O2)
Osat   = 3.0e-4       # mol/L (O2 sat)

# Controls piecewise constants (como tu main.jl)
# uk[0,:] = DO (mol/L), uk[1,:] = F (L/h)
uk1 = np.full(nfe, Osat/2.0)   # DO
uk2 = np.zeros(nfe)            # F
uk1[2: ] = 0.0
uk2[2: ] = 0.01

# ----------------------------
# Initial conditions & scaling
# ----------------------------
nc = 4  # [X, G, E, V]
x0 = 0.2             # g
g0 = 2.0/conv_glu    # mmol
e0 = 0.0/conv_eth    # mmol
v0 = 1.0             # L

c0 = np.array([x0, g0, e0, v0], dtype=float)
cs = np.ones(nc)     # escalado estado
vs = np.ones(nv)     # escalado flujos

# pFBA weights / penalties
w    = 1e-20
phi1 = 1.0
phi2 = 1.0
phi3 = 1.0

# ----------------------------
# Model
# ----------------------------
m = ConcreteModel()

m.NFE = Set(initialize=range(nfe))
m.NCP = Set(initialize=range(ncp))
m.NM  = Set(initialize=range(nm))
m.NV  = Set(initialize=range(nv))
m.NC  = Set(initialize=range(nc))

# Time step (variable mesh like in JuMP)
m.hv = Var(m.NFE, domain=NonNegativeReals, initialize=th/nfe)

# States & derivatives at collocation points
m.c    = Var(m.NC, m.NFE, m.NCP, domain=Reals, initialize=lambda m,i,k,j: float(c0[i]))
m.cdot = Var(m.NC, m.NFE, m.NCP, domain=Reals, initialize=0.0)

# Fluxes per finite element (piecewise const in each FE)
m.v = Var(m.NV, m.NFE, domain=Reals, initialize=0.0)

# KKT multipliers & aux
m.lmbda    = Var(m.NM, m.NFE, domain=Reals, initialize=0.0)
m.alpha_U  = Var(m.NV, m.NFE, domain=Reals, initialize=0.0)
m.alpha_L  = Var(m.NV, m.NFE, domain=Reals, initialize=0.0)
m.alpha_upt= Var(m.NFE,       domain=Reals, initialize=0.0)

m.FO_U   = Var(m.NV, m.NFE, domain=Reals, initialize=0.0)
m.FO_L   = Var(m.NV, m.NFE, domain=Reals, initialize=0.0)
m.FO_upt = Var(m.NFE,       domain=Reals, initialize=0.0)

# Uptake kinetics as expressions per FE (use last collocation node, like JuMP c[*,i,2])
def _G_over_V(m,i):  # at node j = last (2 for 0-based)
    return m.c[1,i,2]*cs[1] / (m.c[3,i,2]*cs[3])
def _E_over_V(m,i):
    return m.c[2,i,2]*cs[2] / (m.c[3,i,2]*cs[3])

def vg_i(m,i):
    G_V = _G_over_V(m,i)
    num = G_V
    den = Kg + G_V + (G_V**2)/Kig
    inhE = 1.0 + _E_over_V(m,i)/Kie
    return vg_max * (num/den) * (1.0/inhE)

def vo_i(m,i):
    DO = uk1[i]
    return vo_max * (DO/(Ko+DO))

# Differential equations at each collocation node
def _m1(m,i,j): # dX/dt = mu * X ; mu = v[obj]
    return m.cdot[0,i,j] == vs[obj]*m.v[obj,i] * m.c[0,i,j]
m.m1 = Constraint(m.NFE, m.NCP, rule=_m1)

def _m2(m,i,j): # dG/dt = F*Gf + v_glu * X
    return m.cdot[1,i,j] == (uk2[i]*Gf + vs[glu]*m.v[glu,i]*m.c[0,i,j]*cs[0]) / cs[1]
m.m2 = Constraint(m.NFE, m.NCP, rule=_m2)

def _m3(m,i,j): # dE/dt = v_eth * X
    return m.cdot[2,i,j] == (vs[eth]*m.v[eth,i]*m.c[0,i,j]*cs[0]) / cs[2]
m.m3 = Constraint(m.NFE, m.NCP, rule=_m3)

def _m4(m,i,j): # dV/dt = F
    return m.cdot[3,i,j] == uk2[i] / cs[3]
m.m4 = Constraint(m.NFE, m.NCP, rule=_m4)

# Collocation equations
def _coll_n(m,l,i,j):
    if i == 0:
        # first element: c = c0 + h * sum(N * cdot)
        return m.c[l,i,j] == c0[l]/cs[l] + m.hv[i]*sum(colmat[j,k]*m.cdot[l,i,k] for k in m.NCP)
    else:
        return m.c[l,i,j] == m.c[l,i-1,ncp-1] + m.hv[i]*sum(colmat[j,k]*m.cdot[l,i,k] for k in m.NCP)
m.coll = Constraint(m.NC, m.NFE, m.NCP, rule=_coll_n)

# Sum of FE sizes equals horizon; element size bounds (like var_h=1.0 in JuMP)
m.sum_h = Constraint(expr=sum(m.hv[i] for i in m.NFE) == th)

# Steady-state constraints of FBA (S v = 0) and simple bounds v in [vlb, vub]
def _Sc(m,mc,i):
    return sum(S[mc,k]*m.v[k,i]*vs[k] for k in m.NV) == 0.0
m.Sc = Constraint(m.NM, m.NFE, rule=_Sc)

m.vUB = Constraint(m.NV, m.NFE, rule=lambda m,k,i:  m.v[k,i]*vs[k] - vub[k] <= 0.0)
m.vLB = Constraint(m.NV, m.NFE, rule=lambda m,k,i: -m.v[k,i]*vs[k] + vlb[k] <= 0.0)

# Non-negativity states
m.cLB = Constraint(m.NC, m.NFE, m.NCP, rule=lambda m,l,i,j: -m.c[l,i,j] <= 0)

# Uptake bounding vs kinetics (glucose and oxygen)
m.glc_upt = Constraint(m.NFE, rule=lambda m,i: -m.v[glu,i]*vs[glu] - vg_i(m,i) <= 0.0)
m.o2_upt  = Constraint(m.NFE, rule=lambda m,i: -m.v[o2 ,i]*vs[o2 ] - vo_i(m,i) <= 0.0)

# KKT stationarity for pFBA: d + w v + alpha_L + alpha_U + S^T lambda + up*alpha_upt = 0
# d has -1 in 'obj' (maximize mu -> minimize -mu) and 0 otherwise
d = np.zeros(nv); d[obj] = -1.0
up = np.zeros(nv); up[glu] = 1.0

def _stationarity(m,k,i):
    return ( d[k] + w*m.v[k,i]*vs[k]
             + m.alpha_L[k,i] + m.alpha_U[k,i]
             + up[k]*m.alpha_upt[i]
             + sum(S[r,k]*m.lmbda[r,i] for r in m.NM) ) == 0.0
m.Lagr = Constraint(m.NV, m.NFE, rule=_stationarity)

# Sign constraints for multipliers (match your JuMP signs)
m.alphaL_sign = Constraint(m.NV, m.NFE, rule=lambda m,k,i: m.alpha_L[k,i] <= 0.0)
m.alphaU_sign = Constraint(m.NV, m.NFE, rule=lambda m,k,i: m.alpha_U[k,i] >= 0.0)
m.alphaupt_sign = Constraint(m.NFE,      rule=lambda m,i:   m.alpha_upt[i] <= 0.0)

# Complementarity products with auxiliary FO_* and penalization
m.FO_L_def   = Constraint(m.NV, m.NFE, rule=lambda m,k,i: m.FO_L[k,i]   == ( m.v[k,i]*vs[k] - vlb[k] ) * m.alpha_L[k,i])
m.FO_U_def   = Constraint(m.NV, m.NFE, rule=lambda m,k,i: m.FO_U[k,i]   == ( m.v[k,i]*vs[k] - vub[k] ) * m.alpha_U[k,i])
m.FO_upt_def = Constraint(m.NFE,       rule=lambda m,i:     m.FO_upt[i] == (-m.v[glu,i]*vs[glu] - vg_i(m,i)) * m.alpha_upt[i])

m.OBJ = Objective(expr=sum(
    sum( -phi1*m.FO_L[k,i] - phi3*m.FO_U[k,i] for k in m.NV ) + phi2*m.FO_upt[i]
    for i in m.NFE
), sense=minimize)

# ----------------------------
# Solve
# ----------------------------
def solve_model(verbose: bool=True):
    solver = SolverFactory("ipopt")
    # Puedes ajustar tolerancias como en JuMP si lo necesitas:
    # solver.options.update(dict(tol=1e-4, acceptable_tol=1e-2, acceptable_iter=5))
    res = solver.solve(m, tee=verbose)
    return res

# ----------------------------
# Post-process (Radau time series)
# ----------------------------
def extract_timeseries():
    # nodos "globales": FE starts + last collocation
    ts = np.zeros(nfe+1)
    for i in range(1, nfe+1):
        ts[i] = ts[i-1] + value(m.hv[i-1])

    # times at collocation nodes
    tsn = [0.0]
    X = []
    for i in range(nfe):
        for j in range(ncp):
            t = ts[i] + radau[j]*value(m.hv[i])
            tsn.append(t)
            X.append([ value(m.c[0,i,j])*cs[0],
                       value(m.c[1,i,j])*cs[1],
                       value(m.c[2,i,j])*cs[2],
                       value(m.c[3,i,j])*cs[3] ])
    tsn = np.array(tsn)
    X = np.array([[c0[0], c0[1], c0[2], c0[3]]] + X)  # prepend ICs
    V = X[:,3]

    return tsn, X, V

if __name__ == "__main__":
    solve_model(verbose=True)
    tsn, X, V = extract_timeseries()
    # Guardar para graficar como en tus scripts
    np.savetxt(HERE/"tsn_py.csv", tsn, delimiter=",")
    np.savetxt(HERE/"xk_py.csv", X, delimiter=",")
