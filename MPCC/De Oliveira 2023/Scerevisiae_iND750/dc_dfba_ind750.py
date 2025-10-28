"""
DC dFBA (KKT pFBA) for iND750 in Pyomo + IPOPT
Replicates the Julia JuMP setup in /main.jl and /pFBA_KKT_flux.jl.

Files expected in the same folder:
- S.csv   (stoichiometric matrix, shape (nm, nv) = (1061, 1266))
- lb.csv  (lower bounds, length nv)
- ub.csv  (upper bounds, length nv)

Outputs:
- tsn_py.csv : time nodes (collocation) [h]
- xk_py.csv  : states [X(g), G(mmol), E(mmol), V(L)] at those nodes
"""

from pathlib import Path
import numpy as np
from pyomo.environ import (ConcreteModel, Var, Set, NonNegativeReals, Reals,
                           Constraint, Objective, minimize, SolverFactory)

HERE = Path(__file__).parent

# ----------------------------
# Data loading
# ----------------------------
S   = np.loadtxt(HERE / "S.csv" , delimiter=",")
vlb = np.loadtxt(HERE / "lb.csv", delimiter=",")
vub = np.loadtxt(HERE / "ub.csv", delimiter=",")
nm, nv = S.shape
assert vlb.shape == (nv,) and vub.shape == (nv,), "lb/ub must be length nv"

# ----------------------------
# Indices (1-based in Julia -> 0-based here)
# ----------------------------
ETH = 420-1
OBJ = 1266-1
GLU = 428-1
O2  = 458-1

# ----------------------------
# Collocation & mesh (Radau)
# ----------------------------
COLMAT = np.array([
    [ 0.19681547722366, -0.06553542585020, 0.02377097434822],
    [ 0.39442431473909,  0.29207341166523,-0.04154875212600],
    [ 0.37640306270047,  0.51248582618842, 0.11111111111111]
])
RADAU = np.array([0.15505, 0.64495, 1.0])

# Mesh per Julia main.jl (cámbialo a 10 para Fig. 8 del paper)
NFE = 4              # number of finite elements
NCP = 3              # Radau points per element
TH  = 3.0            # time horizon [h]

# ----------------------------
# Parameters from main.jl
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

# Controls piecewise constantes (uk[0,:] = DO, uk[1,:] = F) copiados de main.jl
uk_DO = np.full(NFE, Osat/2.0)   # DO
uk_F  = np.zeros(NFE)            # F
uk_DO[2: ] = 0.0                 # FE 3..4 -> 0
uk_F [2: ] = 0.01                # FE 3..4 -> 0.01 L/h

# ----------------------------
# Initial conditions
# ----------------------------
nc = 4  # X,G,E,V
x0 = 0.2             # g
g0 = 2.0/conv_glu    # mmol
e0 = 0.0/conv_eth    # mmol
v0 = 1.0             # L
c0 = np.array([x0, g0, e0, v0], dtype=float)
cs = np.ones(nc)   # scaling states
vs = np.ones(nv)   # scaling fluxes

# pFBA penalty weight (small L2)
w    = 1e-6  # stronger L2 regularization on v for numerical stability
phi1 = 1.0  # FO_L weight
phi2 = 1.0  # FO_upt weight
phi3 = 1.0  # FO_U weight

# ----------------------------
# Kinetics helpers
# ----------------------------
def vg_from_states(G_over_V, E_over_V):
    num = G_over_V
    den = Kg + G_over_V + (G_over_V**2)/Kig
    return vg_max * (num/den) * (1.0/(1.0 + E_over_V/Kie))

def vo_from_DO(DO):
    return vo_max * (DO/(Ko+DO))

# ----------------------------
# Build Pyomo model
# ----------------------------
m = ConcreteModel()

m.NFE = Set(initialize=range(NFE))
m.NCP = Set(initialize=range(NCP))
m.NM  = Set(initialize=range(S.shape[0]))
m.NV  = Set(initialize=range(S.shape[1]))
m.NC  = Set(initialize=range(nc))

# Element sizes
m.hv = Var(m.NFE, domain=NonNegativeReals, initialize=TH/NFE)

# States & derivatives
m.c    = Var(m.NC, m.NFE, m.NCP, domain=Reals, initialize=0.0)
m.cdot = Var(m.NC, m.NFE, m.NCP, domain=Reals, initialize=0.0)

# Fluxes (piecewise-constant in each FE)
m.v = Var(m.NV, m.NFE, domain=Reals, initialize=0.0)

# KKT vars
m.lmbda    = Var(m.NM, m.NFE, domain=Reals, initialize=0.0)
m.alpha_U  = Var(m.NV, m.NFE, domain=Reals, initialize=0.0)
m.alpha_L  = Var(m.NV, m.NFE, domain=Reals, initialize=0.0)
m.alpha_upt= Var(m.NFE,       domain=Reals, initialize=0.0)

m.FO_U   = Var(m.NV, m.NFE, domain=Reals, initialize=0.0)
m.FO_L   = Var(m.NV, m.NFE, domain=Reals, initialize=0.0)
m.FO_upt = Var(m.NFE,       domain=Reals, initialize=0.0)

# ---- Inicialización robusta en TODOS los nodos (evitar V=0, etc.) ----
for i in range(NFE):
    for j in range(NCP):
        m.c[0,i,j].value = x0/cs[0]  # X
        m.c[1,i,j].value = g0/cs[1]  # G
        m.c[2,i,j].value = e0/cs[2]  # E
        m.c[3,i,j].value = v0/cs[3]  # V

# ODEs
def _m1(m,i,j):  # dX/dt = v[obj] * X
    return m.cdot[0,i,j] == m.v[OBJ,i] * m.c[0,i,j]
m.ode_X = Constraint(m.NFE, m.NCP, rule=_m1)

def _m2(m,i,j):  # dG/dt = F*Gf + v_glu * X
    return m.cdot[1,i,j] == (uk_F[i]*Gf + m.v[GLU,i] * m.c[0,i,j])
m.ode_G = Constraint(m.NFE, m.NCP, rule=_m2)

def _m3(m,i,j):  # dE/dt = v_eth * X
    return m.cdot[2,i,j] == (m.v[ETH,i] * m.c[0,i,j])
m.ode_E = Constraint(m.NFE, m.NCP, rule=_m3)

def _m4(m,i,j):  # dV/dt = F
    return m.cdot[3,i,j] == uk_F[i]
m.ode_V = Constraint(m.NFE, m.NCP, rule=_m4)

# Collocation
def _coll(m,l,i,j):
    if i == 0:
        return m.c[l,i,j] == c0[l]/cs[l] + m.hv[i]*sum(COLMAT[j,k]*m.cdot[l,i,k] for k in m.NCP)
    else:
        return m.c[l,i,j] == m.c[l,i-1,NCP-1] + m.hv[i]*sum(COLMAT[j,k]*m.cdot[l,i,k] for k in m.NCP)
m.coll = Constraint(m.NC, m.NFE, m.NCP, rule=_coll)

# Horizon
m.hsum = Constraint(expr=sum(m.hv[i] for i in m.NFE) == TH)

# FBA steady-state
def _Sc(m,mc,i):
    return sum(S[mc,k]*m.v[k,i] for k in m.NV) == 0.0
m.Sv0 = Constraint(m.NM, m.NFE, rule=_Sc)

# v bounds
m.vUB = Constraint(m.NV, m.NFE, rule=lambda m,k,i:  m.v[k,i] - vub[k] <= 0.0)
m.vLB = Constraint(m.NV, m.NFE, rule=lambda m,k,i: -m.v[k,i] + vlb[k] <= 0.0)

# Non-negativity of states
m.cLB = Constraint(m.NC, m.NFE, m.NCP, rule=lambda m,l,i,j: -m.c[l,i,j] <= 0)

# Uptake bounds vs kinetics at last node in the element
EPS = 1e-8  # protege divisiones
def _G_over_V(m,i): return m.c[1,i,NCP-1] / (m.c[3,i,NCP-1] + EPS)
def _E_over_V(m,i): return m.c[2,i,NCP-1] / (m.c[3,i,NCP-1] + EPS)

def _glu_upt(m,i):
    vg = vg_from_states(_G_over_V(m,i), _E_over_V(m,i))
    return -m.v[GLU,i] - vg <= 0.0
m.glc_upt = Constraint(m.NFE, rule=_glu_upt)

def _o2_upt(m,i):
    vo = vo_from_DO(uk_DO[i])
    return -m.v[O2,i] - vo <= 0.0
m.o2_upt = Constraint(m.NFE, rule=_o2_upt)

# KKT stationarity
d  = np.zeros(nv); d[OBJ] = -1.0  # maximize mu -> minimize -mu
up = np.zeros(nv); up[GLU] = 1.0

def _stationarity(m,k,i):
    return ( d[k] + 1e-20*m.v[k,i]
             + m.alpha_L[k,i] + m.alpha_U[k,i]
             + up[k]*m.alpha_upt[i]
             + sum(S[r,k]*m.lmbda[r,i] for r in m.NM) ) == 0.0
m.KKT = Constraint(m.NV, m.NFE, rule=_stationarity)

# Signs
m.alphaL_sign = Constraint(m.NV, m.NFE, rule=lambda m,k,i: m.alpha_L[k,i] <= 0.0)
m.alphaU_sign = Constraint(m.NV, m.NFE, rule=lambda m,k,i: m.alpha_U[k,i] >= 0.0)
m.alphaupt_sign = Constraint(m.NFE,      rule=lambda m,i:   m.alpha_upt[i] <= 0.0)

# Complementarity penalties
m.FO_L_def   = Constraint(m.NV, m.NFE, rule=lambda m,k,i: m.FO_L[k,i]   == ( m.v[k,i] - vlb[k] ) * m.alpha_L[k,i])
m.FO_U_def   = Constraint(m.NV, m.NFE, rule=lambda m,k,i: m.FO_U[k,i]   == ( m.v[k,i] - vub[k] ) * m.alpha_U[k,i])
m.FO_upt_def = Constraint(m.NFE,       rule=lambda m,i:     m.FO_upt[i] == (-m.v[GLU,i] - vg_from_states(_G_over_V(m,i), _E_over_V(m,i))) * m.alpha_upt[i])

m.OBJ = Objective(expr=sum(
    sum( phi1*(m.FO_L[k,i]**2) + phi3*(m.FO_U[k,i]**2) for k in m.NV ) + phi2*(m.FO_upt[i]**2)
    for i in m.NFE
), sense=minimize)
# Optional: add mild L2 on multipliers to help conditioning
# from pyomo.environ import summation
# rho = 1e-8
# m.OBJ_extra = Objective(expr=rho*(
#     sum(m.lmbda[r,i]**2 for r in m.NM for i in m.NFE) +
#     sum(m.alpha_L[k,i]**2 for k in m.NV for i in m.NFE) +
#     sum(m.alpha_U[k,i]**2 for k in m.NV for i in m.NFE) +
#     sum(m.alpha_upt[i]**2 for i in m.NFE)
# ), sense=minimize)

def solve_and_export(log=True):
    solver = SolverFactory("ipopt")
    # Límite de iteraciones solicitado
    solver.options.update(dict(max_iter=100, tol=1e-6, acceptable_tol=1e-3, acceptable_iter=5))
    res = solver.solve(m, tee=log)
    # Build collocation time series
    ts = np.zeros(NFE+1)
    for i in range(1, NFE+1):
        ts[i] = ts[i-1] + float(m.hv[i-1].value)

    tsn = [0.0]
    X = []
    for i in range(NFE):
        for j in range(NCP):
            t = ts[i] + RADAU[j]*float(m.hv[i].value)
            tsn.append(t)
            X.append([ float(m.c[0,i,j].value),
                       float(m.c[1,i,j].value),
                       float(m.c[2,i,j].value),
                       float(m.c[3,i,j].value) ])
    tsn = np.array(tsn)
    X = np.array([[x0, g0, e0, v0]] + X)  # prepend ICs

    np.savetxt(HERE/"tsn_py.csv", tsn, delimiter=",")
    np.savetxt(HERE/"xk_py.csv",  X,   delimiter=",")
    return res

if __name__ == "__main__":
    solve_and_export(log=True)