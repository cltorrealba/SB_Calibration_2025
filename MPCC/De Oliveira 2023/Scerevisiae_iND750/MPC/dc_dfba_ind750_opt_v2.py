"""
Version 2: DC dFBA dynamic optimization exporting global flux vector v per FE.

Adds:
- Export of v matrix per finite element: v_py.csv of shape (NV, NFE).
- Objective: maximize ethanol at final time minus KKT penalties (same signs as JuMP).
- Decision variables: uk[0,:]=DO (mol/L), uk[1,:]=F (L/h).

Required files in the same folder:
- S.csv, lb.csv, ub.csv

Outputs:
- tsn_py.csv, xk_py.csv, uk_py.csv, v_py.csv
"""

from pathlib import Path
import numpy as np
from pyomo.environ import (ConcreteModel, Var, Set, NonNegativeReals, Reals,
                           Constraint, Objective, maximize, SolverFactory)

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
# Indices (Julia 1-based -> 0-based here)
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

NFE = 10
NCP = 3
TH  = 16.0
h   = TH/NFE
hm  = np.full(NFE, h)
var_h = 0.0

# ----------------------------
# Parameters (as in v1)
# ----------------------------
conv_eth = 0.04607    # g/mmol
conv_glu = 0.180156   # g/mmol
Gf  = 50/conv_glu
Kie = 10/conv_eth
Kg  = 0.5/conv_glu
Ko  = 3.00e-6
Kig = 10/conv_glu
vg_max = 20.0
vo_max = 8.0
Osat   = 3.0e-4

# Adjust some bounds
vlb[GLU] = -vg_max
vlb[O2 ] = -vo_max

# KKT weights / penalties
w    = 1e-20
phi1 = 1e-1
phi2 = 1e-1
phi3 = 1e-1

NUPT = 2

# ----------------------------
# Initial conditions
# ----------------------------
nc = 4
x0 = 0.2
g0 = 14.64/conv_glu
e0 = 0.0/conv_eth
v0 = 0.5
c0 = np.array([x0, g0, e0, v0], dtype=float)

cs = np.ones(nc)
vs = np.ones(nv)

# ----------------------------
# Controls (normalized with scaling 'us' as in Julia)
# ----------------------------
NCV    = 2
uk_max = np.array([Osat*0.3, 0.2])  # [DO (mol/L), F (L/h)]
uk_min = np.array([0.0,      0.0])
u0     = np.array([Osat*0.3, 0.0])
DuMax  = np.array([Osat*0.3, 0.2])  # (not used here, left for parity)

us = uk_max.copy()
uk_max_n = uk_max/us
uk_min_n = uk_min/us
u0_n     = u0/us

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
m.NM  = Set(initialize=range(nm))
m.NV  = Set(initialize=range(nv))
m.NC  = Set(initialize=range(nc))
m.NUPT= Set(initialize=range(NUPT))
m.NCV = Set(initialize=range(NCV))

m.hv = Var(m.NFE, domain=NonNegativeReals, initialize=h)

m.c    = Var(m.NC, m.NFE, m.NCP, domain=Reals, initialize=0.0)
m.cdot = Var(m.NC, m.NFE, m.NCP, domain=Reals, initialize=0.0)

m.v = Var(m.NV, m.NFE, domain=Reals, initialize=0.0)

m.lmbda     = Var(m.NM, m.NFE, domain=Reals, initialize=0.0)
m.alpha_U   = Var(m.NV, m.NFE, domain=Reals, initialize=0.0)
m.alpha_L   = Var(m.NV, m.NFE, domain=Reals, initialize=0.0)
m.alpha_upt = Var(m.NFE, m.NUPT, domain=Reals, initialize=0.0)

m.FO_U   = Var(m.NV, m.NFE, domain=Reals, initialize=0.0)
m.FO_L   = Var(m.NV, m.NFE, domain=Reals, initialize=0.0)
m.FO_upt = Var(m.NFE, m.NUPT, domain=Reals, initialize=0.0)

def _uk_bounds(m, k, i):
    return (float(uk_min_n[k]), float(uk_max_n[k]))
m.uk = Var(m.NCV, m.NFE, domain=Reals, bounds=_uk_bounds, initialize=lambda m,k,i: float(u0_n[k]))

# init states
for i in range(NFE):
    for j in range(NCP):
        m.c[0,i,j].value = x0
        m.c[1,i,j].value = g0
        m.c[2,i,j].value = e0
        m.c[3,i,j].value = v0
    m.hv[i].value = hm[i]

# ODEs
def _m1(m,i,j):  # dX/dt = mu * X
    return m.cdot[0,i,j] == m.v[OBJ,i] * m.c[0,i,j]
m.ode_X = Constraint(m.NFE, m.NCP, rule=_m1)

def _m2(m,i,j):  # dG/dt = F*Gf + v_glu*X
    F = m.uk[1,i]*float(us[1])
    return m.cdot[1,i,j] == (F*Gf + m.v[GLU,i] * m.c[0,i,j])
m.ode_G = Constraint(m.NFE, m.NCP, rule=_m2)

def _m3(m,i,j):  # dE/dt = v_eth*X
    return m.cdot[2,i,j] == (m.v[ETH,i] * m.c[0,i,j])
m.ode_E = Constraint(m.NFE, m.NCP, rule=_m3)

def _m4(m,i,j):  # dV/dt = F
    F = m.uk[1,i]*float(us[1])
    return m.cdot[3,i,j] == F
m.ode_V = Constraint(m.NFE, m.NCP, rule=_m4)

def _coll(m,l,i,j):
    if i == 0:
        return m.c[l,i,j] == (c0[l]) + m.hv[i]*sum(COLMAT[j,k]*m.cdot[l,i,k] for k in m.NCP)
    else:
        return m.c[l,i,j] == m.c[l,i-1,NCP-1] + m.hv[i]*sum(COLMAT[j,k]*m.cdot[l,i,k] for k in m.NCP)
m.coll = Constraint(m.NC, m.NFE, m.NCP, rule=_coll)

m.hsum = Constraint(expr=sum(m.hv[i] for i in m.NFE) == TH)
m.hLB  = Constraint(m.NFE, rule=lambda m,i: m.hv[i] >= (1.0 - var_h)*hm[0])
m.hUB  = Constraint(m.NFE, rule=lambda m,i: m.hv[i] <= (1.0 + var_h)*hm[0])

# FBA steady-state
def _Sv0(m,mc,i):
    return sum(S[mc,k]*m.v[k,i] for k in m.NV) == 0.0
m.Sv0 = Constraint(m.NM, m.NFE, rule=_Sv0)

m.vUB = Constraint(m.NV, m.NFE, rule=lambda m,k,i:  m.v[k,i] - vub[k] <= 0.0)
m.vLB = Constraint(m.NV, m.NFE, rule=lambda m,k,i: -m.v[k,i] + vlb[k] <= 0.0)

# States >= 0 and V cap
m.cLB   = Constraint(m.NC, m.NFE, m.NCP, rule=lambda m,l,i,j: -m.c[l,i,j] <= 0.0)
m.cUB_V = Constraint(m.NFE, m.NCP, rule=lambda m,i,j: m.c[3,i,j] <= 1.2)

# Uptake constraints (last node per FE)
EPS = 1e-8
def _G_over_V(m,i): return m.c[1,i,NCP-1]/(m.c[3,i,NCP-1] + EPS)
def _E_over_V(m,i): return m.c[2,i,NCP-1]/(m.c[3,i,NCP-1] + EPS)

def _glu_upt(m,i):
    vg = vg_from_states(_G_over_V(m,i), _E_over_V(m,i))
    return -m.v[GLU,i] - vg <= 0.0
m.glc_upt = Constraint(m.NFE, rule=_glu_upt)

def _o2_upt(m,i):
    DO = m.uk[0,i]*float(us[0])
    vo = vo_from_DO(DO)
    return -m.v[O2,i] - vo <= 0.0
m.o2_upt = Constraint(m.NFE, rule=_o2_upt)

# KKT stationarity
d  = np.zeros(nv); d[OBJ] = -1.0
up = np.zeros(nv); up[GLU] = 1.0
up2= np.zeros(nv); up2[O2 ] = 1.0

def _stationarity(m,k,i):
    return ( d[k] + w*m.v[k,i]
             + m.alpha_L[k,i] + m.alpha_U[k,i]
             + up[k]*m.alpha_upt[i,0] + up2[k]*m.alpha_upt[i,1]
             + sum(S[r,k]*m.lmbda[r,i] for r in m.NM) ) == 0.0
m.KKT = Constraint(m.NV, m.NFE, rule=_stationarity)

m.alphaL_sign   = Constraint(m.NV, m.NFE, rule=lambda m,k,i: m.alpha_L[k,i] <= 0.0)
m.alphaU_sign   = Constraint(m.NV, m.NFE, rule=lambda m,k,i: m.alpha_U[k,i] >= 0.0)
m.alphaupt_sign = Constraint(m.NFE, m.NUPT, rule=lambda m,i,u: m.alpha_upt[i,u] <= 0.0)

m.FO_L_def    = Constraint(m.NV, m.NFE, rule=lambda m,k,i: m.FO_L[k,i]   == ( m.v[k,i] - vlb[k] ) * m.alpha_L[k,i])
m.FO_U_def    = Constraint(m.NV, m.NFE, rule=lambda m,k,i: m.FO_U[k,i]   == ( m.v[k,i] - vub[k] ) * m.alpha_U[k,i])
m.FO_upt1_def = Constraint(m.NFE,       rule=lambda m,i:     m.FO_upt[i,0] == (-m.v[GLU,i] - vg_from_states(_G_over_V(m,i), _E_over_V(m,i))) * m.alpha_upt[i,0])
m.FO_upt2_def = Constraint(m.NFE,       rule=lambda m,i:     m.FO_upt[i,1] == (-m.v[O2 ,i] - vo_from_DO(m.uk[0,i]*float(us[0]))) * m.alpha_upt[i,1])

def ethanol_final(m):
    return m.c[2, NFE-1, NCP-1]

penalty = sum( sum( -phi1*m.FO_L[k,i] + -phi3*m.FO_U[k,i] for k in m.NV ) +
               sum(  phi2*m.FO_upt[i,u]                  for u in m.NUPT )
               for i in m.NFE )

m.OBJ = Objective(expr= ethanol_final(m) - penalty, sense=maximize)

def solve_and_export(log=True):
    solver = SolverFactory("ipopt")
    solver.options.update(dict(max_iter=300, tol=1e-4, acceptable_tol=1e-2, acceptable_iter=5))
    res = solver.solve(m, tee=log)

    # collocation time series
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
    X = np.array([[x0, g0, e0, v0]] + X)

    uk_opt = np.zeros((NCV, NFE))
    for i in range(NFE):
        for k in range(NCV):
            uk_opt[k,i] = float(m.uk[k,i].value) * float(us[k])

    # Export v as (NV, NFE)
    Vmat = np.zeros((nv, NFE))
    for i in range(NFE):
        for k in range(nv):
            Vmat[k,i] = float(m.v[k,i].value)

    np.savetxt(HERE/"tsn_py.csv", tsn, delimiter=",")
    np.savetxt(HERE/"xk_py.csv",  X,   delimiter=",")
    np.savetxt(HERE/"uk_py.csv",  uk_opt, delimiter=",")
    np.savetxt(HERE/"v_py.csv",   Vmat,   delimiter=",")
    return res

if __name__ == "__main__":
    solve_and_export(log=True)
