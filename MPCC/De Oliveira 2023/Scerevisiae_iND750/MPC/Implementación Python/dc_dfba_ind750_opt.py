
"""
DC dFBA (iND750) - Dynamic optimization in Pyomo + IPOPT
Replicates the latest Julia setup:
- Objective: maximize ethanol at final time (c_E at last FE/last node)
- Decision variables: uk[DO, Feed] piecewise-constant over NFE = 10 elements
- Process constraints: V <= 1.2 L, u_min <= u <= u_max
- Embedded pFBA via KKT with complementarity penalties FO_* (same signs as Julia)
- Orthogonal collocation (Radau, 3 points) on variable FE sizes (hv), but var_h = 0 -> hv fixed to hm

Files required in the same folder:
- S.csv, lb.csv, ub.csv

Outputs:
- tsn_py.csv : collocation times
- xk_py.csv  : states [X(g), G(mmol), E(mmol), V(L)] at those nodes
- uk_py.csv  : optimal controls [DO (mol/L), F (L/h)] per FE (in physical units)
"""

from pathlib import Path
import numpy as np
from pyomo.environ import (ConcreteModel, Var, Set, NonNegativeReals, Reals,
                           Constraint, Objective, maximize, SolverFactory, value)

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
ETH = 420-1   # ethanol reaction index (for flux)
OBJ = 1266-1  # growth objective flux index (mu)
GLU = 428-1   # glucose uptake rxn index
O2  = 458-1   # oxygen uptake rxn index

# Extra indices whose bounds are overridden in Julia main.jl
zymst = 500-1
ergst = 419-1
hdcea = 439-1
ocdca = 459-1
ocdcya= 461-1
ocdcea= 460-1

# ----------------------------
# Collocation & mesh (Radau)
# ----------------------------
COLMAT = np.array([
    [ 0.19681547722366, -0.06553542585020, 0.02377097434822],
    [ 0.39442431473909,  0.29207341166523,-0.04154875212600],
    [ 0.37640306270047,  0.51248582618842, 0.11111111111111]
])
RADAU = np.array([0.15505, 0.64495, 1.0])

# Integration parameters (match Julia main.jl)
NFE = 10          # control intervals
NCP = 3           # collocation points
TH  = 16.0        # time horizon [h]
h   = TH/NFE
hm  = np.full(NFE, h) # nominal FE sizes
var_h = 0.0

# ----------------------------
# Parameters (match Julia main.jl)
# ----------------------------
conv_eth = 0.04607   # g/mmol
conv_glu = 0.180156  # g/mmol

Gf  = 50/conv_glu     # mmol/L feed glucose
Kie = 10/conv_eth     # mmol/L ethanol inhibition
Kg  = 0.5/conv_glu    # mmol/L saturation
Ko  = 3.00e-6         # mol/L DO half-sat
Kig = 10/conv_glu     # mmol/L quadratic term
vg_max = 20.0         # mmol/gDW/h
vo_max = 8.0          # mmol/gDW/h
Osat   = 3.0e-4       # mol/L

# Modify bounds like Julia
vlb[zymst] = -1000.0
vlb[ergst] = -1000.0
vlb[hdcea] = -1000.0
vlb[ocdca] = -1000.0
vlb[ocdcya] = -1000.0
vlb[ocdcea] = -1000.0
vlb[GLU] = -vg_max
vlb[O2 ] = -vo_max

# KKT weights / penalties (as in Julia)
w    = 1e-20  # small L2 on v in stationarity
phi1 = 1e-1
phi2 = 1e-1
phi3 = 1e-1

# Uptake complementarity: two constraints (GLU, O2)
NUPT = 2

# ----------------------------
# Initial conditions (match Julia main.jl)
# ----------------------------
nc = 4  # [X, G, E, V]
x0 = 0.2               # g
g0 = 14.64/conv_glu    # mmol
e0 = 0.0/conv_eth      # mmol
v0 = 0.5               # L
c0 = np.array([x0, g0, e0, v0], dtype=float)

# Scaling (kept as ones like in Julia)
cs = np.ones(nc)
vs = np.ones(nv)

# ----------------------------
# Control variables (normalized like Julia)
# ----------------------------
NCV    = 2
uk_max = np.array([Osat*0.3, 0.2])
uk_min = np.array([0.0,      0.0])
u0     = np.array([Osat*0.3, 0.0])
DuMax  = np.array([Osat*0.3, 0.2])

us = uk_max.copy()         # scaling
uk_max_n = uk_max/us       # normalized bounds
uk_min_n = uk_min/us
u0_n     = u0/us
DuMax_n  = DuMax/us

# ----------------------------
# Helper kinetics
# ----------------------------
def vg_from_states(G_over_V, E_over_V):
    # vg_max * (G/V)/(Kg + G/V + (G/V)^2/Kig) * 1/(1 + (E/V)/Kie)
    num = G_over_V
    den = Kg + G_over_V + (G_over_V**2)/Kig
    return vg_max * (num/den) * (1.0/(1.0 + E_over_V/Kie))

def vo_from_DO(DO):
    # vo_max * DO/(Ko+DO)
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

# FE sizes
m.hv = Var(m.NFE, domain=NonNegativeReals, initialize=h)

# States & derivatives
m.c    = Var(m.NC, m.NFE, m.NCP, domain=Reals, initialize=0.0)
m.cdot = Var(m.NC, m.NFE, m.NCP, domain=Reals, initialize=0.0)

# Fluxes per FE (piecewise constant)
m.v = Var(m.NV, m.NFE, domain=Reals, initialize=0.0)

# KKT multipliers & FO terms
m.lmbda     = Var(m.NM, m.NFE, domain=Reals, initialize=0.0)
m.alpha_U   = Var(m.NV, m.NFE, domain=Reals, initialize=0.0)
m.alpha_L   = Var(m.NV, m.NFE, domain=Reals, initialize=0.0)
m.alpha_upt = Var(m.NFE, m.NUPT, domain=Reals, initialize=0.0)

m.FO_U   = Var(m.NV, m.NFE, domain=Reals, initialize=0.0)
m.FO_L   = Var(m.NV, m.NFE, domain=Reals, initialize=0.0)
m.FO_upt = Var(m.NFE, m.NUPT, domain=Reals, initialize=0.0)

# Controls (normalized)
def _uk_bounds(m, k, i):
    lb = float(uk_min_n[k])
    ub = float(uk_max_n[k])
    return (lb, ub)
m.uk = Var(m.NCV, m.NFE, domain=Reals, bounds=_uk_bounds, initialize=lambda m,k,i: float(u0_n[k]))

# ---- Robust initialization on ALL nodes ----
for i in range(NFE):
    for j in range(NCP):
        m.c[0,i,j].value = x0/cs[0]  # X
        m.c[1,i,j].value = g0/cs[1]  # G
        m.c[2,i,j].value = e0/cs[2]  # E
        m.c[3,i,j].value = v0/cs[3]  # V
    m.hv[i].value = hm[i]

# ----------------------------
# ODEs
# ----------------------------
def _m1(m,i,j):  # dX/dt = mu * X ; mu = v[OBJ,i]
    return m.cdot[0,i,j] == m.v[OBJ,i] * m.c[0,i,j]
m.ode_X = Constraint(m.NFE, m.NCP, rule=_m1)

def _m2(m,i,j):  # dG/dt = (F*Gf) + v_glu * X
    F = m.uk[1,i]*float(us[1])     # physical units
    return m.cdot[1,i,j] == (F*Gf + m.v[GLU,i] * m.c[0,i,j])
m.ode_G = Constraint(m.NFE, m.NCP, rule=_m2)

def _m3(m,i,j):  # dE/dt = v_eth * X
    return m.cdot[2,i,j] == (m.v[ETH,i] * m.c[0,i,j])
m.ode_E = Constraint(m.NFE, m.NCP, rule=_m3)

def _m4(m,i,j):  # dV/dt = F
    F = m.uk[1,i]*float(us[1])     # physical units
    return m.cdot[3,i,j] == F
m.ode_V = Constraint(m.NFE, m.NCP, rule=_m4)

# ----------------------------
# Collocation
# ----------------------------
def _coll(m,l,i,j):
    if i == 0:
        return m.c[l,i,j] == (c0[l]/cs[l]) + m.hv[i]*sum(COLMAT[j,k]*m.cdot[l,i,k] for k in m.NCP)
    else:
        return m.c[l,i,j] == m.c[l,i-1,NCP-1] + m.hv[i]*sum(COLMAT[j,k]*m.cdot[l,i,k] for k in m.NCP)
m.coll = Constraint(m.NC, m.NFE, m.NCP, rule=_coll)

# Sum FE sizes and individual bounds
m.hsum = Constraint(expr=sum(m.hv[i] for i in m.NFE) == TH)
m.hLB  = Constraint(m.NFE, rule=lambda m,i: m.hv[i] >= (1.0 - var_h)*hm[0])
m.hUB  = Constraint(m.NFE, rule=lambda m,i: m.hv[i] <= (1.0 + var_h)*hm[0])

# ----------------------------
# FBA steady state & bounds
# ----------------------------
def _Sv0(m,mc,i):
    return sum(S[mc,k]*m.v[k,i] for k in m.NV) == 0.0
m.Sv0 = Constraint(m.NM, m.NFE, rule=_Sv0)

m.vUB = Constraint(m.NV, m.NFE, rule=lambda m,k,i:  m.v[k,i] - vub[k] <= 0.0)
m.vLB = Constraint(m.NV, m.NFE, rule=lambda m,k,i: -m.v[k,i] + vlb[k] <= 0.0)

# States >= 0 and volume cap
m.cLB = Constraint(m.NC, m.NFE, m.NCP, rule=lambda m,l,i,j: -m.c[l,i,j] <= 0.0)
m.cUB_V = Constraint(m.NFE, m.NCP, rule=lambda m,i,j: m.c[3,i,j] <= 1.2)

# ----------------------------
# Uptake kinetics & constraints (at last collocation node per FE)
# ----------------------------
EPS = 1e-8

def _G_over_V(m,i): return m.c[1,i,NCP-1]/(m.c[3,i,NCP-1] + EPS)
def _E_over_V(m,i): return m.c[2,i,NCP-1]/(m.c[3,i,NCP-1] + EPS)

def _glu_upt(m,i):
    vg = vg_from_states(_G_over_V(m,i), _E_over_V(m,i))
    return -m.v[GLU,i] - vg <= 0.0
m.glc_upt = Constraint(m.NFE, rule=_glu_upt)

def _o2_upt(m,i):
    DO = m.uk[0,i]*float(us[0])              # DO in physical units
    vo = vo_from_DO(DO)
    return -m.v[O2,i] - vo <= 0.0
m.o2_upt = Constraint(m.NFE, rule=_o2_upt)

# ----------------------------
# KKT stationarity & signs
# ----------------------------
d  = np.zeros(nv); d[OBJ] = -1.0  # maximize mu -> minimize -mu in inner pFBA
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

# Complementarity product variables
m.FO_L_def   = Constraint(m.NV, m.NFE, rule=lambda m,k,i: m.FO_L[k,i]   == ( m.v[k,i] - vlb[k] ) * m.alpha_L[k,i])
m.FO_U_def   = Constraint(m.NV, m.NFE, rule=lambda m,k,i: m.FO_U[k,i]   == ( m.v[k,i] - vub[k] ) * m.alpha_U[k,i])
m.FO_upt1_def= Constraint(m.NFE,       rule=lambda m,i:     m.FO_upt[i,0] == (-m.v[GLU,i] - vg_from_states(_G_over_V(m,i), _E_over_V(m,i))) * m.alpha_upt[i,0])
m.FO_upt2_def= Constraint(m.NFE,       rule=lambda m,i:     m.FO_upt[i,1] == (-m.v[O2 ,i] - vo_from_DO(m.uk[0,i]*float(us[0]))) * m.alpha_upt[i,1])

# ----------------------------
# Objective: Max ethanol_end - penalties (same algebraic signs as Julia)
# ----------------------------
def ethanol_final(m):
    return m.c[2, NFE-1, NCP-1]  # E at last FE, last collocation node

penalty = sum( sum( -phi1*m.FO_L[k,i] + -phi3*m.FO_U[k,i] for k in m.NV ) +
               sum(  phi2*m.FO_upt[i,u]                  for u in m.NUPT )
               for i in m.NFE )

m.OBJ = Objective(expr= ethanol_final(m) - penalty, sense=maximize)

# ----------------------------
# Solve & export
# ----------------------------
def solve_and_export(log=True):
    solver = SolverFactory("ipopt")
    solver.options.update(dict(
        max_iter=1000,
        tol=1e-4,
        acceptable_tol=1e-2,
        acceptable_iter=5,
        warm_start_init_point='yes'
    ))
    res = solver.solve(m, tee=log)

    # Build collocation time series from hv
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

    # Export optimal controls (physical units)
    uk_opt = np.zeros((NCV, NFE))
    for i in range(NFE):
        for k in range(NCV):
            uk_opt[k,i] = float(m.uk[k,i].value) * float(us[k])

    np.savetxt(HERE/"tsn_py.csv", tsn, delimiter=",")
    np.savetxt(HERE/"xk_py.csv",  X,   delimiter=",")
    np.savetxt(HERE/"uk_py.csv",  uk_opt, delimiter=",")
    
    
    return res

if __name__ == "__main__":
    solve_and_export(log=True)
