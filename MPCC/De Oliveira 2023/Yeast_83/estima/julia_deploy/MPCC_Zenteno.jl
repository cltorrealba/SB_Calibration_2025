#=
MPCC_Zenteno (relaxed) — Julia/JuMP port of the Pyomo script
"pyomo_deploy/MPCC_Zenteno_relax.py" using main.jl as structural reference.

Highlights
- Zenteno ODE (5 states: X, N, G, F, E) via Radau-3 collocation
- Relaxed MPCC/dFBA coupling:
  - Flux bounds via alpha_L/alpha_U with FO_L/FO_U complementarity terms
  - Sugar uptake inequalities using FE-end Zenteno rates rG_fe, rF_fe with alpha_upt and FO_upt
  - Lagrangian stationarity with small pFBA ridge w and growth-only d-vector
- Pure simulation mode (no data-fit SSE) with linear complementarity penalties (as in relaxed Pyomo)
- Ipopt configured in-file for a 5-minute feasibility-first run (LBFGS on, wall-clock 300 s)

Files expected (same folder as main.jl):
- S.csv, lb.csv, ub.csv (stoichiometry and bounds)
- data.jld2 with variable "data" shaped (nc=5, ph=nfe, ncp=3) for FO

Optional environment variables:
    (Not required for defaults; script sets recommended options internally.)

Note: This is a relaxed MPCC variant (no equality FE-end ties). For robust runs,
start with LBFGS (J_LBFGS=1), then optionally rerun with exact Hessian for a short
polish using tighter tolerances.
=#

using JuMP
using Ipopt
using LinearAlgebra
using DelimitedFiles
using FileIO, JLD2
using Dates
using Printf
# Optional ODE integration and plotting
try
    using DifferentialEquations
    using Plots
catch err
    @warn "Plot/ODE packages missing; run Pkg.add([\"DifferentialEquations\",\"Plots\"]) to enable ODE simulation and plotting" err
end

# ---------------------------------------------
# Paths and IO
# ---------------------------------------------
const BASE_DIR = @__DIR__
const ESTIMA_DIR = normpath(joinpath(BASE_DIR, ".."))
const RESULTS_DIR = joinpath(BASE_DIR, "results")
isdir(RESULTS_DIR) || mkpath(RESULTS_DIR)

S = readdlm(joinpath(ESTIMA_DIR, "S.csv"), ',')
lb_raw = readdlm(joinpath(ESTIMA_DIR, "lb.csv"), ',')
ub_raw = readdlm(joinpath(ESTIMA_DIR, "ub.csv"), ',')
# Handle possible (n,1) vs (n,) shapes
lb = lb_raw isa AbstractVector ? copy(lb_raw) : copy(lb_raw[:,1])
ub = ub_raw isa AbstractVector ? copy(ub_raw) : copy(ub_raw[:,1])

# Optional data for FO (nc x ph x ncp); expected in data.jld2 as variable "data"
# If unavailable, we'll fall back to zeros of matching shape.
function load_data_default(nc::Int, ph::Int, ncp::Int)
    path = joinpath(BASE_DIR, "data.jld2")
    if isfile(path)
        d = FileIO.load(path, "data")
        return d
    else
        return zeros(nc, ph, ncp)
    end
end

# ---------------------------------------------
# Problem sizes and key indices
# ---------------------------------------------
nm = size(S, 1)  # metabolites
nv = size(S, 2)  # reactions

# Indices (1-based) matching Pyomo and main.jl conventions
eth = 2630     # ethanol reaction
obj = 3414     # growth/objective reaction
glu = 2588     # glucose uptake
fru = 2583     # fructose uptake

# Special bounds adjustments (as in relaxed Python)
o2 = 2816
ATP = 3415
if 1 <= o2 <= nv
    lb[o2] = 0.0
    ub[o2] = 0.0
end
if 1 <= ATP <= nv
    lb[ATP] = 0.0
end

# ---------------------------------------------
# Collocation and run configuration
# ---------------------------------------------
nc = 5              # X,N,G,F,E
nfe = 12            # number of finite elements
ncp = 3             # collocation points (Radau-3)
th = 22.0           # total horizon
hm = fill(th / nfe, nfe)  # nominal element length
var_h = 1.0         # allow hv ∈ [(1-var_h)h, (1+var_h)h]

# Radau-3 collocation matrix (same as Python)
colmat = [
    0.19681547722366   -0.06553542585020   0.02377097434822;
    0.39442431473909    0.29207341166523  -0.04154875212600;
    0.37640306270047    0.51248582618842   0.11111111111111
]

# Default kinetics temperature and constants
T_const = 293.15
R = 8.314

# pFBA ridge (in stationarity) and linear penalty weights (mirror relaxed Pyomo)
# Fixed defaults (no toggles): feasibility-first configuration
w = 1e-20
phi1 = 1.0
phi2 = 1.0
phi3 = 1.0

# pFBA d-vector (encourage positive growth reaction)
d = zeros(nv); d[obj] = -1.0
vs = ones(nv)

# Uptake selector vectors for stationarity
up_glu = zeros(nv); up_glu[glu] = 1.0
up_fru = zeros(nv); up_fru[fru] = 1.0

# Initial conditions (Zenteno typical)
X0, N0, G0, F0, E0 = 0.5, 0.14, 110.0, 110.0, 0.0
c0 = [X0, N0, G0, F0, E0]

# Nominal parameters (log-parametrized with [0.5x, 2x] bounds)
const Pnames = (
    :mu0, :betaG0, :betaF0, :Kn0, :Kg0, :Kf0, :Kig0, :Kie0, :Kd0,
    :Yxn, :Yxg, :Yxf, :Yeg, :Yef
)
const Pnom = Dict(
    :mu0=>0.141665, :betaG0=>1.41182, :betaF0=>8.49482, :Kn0=>0.226882,
    :Kg0=>3.1514, :Kf0=>2.97625, :Kig0=>29.5276, :Kie0=>2.99809, :Kd0=>3.11736e-5,
    :Yxn=>9.80576, :Yxg=>0.394345, :Yxf=>0.18622, :Yeg=>0.14133, :Yef=>0.96932
)
np = length(Pnames)
LB = similar(zeros(np)); UB = similar(zeros(np)); T0 = similar(zeros(np))
for (i, k) in enumerate(Pnames)
    LB[i] = log(max(1e-12, 0.5 * Pnom[k]))
    UB[i] = log(max(2e-12, 2.0 * Pnom[k]))
    T0[i] = log(Pnom[k])
end

# Load data for FO (nc x ph x ncp)
data = load_data_default(nc, nfe, ncp)

# ---------------------------------------------
# JuMP model
# ---------------------------------------------
m = Model(Ipopt.Optimizer)

# Ipopt options
set_optimizer_attribute(m, "warm_start_init_point", "yes")
set_optimizer_attribute(m, "print_level", 5)
set_optimizer_attribute(m, "tol", 1e-4)
set_optimizer_attribute(m, "acceptable_iter", 5)
set_optimizer_attribute(m, "acceptable_tol", 1e-2)
set_optimizer_attribute(m, "linear_solver", "mumps")
set_optimizer_attribute(m, "mu_strategy", "adaptive")
set_optimizer_attribute(m, "nlp_scaling_method", "gradient-based")
# Use exact Hessian (removed LBFGS approximation) and cap wall-clock at 300 s
set_optimizer_attribute(m, "max_wall_time", 300.0)

# ---------------------------------------------
# Variables
# ---------------------------------------------
@variables(m, begin
    c[1:nc, 1:nfe, 1:ncp]           # states
    cdot[1:nc, 1:nfe, 1:ncp]        # time derivatives
    # Simulation-only: no FO (data-fit) term
    teta[1:np]                       # log-parameters
    hv[1:nfe]                        # element lengths

    v[1:nv, 1:nfe]                   # fluxes
    lambda_[1:nm, 1:nfe]             # multipliers (stoichiometry)
    alpha_U[1:nv, 1:nfe]
    alpha_L[1:nv, 1:nfe]
    alpha_upt[1:2, 1:nfe]            # [glu, fru]

    FO_U[1:nv, 1:nfe]
    FO_L[1:nv, 1:nfe]
    FO_upt[1:2, 1:nfe]
end)

# Start values
for i in 1:nfe, j in 1:ncp
    for l in 1:nc
        set_start_value(c[l, i, j], c0[l])
        set_start_value(cdot[l, i, j], 0.0)
    end
end
for i in 1:nfe
    set_start_value(hv[i], hm[i])
end

# ---------------------------------------------
# Objective: linear complementarity penalties (relaxed Pyomo style, no SSE)
# ---------------------------------------------
@NLobjective(m, Min, sum( sum( -phi1 * FO_L[k,i] - phi3 * FO_U[k,i] for k in 1:nv )
                        +  phi2 * FO_upt[1,i] + phi2 * FO_upt[2,i]
                    for i in 1:nfe ))

# ---------------------------------------------
# Start values
# ---------------------------------------------
# Unpack exponentiated parameters for readability
# Index map for Pnames
function Pidx(sym)
    for (i, s) in enumerate(Pnames)
        s == sym && return i
    end
    error("Parameter $sym not found")
end

# Parameter starts (log-space)
for p in 1:np
    set_start_value(teta[p], T0[p])
end

# Temperature scalars
@NLexpression(m, mu_T,  exp(59453.0 * (T_const - 300.0) / (300.0 * R * T_const)))
@NLexpression(m, Kg_T,  exp(46055.0 * (T_const - 293.15) / (293.15 * R * T_const)))
@NLexpression(m, b_T,   exp(11000.0 * (T_const - 296.15) / (296.15 * R * T_const)))
@NLexpression(m, mrate, 0.01 * exp(37681.0 * (T_const - 293.30) / (293.30 * R * T_const)))

# FE-end shortcuts
@NLexpression(m, Xe[i=1:nfe], c[1, i, ncp])
@NLexpression(m, Ne[i=1:nfe], c[2, i, ncp])
@NLexpression(m, Ge[i=1:nfe], c[3, i, ncp])
@NLexpression(m, Fe[i=1:nfe], c[4, i, ncp])
@NLexpression(m, Ee[i=1:nfe], c[5, i, ncp])
@NLexpression(m, phiG_fe[i=1:nfe], Ge[i] / (Ge[i] + Fe[i] + 1e-9))
@NLexpression(m, phiF_fe[i=1:nfe], Fe[i] / (Ge[i] + Fe[i] + 1e-9))

# Pointwise fractions
@NLexpression(m, phiG_j[i=1:nfe, j=1:ncp], c[3, i, j] / (c[3, i, j] + c[4, i, j] + 1e-9))
@NLexpression(m, phiF_j[i=1:nfe, j=1:ncp], c[4, i, j] / (c[3, i, j] + c[4, i, j] + 1e-9))

# Kinetic rates at collocation points
@NLexpression(m, mu_j[i=1:nfe, j=1:ncp], exp(teta[Pidx(:mu0)]) * mu_T * (
    c[2, i, j] / (c[2, i, j] + exp(teta[Pidx(:Kn0)]) * Kg_T + 1e-9)
))
@NLexpression(m, betaG_j[i=1:nfe, j=1:ncp], exp(teta[Pidx(:betaG0)]) * b_T *
    (c[3, i, j] / (c[3, i, j] + exp(teta[Pidx(:Kg0)]) * Kg_T + 1e-9)) *
    ((exp(teta[Pidx(:Kie0)]) * Kg_T) / (c[5, i, j] + exp(teta[Pidx(:Kie0)]) * Kg_T + 1e-9))
)
@NLexpression(m, betaF_j[i=1:nfe, j=1:ncp], exp(teta[Pidx(:betaF0)]) * b_T *
    (c[4, i, j] / (c[4, i, j] + exp(teta[Pidx(:Kf0)]) * Kg_T + 1e-9)) *
    ((exp(teta[Pidx(:Kig0)]) * Kg_T) / (c[3, i, j] + exp(teta[Pidx(:Kig0)]) * Kg_T + 1e-9)) *
    ((exp(teta[Pidx(:Kie0)]) * Kg_T) / (c[5, i, j] + exp(teta[Pidx(:Kie0)]) * Kg_T + 1e-9))
)

# Death function
@NLexpression(m, Td_[i=1:nfe, j=1:ncp], -0.0001 * c[5, i, j]^3 + 0.0049 * c[5, i, j]^2 - 0.1279 * c[5, i, j] + 315.89)
@NLexpression(m, sw_[i=1:nfe, j=1:ncp], 0.5 * (1.0 + tanh(0.5 * (T_const - Td_[i, j]))))
@NLexpression(m, Kd_j[i=1:nfe, j=1:ncp], exp(teta[Pidx(:Kd0)]) * exp(0.0415 * c[5, i, j] + (130000.0 * (T_const - 305.65)) / (305.65 * R * T_const)) * sw_[i, j])

# FE-end kinetics for sugar demands
@NLexpression(m, mu_fe[i=1:nfe], exp(teta[Pidx(:mu0)]) * mu_T * (Ne[i] / (Ne[i] + exp(teta[Pidx(:Kn0)]) * Kg_T + 1e-9)))
@NLexpression(m, betaG_fe[i=1:nfe], exp(teta[Pidx(:betaG0)]) * b_T *
    (Ge[i] / (Ge[i] + exp(teta[Pidx(:Kg0)]) * Kg_T + 1e-9)) *
    ((exp(teta[Pidx(:Kie0)]) * Kg_T) / (Ee[i] + exp(teta[Pidx(:Kie0)]) * Kg_T + 1e-9))
)
@NLexpression(m, betaF_fe[i=1:nfe], exp(teta[Pidx(:betaF0)]) * b_T *
    (Fe[i] / (Fe[i] + exp(teta[Pidx(:Kf0)]) * Kg_T + 1e-9)) *
    ((exp(teta[Pidx(:Kig0)]) * Kg_T) / (Ge[i] + exp(teta[Pidx(:Kig0)]) * Kg_T + 1e-9)) *
    ((exp(teta[Pidx(:Kie0)]) * Kg_T) / (Ee[i] + exp(teta[Pidx(:Kie0)]) * Kg_T + 1e-9))
)

@NLexpression(m, rG[i=1:nfe], (mu_fe[i] / exp(teta[Pidx(:Yxg)])) + (betaG_fe[i] / exp(teta[Pidx(:Yeg)])) + (mrate * phiG_fe[i]))
@NLexpression(m, rF[i=1:nfe], (mu_fe[i] / exp(teta[Pidx(:Yxf)])) + (betaF_fe[i] / exp(teta[Pidx(:Yef)])) + (mrate * phiF_fe[i]))

# ---------------------------------------------
# Constraints
# ---------------------------------------------
# Collocation with variable hv (bilinear)
@NLconstraints(m, begin
    coll_c_n[l=1:nc, i=2:nfe, j=1:ncp], c[l, i, j] == c[l, i-1, ncp] + hv[i] * sum(colmat[j, k] * cdot[l, i, k] for k in 1:ncp)
    coll_c_0[l=1:nc, j=1:ncp],        c[l, 1, j]   == c0[l] + hv[1] * sum(colmat[j, k] * cdot[l, 1, k] for k in 1:ncp)
end)

@constraints(m, begin
    # Time partitioning for hv
    MFE1, sum(hv[i] for i in 1:nfe) == th
    MFE3[i=1:nfe], hv[i]  >= 0.0
    MFE4[i=1:nfe], hv[i]  >= (1.0 - var_h) * hm[1]
    MFE5[i=1:nfe], hv[i]  <= (1.0 + var_h) * hm[1]

    # State nonnegativity
    c_LB[l=1:nc, i=1:nfe, j=1:ncp], -c[l, i, j] <= 0

    # Bounds on parameters (log-space)
    teta_LB[p=1:np], teta[p] >= LB[p]
    teta_UB[p=1:np], teta[p] <= UB[p]
end)

# ODEs (Zenteno core)
@NLconstraints(m, begin
    dX[i=1:nfe, j=1:ncp], cdot[1, i, j] == (mu_j[i, j] - Kd_j[i, j]) * c[1, i, j]
    dN[i=1:nfe, j=1:ncp], cdot[2, i, j] == -(mu_j[i, j] / exp(teta[Pidx(:Yxn)])) * c[1, i, j]
    dG[i=1:nfe, j=1:ncp], cdot[3, i, j] == -((mu_j[i, j] / exp(teta[Pidx(:Yxg)])) + (betaG_j[i, j] / exp(teta[Pidx(:Yeg)])) + mrate * phiG_j[i, j]) * c[1, i, j]
    dF[i=1:nfe, j=1:ncp], cdot[4, i, j] == -((mu_j[i, j] / exp(teta[Pidx(:Yxf)])) + (betaF_j[i, j] / exp(teta[Pidx(:Yef)])) + mrate * phiF_j[i, j]) * c[1, i, j]
    dE[i=1:nfe, j=1:ncp], cdot[5, i, j] ==  (betaG_j[i, j] + betaF_j[i, j]) * c[1, i, j]
end)

# Stoichiometric balances and flux bounds
@constraints(m, begin
    Sc[mc=1:nm, i=1:nfe],  sum(S[mc, k] * v[k, i] for k in 1:nv) == 0
    v_UB[k=1:nv, i=1:nfe], v[k, i] - ub[k] <= 0
    v_LB[k=1:nv, i=1:nfe], -v[k, i] + lb[k] <= 0

    # Signs for multipliers
    alphaL_sign[k=1:nv, i=1:nfe], alpha_L[k, i] <= 0
    alphaU_sign[k=1:nv, i=1:nfe], alpha_U[k, i] >= 0
    alphaUPT_sign[u=1:2, i=1:nfe], alpha_upt[u, i] <= 0
end)

# Lagrangian stationarity (relaxed)
@constraints(m, begin
    Lagr[k=1:nv, i=1:nfe], + d[k] + w * v[k, i] * vs[k] + alpha_L[k, i] + alpha_U[k, i] +
                            up_glu[k] * alpha_upt[1, i] + up_fru[k] * alpha_upt[2, i] +
                            sum(S[r, k] * lambda_[r, i] for r in 1:nm) == 0
end)

# Complementarity product definitions and uptake inequalities
@NLconstraints(m, begin
    FO_L_def[k=1:nv, i=1:nfe],  FO_L[k, i]   == (v[k, i] - lb[k]) * alpha_L[k, i]
    FO_U_def[k=1:nv, i=1:nfe],  FO_U[k, i]   == (v[k, i] - ub[k]) * alpha_U[k, i]

    v_LB_g[i=1:nfe],            -v[glu, i] - rG[i] <= 0
    v_LB_f[i=1:nfe],            -v[fru, i] - rF[i] <= 0

    FO_upt1[i=1:nfe],           FO_upt[1, i] == (-v[glu, i] - rG[i]) * alpha_upt[1, i]
    FO_upt2[i=1:nfe],           FO_upt[2, i] == (-v[fru, i] - rF[i]) * alpha_upt[2, i]
end)

# (No FO equality: simulation-only mode)

# ---------------------------------------------
# Light-touch initialization mirrored from Pyomo (always on)
# - Fluxes: seed sugar uptakes v_glu, v_fru at FE ends to -rG_fe, -rF_fe
# - Alphas: activity-based seeds at bounds; uptake alphas when ineq is tight
# - Lambdas: least-squares init for S^T * lambda ≈ -(d + w*v + alphas + uptake)
# ---------------------------------------------
try
    # Helpers pulling initial state/param starts (FE-end uses j=ncp)
    function _phiGF(G::Float64, F::Float64)
        denom = G + F + 1e-9
        return G / denom, F / denom
    end
    # Safe getter for start values (default 0.0 when unset)
    _sv(x) = (v = start_value(x); v === nothing ? 0.0 : v)
    mu_T0 = exp(59453.0 * (T_const - 300.0) / (300.0 * R * T_const))
    Kg_T0 = exp(46055.0 * (T_const - 293.15) / (293.15 * R * T_const))
    b_T0  = exp(11000.0 * (T_const - 296.15) / (296.15 * R * T_const))
    mrate0 = 0.01 * exp(37681.0 * (T_const - 293.30) / (293.30 * R * T_const))

    # Parameter exponentials once
    function pexp(sym)
        return exp(T0[findfirst(==(sym), Pnames)])
    end
    mu0   = pexp(:mu0);  Kn0 = pexp(:Kn0);  Kg0 = pexp(:Kg0);  Kf0 = pexp(:Kf0)
    Kig0  = pexp(:Kig0); Kie0 = pexp(:Kie0); Kd0 = pexp(:Kd0)
    Yxn0  = pexp(:Yxn);  Yxg0 = pexp(:Yxg);  Yxf0 = pexp(:Yxf); Yeg0 = pexp(:Yeg); Yef0 = pexp(:Yef)
    bG0   = pexp(:betaG0); bF0 = pexp(:betaF0)

    # FE-end initial states are c0 by construction
    alpha0 = 1e-2
    # Precompute S^T for lambda LS
    ST = Array{Float64}(S)'

    for i in 1:nfe
        # FE-end states
        X = c0[1]; N = c0[2]; G = c0[3]; F = c0[4]; E = c0[5]
        phiG, phiF = _phiGF(G, F)
        mu_fe0   = mu0 * mu_T0 * (N / (N + Kn0 * Kg_T0 + 1e-9))
        betaG_fe0 = bG0 * b_T0 * (G / (G + Kg0 * Kg_T0 + 1e-9)) * ((Kie0 * Kg_T0) / (E + Kie0 * Kg_T0 + 1e-9))
        betaF_fe0 = bF0 * b_T0 * (F / (F + Kf0 * Kg_T0 + 1e-9)) * ((Kig0 * Kg_T0) / (G + Kig0 * Kg_T0 + 1e-9)) * ((Kie0 * Kg_T0) / (E + Kie0 * Kg_T0 + 1e-9))
        rG0 = (mu_fe0 / Yxg0) + (betaG_fe0 / Yeg0) + (mrate0 * phiG)
        rF0 = (mu_fe0 / Yxf0) + (betaF_fe0 / Yef0) + (mrate0 * phiF)

        # Seed uptakes and clip to bounds
        set_start_value(v[glu, i], -rG0)
        set_start_value(v[fru, i], -rF0)
        for k in 1:nv
            vk = _sv(v[k, i])
            if vk < lb[k]
                set_start_value(v[k, i], lb[k])
            elseif vk > ub[k]
                set_start_value(v[k, i], ub[k])
            end
        end

        # Activity-based alphas
        tol = 1e-8
        for k in 1:nv
            vk = _sv(v[k, i])
            aL = (abs(vk - lb[k]) <= tol) ? (-alpha0) : 0.0
            aU = (abs(vk - ub[k]) <= tol) ? (+alpha0) : 0.0
            set_start_value(alpha_L[k, i], aL)
            set_start_value(alpha_U[k, i], aU)
        end
        sG = -_sv(v[glu, i]) - rG0
        sF = -_sv(v[fru, i]) - rF0
        set_start_value(alpha_upt[1, i], (abs(sG) <= tol) ? (-alpha0) : 0.0)
        set_start_value(alpha_upt[2, i], (abs(sF) <= tol) ? (-alpha0) : 0.0)

        # Lambda least-squares
        rhs = zeros(nv)
        a_upt1 = _sv(alpha_upt[1, i])
        a_upt2 = _sv(alpha_upt[2, i])
        for k in 1:nv
            vk = _sv(v[k, i])
            aLk = _sv(alpha_L[k, i])
            aUk = _sv(alpha_U[k, i])
            upt = (k == glu ? a_upt1 : 0.0) + (k == fru ? a_upt2 : 0.0)
            rhs[k] = -(d[k] + w * vk * vs[k] + aLk + aUk + upt)
        end
        # Solve ST * lambda ≈ rhs
        lam = ST \ rhs
        for r in 1:nm
            set_start_value(lambda_[r, i], lam[r])
        end
    end
catch err
    @warn "Initialization skipped" err
end

# ---------------------------------------------
# Solve
# ---------------------------------------------
println("[INFO] Starting solve @ ", Dates.now())
optimize!(m)
status = termination_status(m)
pr_status = primal_status(m)
println("[INFO] Solver status: ", status, ", primal: ", pr_status)

# Report objective (penalty) value
try
    println("[INFO] Penalty objective: ", objective_value(m))
catch
end

# Save a small summary file
try
    summary_path = joinpath(RESULTS_DIR, "zenteno_relax_summary_" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS") * ".txt")
    open(summary_path, "w") do io
        println(io, "status=", status)
        println(io, "primal_status=", pr_status)
    try println(io, "objective=", objective_value(m)) catch end
    end
    println("[SAVE] ", summary_path)
catch err
    @warn "Failed to save summary" err
end

# ---------------------------------------------
# ODE integration (Zenteno) and overlay plot with MPCC FE-end points
# - Continuous lines: ODE solution (X,N,G,F,E)
# - Scatter points: MPCC FE-end values for X, G, F, E
# Saved under results/ as zenteno_ode_vs_mpcc_*.png
# ---------------------------------------------
try
    # Extract parameter values: prefer optimized values, fallback to starts
    function param_exp(sym)
        i = Pidx(sym)
        try
            v = value(teta[i])
            return exp((v === nothing || !isfinite(v)) ? T0[i] : v)
        catch
            return exp(T0[i])
        end
    end

    mu0   = param_exp(:mu0)
    betaG0= param_exp(:betaG0)
    betaF0= param_exp(:betaF0)
    Kn0   = param_exp(:Kn0)
    Kg0   = param_exp(:Kg0)
    Kf0   = param_exp(:Kf0)
    Kig0  = param_exp(:Kig0)
    Kie0  = param_exp(:Kie0)
    Kd0   = param_exp(:Kd0)
    Yxn   = param_exp(:Yxn)
    Yxg   = param_exp(:Yxg)
    Yxf   = param_exp(:Yxf)
    Yeg   = param_exp(:Yeg)
    Yef   = param_exp(:Yef)

    # Temperature scalars (reuse definitions)
    mu_T0 = exp(59453.0 * (T_const - 300.0) / (300.0 * R * T_const))
    Kg_T0 = exp(46055.0 * (T_const - 293.15) / (293.15 * R * T_const))
    b_T0  = exp(11000.0 * (T_const - 296.15) / (296.15 * R * T_const))
    mrate0 = 0.01 * exp(37681.0 * (T_const - 293.30) / (293.30 * R * T_const))

    # ODE RHS
    function zenteno_rhs!(du, u, p, t)
        X, N, G, F, E = u
        mu   = mu0 * mu_T0 * (N / (N + Kn0 * Kg_T0 + 1e-9))
        betaG = betaG0 * b_T0 * (G / (G + Kg0 * Kg_T0 + 1e-9)) * ((Kie0 * Kg_T0) / (E + Kie0 * Kg_T0 + 1e-9))
        betaF = betaF0 * b_T0 * (F / (F + Kf0 * Kg_T0 + 1e-9)) * ((Kig0 * Kg_T0) / (G + Kig0 * Kg_T0 + 1e-9)) * ((Kie0 * Kg_T0) / (E + Kie0 * Kg_T0 + 1e-9))
        Td   = -0.0001 * E^3 + 0.0049 * E^2 - 0.1279 * E + 315.89
        sw   = 0.5 * (1.0 + tanh(0.5 * (T_const - Td)))
        Kd   = Kd0 * exp(0.0415 * E + (130000.0 * (T_const - 305.65)) / (305.65 * R * T_const)) * sw
        denom = G + F + 1e-9
        phiG = G / denom
        phiF = F / denom

        du[1] = (mu - Kd) * X
        du[2] = -(mu / Yxn) * X
        du[3] = -((mu / Yxg) + (betaG / Yeg) + mrate0 * phiG) * X
        du[4] = -((mu / Yxf) + (betaF / Yef) + mrate0 * phiF) * X
        du[5] = (betaG + betaF) * X
        return nothing
    end

    # Integrate ODE
    u0 = copy(c0)
    tspan = (0.0, th)
    prob = DifferentialEquations.ODEProblem(zenteno_rhs!, u0, tspan)
    sol = DifferentialEquations.solve(prob, DifferentialEquations.Tsit5(), reltol=1e-6, abstol=1e-8)

    # Extract MPCC FE-end points and time grid from hv
    hv_val = [try value(hv[i]) catch; hm[i] end for i in 1:nfe]
    t_nodes = cumsum(hv_val)
    # FE-end states
    function safe_val(x)
        try
            return value(x)
        catch
            v = start_value(x)
            return v === nothing ? 0.0 : v
        end
    end
    X_fe = [safe_val(c[1, i, ncp]) for i in 1:nfe]
    N_fe = [safe_val(c[2, i, ncp]) for i in 1:nfe]
    G_fe = [safe_val(c[3, i, ncp]) for i in 1:nfe]
    F_fe = [safe_val(c[4, i, ncp]) for i in 1:nfe]
    E_fe = [safe_val(c[5, i, ncp]) for i in 1:nfe]

    # Compute ODE values at FE-end times and R^2 for X,G,F,E (MPCC vs ODE)
    function r2_score(y::Vector{<:Real}, yhat::Vector{<:Real})
        n = length(y)
        if n == 0
            return NaN
        end
        ȳ = sum(y) / n
        sst = sum((yi - ȳ)^2 for yi in y)
        sse = sum((y[i] - yhat[i])^2 for i in 1:n)
        return sst <= 1e-16 ? (sse <= 1e-16 ? 1.0 : 0.0) : 1 - sse / sst
    end
    # Interpolate ODE at FE times
    X_hat = [DifferentialEquations.solve(sol.prob, sol.alg, save_everystep=false, tstops=[t]).u[end][1] for t in t_nodes]  # robust single-step
    G_hat = [DifferentialEquations.solve(sol.prob, sol.alg, save_everystep=false, tstops=[t]).u[end][3] for t in t_nodes]
    F_hat = [DifferentialEquations.solve(sol.prob, sol.alg, save_everystep=false, tstops=[t]).u[end][4] for t in t_nodes]
    E_hat = [DifferentialEquations.solve(sol.prob, sol.alg, save_everystep=false, tstops=[t]).u[end][5] for t in t_nodes]
    # Prefer fast interpolation if available
    try
        X_hat = [sol(t)[1] for t in t_nodes]
        G_hat = [sol(t)[3] for t in t_nodes]
        F_hat = [sol(t)[4] for t in t_nodes]
        E_hat = [sol(t)[5] for t in t_nodes]
    catch
    end
    r2_X = r2_score(X_fe, X_hat)
    r2_G = r2_score(G_fe, G_hat)
    r2_F = r2_score(F_fe, F_hat)
    r2_E = r2_score(E_fe, E_hat)

    # Build plot with subplots (5 rows), include R^2 in titles
    plt = Plots.plot(layout=(5,1), size=(1000,1200))
    # X
    Plots.plot!(plt[1], sol.t, sol[1,:], label="ODE X", color=:blue, lw=2)
    Plots.scatter!(plt[1], t_nodes, X_fe, label="MPCC X", color=:black, m=:circle)
    Plots.title!(plt[1], @sprintf("X (R^2=%.3f)", r2_X))
    Plots.ylabel!(plt[1], "X")
    # N
    Plots.plot!(plt[2], sol.t, sol[2,:], label="ODE N", color=:blue, lw=2)
    # (sin puntos MPCC, no requerido)
    Plots.ylabel!(plt[2], "N")
    # G
    Plots.plot!(plt[3], sol.t, sol[3,:], label="ODE G", color=:blue, lw=2)
    Plots.scatter!(plt[3], t_nodes, G_fe, label="MPCC G", color=:red, m=:diamond)
    Plots.title!(plt[3], @sprintf("G (R^2=%.3f)", r2_G))
    Plots.ylabel!(plt[3], "G")
    # F
    Plots.plot!(plt[4], sol.t, sol[4,:], label="ODE F", color=:blue, lw=2)
    Plots.scatter!(plt[4], t_nodes, F_fe, label="MPCC F", color=:green, m=:utriangle)
    Plots.title!(plt[4], @sprintf("F (R^2=%.3f)", r2_F))
    Plots.ylabel!(plt[4], "F")
    # E
    Plots.plot!(plt[5], sol.t, sol[5,:], label="ODE E", color=:blue, lw=2)
    Plots.scatter!(plt[5], t_nodes, E_fe, label="MPCC E", color=:purple, m=:star5)
    Plots.title!(plt[5], @sprintf("E (R^2=%.3f)", r2_E))
    Plots.ylabel!(plt[5], "E")
    Plots.xlabel!(plt[5], "time")

    fig_path = joinpath(RESULTS_DIR, "zenteno_ode_vs_mpcc_" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS") * ".png")
    Plots.png(plt, fig_path)
    println("[PLOT] Saved ", fig_path)

    # Save R^2 metrics
    r2_path = joinpath(RESULTS_DIR, "zenteno_r2_" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS") * ".txt")
    open(r2_path, "w") do io
        println(io, @sprintf("R2_X=%.6f", r2_X))
        println(io, @sprintf("R2_G=%.6f", r2_G))
        println(io, @sprintf("R2_F=%.6f", r2_F))
        println(io, @sprintf("R2_E=%.6f", r2_E))
    end
    println("[METRICS] Saved ", r2_path)
catch err
    @warn "ODE integration/plotting failed. Install required packages?" err
end
