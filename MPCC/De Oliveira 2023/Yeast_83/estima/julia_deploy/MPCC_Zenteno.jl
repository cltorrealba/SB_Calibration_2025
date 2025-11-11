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
using Statistics: mean
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

# Synthetic data integration (will be used for future SSE objective)
# Expected file: synthetic_data.jld2 with variables:
#   states :: Matrix{Float64} of size (nc, n_time)
#   time   :: Vector{Float64} length n_time
# Only used for plotting now; does NOT affect objective yet.
const SYN_DATA_PATH = joinpath(BASE_DIR, "synthetic_data.jld2")
mutable struct SyntheticData
    t::Vector{Float64}
    Y::Matrix{Float64}   # (nc x n_time) rows match [X,N,G,F,E]
end
function load_synthetic_data(path::String, nc::Int)
    if isfile(path)
        try
            d = FileIO.load(path)
            if all(k -> k in keys(d), ["states","time"])
                Y = d["states"]
                t = d["time"]
                if size(Y,1) == nc && length(t) == size(Y,2)
                    return SyntheticData(t, Y)
                else
                    @warn "Synthetic data dimensions mismatch; ignoring" size(Y), length(t)
                end
            else
                @warn "synthetic_data.jld2 missing expected keys 'states' and 'time'" keys(d)
            end
        catch err
            @warn "Failed loading synthetic_data.jld2" err
        end
    end
    return nothing
end
syn_data = load_synthetic_data(SYN_DATA_PATH, 5)

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
# Test configuration: 240 h horizon, 12 finite elements (20 h per element)
nfe = 12            # number of finite elements
ncp = 3             # collocation points (Radau-3)
th = 240.0          # total horizon (hours)

# Toggle adaptivity via ENV["HV_ADAPTIVE"] == "1" (default fixed uniform)
const HV_ADAPTIVE = get(ENV, "HV_ADAPTIVE", "0") == "1"
var_h = HV_ADAPTIVE ? 1.0 : 0.0
hm = fill(th / nfe, nfe)  # nominal element length
println("[CFG] th=", th, ", nfe=", nfe, ", hv_mode=", (HV_ADAPTIVE ? "adaptive" : "fixed"))

# Default kinetics temperature and constants (overridable via ENV["T_CONST"])
T_const = try parse(Float64, get(ENV, "T_CONST", "296.15")) catch; 296.15 end
println("[CFG] T_const=", T_const)

# Radau-3 collocation matrix (same as Python)
colmat = [
    0.19681547722366   -0.06553542585020   0.02377097434822;
    0.39442431473909    0.29207341166523  -0.04154875212600;
    0.37640306270047    0.51248582618842   0.11111111111111
]

# Default kinetics constants
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

# Nominal parameters (log-parametrized). Bounds will be set based on identifiability/estimation set.
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

# Parse estimable parameters from ENV (comma-separated symbols). Default to a small set.
function _parse_est_params()::Vector{Symbol}
    # Default now only mu0 for initial verification run
    s = get(ENV, "EST_PARAMS", "mu0")
    parts = filter(!isempty, split(s, [',',';',' ']))
    syms = Symbol[]
    for p in parts
        push!(syms, Symbol(strip(p)))
    end
    # keep only valid names
    valid = Set(Pnames)
    return [x for x in syms if x in valid]
end
const EST_SET = _parse_est_params()
println("[CFG] Estimable params=", EST_SET)

# Bounds: estimables in [0.1x, 10x] (one log cycle); fixed at nominal (lb=ub)
for (i, k) in enumerate(Pnames)
    T0[i] = log(Pnom[k])
    if k in EST_SET
        LB[i] = log(max(1e-12, 0.1 * Pnom[k]))
        UB[i] = log(max(1e-12, 10.0 * Pnom[k]))
    else
        LB[i] = T0[i]
        UB[i] = T0[i]
    end
end

# Override narrower bounds for mu0 if estimable (diagnostic refinement)
if :mu0 in EST_SET
    i_mu = findfirst(==( :mu0), Pnames)
    LB[i_mu] = log(max(1e-12, 0.5 * Pnom[:mu0]))
    UB[i_mu] = log(max(1e-12, 2.0 * Pnom[:mu0]))
end

# Load data for FO (nc x ph x ncp)
data = load_data_default(nc, nfe, ncp)

# Measurement selection: use only X(1), G(3), F(4), E(5) in SSE
const MEAS_IDX = [1,3,4,5]

# Weights for objective terms
const W_SSE = 1.0
const W_PEN = 0.2
const W_REG = 1e-8

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
# Wall-clock time configurable via ENV["WALL_TIME"] (default 600 s). Use 60 for quick diagnostic runs.
wall_time = try parse(Float64, get(ENV, "WALL_TIME", "600")) catch; 600.0 end
set_optimizer_attribute(m, "max_wall_time", wall_time)
println("[CFG] Ipopt wall_time=", wall_time)

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
# Objective: SSE over measured states + penalties + small parameter regularization
# ---------------------------------------------
@NLexpression(m, SSE, sum( (c[l,i,j] - data[l,i,j])^2 for l in MEAS_IDX, i in 1:nfe, j in 1:ncp ))
@NLexpression(m, PEN, sum( sum( -phi1 * FO_L[k,i] - phi3 * FO_U[k,i] for k in 1:nv )
                        +  phi2 * FO_upt[1,i] + phi2 * FO_upt[2,i]
                    for i in 1:nfe ))
# Regularize only estimable params around nominal
const EST_POS = [findfirst(==(k), Pnames) for k in EST_SET]
@NLexpression(m, REG, sum( (teta[p] - T0[p])^2 for p in EST_POS ))
@NLobjective(m, Min, W_SSE * SSE + W_PEN * PEN + W_REG * REG)

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

# Parameter starts (log-space). For estimables, perturb away from nominal within bounds.
for (i,k) in enumerate(Pnames)
    if k in EST_SET
        # deterministic offset: 1.5x, clipped to [LB, UB]
        vstart = log(clamp(Pnom[k] * 1.5, exp(LB[i]), exp(UB[i])))
        set_start_value(teta[i], vstart)
    else
        set_start_value(teta[i], T0[i])
    end
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

if HV_ADAPTIVE
    @constraints(m, begin
        # Adaptive time partitioning
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
else
    @constraints(m, begin
        # Uniform time partitioning: fix each hv[i] to hm[i]
        MFE_fix[i=1:nfe], hv[i] == hm[i]

        # State nonnegativity
        c_LB[l=1:nc, i=1:nfe, j=1:ncp], -c[l, i, j] <= 0

        # Bounds on parameters (log-space)
        teta_LB[p=1:np], teta[p] >= LB[p]
        teta_UB[p=1:np], teta[p] <= UB[p]
    end)
end

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
println("[INFO] Pre-optimization ODE simulation @ ", Dates.now())

# ---------------------------------------------
# Pre-optimization ODE simulation (nominal for fixed params, start values for estimables)
# Saves plot: zenteno_pre_ode_vs_data_*.png (continuous ODE line + synthetic data scatter)
# ---------------------------------------------
try
    function param_start_exp(sym)
        i = Pidx(sym)
        st = try start_value(teta[i]) catch; nothing end
        if st === nothing
            return exp(T0[i])
        end
        return exp(st)
    end
    mu0_s   = param_start_exp(:mu0)
    betaG0_s= param_start_exp(:betaG0)
    betaF0_s= param_start_exp(:betaF0)
    Kn0_s   = param_start_exp(:Kn0)
    Kg0_s   = param_start_exp(:Kg0)
    Kf0_s   = param_start_exp(:Kf0)
    Kig0_s  = param_start_exp(:Kig0)
    Kie0_s  = param_start_exp(:Kie0)
    Kd0_s   = param_start_exp(:Kd0)
    Yxn_s   = param_start_exp(:Yxn)
    Yxg_s   = param_start_exp(:Yxg)
    Yxf_s   = param_start_exp(:Yxf)
    Yeg_s   = param_start_exp(:Yeg)
    Yef_s   = param_start_exp(:Yef)

    mu_T0 = exp(59453.0 * (T_const - 300.0) / (300.0 * R * T_const))
    Kg_T0 = exp(46055.0 * (T_const - 293.15) / (293.15 * R * T_const))
    b_T0  = exp(11000.0 * (T_const - 296.15) / (296.15 * R * T_const))
    mrate0 = 0.01 * exp(37681.0 * (T_const - 293.30) / (293.30 * R * T_const))

    function zenteno_rhs_pre!(du,u,p,t)
        X,N,G,F,E = u
        mu   = mu0_s * mu_T0 * (N / (N + Kn0_s * Kg_T0 + 1e-9))
        betaG = betaG0_s * b_T0 * (G / (G + Kg0_s * Kg_T0 + 1e-9)) * ((Kie0_s * Kg_T0) / (E + Kie0_s * Kg_T0 + 1e-9))
        betaF = betaF0_s * b_T0 * (F / (F + Kf0_s * Kg_T0 + 1e-9)) * ((Kig0_s * Kg_T0) / (G + Kig0_s * Kg_T0 + 1e-9)) * ((Kie0_s * Kg_T0) / (E + Kie0_s * Kg_T0 + 1e-9))
        Td   = -0.0001 * E^3 + 0.0049 * E^2 - 0.1279 * E + 315.89
        sw   = 0.5 * (1.0 + tanh(0.5 * (T_const - Td)))
        Kd   = Kd0_s * exp(0.0415 * E + (130000.0 * (T_const - 305.65)) / (305.65 * R * T_const)) * sw
        denom = G + F + 1e-9
        phiG = G / denom
        phiF = F / denom
        du[1] = (mu - Kd) * X
        du[2] = -(mu / Yxn_s) * X
        du[3] = -((mu / Yxg_s) + (betaG / Yeg_s) + mrate0 * phiG) * X
        du[4] = -((mu / Yxf_s) + (betaF / Yef_s) + mrate0 * phiF) * X
        du[5] = (betaG + betaF) * X
        return nothing
    end
    u0 = copy(c0)
    prob_pre = DifferentialEquations.ODEProblem(zenteno_rhs_pre!, u0, (0.0, th))
    sol_pre = DifferentialEquations.solve(prob_pre, DifferentialEquations.Tsit5(), reltol=1e-6, abstol=1e-8)

    # Synthetic data filtering (prefer synthetic_data.jld2; fallback to data.jld2 at FE-end times)
    t_syn = Float64[]; Y_syn = Matrix{Float64}(undef, 0, 0)
    if syn_data !== nothing
        idx = findall(t -> (t >= 0.0) && (t <= th + 1e-9), syn_data.t)
        if !isempty(idx)
            t_syn = syn_data.t[idx]
            Y_syn = syn_data.Y[:, idx]
        end
    end
    if isempty(t_syn)
        # Fallback: use FE-end grid (uniform pre-solve) and data at j=ncp
        t_syn = cumsum(hm)
        Y_syn = zeros(nc, length(t_syn))
        for i in 1:nfe
            for l in 1:nc
                Y_syn[l, i] = data[l, i, ncp]
            end
        end
        # If fallback is all zeros, keep but use a distinct marker to avoid confusion
    end
    # Plot only measured states [X,G,F,E] as requested
    plt_pre = Plots.plot(layout=(4,1), size=(1000,1000))
    # X (1)
    Plots.plot!(plt_pre[1], sol_pre.t, sol_pre[1,:], label="ODE X (start)", color=:navy, lw=2)
    if !isempty(t_syn); Plots.scatter!(plt_pre[1], t_syn, Y_syn[1,:], label="DATA X", color=:orange, m=:xcross); end
    Plots.ylabel!(plt_pre[1], "X")
    # G (3)
    Plots.plot!(plt_pre[2], sol_pre.t, sol_pre[3,:], label="ODE G (start)", color=:navy, lw=2)
    if !isempty(t_syn); Plots.scatter!(plt_pre[2], t_syn, Y_syn[3,:], label="DATA G", color=:orange, m=:xcross); end
    Plots.ylabel!(plt_pre[2], "G")
    # F (4)
    Plots.plot!(plt_pre[3], sol_pre.t, sol_pre[4,:], label="ODE F (start)", color=:navy, lw=2)
    if !isempty(t_syn); Plots.scatter!(plt_pre[3], t_syn, Y_syn[4,:], label="DATA F", color=:orange, m=:xcross); end
    Plots.ylabel!(plt_pre[3], "F")
    # E (5)
    Plots.plot!(plt_pre[4], sol_pre.t, sol_pre[5,:], label="ODE E (start)", color=:navy, lw=2)
    if !isempty(t_syn); Plots.scatter!(plt_pre[4], t_syn, Y_syn[5,:], label="DATA E", color=:orange, m=:xcross); end
    Plots.ylabel!(plt_pre[4], "E"); Plots.xlabel!(plt_pre[4], "time")
    pre_path = joinpath(RESULTS_DIR, "zenteno_pre_ode_vs_data_" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS") * ".png")
    # R² diagnostics (ODE vs synthetic data) for measured states
    if !isempty(t_syn)
        _r2(y_obs, y_pred) = (length(y_obs) <= 1 ? NaN : (1 - sum((y_obs .- y_pred).^2) / sum((y_obs .- mean(y_obs)).^2)))
        predX = [sol_pre(t)[1] for t in t_syn]
        predG = [sol_pre(t)[3] for t in t_syn]
        predF = [sol_pre(t)[4] for t in t_syn]
        predE = [sol_pre(t)[5] for t in t_syn]
        r2X = _r2(Y_syn[1,:], predX); r2G = _r2(Y_syn[3,:], predG); r2F = _r2(Y_syn[4,:], predF); r2E = _r2(Y_syn[5,:], predE)
        Plots.title!(plt_pre[1], @sprintf("X (R²=%.3f)", r2X))
        Plots.title!(plt_pre[2], @sprintf("G (R²=%.3f)", r2G))
        Plots.title!(plt_pre[3], @sprintf("F (R²=%.3f)", r2F))
        Plots.title!(plt_pre[4], @sprintf("E (R²=%.3f)", r2E))
    end
    Plots.png(plt_pre, pre_path)
    println("[PLOT] Saved pre-optimization ODE plot ", pre_path)
catch err
    @warn "Pre-optimization ODE simulation failed" err
end

println("[INFO] Starting optimization @ ", Dates.now())
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
# Estimation report (parameters, bounds, starts, solution, metrics)
# ---------------------------------------------
try
    rep_path = joinpath(RESULTS_DIR, "zenteno_estimation_report_" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS") * ".txt")
    open(rep_path, "w") do io
        println(io, "th=", th, ", nfe=", nfe, ", hv_mode=", (HV_ADAPTIVE ? "adaptive" : "fixed"))
        println(io, "measured_states=", MEAS_IDX)
        println(io, "estimable_params=", EST_SET)
        # Metrics
        sse_val = try value(SSE) catch; NaN end
        pen_val = try value(PEN) catch; NaN end
        reg_val = try value(REG) catch; NaN end
        println(io, @sprintf("SSE=%.6e", sse_val))
        println(io, @sprintf("PEN=%.6e", pen_val))
        println(io, @sprintf("REG=%.6e", reg_val))
        println(io, @sprintf("OBJ=%.6e", try objective_value(m) catch; NaN end))
        # Complementarity diagnostics (FO products magnitude summaries)
        foL_max = try maximum(abs(value(FO_L[k,i])) for k in 1:nv, i in 1:nfe) catch; NaN end
        foU_max = try maximum(abs(value(FO_U[k,i])) for k in 1:nv, i in 1:nfe) catch; NaN end
        foupt_max = try maximum(abs(value(FO_upt[u,i])) for u in 1:2, i in 1:nfe) catch; NaN end
        fo_sum = try sum(abs(value(FO_L[k,i])) + abs(value(FO_U[k,i])) for k in 1:nv, i in 1:nfe) catch; NaN end
        foupt_sum = try sum(abs(value(FO_upt[u,i])) for u in 1:2, i in 1:nfe) catch; NaN end
        println(io, @sprintf("FO_L_max=%.6e", foL_max))
        println(io, @sprintf("FO_U_max=%.6e", foU_max))
        println(io, @sprintf("FO_upt_max=%.6e", foupt_max))
        println(io, @sprintf("FO_LU_sum=%.6e", fo_sum))
        println(io, @sprintf("FO_upt_sum=%.6e", foupt_sum))
        println(io)
        println(io, "# Parameters (name, nominal, lb, ub, start, opt_log, opt_real)")
        for (i,k) in enumerate(Pnames)
            lb_i = LB[i]; ub_i = UB[i]; t0_i = T0[i]
            st_i = try start_value(teta[i]) catch; t0_i end
            opt_i = try value(teta[i]) catch; NaN end
            @printf(io, "%8s  nom=% .6e  lb=% .6e  ub=% .6e  start=% .6e  opt_log=% .6e  opt=%.6e\n",
                   String(k), exp(t0_i), exp(lb_i), exp(ub_i), exp(st_i), opt_i, (isfinite(opt_i) ? exp(opt_i) : NaN))
        end
    end
    println("[REPORT] Saved ", rep_path)
catch err
    @warn "Failed to save estimation report" err
end

    # ---------------------------------------------
    # Post-optimization ODE simulation (optimized for estimables, nominal for fixed)
    # Saves plot: zenteno_post_ode_vs_data_*.png (continuous ODE line + synthetic data scatter)
    # ---------------------------------------------
    try
        function param_opt_exp(sym)
            i = Pidx(sym)
            v = try value(teta[i]) catch; nothing end
            if v === nothing || !isfinite(v)
                return exp(T0[i])
            end
            return exp(v)
        end
        mu0_o   = param_opt_exp(:mu0)
        betaG0_o= param_opt_exp(:betaG0)
        betaF0_o= param_opt_exp(:betaF0)
        Kn0_o   = param_opt_exp(:Kn0)
        Kg0_o   = param_opt_exp(:Kg0)
        Kf0_o   = param_opt_exp(:Kf0)
        Kig0_o  = param_opt_exp(:Kig0)
        Kie0_o  = param_opt_exp(:Kie0)
        Kd0_o   = param_opt_exp(:Kd0)
        Yxn_o   = param_opt_exp(:Yxn)
        Yxg_o   = param_opt_exp(:Yxg)
        Yxf_o   = param_opt_exp(:Yxf)
        Yeg_o   = param_opt_exp(:Yeg)
        Yef_o   = param_opt_exp(:Yef)

        mu_T0 = exp(59453.0 * (T_const - 300.0) / (300.0 * R * T_const))
        Kg_T0 = exp(46055.0 * (T_const - 293.15) / (293.15 * R * T_const))
        b_T0  = exp(11000.0 * (T_const - 296.15) / (296.15 * R * T_const))
        mrate0 = 0.01 * exp(37681.0 * (T_const - 293.30) / (293.30 * R * T_const))

        function zenteno_rhs_post!(du,u,p,t)
            X,N,G,F,E = u
            mu   = mu0_o * mu_T0 * (N / (N + Kn0_o * Kg_T0 + 1e-9))
            betaG = betaG0_o * b_T0 * (G / (G + Kg0_o * Kg_T0 + 1e-9)) * ((Kie0_o * Kg_T0) / (E + Kie0_o * Kg_T0 + 1e-9))
            betaF = betaF0_o * b_T0 * (F / (F + Kf0_o * Kg_T0 + 1e-9)) * ((Kig0_o * Kg_T0) / (G + Kig0_o * Kg_T0 + 1e-9)) * ((Kie0_o * Kg_T0) / (E + Kie0_o * Kg_T0 + 1e-9))
            Td   = -0.0001 * E^3 + 0.0049 * E^2 - 0.1279 * E + 315.89
            sw   = 0.5 * (1.0 + tanh(0.5 * (T_const - Td)))
            Kd   = Kd0_o * exp(0.0415 * E + (130000.0 * (T_const - 305.65)) / (305.65 * R * T_const)) * sw
            denom = G + F + 1e-9
            phiG = G / denom
            phiF = F / denom
            du[1] = (mu - Kd) * X
            du[2] = -(mu / Yxn_o) * X
            du[3] = -((mu / Yxg_o) + (betaG / Yeg_o) + mrate0 * phiG) * X
            du[4] = -((mu / Yxf_o) + (betaF / Yef_o) + mrate0 * phiF) * X
            du[5] = (betaG + betaF) * X
            return nothing
        end
        u0 = copy(c0)
        prob_post = DifferentialEquations.ODEProblem(zenteno_rhs_post!, u0, (0.0, th))
        sol_post = DifferentialEquations.solve(prob_post, DifferentialEquations.Tsit5(), reltol=1e-6, abstol=1e-8)

        # Synthetic data filtering (prefer synthetic_data; fallback to data on FE grid)
        t_syn = Float64[]; Y_syn = Matrix{Float64}(undef, 0, 0)
        if syn_data !== nothing
            idx = findall(t -> (t >= 0.0) && (t <= th + 1e-9), syn_data.t)
            if !isempty(idx)
                t_syn = syn_data.t[idx]
                Y_syn = syn_data.Y[:, idx]
            end
        end
        if isempty(t_syn)
            t_syn = cumsum([try value(hv[i]) catch; hm[i] end for i in 1:nfe])
            Y_syn = zeros(nc, length(t_syn))
            for i in 1:nfe
                for l in 1:nc
                    Y_syn[l, i] = data[l, i, ncp]
                end
            end
        end
        # Extract MPCC FE-end solution snapshots for measured states.
        # Define time nodes (FE ends) and corresponding state values. Handles infeasible/partial solutions.
        t_nodes = cumsum([try value(hv[i]) catch; hm[i] end for i in 1:nfe])
        X_fe = [try value(c[1,i,ncp]) catch; NaN end for i in 1:nfe]
        G_fe = [try value(c[3,i,ncp]) catch; NaN end for i in 1:nfe]
        F_fe = [try value(c[4,i,ncp]) catch; NaN end for i in 1:nfe]
        E_fe = [try value(c[5,i,ncp]) catch; NaN end for i in 1:nfe]
        # Build combined plot for measured states [X,G,F,E]: data scatter + ODE line + MPCC FE-end markers
        plt_post = Plots.plot(layout=(4,1), size=(1100,1100))
        # X (1)
        Plots.plot!(plt_post[1], sol_post.t, sol_post[1,:], label="ODE X (opt)", color=:navy, lw=2)
        if !isempty(t_syn); Plots.scatter!(plt_post[1], t_syn, Y_syn[1,:], label="DATA X", color=:orange, m=:xcross); end
        Plots.scatter!(plt_post[1], t_nodes, X_fe, label="MPCC X", color=:black, m=:circle)
        Plots.ylabel!(plt_post[1], "X")
        # G (3)
        Plots.plot!(plt_post[2], sol_post.t, sol_post[3,:], label="ODE G (opt)", color=:navy, lw=2)
        if !isempty(t_syn); Plots.scatter!(plt_post[2], t_syn, Y_syn[3,:], label="DATA G", color=:orange, m=:xcross); end
        Plots.scatter!(plt_post[2], t_nodes, G_fe, label="MPCC G", color=:red, m=:diamond)
        Plots.ylabel!(plt_post[2], "G")
        # F (4)
        Plots.plot!(plt_post[3], sol_post.t, sol_post[4,:], label="ODE F (opt)", color=:navy, lw=2)
        if !isempty(t_syn); Plots.scatter!(plt_post[3], t_syn, Y_syn[4,:], label="DATA F", color=:orange, m=:xcross); end
        Plots.scatter!(plt_post[3], t_nodes, F_fe, label="MPCC F", color=:green, m=:utriangle)
        Plots.ylabel!(plt_post[3], "F")
        # E (5)
        Plots.plot!(plt_post[4], sol_post.t, sol_post[5,:], label="ODE E (opt)", color=:navy, lw=2)
        if !isempty(t_syn); Plots.scatter!(plt_post[4], t_syn, Y_syn[5,:], label="DATA E", color=:orange, m=:xcross); end
        Plots.scatter!(plt_post[4], t_nodes, E_fe, label="MPCC E", color=:purple, m=:star5)
        Plots.ylabel!(plt_post[4], "E"); Plots.xlabel!(plt_post[4], "time")
        post_path = joinpath(RESULTS_DIR, "zenteno_post_ode_vs_data_mpcc_" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS") * ".png")
        # R² diagnostics post (ODE opt vs data)
        if !isempty(t_syn)
            _r2(y_obs, y_pred) = (length(y_obs) <= 1 ? NaN : (1 - sum((y_obs .- y_pred).^2) / sum((y_obs .- mean(y_obs)).^2)))
            predX = [sol_post(t)[1] for t in t_syn]
            predG = [sol_post(t)[3] for t in t_syn]
            predF = [sol_post(t)[4] for t in t_syn]
            predE = [sol_post(t)[5] for t in t_syn]
            r2X = _r2(Y_syn[1,:], predX); r2G = _r2(Y_syn[3,:], predG); r2F = _r2(Y_syn[4,:], predF); r2E = _r2(Y_syn[5,:], predE)
            Plots.title!(plt_post[1], @sprintf("X (R²=%.3f)", r2X))
            Plots.title!(plt_post[2], @sprintf("G (R²=%.3f)", r2G))
            Plots.title!(plt_post[3], @sprintf("F (R²=%.3f)", r2F))
            Plots.title!(plt_post[4], @sprintf("E (R²=%.3f)", r2E))
        end
        Plots.png(plt_post, post_path)
        println("[PLOT] Saved post-optimization ODE plot ", post_path)
    catch err
        @warn "Post-optimization ODE simulation failed" err
    end

## (Removed standalone ODE vs MPCC plot; now integrated into the post-optimization plot.)
