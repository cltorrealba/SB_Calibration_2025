#=
MPCC_Zenteno (relaxed) — Julia/JuMP port of the Pyomo script
"pyomo_deploy/MPCC_Zenteno_relax.py" using main.jl as structural reference.

Highlights
- Zenteno ODE (5 states: X, N, G, F, E) via Radau-3 collocation
- Relaxed MPCC/dFBA coupling:
    - Flux bounds via alpha_L/alpha_U with FO_L/FO_U complementarity terms
    - Sugar uptake inequalities using FE-end Zenteno rates rG_fe, rF_fe with alpha_upt and FO_upt
    - Lagrangian stationarity with small pFBA ridge w and growth-only d-vector
- Pure simulation mode (no data-fit SSE) with linear complementarity penalties (relaxed Pyomo parity)
- Ipopt configured with exact Hessian (no LBFGS) and 5 min wall-clock limit (modifiable in-code)

Files expected (same folder as main.jl):
- S.csv, lb.csv, ub.csv (stoichiometry and bounds)
- data.jld2 with variable "data" shaped (nc=5, ph=nfe, ncp=3) for FO (not used in objective here)

Note: This is a relaxed MPCC variant (no equality FE-end ties). Warm-start heuristics are limited to
simple deterministic seeding; we always use exact Hessian as requested.
=#

using JuMP
using Ipopt
using LinearAlgebra
using DelimitedFiles
using FileIO, JLD2
using Dates
using DifferentialEquations
using Plots

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
th = 200.0          # total horizon (hours)
hm = fill(th / nfe, nfe)  # nominal element length
var_h = 1.0         # allow hv ∈ [(1-var_h)h, (1+var_h)h]

# Radau-3 collocation matrix (same as Python)
colmat = [
    0.19681547722366   -0.06553542585020   0.02377097434822;
    0.39442431473909    0.29207341166523  -0.04154875212600;
    0.37640306270047    0.51248582618842   0.11111111111111
]

# Radau-3 nodes for plotting/sampling
const RADAU3_NODES = [0.155051025721682, 0.644948974278318, 1.0]

# Default kinetics temperature and constants
T_const = 293.15
R = 8.314

# Run mode toggle:
# - "estimate" (default): fit with SSE + penalties
# - "simulate": ODE-only simulation and exit (no optimization)
# - "simulate_mpcc": solve MPCC NLP with penalty-only objective (no SSE) using nominal parameters
const RUN_MODE = get(ENV, "MPCC_RUN_MODE", "estimate")

# pFBA ridge (in stationarity) and linear penalty weights (mirror relaxed Pyomo)
# Fixed defaults (no toggles): feasibility-first configuration
w = 1e-20
phi1 = 1.0
phi2 = 1.0
phi3 = 1.0

# Data-fitting vs penalty weights and regularization (used in estimate mode)
const W_SSE = 1.0
const W_PEN = 0.2
const W_REG = 1e-6

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
const EST_SYMS = (:mu0, :Yeg, :Yef)   # parameters to estimate
const Pnom = Dict(
    :mu0=>0.141665, :betaG0=>1.41182, :betaF0=>8.49482, :Kn0=>0.226882,
    :Kg0=>3.1514, :Kf0=>2.97625, :Kig0=>29.5276, :Kie0=>2.99809, :Kd0=>3.11736e-5,
    :Yxn=>9.80576, :Yxg=>0.394345, :Yxf=>0.18622, :Yeg=>0.14133, :Yef=>0.96932
)
np = length(Pnames)
LB = similar(zeros(np)); UB = similar(zeros(np)); T0 = similar(zeros(np))
for (i, k) in enumerate(Pnames)
    if k in EST_SYMS
        # bounds only for estimated parameters
        LB[i] = log(max(1e-12, 0.2 * Pnom[k]))
        UB[i] = log(max(1e-12, 5.0 * Pnom[k]))
    else
        # Fix: lb=ub=log(Pnom)
        LB[i] = log(Pnom[k])
        UB[i] = log(Pnom[k])
    end
    T0[i] = log(Pnom[k])
end

# In MPCC simulation mode, fix all parameters at nominal (no estimation)
if RUN_MODE == "simulate_mpcc"
    for i in 1:np
        LB[i] = log(Pnom[Pnames[i]])
        UB[i] = LB[i]
    end
end

# Prior (initial guess) shifted away from nominal (deterministic factors)
const PRIOR_MUL = Dict(
    :mu0=>1.5, :betaG0=>0.6, :betaF0=>1.8, :Kn0=>1.3, :Kg0=>0.8, :Kf0=>1.2,
    :Kig0=>0.7, :Kie0=>1.4, :Kd0=>2.0, :Yxn=>0.9, :Yxg=>1.4, :Yxf=>0.8,
    :Yeg=>1.2, :Yef=>0.7
)
Tstart = similar(T0)
for (i,k) in enumerate(Pnames)
    mul = get(PRIOR_MUL, k, 1.5)
    v = if k in EST_SYMS
        clamp(Pnom[k]*mul, exp(LB[i]), exp(UB[i]))
    else
        Pnom[k]  # fixed parameters start at true value
    end
    Tstart[i] = log(v)
end

# Load data for FO (nc x ph x ncp) and tgrid if available
function load_data_and_tgrid(nc::Int, ph::Int, ncp::Int)
    path = joinpath(BASE_DIR, "data.jld2")
    if isfile(path)
        d = FileIO.load(path)
        data = haskey(d, "data") ? d["data"] : zeros(nc, ph, ncp)
        tgrid = haskey(d, "tgrid") ? d["tgrid"] : begin
            hv = th / ph
            tg = Array{Float64}(undef, ph, ncp)
            for i in 1:ph
                tstart = (i-1) * hv
                for (j, τ) in enumerate(RADAU3_NODES)
                    tg[i, j] = tstart + τ * hv
                end
            end
            tg
        end
        return data, tgrid
    else
        hv = th / ph
        tg = Array{Float64}(undef, ph, ncp)
        for i in 1:ph
            tstart = (i-1) * hv
            for (j, τ) in enumerate(RADAU3_NODES)
                tg[i, j] = tstart + τ * hv
            end
        end
        return zeros(nc, ph, ncp), tg
    end
end

data, tgrid = load_data_and_tgrid(nc, nfe, ncp)

# ---------------------------------------------
# ODE for standalone simulation (for plotting)
# ---------------------------------------------
struct ZentenoParams
    mu0::Float64; betaG0::Float64; betaF0::Float64
    Kn0::Float64; Kg0::Float64; Kf0::Float64; Kig0::Float64; Kie0::Float64; Kd0::Float64
    Yxn::Float64; Yxg::Float64; Yxf::Float64; Yeg::Float64; Yef::Float64
end

function zenteno_ode!(du, u, p::ZentenoParams, t)
    X, N, G, F, E = u
    # Temperature scalars
    mu_T =  exp(59453.0 * (T_const - 300.0) / (300.0 * R * T_const))
    Kg_T =  exp(46055.0 * (T_const - 293.15) / (293.15 * R * T_const))
    b_T  =  exp(11000.0 * (T_const - 296.15) / (296.15 * R * T_const))
    mrate = 0.01 * exp(37681.0 * (T_const - 293.30) / (293.30 * R * T_const))
    eps = 1e-9
    denom = G + F + eps
    phiG = G / denom
    phiF = F / denom
    mu   = p.mu0 * mu_T * (N / (N + p.Kn0 * Kg_T + eps))
    betaG = p.betaG0 * b_T * (G / (G + p.Kg0 * Kg_T + eps)) * ((p.Kie0 * Kg_T) / (E + p.Kie0 * Kg_T + eps))
    betaF = p.betaF0 * b_T * (F / (F + p.Kf0 * Kg_T + eps)) * ((p.Kig0 * Kg_T) / (G + p.Kig0 * Kg_T + eps)) * ((p.Kie0 * Kg_T) / (E + p.Kie0 * Kg_T + eps))
    Td = -0.0001 * E^3 + 0.0049 * E^2 - 0.1279 * E + 315.89
    sw = 0.5 * (1.0 + tanh(0.5 * (T_const - Td)))
    Kd = p.Kd0 * exp(0.0415 * E + (130000.0 * (T_const - 305.65)) / (305.65 * R * T_const)) * sw
    du[1] = (mu - Kd) * X
    du[2] = -(mu / p.Yxn) * X
    du[3] = -((mu / p.Yxg) + (betaG / p.Yeg) + mrate * phiG) * X
    du[4] = -((mu / p.Yxf) + (betaF / p.Yef) + mrate * phiF) * X
    du[5] =  (betaG + betaF) * X
    return nothing
end

function simulate_zenteno(params::ZentenoParams; tspan=(0.0, th))
    u0 = c0
    prob = ODEProblem(zenteno_ode!, u0, tspan, params)
    sol = solve(prob, Rodas5(); reltol=1e-8, abstol=1e-10)
    t_dense = collect(range(tspan[1], tspan[2]; length=2001))
    U = reduce(hcat, [sol(t) for t in t_dense])
    return t_dense, U
end

# Create a time grid based on optimized hv values (Radau-3 nodes per FE)
function build_time_grid_from_hv(hv_vals::Vector{Float64})
    ph = length(hv_vals)
    tg = Array{Float64}(undef, ph, ncp)
    tacc = 0.0
    for i in 1:ph
        for (j, τ) in enumerate(RADAU3_NODES)
            tg[i, j] = tacc + τ * hv_vals[i]
        end
        tacc += hv_vals[i]
    end
    return tg
end

function plot_fit_overlay(t_dense, U, tgrid, data; title_str::String, save_path::AbstractString)
    plt = plot(layout=(5,1), size=(900, 1200))
    state_labels = ["X", "N", "G", "F", "E"]
    colors = [:blue, :green, :red, :orange, :purple]
    for s in 1:5
        plot!(plt[s], t_dense, U[s, :]; color=colors[s], label="$(state_labels[s]) clean", lw=2)
        ts = vec(tgrid)
        ys = reshape(data[s, :, :], size(tgrid,1)*size(tgrid,2))
        scatter!(plt[s], ts, ys; color=:black, ms=3, alpha=0.7, label="data")
        ylabel!(plt[s], state_labels[s])
        if s == 1
            title!(plt[s], title_str)
        end
    end
    xlabel!(plt[5], "time [h]")
    savefig(plt, save_path)
    println("[PLOT] Saved ", save_path)
end

# ---------------------------------------------
# Early exit path: simulation-only (no optimization)
# ---------------------------------------------
if RUN_MODE == "simulate"
    # Simulate with original (nominal) parameter vector from the dataset table
    pnom = ZentenoParams(Pnom[:mu0], Pnom[:betaG0], Pnom[:betaF0], Pnom[:Kn0], Pnom[:Kg0], Pnom[:Kf0], Pnom[:Kig0], Pnom[:Kie0], Pnom[:Kd0],
                         Pnom[:Yxn], Pnom[:Yxg], Pnom[:Yxf], Pnom[:Yeg], Pnom[:Yef])
    t_dense, U = simulate_zenteno(pnom)
    sim_path = joinpath(RESULTS_DIR, "sim_only_nominal_" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS") * ".png")
    plot_fit_overlay(t_dense, U, tgrid, data; title_str="Simulation-only (nominal) vs data (200h)", save_path=sim_path)
    println("[SIM_ONLY] Saved ", sim_path)
    # Terminate script without building optimization model
    exit()
end

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
# Exact Hessian and wall-clock 300 s (override with ENV["MPCC_SHORT_TEST"])
# Wall-clock time limit: 5 min for simulate_mpcc, longer (30 min) for estimate
if RUN_MODE == "simulate_mpcc"
    set_optimizer_attribute(m, "max_wall_time", 300.0)
else
    set_optimizer_attribute(m, "max_wall_time", 1800.0)
end

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
# Objective: conditional
# - estimate: SSE (data fit) + linear complementarity penalties + regularization
# - simulate_mpcc: penalty-only (complementarity + uptake penalties), parameters fixed at nominal
# ---------------------------------------------
@NLexpression(m, SSE, sum( (c[l,i,j] - data[l,i,j])^2 for l in 1:nc, i in 1:nfe, j in 1:ncp ))
@NLexpression(m, PEN, sum( sum( -phi1 * FO_L[k,i] - phi3 * FO_U[k,i] for k in 1:nv )
                        +  phi2 * FO_upt[1,i] + phi2 * FO_upt[2,i]
                    for i in 1:nfe ))
@NLexpression(m, REG, sum( (teta[p] - T0[p])^2 for p in 1:np ))
if RUN_MODE == "simulate_mpcc"
    @NLobjective(m, Min, PEN)
else
    @NLobjective(m, Min, W_SSE * SSE + W_PEN * PEN + W_REG * REG)
end

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

# Parameter starts (log-space) using shifted prior Tstart
for p in 1:np
    set_start_value(teta[p], Tstart[p])
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
# Pre-optimization plotting
# - estimate: plot prior vs data (as before)
# - simulate_mpcc: plot ODE with nominal parameters vs data (kept as requested)
# ---------------------------------------------
if RUN_MODE == "estimate"
    let
        # Build prior param set: fixed params at Pnom, estimated params shifted by PRIOR_MUL
        prior_vals = Dict{Symbol,Float64}()
        for k in Pnames
            if k in EST_SYMS
                prior_vals[k] = Pnom[k] * get(PRIOR_MUL, k, 1.5)
            else
                prior_vals[k] = Pnom[k]
            end
        end
        pprior = ZentenoParams(prior_vals[:mu0], prior_vals[:betaG0], prior_vals[:betaF0], prior_vals[:Kn0], prior_vals[:Kg0], prior_vals[:Kf0], prior_vals[:Kig0], prior_vals[:Kie0], prior_vals[:Kd0],
                               prior_vals[:Yxn], prior_vals[:Yxg], prior_vals[:Yxf], prior_vals[:Yeg], prior_vals[:Yef])
        t_dense, U = simulate_zenteno(pprior)
        pre_path = joinpath(RESULTS_DIR, "fit_prior3_" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS") * ".png")
        plot_fit_overlay(t_dense, U, tgrid, data; title_str="Zenteno prior(3 vars) vs data (200h)", save_path=pre_path)
    end
elseif RUN_MODE == "simulate_mpcc"
    let
        pnom = ZentenoParams(Pnom[:mu0], Pnom[:betaG0], Pnom[:betaF0], Pnom[:Kn0], Pnom[:Kg0], Pnom[:Kf0], Pnom[:Kig0], Pnom[:Kie0], Pnom[:Kd0],
                             Pnom[:Yxn], Pnom[:Yxg], Pnom[:Yxf], Pnom[:Yeg], Pnom[:Yef])
        t_dense, U = simulate_zenteno(pnom)
        pre_path = joinpath(RESULTS_DIR, "ode_nominal_simulate_mpcc_" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS") * ".png")
        plot_fit_overlay(t_dense, U, tgrid, data; title_str="ODE (nominal) vs data (200h)", save_path=pre_path)
    end
end

# ---------------------------------------------
# Solve
# ---------------------------------------------
solve_t0 = time()
println("[INFO] Starting solve @ ", Dates.now())
optimize!(m)
status = termination_status(m)
pr_status = primal_status(m)
println("[INFO] Solver status: ", status, ", primal: ", pr_status)

# Report objective (SSE+penalty) value
try
    println("[INFO] Objective (SSE+pen): ", objective_value(m))
catch
end

# ---------------------------------------------
# Post-optimization plotting
# - estimate: simulate ODE with optimal parameters (as before)
# - simulate_mpcc: overlay ODE (nominal) with MPCC collocation states
# ---------------------------------------------
if RUN_MODE == "estimate"
    begin
        # Extract parameter values (exp of teta)
        pvals = Dict{Symbol,Float64}()
        for (i, k) in enumerate(Pnames)
            tv = try value(teta[i]) catch; T0[i] end
            pvals[k] = exp(tv)
        end
        popt = ZentenoParams(pvals[:mu0], pvals[:betaG0], pvals[:betaF0], pvals[:Kn0], pvals[:Kg0], pvals[:Kf0], pvals[:Kig0], pvals[:Kie0], pvals[:Kd0],
                             pvals[:Yxn], pvals[:Yxg], pvals[:Yxf], pvals[:Yeg], pvals[:Yef])
        t_dense2, U2 = simulate_zenteno(popt)
        post_path = joinpath(RESULTS_DIR, "fit_optimal_" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS") * ".png")
        plot_fit_overlay(t_dense2, U2, tgrid, data; title_str="Zenteno optimal vs data (200h)", save_path=post_path)
    end
elseif RUN_MODE == "simulate_mpcc"
    begin
        # Extract hv and collocation states from MPCC
        hv_vals = [try value(hv[i]) catch; hm[i] end for i in 1:nfe]
        tgrid_mpcc = build_time_grid_from_hv(hv_vals)
        Cmpcc = Array{Float64}(undef, nc, nfe, ncp)
        for l in 1:nc, i in 1:nfe, j in 1:ncp
            Cmpcc[l, i, j] = try value(c[l, i, j]) catch; NaN; end
        end
        # ODE with nominal
        pnom = ZentenoParams(Pnom[:mu0], Pnom[:betaG0], Pnom[:betaF0], Pnom[:Kn0], Pnom[:Kg0], Pnom[:Kf0], Pnom[:Kig0], Pnom[:Kie0], Pnom[:Kd0],
                             Pnom[:Yxn], Pnom[:Yxg], Pnom[:Yxf], Pnom[:Yeg], Pnom[:Yef])
        t_dense, U = simulate_zenteno(pnom)

        # Build overlay plot: ODE line + MPCC states (markers/lines at collocation points) + data
        plt = plot(layout=(5,1), size=(900, 1200))
        state_labels = ["X", "N", "G", "F", "E"]
        colors = [:blue, :green, :red, :orange, :purple]
        for s in 1:5
            # ODE continuous
            plot!(plt[s], t_dense, U[s, :]; color=colors[s], label="$(state_labels[s]) ODE", lw=2)
            # MPCC collocation (scatter + connecting lines per FE)
            ts = vec(tgrid_mpcc)
            ys = reshape(Cmpcc[s, :, :], nfe*ncp)
            scatter!(plt[s], ts, ys; color=:black, ms=3, alpha=0.7, label="MPCC")
            # Optional: also overlay data if available
            if size(data, 2) == nfe && size(data, 3) == ncp
                ds = reshape(data[s, :, :], nfe*ncp)
                scatter!(plt[s], vec(tgrid), ds; color=:gray, ms=3, alpha=0.5, label=(s==1 ? "data" : ""))
            end
            ylabel!(plt[s], state_labels[s])
            if s == 1
                title!(plt[s], "ODE (nominal) vs MPCC (penalty-only) vs data")
            end
        end
        xlabel!(plt[5], "time [h]")
        post_path = joinpath(RESULTS_DIR, "simulate_mpcc_overlay_" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS") * ".png")
        savefig(plt, post_path)
        println("[PLOT] Saved ", post_path)
    end
end

# ---------------------------------------------
# Complementary diagnostics & summary (robust)
# ---------------------------------------------
let
    solve_time = time() - solve_t0
    basic_summary_path = joinpath(RESULTS_DIR, "zenteno_relax_summary_" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS") * ".txt")
    # Always write a minimal summary first (so even if metrics fail we have status)
    open(basic_summary_path, "w") do io
        println(io, "status=", status)
        println(io, "primal_status=", pr_status)
        try println(io, "objective=", objective_value(m)) catch end
        println(io, "solve_time_s=", solve_time)
    end
    # Attempt extended metrics only for solved or nearly feasible cases
    do_metrics = status == LOCALLY_SOLVED || pr_status == FEASIBLE_POINT || pr_status == NEARLY_FEASIBLE_POINT
    if do_metrics
        # Safe accessor
    safe_val(x) = try value(x) catch; NaN; end
        sum_abs_FO_L = 0.0
        sum_abs_FO_U = 0.0
        sum_abs_FO_upt = 0.0
        cnt_FO_L_zero = 0
        cnt_FO_U_zero = 0
        cnt_FO_upt_zero = 0
        tol_fo = 1e-8
        for i in 1:nfe
            for k in 1:nv
                vFO = safe_val(FO_L[k,i]); sum_abs_FO_L += abs(vFO); cnt_FO_L_zero += (abs(vFO) <= tol_fo) ? 1 : 0
                vFOU = safe_val(FO_U[k,i]); sum_abs_FO_U += abs(vFOU); cnt_FO_U_zero += (abs(vFOU) <= tol_fo) ? 1 : 0
            end
            for u in 1:2
                vFOu = safe_val(FO_upt[u,i]); sum_abs_FO_upt += abs(vFOu); cnt_FO_upt_zero += (abs(vFOu) <= tol_fo) ? 1 : 0
            end
        end
        total_FO_terms = nfe*nv*2 + nfe*2
        frac_zero_FO = (cnt_FO_L_zero + cnt_FO_U_zero + cnt_FO_upt_zero) / total_FO_terms
        max_sc_violation = 0.0
        for i in 1:nfe, mc in 1:nm
            res = 0.0
            for k in 1:nv
                vk = safe_val(v[k,i]); res += S[mc,k]*vk
            end
            max_sc_violation = max(max_sc_violation, abs(res))
        end
        max_lagr_residual = 0.0
        for i in 1:nfe, k in 1:nv
            term_v = safe_val(v[k,i])
            term_aL = safe_val(alpha_L[k,i])
            term_aU = safe_val(alpha_U[k,i])
            term_upt = up_glu[k]*safe_val(alpha_upt[1,i]) + up_fru[k]*safe_val(alpha_upt[2,i])
            lam_sum = 0.0
            for r in 1:nm
                lam_sum += S[r,k]*safe_val(lambda_[r,i])
            end
            expr = d[k] + w*term_v*vs[k] + term_aL + term_aU + term_upt + lam_sum
            max_lagr_residual = max(max_lagr_residual, abs(expr))
        end
        avg_abs_FO = (sum_abs_FO_L + sum_abs_FO_U + sum_abs_FO_upt) / total_FO_terms
        println("[METRIC] sum|FO_L|=", sum_abs_FO_L)
        println("[METRIC] sum|FO_U|=", sum_abs_FO_U)
        println("[METRIC] sum|FO_upt|=", sum_abs_FO_upt)
        println("[METRIC] avg|FO|=", avg_abs_FO, ", frac_zero_FO=", frac_zero_FO)
        println("[METRIC] max_sc_violation=", max_sc_violation)
        println("[METRIC] max_lagr_residual=", max_lagr_residual)
        # Append metrics to existing summary file
        open(basic_summary_path, "a") do io
            println(io, "sum_abs_FO_L=", sum_abs_FO_L)
            println(io, "sum_abs_FO_U=", sum_abs_FO_U)
            println(io, "sum_abs_FO_upt=", sum_abs_FO_upt)
            println(io, "avg_abs_FO=", avg_abs_FO)
            println(io, "frac_zero_FO=", frac_zero_FO)
            println(io, "max_sc_violation=", max_sc_violation)
            println(io, "max_lagr_residual=", max_lagr_residual)
        end
    else
        println("[INFO] Metrics skipped (status/primal not solved/nearly feasible): status=", status, " primal=", pr_status)
    end
    println("[SAVE] ", basic_summary_path)
end
