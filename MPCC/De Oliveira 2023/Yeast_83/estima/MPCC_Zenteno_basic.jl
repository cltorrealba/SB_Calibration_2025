#!/usr/bin/env julia
# MPCC_Zenteno_basic.jl
#
# MPCC estilo original.jl, pero con el modelo macroscÃ³pico de Zenteno
# y datos sintÃ©ticos generados con el mismo modelo para asegurar convergencia.

using JuMP
using Ipopt
using LinearAlgebra
using DelimitedFiles
using DifferentialEquations
using Plots
using Dates
using Random
using Printf
const MOI = JuMP.MathOptInterface

# ---------------------------------------------
# Paths e IO
# ---------------------------------------------
const BASE_DIR   = @__DIR__
const ESTIMA_DIR = BASE_DIR
const PLOTS_DIR  = joinpath(ESTIMA_DIR, "plots")
isdir(PLOTS_DIR) || mkpath(PLOTS_DIR)

# Matriz estequiometrica y cotas de flujos
S     = readdlm(joinpath(ESTIMA_DIR, "S.csv"), ',')
lbraw = readdlm(joinpath(ESTIMA_DIR, "lb.csv"), ',')
ubraw = readdlm(joinpath(ESTIMA_DIR, "ub.csv"), ',')
lb    = lbraw isa AbstractVector ? copy(lbraw) : copy(lbraw[:,1])
ub    = ubraw isa AbstractVector ? copy(ubraw) : copy(ubraw[:,1])

# ---------------------------------------------
# Indices y tamanos del GEM
# ---------------------------------------------
nm = size(S, 1)
nv = size(S, 2)

const eth = 2630
const obj = 3414
const glu = 2588
const fru = 2583
const o2  = 2816
const ATP = 3415

if 1 <= o2 <= nv
    lb[o2] = 0.0
    ub[o2] = 0.0
end
if 1 <= ATP <= nv
    lb[ATP] = 0.0
end

# ---------------------------------------------
# Modelo Zenteno (param nominal)
# ---------------------------------------------
const nc = 5 # X,N,G,F,E
const NOISE_SEED = 1234

const MU0_nom   = 0.141665
const YXN_nom   = 9.80576
const YXG_nom   = 0.394345
const YXF_nom   = 0.18622
const YEG_nom   = 0.14133
const YEF_nom   = 0.96932
const Kn0_nom   = 0.226882
const Kg0_nom   = 3.1514
const Kf0_nom   = 2.97625
const Kig0_nom  = 29.5276
const Kie0_nom  = 2.99809
const Kd0_nom   = 0.0000311736
const betaG0_nom = 1.41182
const betaF0_nom = 8.49482

const R = 8.314
const T_const = try parse(Float64, get(ENV, "T_CONST", "293.15")) catch; 293.15 end
const eps = 1e-9

death_rate(E) = begin
    Td = -0.0001 * E^3 + 0.0049 * E^2 - 0.1279 * E + 315.89
    s = 0.5 * (1.0 + tanh(0.5 * (T_const - Td)))
    base = Kd0_nom * exp(0.0415 * E + (130000.0 * (T_const - 305.65)) / (305.65 * R * T_const))
    base * s
end

# Parametros estimables: mu0, Yeg, Yef (log)
const np = 3
const theta0 = log.([0.1, 0.1, 0.8])
LB = log.([0.5*MU0_nom, 0.5*YEG_nom, 0.5*YEF_nom])
UB = log.([2.0*MU0_nom, 2.0*YEG_nom, 2.0*YEF_nom])

# Condiciones iniciales
X0 = 0.5; N0 = 0.14; G0 = 110.0; F0 = 110.0; E0 = 0.0
const C0_INIT = [X0, N0, G0, F0, E0]
c0 = copy(C0_INIT)

# Discretizacion
nfe = 12
ncp = 3
th  = 48.0
h   = th / nfe
ph  = nfe
hm    = fill(h, nfe)'
var_h = 1.0

# Pesos MPCC
w    = 1e-20
omega = 1e2
phi1  = 1.0
phi2  = 1.0
phi3  = 1.0

d   = zeros(nv); d[obj] = -1.0
up  = zeros(nv); up[glu] = 1.0
up2 = zeros(nv); up2[fru] = 1.0
const n_up = 2

cs = ones(nc)
vs = ones(nv)

const MEAS_STATES = (3, 4, 5) # G, F, E
const NOISE_REL_STD = 0.10

colmat = [
    0.19681547722366   -0.06553542585020   0.02377097434822;
    0.39442431473909    0.29207341166523  -0.04154875212600;
    0.37640306270047    0.51248582618842   0.11111111111111
]
const radau_nodes = (0.15505, 0.64495, 1.0)
Random.seed!(NOISE_SEED)

# ---------------------------------------------
# Herramientas de simulacion/plot
# ---------------------------------------------
const STATE_LABELS = ("X", "N", "G", "F", "E")
const STATE_COLORS = (:royalblue, :forestgreen, :firebrick, :darkorange, :purple)

struct ZentenoPlotParams
    mu0::Float64
    Yeg::Float64
    Yef::Float64
end

function build_time_grid_from_lengths(lengths::AbstractVector{<:Real})
    tgrid = Array{Float64}(undef, length(lengths), ncp)
    acc = 0.0
    for i in 1:length(lengths)
        for (j, tau) in enumerate(radau_nodes)
            tgrid[i, j] = acc + tau * lengths[i]
        end
        acc += lengths[i]
    end
    return tgrid
end

const DATA_TIME_GRID = build_time_grid_from_lengths(fill(h, nfe))

function zenteno_ode!(du, u, p::ZentenoPlotParams, t)
    X, N, G, F, E = u
    mu_T =  exp(59453.0 * (T_const - 300.0) / (300.0 * R * T_const))
    Kg_T =  exp(46055.0 * (T_const - 293.15) / (293.15 * R * T_const))
    b_T  =  exp(11000.0 * (T_const - 296.15) / (296.15 * R * T_const))
    mrate = 0.01 * exp(37681.0 * (T_const - 293.30) / (293.30 * R * T_const))
    denom = G + F + eps
    phiG = G / denom
    phiF = F / denom
    mu   = p.mu0 * mu_T * (N / (N + Kn0_nom * Kg_T + eps))
    betaG = betaG0_nom * b_T *
        (G / (G + Kg0_nom * Kg_T + eps)) *
        (Kie0_nom * Kg_T / (E + Kie0_nom * Kg_T + eps))
    betaF = betaF0_nom * b_T *
        (F / (F + Kf0_nom * Kg_T + eps)) *
        (Kig0_nom * Kg_T / (G + Kig0_nom * Kg_T + eps)) *
        (Kie0_nom * Kg_T / (E + Kie0_nom * Kg_T + eps))
    Td = -0.0001 * E^3 + 0.0049 * E^2 - 0.1279 * E + 315.89
    s_sw = 0.5 * (1.0 + tanh(0.5 * (T_const - Td)))
    Kd_val = Kd0_nom * exp(0.0415 * E + (130000.0 * (T_const - 305.65)) / (305.65 * R * T_const)) * s_sw
    du[1] = (mu - Kd_val) * X
    du[2] = -(mu / YXN_nom) * X
    du[3] = -((mu / YXG_nom) + (betaG / p.Yeg) + mrate * phiG) * X
    du[4] = -((mu / YXF_nom) + (betaF / p.Yef) + mrate * phiF) * X
    du[5] = (betaG + betaF) * X
    return nothing
end

function simulate_zenteno(params::ZentenoPlotParams; tspan=(0.0, th))
    u0 = copy(C0_INIT)
    prob = ODEProblem(zenteno_ode!, u0, tspan, params)
    sol = solve(prob, Rodas5(); reltol=1e-8, abstol=1e-10, saveat=0.1)
    t_dense = collect(range(tspan[1], tspan[2]; length=2001))
    U = reduce(hcat, (sol(t) for t in t_dense))
    return t_dense, U
end

function plot_post_solution(t_pre, states_pre, t_post, states_post, tgrid_data, data_vals,
        mpcc_tgrid, mpcc_states; title_str::AbstractString, save_path::AbstractString)
    plt = plot(layout=(nc, 1), size=(900, 1200))
    ts_data = vec(tgrid_data)
    for s in 1:nc
        if t_pre !== nothing && states_pre !== nothing
            plot!(plt[s], t_pre, states_pre[s, :];
                color=STATE_COLORS[s], lw=2, linestyle=:dashdot, label=(s == 1 ? "ODE pre" : nothing))
        end
        plot!(plt[s], t_post, states_post[s, :];
            color=STATE_COLORS[s], lw=3, label=(s == 1 ? "ODE post" : nothing))
        if s in MEAS_STATES
            ys_data = reshape(data_vals[s, :, :], length(ts_data))
            scatter!(plt[s], ts_data, ys_data;
                color=:black, ms=4, alpha=0.8, label=(s == MEAS_STATES[1] ? "Datos exp." : nothing))
        end
        if mpcc_tgrid !== nothing && mpcc_states !== nothing
            ts_mpcc = vec(mpcc_tgrid)
            ys_mpcc = reshape(mpcc_states[s, :, :], length(ts_mpcc))
            scatter!(plt[s], ts_mpcc, ys_mpcc;
                color=:purple, ms=5, alpha=0.9, marker=:diamond, label=(s == 1 ? "MPCC" : nothing))
        end
        ylabel!(plt[s], STATE_LABELS[s])
        if s == 1
            title!(plt[s], title_str)
        end
    end
    xlabel!(plt[nc], "tiempo [h]")
    savefig(plt, save_path)
    println("[PLOT] Guardado ", save_path)
end

sanitize_token(str::AbstractString) = replace(str, r"[^0-9A-Za-z]+" => "_")

function short_token(str::AbstractString; maxlen::Int=10)
    clean = sanitize_token(str)
    return clean[1:min(length(clean), maxlen)]
end

function format_float_token(val::Float64; digits::Int=1)
    replace(@sprintf("%.*f", digits, val), "." => "p")
end

function result_file_prefix(; wall_time::Float64, nfe::Int, status, primal_status)
    wall_tok = "w$(format_float_token(wall_time))s"
    feas_tok = "f$(short_token(string(primal_status); maxlen=6))"
    term_tok = "t$(short_token(string(status); maxlen=6))"
    stamp = Dates.format(Dates.now(), "yyyymmdd_HHmmss")
    return joinpath(PLOTS_DIR, "MPCCpost_$(wall_tok)_nfe$(nfe)_$(feas_tok)_$(term_tok)_$(stamp)")
end

function write_diagnostic_report(path_prefix::AbstractString; wall_time::Float64, status, primal_status,
        objective::Float64, dual_inf, primal_inf, compl, constr_viol, iter_count, fo_value::Float64)
    report_path = path_prefix * ".txt"
    open(report_path, "w") do io
        println(io, "timestamp=", Dates.now())
        println(io, "termination_status=", status)
        println(io, "primal_status=", primal_status)
        println(io, @sprintf("wall_time_s=%.4f", wall_time))
        println(io, "objective=", objective)
        println(io, "FO_value=", fo_value)
        println(io, "iter_count=", iter_count)
        println(io, "dual_infeasibility=", dual_inf)
        println(io, "primal_infeasibility=", primal_inf)
        println(io, "constraint_violation=", constr_viol)
        println(io, "complementarity=", compl)
    end
    println("[REPORT] Guardado ", report_path)
end

function optimizer_attr(m, attr::AbstractString, default=NaN)
    try
        return get_optimizer_attribute(m, attr)
    catch
        return default
    end
end

safe_value(x, default=NaN) = try
    value(x)
catch
    default
end

maxabs(arr::AbstractArray) = isempty(arr) ? 0.0 : maximum(abs, arr)

function collect_ipopt_stats(model::Model)
    backend = JuMP.backend(model)
    optimizer = getfield(backend, :optimizer)
    inner = optimizer.inner
    inner isa Ipopt.IpoptProblem || return nothing
    nvar = inner.n
    ncon = inner.m
    x_L_violation = zeros(Float64, nvar)
    x_U_violation = zeros(Float64, nvar)
    compl_x_L = zeros(Float64, nvar)
    compl_x_U = zeros(Float64, nvar)
    grad_lag_x = zeros(Float64, nvar)
    constr_violation = zeros(Float64, ncon)
    compl_g = zeros(Float64, ncon)
    try
        Ipopt.GetIpoptCurrentViolations(
            inner,
            false,
            nvar,
            x_L_violation,
            x_U_violation,
            compl_x_L,
            compl_x_U,
            grad_lag_x,
            ncon,
            constr_violation,
            compl_g,
        )
    catch err
        @warn "No se pudo obtener violaciones de Ipopt" err
        return nothing
    end
    dual_inf = maxabs(grad_lag_x)
    primal_inf = max(maxabs(x_L_violation), maxabs(x_U_violation))
    constraint_violation = maxabs(constr_violation)
    complementarity = maximum((
        maxabs(compl_x_L),
        maxabs(compl_x_U),
        maxabs(compl_g),
    ))
    iter_count = try
        MOI.get(backend, MOI.BarrierIterations())
    catch
        NaN
    end
    return (
        dual_inf=dual_inf,
        primal_inf=primal_inf,
        constraint_violation=constraint_violation,
        complementarity=complementarity,
        iter_count=iter_count,
    )
end

# ---------------------------------------------
# Datos sinteticos (modelo Zenteno nominal)
# ---------------------------------------------
function _simulate_zenteno_synthetic(; nfe::Int, ncp::Int, th::Float64, c0_vec::Vector{Float64})
    nc = length(c0_vec)
    nsteps = nfe * ncp * 5
    dt = th / (nsteps - 1)
    X = zeros(nc, nsteps)
    X[:, 1] .= c0_vec

    mu_T_val  = exp(59453.0 * (T_const - 300.0) / (300.0 * R * T_const))
    Kg_T_val  = exp(46055.0 * (T_const - 293.15) / (293.15 * R * T_const))
    b_T_val   = exp(11000.0 * (T_const - 296.15) / (296.15 * R * T_const))
    mrate_val = 0.01 * exp(37681.0 * (T_const - 293.30) / (293.30 * R * T_const))

    for s in 1:(nsteps-1)
        x = X[1, s]; n = X[2, s]; g = X[3, s]; f = X[4, s]; e = X[5, s]

        mu_val = MU0_nom * mu_T_val * (n / (n + Kn0_nom * Kg_T_val + eps))
        betaG_val = betaG0_nom * b_T_val *
            (g / (g + Kg0_nom * Kg_T_val + eps)) *
            (Kie0_nom * Kg_T_val / (e + Kie0_nom * Kg_T_val + eps))
        betaF_val = betaF0_nom * b_T_val *
            (f / (f + Kf0_nom * Kg_T_val + eps)) *
            (Kig0_nom * Kg_T_val / (g + Kig0_nom * Kg_T_val + eps)) *
            (Kie0_nom * Kg_T_val / (e + Kie0_nom * Kg_T_val + eps))

        Td = -0.0001 * e^3 + 0.0049 * e^2 - 0.1279 * e + 315.89
        s_sw = 0.5 * (1 + tanh(0.5 * (T_const - Td)))
        Kd_val = Kd0_nom * exp(0.0415 * e + (130000.0 * (T_const - 305.65)) / (305.65 * R * T_const)) * s_sw

        phiG = g / (g + f + eps)
        phiF = f / (g + f + eps)

        dX = (mu_val - Kd_val) * x
        dN = -(mu_val / YXN_nom) * x
        dG = -((mu_val / YXG_nom) + (betaG_val / YEG_nom) + mrate_val * phiG) * x
        dF = -((mu_val / YXF_nom) + (betaF_val / YEF_nom) + mrate_val * phiF) * x
        dE = (betaG_val + betaF_val) * x

        X[1, s+1] = max(0.0, x + dt * dX)
        X[2, s+1] = max(0.0, n + dt * dN)
        X[3, s+1] = max(0.0, g + dt * dG)
        X[4, s+1] = max(0.0, f + dt * dF)
        X[5, s+1] = max(0.0, e + dt * dE)
    end

    data = zeros(nc, nfe, ncp)
    for i in 1:nfe, (j_idx, tau) in enumerate(radau_nodes)
        t_ij = (i - 1 + tau) * h
        idx  = clamp(round(Int, t_ij / dt) + 1, 1, nsteps)
        @inbounds data[:, i, j_idx] .= X[:, idx]
    end
    if NOISE_REL_STD > 0
        noise = NOISE_REL_STD .* data .* randn(size(data))
        data .+= noise
        @. data = max(data, 0.0)
    end
    return data
end

println("[SYNTH] Generando datos sinteticos con modelo Zenteno basico")
data = _simulate_zenteno_synthetic(nfe=nfe, ncp=ncp, th=th, c0_vec=c0)

t_pre = nothing
states_pre = nothing
try
    pre_params = ZentenoPlotParams(exp(theta0[1]), exp(theta0[2]), exp(theta0[3]))
    t_pre_sim, states_pre_sim = simulate_zenteno(pre_params)
    t_pre = t_pre_sim
    states_pre = states_pre_sim
catch err
    @warn "No se pudo simular la ODE previa a la optimizacion" err
end

# ---------------------------------------------
# Modelo JuMP
# ---------------------------------------------
m = Model(Ipopt.Optimizer)
set_optimizer_attribute(m, "warm_start_init_point", "yes")
set_optimizer_attribute(m, "print_level", 5)
set_optimizer_attribute(m, "tol", 1e-4)
set_optimizer_attribute(m, "acceptable_iter", 5)
set_optimizer_attribute(m, "acceptable_tol", 1e-2)
set_optimizer_attribute(m, "linear_solver", "mumps")

if haskey(ENV, "J_IPOPT_MAX_ITER")
    try
        maxit = parse(Int, ENV["J_IPOPT_MAX_ITER"])
        set_optimizer_attribute(m, "max_iter", maxit)
        @info "Ipopt max_iter set from ENV[J_IPOPT_MAX_ITER]" maxit
    catch err
        @warn "Failed to parse ENV[J_IPOPT_MAX_ITER]; ignoring" error=err
    end
end

@variables(m, begin
    c[1:nc, 1:ph, 1:ncp]
    cdot[1:nc, 1:ph, 1:ncp]
    FO
    teta[1:np]
    v[1:nv, 1:nfe]
    lambda_[1:nm, 1:nfe]
    alpha_U[1:nv, 1:nfe]
    alpha_L[1:nv, 1:nfe]
    alpha_upt[1:n_up, 1:nfe]
    FO_U[1:nv, 1:nfe]
    FO_L[1:nv, 1:nfe]
    FO_upt[1:n_up, 1:nfe]
    hv[1:nfe]
end)

for k in 1:np
    set_start_value(teta[k], theta0[k])
end

for i in 1:ph, j in 1:ncp, l in 1:nc
    set_start_value(c[l, i, j], c0[l])
    set_start_value(cdot[l, i, j], 0.0)
end
for i in 1:nfe
    set_start_value(hv[i], hm[i])
end

for i in 1:nc
    c0[i] = c0[i] / cs[i]
end

@NLobjective(m, Min,
    omega * FO +
    sum(
        sum(-phi1 * FO_L[mc, i] - phi3 * FO_U[mc, i] for mc in 1:nv) +
        phi2 * FO_upt[1, i] + phi2 * FO_upt[2, i]
        for i in 1:nfe
    )
)

JuMP.register(m, :death_rate, 1, death_rate; autodiff = true)
@NLexpression(m, mu_T, exp(59453.0 * (T_const - 300.0) / (300.0 * R * T_const)))
@NLexpression(m, Kg_T, exp(46055.0 * (T_const - 293.15) / (293.15 * R * T_const)))
@NLexpression(m, b_T,  exp(11000.0 * (T_const - 296.15) / (296.15 * R * T_const)))
@NLexpression(m, mrate, 0.01 * exp(37681.0 * (T_const - 293.30) / (293.30 * R * T_const)))

@NLexpression(m, mu0, exp(teta[1]))
@NLexpression(m, Yeg, exp(teta[2]))
@NLexpression(m, Yef, exp(teta[3]))

const Yxn = YXN_nom
const Yxg = YXG_nom
const Yxf = YXF_nom

@NLexpression(m, mu_j[i=1:ph, j=1:ncp],
    mu0 * mu_T * (c[2,i,j] / (c[2,i,j] + Kn0_nom * Kg_T + eps))
)
@NLexpression(m, betaG_j[i=1:ph, j=1:ncp],
    betaG0_nom * b_T *
    (c[3,i,j] / (c[3,i,j] + Kg0_nom * Kg_T + eps)) *
    (Kie0_nom * Kg_T / (c[5,i,j] + Kie0_nom * Kg_T + eps))
)
@NLexpression(m, betaF_j[i=1:ph, j=1:ncp],
    betaF0_nom * b_T *
    (c[4,i,j] / (c[4,i,j] + Kf0_nom * Kg_T + eps)) *
    (Kig0_nom * Kg_T / (c[3,i,j] + Kig0_nom * Kg_T + eps)) *
    (Kie0_nom * Kg_T / (c[5,i,j] + Kie0_nom * Kg_T + eps))
)
@NLexpression(m, phiG_j[i=1:ph, j=1:ncp], c[3,i,j] / (c[3,i,j] + c[4,i,j] + eps))
@NLexpression(m, phiF_j[i=1:ph, j=1:ncp], c[4,i,j] / (c[3,i,j] + c[4,i,j] + eps))
@NLexpression(m, Kd_j[i=1:ph, j=1:ncp], death_rate(c[5,i,j]))

@constraints(m, begin
    coll_c_n[l=1:nc, i=2:ph, j=1:ncp],
        c[l,i,j] == c[l,i-1,ncp] + hv[i] * sum(colmat[j,k] * cdot[l,i,k] for k in 1:ncp)
    coll_c_0[l=1:nc, j=1:ncp],
        c[l,1,j] == c0[l] + hv[1] * sum(colmat[j,k] * cdot[l,1,k] for k in 1:ncp)

    teta_LB[p=1:np], teta[p] >= LB[p]
    teta_UB[p=1:np], teta[p] <= UB[p]

    Sc[mc=1:nm, i=1:nfe],  sum(S[mc,k] * v[k,i] * vs[k] for k in 1:nv) == 0
    v_UB[mc=1:nv, i=1:nfe], v[mc,i]*vs[mc] - ub[mc] <= 0
    v_LB[mc=1:nv, i=1:nfe], -v[mc,i]*vs[mc] + lb[mc] <= 0

    c_LB[l=1:nc, i=1:nfe, j=1:ncp], -c[l,i,j] <= 0

    MFE1, sum(hv[i] for i in 1:nfe) == th
    MFE3[i=1:nfe], hv[i] >= 0.0
    MFE4[i=1:nfe], hv[i] >= (1.0 - var_h) * hm[1]
    MFE5[i=1:nfe], hv[i] <= (1.0 + var_h) * hm[1]

    Lagr[mc=1:nv, i=1:nfe],
        d[mc] + w * v[mc,i] * vs[mc] + alpha_L[mc,i] + alpha_U[mc,i] +
        up[mc] * alpha_upt[1,i] + up2[mc] * alpha_upt[2,i] +
        sum(S[k,mc] * lambda_[k,i] for k in 1:nm) == 0

    alpha1_LB[mc=1:nv, i=1:nfe], alpha_L[mc,i] <= 0
    alpha4_LB[mc=1:n_up, i=1:nfe], alpha_upt[mc,i] <= 0
    alpha1_UB[mc=1:nv, i=1:nfe], alpha_U[mc,i] >= 0
end)

@NLconstraints(m, begin
    dX[i=1:ph, j=1:ncp], cdot[1,i,j] == (mu_j[i,j] - Kd_j[i,j]) * c[1,i,j]
    dN[i=1:ph, j=1:ncp], cdot[2,i,j] == -(mu_j[i,j] / Yxn) * c[1,i,j]
    dG[i=1:ph, j=1:ncp], cdot[3,i,j] == -((mu_j[i,j] / Yxg) + (betaG_j[i,j] / Yeg) + mrate * phiG_j[i,j]) * c[1,i,j]
    dF[i=1:ph, j=1:ncp], cdot[4,i,j] == -((mu_j[i,j] / Yxf) + (betaF_j[i,j] / Yef) + mrate * phiF_j[i,j]) * c[1,i,j]
    dE[i=1:ph, j=1:ncp], cdot[5,i,j] == (betaG_j[i,j] + betaF_j[i,j]) * c[1,i,j]

    FO1[mc=1:nv, i=1:nfe], FO_L[mc,i] == (v[mc,i]*vs[mc] - lb[mc]) * alpha_L[mc,i]
    FO2[mc=1:nv, i=1:nfe], FO_U[mc,i] == (v[mc,i]*vs[mc] - ub[mc]) * alpha_U[mc,i]

    FO3_upt[i=1:nfe], FO_upt[1,i] == (-v[glu,i]*vs[glu]) * alpha_upt[1,i]
    FO4_upt[i=1:nfe], FO_upt[2,i] == (-v[fru,i]*vs[fru]) * alpha_upt[2,i]

    FO_def,
        FO == sum((data[l,i,j] - c[l,i,j])^2 for l in MEAS_STATES, i in 1:ph, j in 1:ncp)
end)

# ---------------------------------------------
# Resolver
# ---------------------------------------------
t_start = time()
optimize!(m)
t_end = time()

wall_time = t_end - t_start
status = termination_status(m)
pr_status = primal_status(m)

println("Solver status = ", status)
println("Primal status = ", pr_status)
println("Objective FO   = ", safe_value(FO))
println("Wall time (s)  = ", wall_time)

objective_val = try
    objective_value(m)
catch
    NaN
end
fo_val = safe_value(FO, NaN)
dual_inf = optimizer_attr(m, "dual infeasibility")
primal_inf = optimizer_attr(m, "primal infeasibility")
compl = optimizer_attr(m, "complementarity")
constr_viol = optimizer_attr(m, "constraint violation")
iter_count = optimizer_attr(m, "iter_count")
if (stats = collect_ipopt_stats(m)) !== nothing
    dual_inf = stats.dual_inf
    primal_inf = stats.primal_inf
    compl = stats.complementarity
    constr_viol = stats.constraint_violation
    iter_count = stats.iter_count
end

result_prefix = result_file_prefix(wall_time=wall_time, nfe=nfe, status=status, primal_status=pr_status)
plot_output_path = result_prefix * ".png"

try
    mu_log  = safe_value(teta[1], theta0[1])
    yeg_log = safe_value(teta[2], theta0[2])
    yef_log = safe_value(teta[3], theta0[3])
    post_params = ZentenoPlotParams(exp(mu_log), exp(yeg_log), exp(yef_log))
    hv_vals = [safe_value(hv[i], hm[i]) for i in 1:nfe]
    mpcc_tgrid = build_time_grid_from_lengths(hv_vals)
    mpcc_states = Array{Float64}(undef, nc, nfe, ncp)
    for l in 1:nc, i in 1:nfe, j in 1:ncp
        mpcc_states[l, i, j] = safe_value(c[l, i, j])
    end
    t_post, states_post = simulate_zenteno(post_params)
    plot_post_solution(
        t_pre, states_pre, t_post, states_post, DATA_TIME_GRID, data,
        mpcc_tgrid, mpcc_states;
        title_str="MPCC post: $(status) / $(pr_status)",
        save_path=plot_output_path,
    )
catch err
    @warn "No se pudo generar el grafico posterior a la optimizacion" err
end

write_diagnostic_report(
    result_prefix;
    wall_time=wall_time,
    status=status,
    primal_status=pr_status,
    objective=objective_val,
    dual_inf=dual_inf,
    primal_inf=primal_inf,
    compl=compl,
    constr_viol=constr_viol,
    iter_count=iter_count,
    fo_value=fo_val,
)
