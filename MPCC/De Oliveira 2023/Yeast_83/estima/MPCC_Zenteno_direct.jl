#!/usr/bin/env julia
# MPCC_Zenteno_direct.jl
#
# Implementación "direct approach" (enfoque secuencial) para el
# mismo problema Zenteno resuelto por MPCC_Zenteno_basic.jl.
# En lugar de optimizar simultáneamente estados, flujos y parámetros,
# aquí integramos el modelo macroscópico paso a paso y ejecutamos
# un esquema de búsqueda directa sobre los parámetros estimables.

using LinearAlgebra
using DelimitedFiles
using DifferentialEquations
using Random
using Printf
using Dates
using Plots
using JuMP
using Ipopt
using MathOptInterface
const MOI = MathOptInterface

# ---------------------------------------------
# Paths e IO
# ---------------------------------------------
const BASE_DIR   = @__DIR__
const ESTIMA_DIR = BASE_DIR
const PLOTS_DIR  = joinpath(ESTIMA_DIR, "plots")
isdir(PLOTS_DIR) || mkpath(PLOTS_DIR)

_sanitize_experiment_name(str::AbstractString) = replace(strip(str), r"[^0-9A-Za-z._-]+" => "_")
const EXPERIMENT_TOKEN = _sanitize_experiment_name(get(ENV, "EXPERIMENT", "direct"))
const EXPERIMENT_DIR = joinpath(PLOTS_DIR, EXPERIMENT_TOKEN * "_direct")
isdir(EXPERIMENT_DIR) || mkpath(EXPERIMENT_DIR)

# ---------------------------------------------
# Matriz estequiométrica y cotas de flujos
# (sólo se usan para comparar resultados y
# eventualmente reproducir FBA si fuese necesario)
# ---------------------------------------------
S     = readdlm(joinpath(ESTIMA_DIR, "S.csv"), ',')
lbraw = readdlm(joinpath(ESTIMA_DIR, "lb.csv"), ',')
ubraw = readdlm(joinpath(ESTIMA_DIR, "ub.csv"), ',')
lb    = lbraw isa AbstractVector ? copy(lbraw) : copy(lbraw[:,1])
ub    = ubraw isa AbstractVector ? copy(ubraw) : copy(ubraw[:,1])

const nm = size(S, 1)
const nv = size(S, 2)

const glu = 2588
const fru = 2583
const obj = 3414
const o2  = 2816
const ATP = 3415

if 1 <= o2 <= nv
    lb[o2] = 0.0
    ub[o2] = 0.0
end
if 1 <= ATP <= nv
    lb[ATP] = 0.0
end

const BASE_LB_VEC = copy(lb)
const BASE_UB_VEC = copy(ub)

# Ensure biomass reaction is nonnegative for the direct dFBA solve
if 1 <= obj <= nv
    BASE_LB_VEC[obj] = max(BASE_LB_VEC[obj], 0.0)
    lb[obj] = BASE_LB_VEC[obj]
end

const BASE_OBJ_LB = BASE_LB_VEC[obj]
const BASE_OBJ_UB = BASE_UB_VEC[obj]
const BASE_GLU_LB = BASE_LB_VEC[glu]
const BASE_GLU_UB = BASE_UB_VEC[glu]
const BASE_FRU_LB = BASE_LB_VEC[fru]
const BASE_FRU_UB = BASE_UB_VEC[fru]

# ---------------------------------------------
# Modelo Zenteno
# ---------------------------------------------
const nc = 5 # X,N,G,F,E
const NOISE_SEED = 1234
const NOISE_REL_STD = 0.10
const R = 8.314
const T_const = try parse(Float64, get(ENV, "T_CONST", "293.15")) catch; 293.15 end
const eps = 1e-9

const MU_T_VAL  = exp(59453.0 * (T_const - 300.0) / (300.0 * R * T_const))
const KG_T_VAL  = exp(46055.0 * (T_const - 293.15) / (293.15 * R * T_const))
const B_T_VAL   = exp(11000.0 * (T_const - 296.15) / (296.15 * R * T_const))
const MRATE_VAL = 0.01 * exp(37681.0 * (T_const - 293.30) / (293.30 * R * T_const))

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

# Parámetros estimables (log)
const np = 3
const theta0 = log.([0.1, 0.1, 0.8])
const LB = log.([0.5*MU0_nom, 0.5*YEG_nom, 0.5*YEF_nom])
const UB = log.([2.0*MU0_nom, 2.0*YEG_nom, 2.0*YEF_nom])

# Condiciones iniciales
const X0 = 0.5
const N0 = 0.14
const G0 = 110.0
const F0 = 110.0
const E0 = 0.0
const C0_INIT = [X0, N0, G0, F0, E0]

# Discretización temporal (igual al MPCC)
const nfe = 12
const ncp = 3
const th  = 72.0
const h   = th / nfe
const HM_REFERENCE = fill(h, nfe)

const radau_nodes = (0.15505, 0.64495, 1.0)

# Estados observados y utilidades para plotting
const MEAS_STATES = (3, 4, 5) # G, F, E
const STATE_LABELS = ("X", "N", "G", "F", "E")
const STATE_COLORS = (:royalblue, :forestgreen, :firebrick, :darkorange, :purple)
const STATE_MIN_CONC = (
    1e-6,
    1e-5,
    1e-4,
    1e-4,
    1e-6,
)

# ---------------------------------------------
# Herramientas de simulación y utilidades
# ---------------------------------------------
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

const DATA_TIME_GRID = build_time_grid_from_lengths(HM_REFERENCE)

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
    prob = ODEProblem(zenteno_ode!, copy(C0_INIT), tspan, params)
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
            color=STATE_COLORS[s], lw=3, label=(s == 1 ? "ODE ajuste" : nothing))
        if s in MEAS_STATES
            ys_data = reshape(data_vals[s, :, :], length(ts_data))
            scatter!(plt[s], ts_data, ys_data;
                color=:black, ms=4, alpha=0.8, label=(s == MEAS_STATES[1] ? "Datos exp." : nothing))
        end
        if mpcc_tgrid !== nothing && mpcc_states !== nothing
            ts_mpcc = vec(mpcc_tgrid)
            ys_mpcc = reshape(mpcc_states[s, :, :], length(ts_mpcc))
            scatter!(plt[s], ts_mpcc, ys_mpcc;
                color=:purple, ms=5, alpha=0.9, marker=:diamond, label=(s == 1 ? "Direct approach" : nothing))
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

function result_file_prefix(; tag::AbstractString, wall_time::Float64, objective::Float64)
    wall_tok = @sprintf("w%.1fs", wall_time)
    obj_tok = @sprintf("obj%.3e", objective)
    stamp = Dates.format(Dates.now(), "yyyymmdd_HHmmss")
    return joinpath(EXPERIMENT_DIR, "$(tag)_$(wall_tok)_$(obj_tok)_$(stamp)")
end

function write_direct_report(path_prefix::AbstractString; wall_time::Float64,
        best_theta::Vector{Float64}, best_sse::Float64, evals::Int, status::AbstractString,
        plot_path::AbstractString, history_path::Union{Nothing,AbstractString})
    report_path = path_prefix * ".txt"
    open(report_path, "w") do io
        println(io, "timestamp=", Dates.now())
        println(io, "status=", status)
        println(io, @sprintf("wall_time_s=%.4f", wall_time))
        println(io, "eval_count=", evals)
        println(io, "best_theta_log=", join(best_theta, ", "))
        println(io, "best_theta_lin=", join(exp.(best_theta), ", "))
        println(io, "objective=", best_sse)
        println(io, "plot_path=", plot_path)
        history_path === nothing || println(io, "history_path=", history_path)
    end
    println("[REPORT] Guardado ", report_path)
end

function write_history_csv(path_prefix::AbstractString, history)
    isempty(history) && return nothing
    csv_path = path_prefix * "_history.csv"
    open(csv_path, "w") do io
        println(io, "generation,best_objective")
        for entry in history
            gen = get(entry, :generation, missing)
            best_val = get(entry, :best, missing)
            println(io, "$(gen),$(best_val)")
        end
    end
    println("[HISTORY] Guardado ", csv_path)
    return csv_path
end

# ---------------------------------------------
# Herramientas dFBA directo
# ---------------------------------------------
function build_fba_model()
    m = Model(Ipopt.Optimizer)
    set_silent(m)
    set_optimizer_attribute(m, "print_level", 0)
    set_optimizer_attribute(m, "tol", 1e-8)
    @variable(m, v[1:nv])
    for k in 1:nv
        set_lower_bound(v[k], BASE_LB_VEC[k])
        set_upper_bound(v[k], BASE_UB_VEC[k])
    end
    @constraint(m, [mc=1:nm], sum(S[mc, k] * v[k] for k in 1:nv) == 0)
    @objective(m, Max, v[obj])
    return m, v
end

function reset_flux_bounds!(v)
    set_lower_bound(v[obj], BASE_OBJ_LB)
    set_upper_bound(v[obj], BASE_OBJ_UB)
    set_lower_bound(v[glu], BASE_GLU_LB)
    set_upper_bound(v[glu], BASE_GLU_UB)
    set_lower_bound(v[fru], BASE_FRU_LB)
    set_upper_bound(v[fru], BASE_FRU_UB)
end

function apply_biomass_cap!(v, cap::Float64)
    cap_val = max(1e-8, cap)
    set_lower_bound(v[obj], 0.0)
    set_upper_bound(v[obj], cap_val)
end

function compute_zenteno_rates(u::Vector{Float64}, theta_lin::Vector{Float64})
    X, N, G, F, E = u
    mu_cap = theta_lin[1] * MU_T_VAL * (N / (N + Kn0_nom * KG_T_VAL + eps))
    denom = G + F + eps
    phiG = G / denom
    phiF = F / denom
    betaG = betaG0_nom * B_T_VAL *
        (G / (G + Kg0_nom * KG_T_VAL + eps)) *
        (Kie0_nom * KG_T_VAL / (E + Kie0_nom * KG_T_VAL + eps))
    betaF = betaF0_nom * B_T_VAL *
        (F / (F + Kf0_nom * KG_T_VAL + eps)) *
        (Kig0_nom * KG_T_VAL / (G + Kig0_nom * KG_T_VAL + eps)) *
        (Kie0_nom * KG_T_VAL / (E + Kie0_nom * KG_T_VAL + eps))
    Td = -0.0001 * E^3 + 0.0049 * E^2 - 0.1279 * E + 315.89
    s_sw = 0.5 * (1.0 + tanh(0.5 * (T_const - Td)))
    Kd_val = Kd0_nom * exp(0.0415 * E + (130000.0 * (T_const - 305.65)) / (305.65 * R * T_const)) * s_sw
    return mu_cap, betaG, betaF, phiG, phiF, Kd_val
end

function advance_state!(u::Vector{Float64}, theta_lin::Vector{Float64},
        model::Model, v, dt::Float64)
    mu_cap, betaG, betaF, phiG, phiF, Kd = compute_zenteno_rates(u, theta_lin)
    apply_biomass_cap!(v, mu_cap)
    optimize!(model)
    status = termination_status(model)
    mu = if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
        val = value(v[obj])
        isfinite(val) ? val : mu_cap
    else
        mu_cap
    end
    mu = clamp(mu, 0.0, max(mu_cap, 1e-8))
    X = u[1]
    Yeg = theta_lin[2]
    Yef = theta_lin[3]
    dX = (mu - Kd) * X
    dN = -(mu / YXN_nom) * X
    dG = -((mu / YXG_nom) + (betaG / Yeg) + MRATE_VAL * phiG) * X
    dF = -((mu / YXF_nom) + (betaF / Yef) + MRATE_VAL * phiF) * X
    dE = (betaG + betaF) * X
    u[1] = max(STATE_MIN_CONC[1], X + dt * dX)
    u[2] = max(STATE_MIN_CONC[2], u[2] + dt * dN)
    u[3] = max(STATE_MIN_CONC[3], u[3] + dt * dG)
    u[4] = max(STATE_MIN_CONC[4], u[4] + dt * dF)
    u[5] = max(STATE_MIN_CONC[5], u[5] + dt * dE)
    return nothing
end

function integrate_interval!(u::Vector{Float64}, theta_lin::Vector{Float64},
        model::Model, v, Δt::Float64; max_step::Float64=0.25)
    remaining = Δt
    while remaining > 1e-9
        step = min(max_step, remaining)
        advance_state!(u, theta_lin, model, v, step)
        remaining -= step
    end
    return u
end

# ---------------------------------------------
# Datos sintéticos (idénticos al MPCC base)
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
    if NOISE_SEED >= 0
        Random.seed!(NOISE_SEED)
    end
    noise = NOISE_REL_STD .* data .* randn(size(data))
    data .+= noise
    @. data = max(data, 0.0)
    return data
end

# ---------------------------------------------
# Evaluación secuencial en la malla MPCC
# ---------------------------------------------
function sample_states_on_grid(theta_log::AbstractVector{<:Real};
        h_lengths::AbstractVector{<:Real}=HM_REFERENCE)
    theta_lin = exp.(theta_log)
    model, v = build_fba_model()
    reset_flux_bounds!(v)
    nfe_local = length(h_lengths)
    states = Array{Float64}(undef, nc, nfe_local, ncp)
    u = copy(C0_INIT)
    current_time = 0.0
    for (i, hi) in enumerate(h_lengths)
        element_start = current_time
        for (j, tau) in enumerate(radau_nodes)
            target_time = element_start + tau * hi
            Δt = target_time - current_time
            if Δt > 0
                integrate_interval!(u, theta_lin, model, v, Δt)
            end
            current_time = target_time
            @inbounds states[:, i, j] .= u
        end
        element_end = element_start + hi
        if current_time < element_end - 1e-9
            integrate_interval!(u, theta_lin, model, v, element_end - current_time)
            current_time = element_end
        end
    end
    return states
end

mutable struct DirectObjective
    data::Array{Float64,3}
    cache::Dict{NTuple{3,Float64}, Float64}
    eval_count::Int
end

DirectObjective(data::Array{Float64,3}) = DirectObjective(data, Dict{NTuple{3,Float64}, Float64}(), 0)

function evaluate!(obj::DirectObjective, theta::AbstractVector{<:Real})
    if length(theta) != np
        error("theta dimension mismatch")
    end
    for k in 1:np
        if theta[k] < LB[k] - 1e-8 || theta[k] > UB[k] + 1e-8
            return Inf
        end
    end
    key = (theta[1], theta[2], theta[3])
    if haskey(obj.cache, key)
        return obj.cache[key]
    end
    val = try
        states = sample_states_on_grid(theta)
        sse = 0.0
        for l in MEAS_STATES, i in 1:nfe, j in 1:ncp
            diff = obj.data[l, i, j] - states[l, i, j]
            sse += diff * diff
        end
        sse
    catch err
        @warn "Fallo al integrar el esquema dFBA directo durante la evaluacion" err
        Inf
    end
    obj.cache[key] = val
    obj.eval_count += 1
    return val
end

# ---------------------------------------------
# Coordinated pattern search (simple direct search)
# ---------------------------------------------
function _rand3(pop_size::Int, exclude::Int)
    idxs = Int[]
    while length(idxs) < 3
        cand = rand(1:pop_size)
        (cand == exclude || cand in idxs) && continue
        push!(idxs, cand)
    end
    return idxs
end

function differential_evolution(obj::DirectObjective;
        lb::Vector{Float64}, ub::Vector{Float64},
        pop_size::Int=30, generations::Int=250,
        F::Float64=0.8, CR::Float64=0.9)
    dim = length(lb)
    pop = Vector{Vector{Float64}}(undef, pop_size)
    pop[1] = clamp.(copy(theta0), lb, ub)
    for i in 2:pop_size
        pop[i] = clamp.(lb .+ rand(dim) .* (ub .- lb), lb, ub)
    end
    fitness = [evaluate!(obj, pop[i]) for i in 1:pop_size]
    best_idx = argmin(fitness)
    best_theta = copy(pop[best_idx])
    best_val = fitness[best_idx]
    history = [(generation=0, best=best_val)]

    for gen in 1:generations
        for i in 1:pop_size
            r1, r2, r3 = _rand3(pop_size, i)
            mutant = clamp.(pop[r1] .+ F .* (pop[r2] .- pop[r3]), lb, ub)
            trial = similar(mutant)
            jrand = rand(1:dim)
            for j in 1:dim
                if rand() < CR || j == jrand
                    trial[j] = mutant[j]
                else
                    trial[j] = pop[i][j]
                end
            end
            trial_val = evaluate!(obj, trial)
            if trial_val <= fitness[i]
                pop[i] = trial
                fitness[i] = trial_val
                if trial_val < best_val
                    best_val = trial_val
                    best_theta = copy(trial)
                end
            end
        end
        push!(history, (generation=gen, best=best_val))
    end

    return best_theta, best_val, history
end

function run_direct_search(data::Array{Float64,3};
        pop_size::Int=30, generations::Int=250, F::Float64=0.8, CR::Float64=0.9)
    objective = DirectObjective(data)
    theta_hat, val, history = differential_evolution(
        objective;
        lb=LB, ub=UB,
        pop_size=pop_size,
        generations=generations,
        F=F,
        CR=CR,
    )
    return (
        theta=theta_hat,
        objective=val,
        evals=objective.eval_count,
        history=history,
    )
end

# ---------------------------------------------
# Programa principal
# ---------------------------------------------
function main()
    println("[DIRECT] Ejecutando enfoque secuencial para Zenteno dFBA (parámetros reducidos)")
    Random.seed!(NOISE_SEED)
    println("[SYNTH] Generando datos sintéticos para calibración directa")
    data = _simulate_zenteno_synthetic(nfe=nfe, ncp=ncp, th=th, c0_vec=copy(C0_INIT))

    t_pre = nothing
    states_pre = nothing
    try
        pre_params = ZentenoPlotParams(exp(theta0[1]), exp(theta0[2]), exp(theta0[3]))
        local_t_pre, local_states_pre = simulate_zenteno(pre_params)
        t_pre = local_t_pre
        states_pre = local_states_pre
    catch err
        @warn "No se pudo simular el estado nominal previo" err
    end

    t_start = time()
    result = run_direct_search(data)
    wall_time = time() - t_start

    best_theta = result.theta
    best_obj = result.objective
    eval_count = result.evals

    println("[DIRECT] Búsqueda finalizada")
    println(@sprintf("          theta (log) = (%.6f, %.6f, %.6f)", best_theta...))
    println(@sprintf("          theta (lin) = (%.6f, %.6f, %.6f)", exp.(best_theta)...))
    println(@sprintf("          SSE = %.6e; evals = %d; tiempo = %.2fs", best_obj, eval_count, wall_time))

    best_params = ZentenoPlotParams(exp(best_theta[1]), exp(best_theta[2]), exp(best_theta[3]))
    predicted_states = sample_states_on_grid(best_theta)
    mpcc_tgrid = build_time_grid_from_lengths(HM_REFERENCE)

    t_post, states_post = simulate_zenteno(best_params)
    result_prefix = result_file_prefix(tag="DirectApproach", wall_time=wall_time, objective=best_obj)
    plot_path = result_prefix * ".png"
    plot_post_solution(
        t_pre, states_pre, t_post, states_post, DATA_TIME_GRID, data,
        mpcc_tgrid, predicted_states;
        title_str="Direct approach Zenteno (SSE=$(round(best_obj, sigdigits=4)))",
        save_path=plot_path,
    )
    history_path = write_history_csv(result_prefix, result.history)
    write_direct_report(
        result_prefix;
        wall_time=wall_time,
        best_theta=best_theta,
        best_sse=best_obj,
        evals=eval_count,
        status="completed",
        plot_path=plot_path,
        history_path=history_path,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
