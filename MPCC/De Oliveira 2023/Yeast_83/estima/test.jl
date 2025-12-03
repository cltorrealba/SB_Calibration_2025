#!/usr/bin/env julia
# MPCC_Zenteno_tuned_v3.jl
# CORRECCIÓN DEFINITIVA: 
# 1. Inicialización "Clean Slate" para flujos y duales (dejar que Ipopt decida).
# 2. Mantenemos relajación robusta (TOL=1.0, omega=1.0).

using JuMP
using Ipopt
using LinearAlgebra
using DelimitedFiles
using DifferentialEquations
using Plots
using Dates
using Random
using Printf
using MathOptInterface
using JLD2
const MOI = MathOptInterface

# ---------------------------------------------
# Paths e IO
# ---------------------------------------------
const BASE_DIR   = @__DIR__
const ESTIMA_DIR = BASE_DIR
const PLOTS_DIR  = joinpath(ESTIMA_DIR, "plots")
isdir(PLOTS_DIR) || mkpath(PLOTS_DIR)

function _sanitize_experiment_name(str::AbstractString)
    clean = strip(str)
    isempty(clean) && return "default"
    return replace(clean, r"[^0-9A-Za-z._-]+" => "_")
end

const EXPERIMENT_TOKEN = _sanitize_experiment_name(get(ENV, "EXPERIMENT", "default"))
const EXPERIMENT_DIR = joinpath(PLOTS_DIR, EXPERIMENT_TOKEN)
isdir(EXPERIMENT_DIR) || mkpath(EXPERIMENT_DIR)

# Matriz estequiometrica y cotas de flujos
S     = readdlm(joinpath(ESTIMA_DIR, "S.csv"), ',')
lbraw = readdlm(joinpath(ESTIMA_DIR, "lb.csv"), ',')
ubraw = readdlm(joinpath(ESTIMA_DIR, "ub.csv"), ',')
lb    = lbraw isa AbstractVector ? copy(lbraw) : copy(lbraw[:,1])
ub    = ubraw isa AbstractVector ? copy(ubraw) : copy(ubraw[:,1])

const ETHYL_ACETATE_ROW = 2023
const ETHYL_ACETATE_EX_RXN = begin
    row = view(S, ETHYL_ACETATE_ROW, :)
    nz = findall(!iszero, row)
    if isempty(nz)
        nothing
    else
        pos = findfirst(k -> row[k] < 0.0, nz)
        nz[pos === nothing ? 1 : pos]
    end
end

const MOLAR_MASS_ETHYL_ACETATE = 88.106 # g / mol

const REDUCED_MODE = get(ENV, "REDUCED_MODE", "0") == "1"
const REDUCED_SETS_PATH = joinpath(BASE_DIR, "julia_deploy", "results", "reduced_sets.jld2")

function _load_reduced_sets(path::String)
    if !isfile(path)
        return nothing
    end
    try
        return JLD2.jldopen(path, "r") do f
            (read(f, "A"), read(f, "C"), read(f, "F"))
        end
    catch err
        @warn "No se pudieron cargar reduced_sets" path err
        return nothing
    end
end

const reduced_sets = REDUCED_MODE ? _load_reduced_sets(REDUCED_SETS_PATH) : nothing
if REDUCED_MODE && reduced_sets === nothing
    @warn "REDUCED_MODE=1 sin reduced_sets.jld2; se usara el modelo completo" REDUCED_SETS_PATH
end

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
const DEFAULT_LINEAR_SOLVER = "mumps"

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

# --- CONFIGURACIÓN ROBUSTA ---
const MPCC_RELAX_TOL = 1.0 

death_rate(E) = begin
    Td = -0.0001 * E^3 + 0.0049 * E^2 - 0.1279 * E + 315.89
    s = 0.5 * (1.0 + tanh(0.5 * (T_const - Td)))
    base = Kd0_nom * exp(0.0415 * E + (130000.0 * (T_const - 305.65)) / (305.65 * R * T_const))
    base * s
end

const SQRT_2PI = sqrt(2*pi)
function smooth_injection(t, t_shot, dose, width=1.0)
    abs(t - t_shot) > 5 * width && return 0.0
    return (dose / (width * SQRT_2PI)) * exp(-0.5 * ((t - t_shot) / width)^2)
end

# Parametros estimables
const np = 4
const THETA_NAMES = ("mu0", "Yeg", "Yef", "Yxn")
const theta_data_params = log.([0.1, 0.1, 0.8, YXN_nom]) # parámetros usados para generar los datos sintéticos
const theta_init_guess = copy(theta_data_params)
LB = log.([0.5*MU0_nom, 0.5*YEG_nom, 0.5*YEF_nom, 0.5*YXN_nom])
UB = log.([5.0*MU0_nom, 5.0*YEG_nom, 5.0*YEF_nom, 5.0*YXN_nom])

# Condiciones iniciales (CRASH TEST CONFIG)
X0 = 0.5; N0 = 0.14; 
G0 = 110.0; 
F0 = 110.0; 
E0 = 0.0
const C0_INIT = [X0, N0, G0, F0, E0]
c0 = copy(C0_INIT)

# Discretizacion (CRASH TEST CONFIG)
nfe = 15   
ncp = 3
th  = 72.0 
h   = th / nfe
ph  = nfe
hm    = fill(h, nfe)'
const HM_REFERENCE = vec(hm)
var_h = 1.0

const T_INJ_1 = 48.0   # horas
const DOSE_1  = 0.08   # g/L (80 mg/L)
const WIDTH_1 = 5.0    # ancho de pulso (h)

# --- OMEGA NEUTRO ---
w     = 1e-20
omega = 1.0 

d   = zeros(nv); d[obj] = -1.0
up  = zeros(nv); up[glu] = 1.0
up2 = zeros(nv); up2[fru] = 1.0
const n_up = 2

cs = ones(nc)
vs = ones(nv)

const ESTIMATE_PARAMS = true

function _build_reduced_axes()
    if !REDUCED_MODE || reduced_sets === nothing
        return collect(1:nv), collect(1:nm)
    end
    A_sets, C_sets, _ = reduced_sets
    K = Int[]
    for i in 1:nfe
        if i <= length(A_sets); append!(K, A_sets[i]); end
        if i <= length(C_sets); append!(K, C_sets[i]); end
    end
    append!(K, (glu, fru))
    K = unique(K)
    sort!(K)
    M = Int[]
    for mc in 1:nm
        for k in K
            if S[mc, k] != 0.0; push!(M, mc); break; end
        end
    end
    M = unique(M)
    sort!(M)
    return K, M
end

const K_AX, M_AX = _build_reduced_axes()

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
# Configuracion Ipopt
# ---------------------------------------------
function configure_pardiso_defaults()
    solver = lowercase(strip(get(ENV, "IPOPT_LINEAR_SOLVER", "")))
    !(solver in ("pardiso", "pardisomkl")) && return
    set_default!(name, val) = isempty(strip(get(ENV, name, ""))) && (ENV[name] = val)
    set_default!("PARDISO_MTYPE", "-2")
    println("[IPOPT] Pardiso defaults configured.")
end
configure_pardiso_defaults()


# ---------------------------------------------
# Herramientas de simulacion/plot
# ---------------------------------------------
const STATE_LABELS = ("X", "N", "G", "F", "E")
const STATE_COLORS = (:royalblue, :forestgreen, :firebrick, :darkorange, :purple)
const STATE_MIN_CONC = (
    1e-6, 
    0.0,   
    0.0,   
    0.0,   
    1e-6, 
)
const FREEZE_DEPLETED_STATES = get(ENV, "FREEZE_DEPLETED_STATES", "1") == "1"
const FREEZE_NITROGEN = get(ENV, "FREEZE_NITROGEN", "1") == "1"

struct ZentenoPlotParams
    mu0::Float64
    Yeg::Float64
    Yef::Float64
    Yxn::Float64
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
    betaG = betaG0_nom * b_T * (G / (G + Kg0_nom * Kg_T + eps)) * (Kie0_nom * Kg_T / (E + Kie0_nom * Kg_T + eps))
    betaF = betaF0_nom * b_T * (F / (F + Kf0_nom * Kg_T + eps)) * (Kig0_nom * Kg_T / (G + Kig0_nom * Kg_T + eps)) * (Kie0_nom * Kg_T / (E + Kie0_nom * Kg_T + eps))
    Td = -0.0001 * E^3 + 0.0049 * E^2 - 0.1279 * E + 315.89
    s_sw = 0.5 * (1.0 + tanh(0.5 * (T_const - Td)))
    Kd_val = Kd0_nom * exp(0.0415 * E + (130000.0 * (T_const - 305.65)) / (305.65 * R * T_const)) * s_sw
    du[1] = (mu - Kd_val) * X
    du[2] = -(mu / p.Yxn) * X + smooth_injection(t, T_INJ_1, DOSE_1, WIDTH_1)
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
            plot!(plt[s], t_pre, states_pre[s, :]; color=STATE_COLORS[s], lw=2, linestyle=:dashdot, label=(s == 1 ? "ODE pre" : nothing))
        end
        plot!(plt[s], t_post, states_post[s, :]; color=STATE_COLORS[s], lw=3, label=(s == 1 ? "ODE post" : nothing))
        if s in MEAS_STATES
            ys_data = reshape(data_vals[s, :, :], length(ts_data))
            scatter!(plt[s], ts_data, ys_data; color=:black, ms=4, alpha=0.8, label=(s == MEAS_STATES[1] ? "Datos exp." : nothing))
        end
        if mpcc_tgrid !== nothing && mpcc_states !== nothing
            ts_mpcc = vec(mpcc_tgrid)
            ys_mpcc = reshape(mpcc_states[s, :, :], length(ts_mpcc))
            scatter!(plt[s], ts_mpcc, ys_mpcc; color=:purple, ms=5, alpha=0.9, marker=:diamond, label=(s == 1 ? "MPCC" : nothing))
        end
        ylabel!(plt[s], STATE_LABELS[s])
        if s == 1; title!(plt[s], title_str); end
    end
    xlabel!(plt[nc], "tiempo [h]")
    save_dir = dirname(save_path)
    isdir(save_dir) || mkpath(save_dir)
    savefig(plt, save_path)
    println("[PLOT] Guardado ", save_path)
end

function _build_ethyl_acetate_series(mpcc_tgrid, mpcc_states, v_var)
    if mpcc_tgrid === nothing || mpcc_states === nothing
        return nothing
    end
    rxn_idx = ETHYL_ACETATE_EX_RXN
    if rxn_idx === nothing
        @warn "No se encontró ninguna reacción asociada a la fila $(ETHYL_ACETATE_ROW) de S"
        return nothing
    end
    nfe = size(mpcc_states, 2)
    flux_vals = Vector{Float64}(undef, nfe)
    for i in 1:nfe
        flux_var = try
            v_var[rxn_idx, i]
        catch err
            @warn "El MPCC no contiene la reacción de ethyl acetate en el conjunto de variables" rxn_idx err
            return nothing
        end
        flux_val = safe_value(flux_var, NaN)
        if !isfinite(flux_val)
            @warn "El flujo de ethyl acetate no es finito" step=i flux=flux_val
            return nothing
        end
        flux_vals[i] = flux_val
    end
    time_rate = _build_time_rate_arrays(mpcc_tgrid, mpcc_states, flux_vals)
    time_rate === nothing && return nothing
    times, rates = time_rate
    isempty(times) && return nothing
    concentrations = _cumulative_trapezoid(times, rates)
    return (times=times, concentrations=concentrations)
end

function _build_time_rate_arrays(mpcc_tgrid, mpcc_states, flux_values)
    nfe = size(mpcc_states, 2)
    ncp = size(mpcc_states, 3)
    if length(flux_values) != nfe || size(mpcc_tgrid, 1) != nfe || size(mpcc_tgrid, 2) != ncp
        @warn "Dimensiones incompatibles al construir la serie de ethyl acetate" nfe ncp mpcc_size=size(mpcc_tgrid) flux_len=length(flux_values)
        return nothing
    end
    npts = nfe * ncp
    times = Vector{Float64}(undef, npts)
    rates = Vector{Float64}(undef, npts)
    idx = 1
    for i in 1:nfe
        flux = Float64(flux_values[i])
        for j in 1:ncp
            biomass = Float64(mpcc_states[1, i, j])
            if !isfinite(biomass)
                @warn "Concentración de biomasa no definida al construir la serie de ethyl acetate" collocation=(i, j)
                return nothing
            end
            times[idx] = mpcc_tgrid[i, j]
            rates[idx] = flux * biomass
            idx += 1
        end
    end
    perm = sortperm(times)
    return times[perm], rates[perm]
end

function _cumulative_trapezoid(times, values)
    n = length(times)
    concentrations = zeros(Float64, n)
    for idx in 2:n
        dt = max(Float64(times[idx] - times[idx - 1]), 0.0)
        concentrations[idx] = concentrations[idx - 1] +
            0.5 * (Float64(values[idx]) + Float64(values[idx - 1])) * dt
    end
    return concentrations
end

function _natural_cubic_coefficients(x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    @assert n == length(y)
    h = diff(x)
    alpha = zeros(Float64, n)
    for i in 2:(n-1)
        alpha[i] = (3.0 / h[i]) * (y[i+1] - y[i]) - (3.0 / h[i-1]) * (y[i] - y[i-1])
    end
    lvec = ones(Float64, n)
    mu = zeros(Float64, n)
    z = zeros(Float64, n)
    for i in 2:(n-1)
        lvec[i] = 2.0 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / lvec[i]
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / lvec[i]
    end
    c = zeros(Float64, n)
    b = zeros(Float64, n-1)
    d = zeros(Float64, n-1)
    a = copy(y[1:n-1])
    for j in (n-1):-1:1
        c[j] = z[j] - mu[j] * c[j+1]
        b[j] = (y[j+1] - y[j]) / h[j] - (h[j] * (c[j+1] + 2.0 * c[j])) / 3.0
        d[j] = (c[j+1] - c[j]) / (3.0 * h[j])
    end
    return (a=a, b=b, c=c, d=d, x=x)
end

function _evaluate_natural_cubic(coeffs, t::Float64)
    idx = clamp(searchsortedlast(coeffs.x, t), 1, length(coeffs.a))
    delta_t = t - coeffs.x[idx]
    return coeffs.a[idx] + coeffs.b[idx] * delta_t + coeffs.c[idx] * delta_t^2 + coeffs.d[idx] * delta_t^3
end

function _sample_cubic_spline(times::Vector{Float64}, values::Vector{Float64}; nsamples::Int=400)
    n = length(times)
    if n < 2 || (times[end] - times[1]) <= 0
        return copy(times), copy(values)
    end
    coeffs = _natural_cubic_coefficients(times, values)
    ts = collect(range(times[1], times[end]; length=nsamples))
    vals = Vector{Float64}(undef, length(ts))
    for (idx, t) in enumerate(ts)
        vals[idx] = _evaluate_natural_cubic(coeffs, t)
    end
    return ts, vals
end

function plot_ethyl_acetate_concentration(mpcc_tgrid, mpcc_states, v_var; title_str::AbstractString, save_path::AbstractString)
    series = _build_ethyl_acetate_series(mpcc_tgrid, mpcc_states, v_var)
    if series === nothing
        @warn "No se pudo construir la serie de concentración de ethyl acetate"
        return false
    end
    times = Float64.(series.times)
    concentrations_mmol = Float64.(series.concentrations)
    isempty(times) && return false
    concentrations_g = concentrations_mmol .* (MOLAR_MASS_ETHYL_ACETATE / 1000.0)
    line_t, line_vals = _sample_cubic_spline(times, concentrations_g)
    plt = plot(size=(900, 400))
    scatter!(plt, times, concentrations_g;
        color=:darkorange, ms=5, alpha=0.9, label="Colocaciones MPCC")
    plot!(plt, line_t, line_vals;
        color=:navy, lw=2, label="Spline interpolado")
    xlabel!(plt, "tiempo [h]")
    ylabel!(plt, "concentración ethyl acetate [g/L]")
    title!(plt, title_str)
    save_dir = dirname(save_path)
    isdir(save_dir) || mkpath(save_dir)
    savefig(plt, save_path)
    println("[PLOT] Guardado ", save_path)
    return true
end

safe_value(x, default=NaN) = try value(x) catch; default end
maxabs(arr::AbstractArray) = isempty(arr) ? 0.0 : maximum(abs, arr)

sanitize_token(str::AbstractString) = replace(str, r"[^0-9A-Za-z]+" => "_")

function short_token(str::AbstractString; maxlen::Int=10)
    clean = sanitize_token(str)
    return clean[1:min(length(clean), maxlen)]
end

format_float_token(val::Float64; digits::Int=1) = replace(@sprintf("%.*f", digits, val), "." => "p")

_stat_display(val) = begin
    if val === nothing
        "unavailable"
    elseif val isa Float64 && !isfinite(val)
        "unavailable"
    else
        string(val)
    end
end

_format_vector(vec::AbstractVector) = "[" * join((@sprintf("%.6g", v) for v in vec), ", ") * "]"

function result_file_prefix(; wall_time::Float64, nfe::Int, status, primal_status)
    wall_tok = "w$(format_float_token(wall_time))s"
    feas_tok = "f$(short_token(string(primal_status); maxlen=6))"
    term_tok = "t$(short_token(string(status); maxlen=6))"
    stamp = Dates.format(Dates.now(), "yyyymmdd_HHmmss")
    base_name = "MPCCpost_$(wall_tok)_nfe$(nfe)_$(feas_tok)_$(term_tok)_$(stamp)"
    full_path = joinpath(EXPERIMENT_DIR, base_name)
    if Sys.iswindows()
        # Agregamos margen para los sufijos (_ethyl_acetate.png, etc.) y evitamos el limite de 260 chars.
        worst_case = length(full_path) + length("_ethyl_acetate.png")
        if worst_case >= 255
            short_name = "MPCCpost_$(stamp)"
            full_path = joinpath(EXPERIMENT_DIR, short_name)
            @info "Ruta de salida abreviada para evitar limites de path en Windows" short_name full_path
        end
    end
    return full_path
end

function write_diagnostic_report(path_prefix::AbstractString; wall_time::Float64, status, primal_status,
        objective::Float64, dual_inf, primal_inf, compl, constr_viol, iter_count, fo_value::Float64,
        nfe::Int, th::Float64, c0_init::AbstractVector, theta_init_log::AbstractVector,
        theta_data_log::AbstractVector, theta_final_log::AbstractVector, estimate_params::Bool)
    report_path = path_prefix * ".txt"
    open(report_path, "w") do io
        println(io, "timestamp=", Dates.now())
        println(io, "termination_status=", status)
        println(io, "primal_status=", primal_status)
        println(io, @sprintf("wall_time_s=%.4f", wall_time))
        println(io, "objective=", objective)
        println(io, "FO_value=", fo_value)
        println(io, "iter_count=", _stat_display(iter_count))
        println(io, "dual_infeasibility=", _stat_display(dual_inf))
        println(io, "primal_infeasibility=", _stat_display(primal_inf))
        println(io, "constraint_violation=", _stat_display(constr_viol))
        println(io, "complementarity=", _stat_display(compl))
        println(io, "nfe=", nfe)
        println(io, "th_h=", th)
        println(io, "initial_conditions=", _format_vector(c0_init))
        println(io, "estimate_params=", estimate_params)
        println(io, "theta_init_guess_log=", _format_vector(theta_init_log))
        println(io, "theta_init_guess_exp=", _format_vector(exp.(theta_init_log)))
        println(io, "theta_data_log=", _format_vector(theta_data_log))
        println(io, "theta_data_exp=", _format_vector(exp.(theta_data_log)))
        println(io, "theta_final_log=", _format_vector(theta_final_log))
        println(io, "theta_final_exp=", _format_vector(exp.(theta_final_log)))
    end
    println("[REPORT] Guardado ", report_path)
end

function optimizer_attr(m, attr::AbstractString)
    try
        return get_optimizer_attribute(m, attr)
    catch
        return nothing
    end
end

function unwrap_ipopt_optimizer(model::Model)
    opt = JuMP.backend(model)
    while true
        if opt isa MOI.Bridges.AbstractBridgeOptimizer
            opt = opt.model
            continue
        elseif opt isa MOI.Utilities.CachingOptimizer
            opt = opt.optimizer
            opt === nothing && return nothing
            continue
        elseif typeof(opt) == Ipopt.Optimizer
            return opt
        else
            return nothing
        end
    end
end

function collect_ipopt_stats(model::Model)
    optimizer = unwrap_ipopt_optimizer(model)
    optimizer === nothing && return nothing
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
    success = try
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
        true
    catch
        false
    end
    success || return nothing
    dual_inf = maxabs(grad_lag_x)
    primal_inf = max(maxabs(x_L_violation), maxabs(x_U_violation))
    constraint_violation = maxabs(constr_violation)
    complementarity = maximum((
        maxabs(compl_x_L),
        maxabs(compl_x_U),
        maxabs(compl_g),
    ))
    backend = JuMP.backend(model)
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
# Datos sinteticos
# ---------------------------------------------
function _simulate_zenteno_synthetic(; nfe::Int, ncp::Int, th::Float64, c0_vec::Vector{Float64})
    params = ZentenoPlotParams(
        exp(theta_data_params[1]),
        exp(theta_data_params[2]),
        exp(theta_data_params[3]),
        exp(theta_data_params[4]),
    )
    t_dense, U = simulate_zenteno(params; tspan=(0.0, th))
    data = zeros(nc, nfe, ncp)
    dt = th / nfe
    for i in 1:nfe
        for (j, tau) in enumerate(radau_nodes)
            t_curr = (i-1)*dt + tau*dt
            idx = findmin(abs.(t_dense .- t_curr))[2]
            data[:, i, j] = U[:, idx]
        end
    end
    if NOISE_REL_STD > 0
        noise = NOISE_REL_STD .* data .* randn(size(data))
        data .+= noise
        @. data = max(data, 0.0)
    end
    return data
end

println("[SYNTH] Generando datos sinteticos...")
data = _simulate_zenteno_synthetic(nfe=nfe, ncp=ncp, th=th, c0_vec=c0)

t_pre = nothing
states_pre = nothing
try
    pre_params = ZentenoPlotParams(
        exp(theta_data_params[1]),
        exp(theta_data_params[2]),
        exp(theta_data_params[3]),
        exp(theta_data_params[4]),
    )
    local_t_pre, local_states_pre = simulate_zenteno(pre_params; tspan=(0.0, th))
    global t_pre = local_t_pre
    global states_pre = local_states_pre
catch err
    @warn "No se pudo simular la ODE previa a la optimizacion" err
end

# ---------------------------------------------
# Modelo JuMP + Configuración de Convergencia "Pragmática"
# ---------------------------------------------
m = Model(Ipopt.Optimizer)

set_optimizer_attribute(m, "warm_start_init_point", "yes")
set_optimizer_attribute(m, "print_level", 5)

# 1. Tolerancia Principal (Objetivo ideal)
# Relajamos de 1e-4 a 1e-3. Es suficiente precisión para dFBA (flujos ~1000 +/- 1)
set_optimizer_attribute(m, "tol", 1e-2)

# 2. Red de Seguridad ("Acceptable Level")
# Si Ipopt no logra 1e-3, pero se mantiene estable en 1e-1 durante 5 iteraciones, termina con éxito.
set_optimizer_attribute(m, "acceptable_tol", 1e-1) 
set_optimizer_attribute(m, "acceptable_iter", 5)
set_optimizer_attribute(m, "acceptable_obj_change_tol", 1e-2)

# 3. Prioridades Específicas (Crucial para MPCC)
# Queremos que el balance de masa (S*v=0) se respete bien (constr_viol),
# pero no nos importa tanto si la optimalidad matemática (gradientes/duales) es perfecta.
set_optimizer_attribute(m, "constr_viol_tol", 1e-3)  # Respetar física (Primal)
set_optimizer_attribute(m, "dual_inf_tol", 1.0)      # Relajar matemáticas (Dual) - Los MPCC tienen duales feos por naturaleza
set_optimizer_attribute(m, "compl_inf_tol", 1e-1)     # Complementariedad (ya relajada manualmente, esto es solo el check interno)

# 4. Limpieza
set_optimizer_attribute(m, "max_iter", 700) # Darle espacio si avanza lento pero seguro

@variables(m, begin
    c[1:nc, 1:ph, 1:ncp]
    cdot[1:nc, 1:ph, 1:ncp]
    FO
    teta[1:np]
    hv[1:nfe]
end)

if !REDUCED_MODE || reduced_sets === nothing
    @variables(m, begin
        v[1:nv, 1:nfe]
        lambda_[1:nm, 1:nfe]
        alpha_U[1:nv, 1:nfe]
        alpha_L[1:nv, 1:nfe]
    end)
else
    @variables(m, begin
        v[K_AX, 1:nfe]
        lambda_[M_AX, 1:nfe]
        alpha_U[K_AX, 1:nfe]
        alpha_L[K_AX, 1:nfe]
    end)
end

@variables(m, begin
    alpha_upt[1:n_up, 1:nfe]
end)

# --- INICIALIZACION SOLO ESTADOS ---
for i in 1:ph, j in 1:ncp, l in 1:nc
    val_init = data[l, i, j]
    val_init = max(val_init, STATE_MIN_CONC[l])
    set_start_value(c[l, i, j], val_init)
    set_start_value(cdot[l, i, j], 0.0)
end
for i in 1:nfe
    set_start_value(hv[i], hm[i])
end
for k in 1:np; set_start_value(teta[k], theta_init_guess[k]); end

# --- NO INICIALIZAR v NI alpha MANUALMENTE ---
# (Dejar que el solver encuentre el punto de silla)

for i in 1:nc
    c0[i] = c0[i] / cs[i]
    c0[i] = max(c0[i], STATE_MIN_CONC[i])
end

@NLobjective(m, Min, omega * FO)

JuMP.register(m, :death_rate, 1, death_rate; autodiff = true)
JuMP.register(m, :smooth_injection, 4, smooth_injection; autodiff = true)
@NLexpression(m, mu_T, exp(59453.0 * (T_const - 300.0) / (300.0 * R * T_const)))
@NLexpression(m, Kg_T, exp(46055.0 * (T_const - 293.15) / (293.15 * R * T_const)))
@NLexpression(m, b_T,  exp(11000.0 * (T_const - 296.15) / (296.15 * R * T_const)))
@NLexpression(m, mrate, 0.01 * exp(37681.0 * (T_const - 293.30) / (293.30 * R * T_const)))
@NLexpression(m, mu0, exp(teta[1]))
@NLexpression(m, Yeg, exp(teta[2]))
@NLexpression(m, Yef, exp(teta[3]))

const Yxg = YXG_nom
const Yxf = YXF_nom

@NLexpression(m, mu_j[i=1:ph, j=1:ncp], mu0 * mu_T * (c[2,i,j] / (c[2,i,j] + Kn0_nom * Kg_T + eps)))
@NLexpression(m, betaG_j[i=1:ph, j=1:ncp], betaG0_nom * b_T * (c[3,i,j] / (c[3,i,j] + Kg0_nom * Kg_T + eps)) * (Kie0_nom * Kg_T / (c[5,i,j] + Kie0_nom * Kg_T + eps)))
@NLexpression(m, betaF_j[i=1:ph, j=1:ncp], betaF0_nom * b_T * (c[4,i,j] / (c[4,i,j] + Kf0_nom * Kg_T + eps)) * (Kig0_nom * Kg_T / (c[3,i,j] + Kig0_nom * Kg_T + eps)) * (Kie0_nom * Kg_T / (c[5,i,j] + Kie0_nom * Kg_T + eps)))
@NLexpression(m, phiG_j[i=1:ph, j=1:ncp], c[3,i,j] / (c[3,i,j] + c[4,i,j] + eps))
@NLexpression(m, phiF_j[i=1:ph, j=1:ncp], c[4,i,j] / (c[3,i,j] + c[4,i,j] + eps))
@NLexpression(m, Kd_j[i=1:ph, j=1:ncp], death_rate(c[5,i,j]))
@NLexpression(m, Yxn, exp(teta[4]))
@NLexpression(m, injection_rate[i=1:nfe, j=1:ncp],
    smooth_injection((i - 1 + radau_nodes[j]) * h, T_INJ_1, DOSE_1, WIDTH_1)
)

@constraints(m, begin
    coll_c_n[l=1:nc, i=2:ph, j=1:ncp], c[l,i,j] == c[l,i-1,ncp] + hv[i] * sum(colmat[j,k] * cdot[l,i,k] for k in 1:ncp)
    coll_c_0[l=1:nc, j=1:ncp], c[l,1,j] == c0[l] + hv[1] * sum(colmat[j,k] * cdot[l,1,k] for k in 1:ncp)
    c_LB[l=1:nc, i=1:nfe, j=1:ncp], STATE_MIN_CONC[l] - c[l,i,j] <= 0
    MFE1, sum(hv[i] for i in 1:nfe) == th
    MFE3[i=1:nfe], hv[i] >= 0.0
    MFE4[i=1:nfe], hv[i] >= (1.0 - var_h) * hm[1]
    MFE5[i=1:nfe], hv[i] <= (1.0 + var_h) * hm[1]
    alpha4_LB[mc=1:n_up, i=1:nfe], alpha_upt[mc,i] <= 0
end)

if ESTIMATE_PARAMS
    @constraints(m, begin
        teta_LB[p=1:np], teta[p] >= LB[p]
        teta_UB[p=1:np], teta[p] <= UB[p]
    end)
else
    @constraints(m, begin
        teta_fix[p=1:np], teta[p] == theta_data_params[p]
    end)
end

# --- SUAVIZADO Y ACOPLAMIENTO DINÁMICO ---

# Constantes de saturación numérica
const K_smooth = 1.0       # Para Glucosa/Fructosa (Escala ~100 g/L)
const K_nit_smooth = 0.001  # Para Nitrógeno (Escala ~0.15 g/L)

# Índices de reacciones de intercambio de nitrógeno
const idx_NH4 = 2536
const idx_Arg = 2729
const idx_Gln = 2740
const idx_Glu = 2738
const idx_Ser = 2754
const idx_Thr = 2759
const idx_Ala = 2723
const idx_Trp = 2760
const NITROGEN_SOURCES = [idx_NH4, idx_Arg, idx_Gln, idx_Glu, idx_Ser, idx_Thr, idx_Ala, idx_Trp]

# Selectores booleanos
const LB_SELECTOR_GLU = [mc == glu ? 1.0 : 0.0 for mc in 1:nv]
const LB_SELECTOR_FRU = [mc == fru ? 1.0 : 0.0 for mc in 1:nv]
const LB_SELECTOR_NIT = [mc in NITROGEN_SOURCES ? 1.0 : 0.0 for mc in 1:nv]

# Límite inferior dinámico multicomponente
@NLexpression(m, lb_eff[mc=1:nv, i=1:nfe],
    lb[mc] * (
        LB_SELECTOR_GLU[mc] * (c[3,i,1] / (c[3,i,1] + K_smooth)) +
        LB_SELECTOR_FRU[mc] * (c[4,i,1] / (c[4,i,1] + K_smooth)) +
        LB_SELECTOR_NIT[mc] * (c[2,i,1] / (c[2,i,1] + K_nit_smooth)) +
        (1.0 - LB_SELECTOR_GLU[mc] - LB_SELECTOR_FRU[mc] - LB_SELECTOR_NIT[mc])
    )
)

if !REDUCED_MODE || reduced_sets === nothing
    @constraints(m, begin
        Sc[mc=1:nm, i=1:nfe],  sum(S[mc,k] * v[k,i] * vs[k] for k in 1:nv) == 0
        v_UB[mc=1:nv, i=1:nfe], v[mc,i]*vs[mc] - ub[mc] <= 0
    end)
    @NLconstraints(m, begin
        v_LB_dyn[mc=1:nv, i=1:nfe], -v[mc,i]*vs[mc] + lb_eff[mc,i] <= 0
    end)
    @constraints(m, begin
        Lagr[mc=1:nv, i=1:nfe],
            d[mc] + w * v[mc,i] * vs[mc] + alpha_L[mc,i] + alpha_U[mc,i] +
            up[mc] * alpha_upt[1,i] + up2[mc] * alpha_upt[2,i] +
            sum(S[k,mc] * lambda_[k,i] for k in 1:nm) == 0
        alpha1_LB[mc=1:nv, i=1:nfe], alpha_L[mc,i] <= 0
        alpha1_UB[mc=1:nv, i=1:nfe], alpha_U[mc,i] >= 0
    end)
else
    # (Bloque REDUCED_MODE omitido)
end

for i in 1:ph, j in 1:ncp
    @NLconstraint(m, cdot[1,i,j] == (mu_j[i,j] - Kd_j[i,j]) * c[1,i,j])
    @NLconstraint(m, cdot[2,i,j] == -(mu_j[i,j] / Yxn) * c[1,i,j] + injection_rate[i,j])
    @NLconstraint(m, cdot[3,i,j] == -((mu_j[i,j] / Yxg) + (betaG_j[i,j] / Yeg) + mrate * phiG_j[i,j]) * c[1,i,j])
    @NLconstraint(m, cdot[4,i,j] == -((mu_j[i,j] / Yxf) + (betaF_j[i,j] / Yef) + mrate * phiF_j[i,j]) * c[1,i,j])
    @NLconstraint(m, cdot[5,i,j] == (betaG_j[i,j] + betaF_j[i,j]) * c[1,i,j])
end

@NLconstraints(m, begin
    FO3_upt_relax[i=1:nfe], (-v[glu,i]*vs[glu]) * alpha_upt[1,i] >= -MPCC_RELAX_TOL
    FO4_upt_relax[i=1:nfe], (-v[fru,i]*vs[fru]) * alpha_upt[2,i] >= -MPCC_RELAX_TOL
    FO_def, FO == sum((data[l,i,j] - c[l,i,j])^2 for l in MEAS_STATES, i in 1:ph, j in 1:ncp)
end)

if !REDUCED_MODE || reduced_sets === nothing
    @NLconstraints(m, begin
        FO1_relax[mc=1:nv, i=1:nfe], (lb_eff[mc,i] - v[mc,i]*vs[mc]) * alpha_L[mc,i] >= -MPCC_RELAX_TOL
        FO2_relax[mc=1:nv, i=1:nfe], (v[mc,i]*vs[mc] - ub[mc]) * alpha_U[mc,i] >= -MPCC_RELAX_TOL
    end)
end

println("Iniciando optimizacion robusta V3...")
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

theta_final_log = [safe_value(teta[k], theta_data_params[k]) for k in 1:np]
theta_final_vals = exp.(theta_final_log)

try
    post_params = ZentenoPlotParams(
        theta_final_vals[1],
        theta_final_vals[2],
        theta_final_vals[3],
        theta_final_vals[4],
    )
    hv_vals = [safe_value(hv[i], hm[i]) for i in 1:nfe]
    mpcc_tgrid = build_time_grid_from_lengths(hv_vals)
    mpcc_states = Array{Float64}(undef, nc, nfe, ncp)
    for l in 1:nc, i in 1:nfe, j in 1:ncp
        mpcc_states[l, i, j] = safe_value(c[l, i, j])
    end
    t_post, states_post = simulate_zenteno(post_params; tspan=(0.0, th))
    plot_post_solution(
        t_pre, states_pre, t_post, states_post, DATA_TIME_GRID, data,
        mpcc_tgrid, mpcc_states;
        title_str="MPCC post: $(status) / $(pr_status)",
        save_path=plot_output_path,
    )
    plot_ethyl_acetate_concentration(
        mpcc_tgrid, mpcc_states, v;
        title_str="Ethyl acetate MPCC: $(status) / $(pr_status)",
        save_path=result_prefix * "_ethyl_acetate.png",
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
    nfe=nfe,
    th=th,
    c0_init=C0_INIT,
    theta_init_log=theta_init_guess,
    theta_data_log=theta_data_params,
    theta_final_log=theta_final_log,
    estimate_params=ESTIMATE_PARAMS,
)
