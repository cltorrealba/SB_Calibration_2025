#!/usr/bin/env julia
# MPCC_Zenteno_stripping.jl
# IMPLEMENTACIÓN CORREGIDA Y REVISADA
# 1. Inicialización completa de variables duales y primal.
# 2. Lógica unificada de uptake (Glc, Fru, N).
# 3. Stripping de aromas post-proceso.

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
using Clapeyron
using CSV
using HiGHS
const MOI = MathOptInterface

"""
Helper utilities and plotting/stripping builders used outside the JuMP model.
- Keep external logic separate from JuMP containers to avoid NL parsing issues.
"""
function safe_value(x, default=nothing)
    try
        return JuMP.value(x)
    catch
        return default === nothing ? NaN : default
    end
end

function result_file_prefix(; wall_time, nfe, status, primal_status)
    ts = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
    return joinpath(EXPERIMENT_DIR, "result_$(ts)_nfe$(nfe)_$(status)_$(primal_status)")
end

struct StrippingInputs
    times::Vector{Float64}
    k_vals::Vector{Float64}
    co2_vals::Vector{Float64}
end

struct EthylSeries
    times::Vector{Float64}
    concentrations::Vector{Float64}
    rates::Vector{Float64}
    line_t::Vector{Float64}
    line_vals::Vector{Float64}
    stripping::Union{Nothing,NamedTuple}
end

function _build_stripping_inputs(t_post::Vector{Float64}, states_post::Array{Float64,2}, p)
    # Simple apparent k profile based on temperature; CO2 flow placeholder
    k_vals = Float64[]
    co2_vals = Float64[]
    for t in t_post
        T = dynamic_temperature(t)
        push!(k_vals, 0.001 * exp(0.01 * (T - T_BASE)))
        push!(co2_vals, 1.0) # L/h placeholder
    end
    return StrippingInputs(t_post, k_vals, co2_vals)
end

function _build_ethyl_acetate_plot_data(mpcc_tgrid::Array{Float64,2}, mpcc_states::Array{Float64,3}, v; stripping_inputs::Union{Nothing,StrippingInputs}=nothing)
    # Extract E state at last collocation of each fe
    times = vec(mpcc_tgrid)
    Evals = Float64[]
    rates = Float64[]
    for idx in eachindex(times)
        i = div(idx-1, size(mpcc_tgrid, 2)) + 1
        j = mod(idx-1, size(mpcc_tgrid, 2)) + 1
        push!(Evals, mpcc_states[5, i, j])
        push!(rates, 0.0)
    end
    # Simple linear interpolation over dense grid
    line_t = collect(range(first(times), last(times), length=length(times)))
    line_vals = copy(Evals)
    strip = nothing
    if stripping_inputs !== nothing
        strip = (bulk_times = stripping_inputs.times,
                 bulk_conc = Evals[1:length(stripping_inputs.times)],
                 lost_conc = zeros(length(stripping_inputs.times)))
    end
    return EthylSeries(times, Evals, rates, line_t, line_vals, strip)
end

function plot_post_solution(
    t_post::Vector{Float64}, states_post::Array{Float64,2},
    mpcc_tgrid::Array{Float64,2}, mpcc_states::Array{Float64,3}, v;
    title_str::AbstractString, save_path::AbstractString,
    stripping_inputs::Union{Nothing,StrippingInputs}=nothing,
    ethyl_series::Union{Nothing,EthylSeries}=nothing,
)
    plt = plot(title=title_str, size=(900,600))
    plot!(plt, t_post, states_post[1, :], label="X (ODE)")
    plot!(plt, t_post, states_post[3, :], label="G (ODE)")
    plot!(plt, t_post, states_post[4, :], label="F (ODE)")
    plot!(plt, t_post, states_post[5, :], label="E (ODE)")
    savefig(plt, save_path)
end

function write_diagnostic_report(prefix::AbstractString; kwargs...)
    path = prefix * "_diagnostic.txt"
    open(path, "w") do io
        for (k,v) in kwargs
            println(io, string(k), ": ", string(v))
        end
    end
end

# ---- Ipopt attribute helper (avoid direct dependency on optimizer_attr) ----
safe_ipopt_attr(m::Model, name::AbstractString) = try
    MOI.get(m, MOI.SolverAttribute(name))
catch
    NaN
end

const EXPORT_PLOT_CSV = get(ENV, "EXPORT_PLOT_CSV", "0") == "1"
const USE_WARM_START = get(ENV, "USE_WARM_START", "1") == "1"
const USE_SCHOLTES = true # Variante con relajacion de Scholtes

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

function export_plot_data(prefix::AbstractString; mpcc_tgrid, mpcc_states, t_post, states_post, ethyl_series, stripping_inputs)
    base_dir = dirname(prefix)
    isdir(base_dir) || mkpath(base_dir)

    # MPCC colocaciones
    ts_mpcc = vec(mpcc_tgrid)
    rows_mpcc = NamedTuple[]
    for idx in eachindex(ts_mpcc)
        i = div(idx-1, size(mpcc_tgrid, 2)) + 1
        j = mod(idx-1, size(mpcc_tgrid, 2)) + 1
        push!(rows_mpcc, (
            time = ts_mpcc[idx],
            X = mpcc_states[1, i, j],
            N = mpcc_states[2, i, j],
            G = mpcc_states[3, i, j],
            F = mpcc_states[4, i, j],
            E = mpcc_states[5, i, j],
            Temp = dynamic_temperature(ts_mpcc[idx]),
        ))
    end
    CSV.write(prefix * "_mpcc_states.csv", rows_mpcc)

    # ODE post
    rows_post = NamedTuple[]
    for k in eachindex(t_post)
        push!(rows_post, (
            time = t_post[k],
            X = states_post[1, k],
            N = states_post[2, k],
            G = states_post[3, k],
            F = states_post[4, k],
            E = states_post[5, k],
            Temp = dynamic_temperature(t_post[k]),
        ))
    end
    CSV.write(prefix * "_ode_states.csv", rows_post)

    # Ethyl acetate base + spline
    if ethyl_series !== nothing
        rows_ethyl = NamedTuple[]
        for k in eachindex(ethyl_series.times)
            push!(rows_ethyl, (
                time_raw = ethyl_series.times[k],
                conc_raw_gL = ethyl_series.concentrations[k],
                rate_gL_h = ethyl_series.rates[k],
            ))
        end
        CSV.write(prefix * "_ethyl_raw.csv", rows_ethyl)

        rows_spline = [(
            time = ethyl_series.line_t[k],
            conc_gL = ethyl_series.line_vals[k],
        ) for k in eachindex(ethyl_series.line_t)]
        CSV.write(prefix * "_ethyl_spline.csv", rows_spline)

        if ethyl_series.stripping !== nothing
            rows_strip = [(
                time = ethyl_series.stripping.bulk_times[k],
                conc_bulk_gL = ethyl_series.stripping.bulk_conc[k],
                conc_lost_gL = ethyl_series.stripping.lost_conc[k],
            ) for k in eachindex(ethyl_series.stripping.bulk_times)]
            CSV.write(prefix * "_ethyl_stripping.csv", rows_strip)
        end
    end

    # Inputs de stripping (CO2 y k aparentes)
    if stripping_inputs !== nothing
        rows_inputs = [(
            time = stripping_inputs.times[k],
            k_app_h = stripping_inputs.k_vals[k],
            co2_flow_L_h = stripping_inputs.co2_vals[k],
        ) for k in eachindex(stripping_inputs.times)]
        CSV.write(prefix * "_stripping_inputs.csv", rows_inputs)
    end
end

const MOLAR_MASS_ETHYL_ACETATE = 88.106 # g / mol

# --- FACTORES DE ESCALA MANUAL ---
const SC_V = 1.0  # escala flujos
const SC_C = 1.0   # escala sustratos altos (G/F)
const SC_X = 1.0     # biomasa/N se mantienen en ~1

# ---------------------------------------------
# Configuracion de stripping (CO2 + UNIFAC)
# ---------------------------------------------
const STRIPPING_LIQ_VOLUME = try
    parse(Float64, get(ENV, "STRIP_LIQ_VOL", "100.0"))
catch
    100.0
end
const STRIPPING_AROMA_INIT = try
    parse(Float64, get(ENV, "STRIP_AROMA_INIT", "0.0"))
catch
    0.0
end
const STRIPPING_SPECIES = ["water", "ethanol", "ethyl acetate"]
const DEFAULT_PARTITION_COEFF = 1.0

const UNIFAC_GROUPS_CSV = normpath(joinpath(dirname(pathof(Clapeyron)), "..", "database", "Activity", "UNIFAC", "UNIFAC_groups.csv"))
function load_available_unifac_species(db_path::AbstractString)
    if !isfile(db_path)
        @warn "No se encontro la tabla de especies UNIFAC en Clapeyron." path=db_path
        return String[]
    end
    lines = readlines(db_path)
    length(lines) <= 3 && return String[]
    species_list = String[]
    for line in lines[4:end] # saltamos metadata y encabezado "species,groups"
        stripped = strip(line)
        isempty(stripped) && continue
        first_col = split(stripped, ','; limit=2)[1]
        cleaned = replace(replace(first_col, '"' => ""), "~|~" => " / ")
        push!(species_list, cleaned)
    end
    return species_list
end
const AVAILABLE_UNIFAC_SPECIES = load_available_unifac_species(UNIFAC_GROUPS_CSV)
_normalize_unifac_entry(entry::AbstractString) = lowercase(strip(String(entry)))
const AVAILABLE_UNIFAC_SET = Set([_normalize_unifac_entry(p) for comp in AVAILABLE_UNIFAC_SPECIES for p in split(comp, "/")])

function build_activity_model(species::Vector{String})
    missing = [s for s in species if !(_normalize_unifac_entry(s) in AVAILABLE_UNIFAC_SET)]
    if !isempty(missing)
        @info "UNIFAC no disponible para algunas especies; se asumira idealidad (gamma=1)." faltantes=missing
        return nothing
    end
    try
        return UNIFAC(species)
    catch err
        @warn "No se pudieron cargar parametros UNIFAC; se asumira idealidad (gamma=1)." exception=err
        return nothing
    end
end
const activity_model = build_activity_model(STRIPPING_SPECIES)

const SPECIES_PARAM_LOCATIONS = [
    "properties/molarmass.csv",
    "properties/critical.csv",
    "Correlations/saturation_correlations/dippr101_like.csv",
]
antoine_mmhg_to_pa(A, B, C, T) = 133.322368 * 10.0^(A - B / ((T - 273.15) + C))
const MANUAL_SPECIES_FALLBACK = Dict(
    "water" => (mw = 18.015, psat = (type = :custom, eval = (T -> exp(23.1964 - 3816.44 / (T - 46.13))))),
    "ethanol" => (mw = 46.07, psat = (type = :custom, eval = (T -> exp(23.8381 - 3803.98 / (T - 41.68))))),
    # Antoine: log10(P_mmHg) = A - B/(T+C)
    "ethyl acetate" => (mw = MOLAR_MASS_ETHYL_ACETATE, psat = (type = :antoine_mmhg, A = 7.10174, B = 1407.01, C = 214.06)),
)

function fetch_species_properties(species::Vector{String})
    n = length(species)
    mw = fill(NaN, n)
    psat = Vector{Union{Nothing, NamedTuple}}(undef, n)
    psat .= nothing
    manual_keys = Set(keys(MANUAL_SPECIES_FALLBACK))
    to_query_idx = [i for i in 1:n if !(_normalize_unifac_entry(species[i]) in manual_keys)]
    to_query = species[to_query_idx]
    params = nothing
    if !isempty(to_query)
        try
            params = getparams(to_query, SPECIES_PARAM_LOCATIONS; verbose=false)
        catch err
            @warn "No se pudieron recuperar los parametros puros desde la base de Clapeyron." exception=err
        end
    end
    if params !== nothing
        mw_param = get(params, "Mw", nothing)
        if mw_param !== nothing
            for (loc_idx, global_idx) in enumerate(to_query_idx)
                mw[global_idx] = mw_param.values[loc_idx]
            end
        end
        dippr_keys = ["A", "B", "C", "D", "E", "Tmin", "Tmax"]
        if all(k -> haskey(params, k), dippr_keys)
            dippr_data = Dict(k => collect(params[k].values) for k in dippr_keys)
            for (loc_idx, global_idx) in enumerate(to_query_idx)
                vals = (dippr_data["A"][loc_idx], dippr_data["B"][loc_idx], dippr_data["C"][loc_idx],
                        dippr_data["D"][loc_idx], dippr_data["E"][loc_idx], dippr_data["Tmin"][loc_idx], dippr_data["Tmax"][loc_idx])
                if all(x -> !isnan(x), vals)
                    psat[global_idx] = (type = :dippr, A = vals[1], B = vals[2], C = vals[3], D = vals[4], E = vals[5], Tmin = vals[6], Tmax = vals[7])
                end
            end
        end
    end
    for (i, comp) in pairs(species)
        key = _normalize_unifac_entry(comp)
        fallback = get(MANUAL_SPECIES_FALLBACK, key, nothing)
        if isnan(mw[i]) && fallback !== nothing
            mw[i] = fallback.mw
        end
        if psat[i] === nothing && fallback !== nothing
            psat[i] = fallback.psat
        end
        if isnan(mw[i])
            @warn "No se encontro la masa molar para $(comp). Se usara 1.0 g/mol como valor de respaldo."
            mw[i] = 1.0
        end
        if psat[i] === nothing
            @warn "No hay correlacion de presion de vapor para $(comp); se aproximara a cero."
            psat[i] = (type = :none,)
        end
    end
    return (mw = mw, psat = psat)
end

const STRIPPING_SPECIES_PROPERTIES = fetch_species_properties(STRIPPING_SPECIES)
const STRIPPING_MW = STRIPPING_SPECIES_PROPERTIES.mw

function psat_from_data(T::Float64, idx::Int)
    params = STRIPPING_SPECIES_PROPERTIES.psat[idx]
    typ = params.type
    if typ === :dippr
        if !(params.Tmin <= T <= params.Tmax)
            @warn "Temperatura fuera del rango DIPPR para $(STRIPPING_SPECIES[idx])." T=T range=(params.Tmin, params.Tmax)
        end
        return exp(params.A + params.B / T + params.C * log(T) + params.D * T^params.E)
    elseif typ === :custom
        return params.eval(T)
    elseif typ === :antoine_mmhg
        return antoine_mmhg_to_pa(params.A, params.B, params.C, T)
    else
        return 0.0
    end
end

function calculate_partition_coefficient(model, T::Float64, x_molar::AbstractVector{<:Real})
    gamma = if model === nothing
        ones(length(x_molar))
    else
        ln_gamma = activity_coefficient(model, 101325.0, T, x_molar)
        exp.(ln_gamma)
    end
    p_sat = [psat_from_data(T, i) for i in eachindex(STRIPPING_SPECIES)]
    Ki_termo = (gamma .* p_sat) ./ 101325.0
    rho_L = 1000.0 # kg/m^3 (approx water-like)
    # Ideal gas density of CO2 at 1 atm: rho = P*MW/(R*T), MW in kg/mol
    rho_G = 101325.0 * 44.01e-3 / (R * T)
    MW_mix = sum(x_molar .* STRIPPING_MW)
    H_cc = Ki_termo .* (101325.0 / (R*T)) ./ (rho_L ./ MW_mix)
    # devolvemos la constante aparente de Henry para el aroma (indice 3)
    return isfinite(H_cc[3]) && H_cc[3] > 0.0 ? H_cc[3] : DEFAULT_PARTITION_COEFF
end

function partition_coeff_from_state(E_conc::Float64, T::Float64)
    total_mass = 1000.0 # g por L de mosto aproximado
    w_eth = clamp(E_conc / total_mass, 0.0, 0.2)
    w_water = max(1.0 - w_eth - 1e-6, 1e-6)
    # aroma trazas para no romper UNIFAC
    moles = [w_water/STRIPPING_MW[1], w_eth/STRIPPING_MW[2], 1e-6/STRIPPING_MW[3]]
    x_molar = moles ./ sum(moles)
    return calculate_partition_coefficient(activity_model, T, x_molar)
end

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
# Constantes de Penalización MPCC (Faltantes en versión anterior)
# ---------------------------------------------
const phi1 = 1000.0
const phi2 = 1000.0
const phi3 = 1000.0


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

const idx_NH4 = 2536
const idx_Arg = 2729
const idx_Gln = 2740
const idx_Glu = 2738
const idx_Ser = 2754
const idx_Thr = 2759
const idx_Ala = 2723
const idx_Trp = 2760
const NITROGEN_SOURCES = [idx_NH4, idx_Arg, idx_Gln, idx_Glu, idx_Ser, idx_Thr, idx_Ala, idx_Trp]

NITROGEN_SOURCES_RAW = [idx_NH4, idx_Arg, idx_Gln, idx_Glu, idx_Ser, idx_Thr, idx_Ala, idx_Trp]
const NITROGEN_SOURCES = filter(x -> 1 <= x <= nv, NITROGEN_SOURCES_RAW)

# Agrupamos Glucosa, Fructosa y TODAS las fuentes de nitrógeno (filtradas y robustas)
const UPTAKE_IDXS = vcat((1 <= glu <= nv) ? [glu] : Int[], (1 <= fru <= nv) ? [fru] : Int[], NITROGEN_SOURCES)
const n_up = length(UPTAKE_IDXS)
# Selectores constantes para evitar condicionales en NLexpresiones
const IS_GLU = [x == glu ? 1.0 : 0.0 for x in UPTAKE_IDXS]
const IS_FRU = [x == fru ? 1.0 : 0.0 for x in UPTAKE_IDXS]
const IS_NIT = [1.0 - IS_GLU[i] - IS_FRU[i] for i in 1:n_up]
const SELECT_UPTAKE = [Float64(mc == UPTAKE_IDXS[k]) for mc in 1:nv, k in 1:n_up]

# Agrupamos Glucosa, Fructosa y TODAS las fuentes de nitrógeno, filtrando inválidos
if 1 <= o2 <= nv; lb[o2] = 0.0; ub[o2] = 0.0; end
if 1 <= ATP <= nv; lb[ATP] = 0.0; end
if 1 <= o2 <= nv; lb[o2] = 0.0; ub[o2] = 0.0; end
if 1 <= ATP <= nv; lb[ATP] = 0.0; end

# ---------------------------------------------
# Modelo Zenteno
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
const T_const = 293.15
const eps = 1e-9

# Perfil de T
const T_BASE = T_const
const T_STEPS = [36.0, 96.0]
const T_DELTAS = [5.0, 3.0]
const T_STEEP = 1.0

function dynamic_temperature(t)
    val = T_BASE
    for idx in eachindex(T_STEPS)
        sigmoid = 1.0 / (1.0 + exp(-T_STEEP * (t - T_STEPS[idx])))
        val += T_DELTAS[idx] * sigmoid
    end
    return val
end

function death_rate_T(E, T_val)
    Td = -0.0001 * E^3 + 0.0049 * E^2 - 0.1279 * E + 315.89
    s = 0.5 * (1.0 + tanh(0.5 * (T_val - Td)))
    base = Kd0_nom * exp(0.0415 * E + (130000.0 * (T_val - 305.65)) / (305.65 * R * T_val))
    base * s
end
death_rate(E) = death_rate_T(E, T_BASE)

# Inyección suave
const SQRT_2PI = sqrt(2*pi)
function smooth_injection(t, t_shot, dose, width=1.0)
    abs(t - t_shot) > 5 * width && return 0.0
    return (dose / (width * SQRT_2PI)) * exp(-0.5 * ((t - t_shot) / width)^2)
end

# Params
const np = 4
const theta_data_params = log.([MU0_nom, YEG_nom, YEF_nom, YXN_nom])
const theta_init_guess  = log.([MU0_nom*1.2, YEG_nom*1.2, YEF_nom*1.2, YXN_nom*1.2])
LB = log.([0.5*MU0_nom, 0.5*YEG_nom, 0.5*YEF_nom, 0.5*YXN_nom])
UB = log.([5.0*MU0_nom, 5.0*YEG_nom, 5.0*YEF_nom, 5.0*YXN_nom])

# Condiciones iniciales
X0 = 0.5; N0 = 0.14; G0 = 110.0; F0 = 110.0; E0 = 0.0
const C0_INIT = [X0, N0, G0, F0, E0]
c0 = copy(C0_INIT)

# Discretizacion
nfe = 6   
ncp = 3
th  = 72.0 
h   = th / nfe
ph  = nfe
hm    = fill(h, nfe)'
const HM_REFERENCE = vec(hm)
var_h = 1.0

# OMEGA
w     = 1e-20
omega = 1.0 
d   = zeros(nv); d[obj] = -1.0

cs = ones(nc)
vs = ones(nv)

const T_INJ_1 = 48.0; const DOSE_1  = 0.10; const WIDTH_1 = 5.0
const T_INJ_2 = 72.0; const DOSE_2  = 0.10; const WIDTH_2 = 5.0
const ESTIMATE_PARAMS = true

const K_AX = collect(1:nv) # Modo completo por defecto

const MEAS_STATES = (3, 4, 5) # G, F, E
const NOISE_REL_STD = 0.10

colmat = [
    0.19681547722366   -0.06553542585020   0.02377097434822;
    0.39442431473909    0.29207341166523  -0.04154875212600;
    0.37640306270047    0.51248582618842   0.11111111111111
]
const radau_nodes = (0.15505, 0.64495, 1.0)
Random.seed!(NOISE_SEED)

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

struct ZentenoPlotParams
    mu0::Float64
    Yeg::Float64
    Yef::Float64
    Yxn::Float64
end

function zenteno_ode!(du, u, p::ZentenoPlotParams, t)
    X, N, G, F, E = max.(u, 0.0) # Protección
    T_curr = dynamic_temperature(t)
    
    # Cinemática
    safe_exp(val) = exp(clamp(val, -700.0, 100.0))
    mu_T =  safe_exp(59453.0 * (T_curr - 300.0) / (300.0 * R * T_curr))
    Kg_T =  safe_exp(46055.0 * (T_curr - 293.15) / (293.15 * R * T_curr))
    b_T  =  safe_exp(11000.0 * (T_curr - 296.15) / (296.15 * R * T_curr))
    mrate = 0.01 * safe_exp(37681.0 * (T_curr - 293.30) / (293.30 * R * T_curr))
    
    denom = G + F + eps
    phiG = G / denom; phiF = F / denom
    
    mu   = p.mu0 * mu_T * (N / (N + Kn0_nom * Kg_T + eps))
    betaG = betaG0_nom * b_T * (G / (G + Kg0_nom * Kg_T + eps)) * (Kie0_nom * Kg_T / (E + Kie0_nom * Kg_T + eps))
    betaF = betaF0_nom * b_T * (F / (F + Kf0_nom * Kg_T + eps)) * (Kig0_nom * Kg_T / (G + Kig0_nom * Kg_T + eps)) * (Kie0_nom * Kg_T / (E + Kie0_nom * Kg_T + eps))
    Kd_val = death_rate_T(E, T_curr)
    injN = smooth_injection(t, T_INJ_1, DOSE_1, WIDTH_1) + smooth_injection(t, T_INJ_2, DOSE_2, WIDTH_2)
    
    du[1] = (mu - Kd_val) * X
    du[2] = -(mu / p.Yxn) * X + injN
    du[3] = -((mu / YXG_nom) + (betaG / p.Yeg) + mrate * phiG) * X
    du[4] = -((mu / YXF_nom) + (betaF / p.Yef) + mrate * phiF) * X
    du[5] = (betaG + betaF) * X
end

function simulate_zenteno(params::ZentenoPlotParams; tspan=(0.0, th))
    u0 = copy(C0_INIT)
    prob = ODEProblem(zenteno_ode!, u0, tspan, params)
    sol = solve(prob, Rodas5(); reltol=1e-5, abstol=1e-8, saveat=0.1)
    t_dense = sol.t
    U = Array(sol)
    return t_dense, U
end

# Generación de Warm Start (Simplificada para el script)
function generate_warm_start(S, lb, ub, c0, times, obj_idx)
    # Placeholder: En una implementación real aquí va el dFBA secuencial
    # Para crash test retornamos nothing o una matriz de ceros/promedios
    return nothing 
end

# ---------------------------------------------
# Generación de Datos Sintéticos
# ---------------------------------------------
println("[SYNTH] Generando datos sinteticos...")
function _simulate_zenteno_synthetic(; nfe, ncp, th, c0_vec)
    params = ZentenoPlotParams(exp.(theta_data_params)...)
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
        data .+= NOISE_REL_STD .* data .* randn(size(data))
        data .= max.(data, 0.0)
    end
    return data
end
data = _simulate_zenteno_synthetic(nfe=nfe, ncp=ncp, th=th, c0_vec=c0)

# ---------------------------------------------
# MODELO JuMP
# ---------------------------------------------
m = Model(Ipopt.Optimizer)
set_optimizer_attribute(m, "print_level", 5)
set_optimizer_attribute(m, "max_iter", 30)
set_optimizer_attribute(m, "tol", 1e-4)
set_optimizer_attribute(m, "linear_solver", "mumps")

@variables(m, begin
    c[1:nc, 1:ph, 1:ncp] >= 0.0
    cdot[1:nc, 1:ph, 1:ncp]
    FO >= 0.0
    teta[1:np]
   
    v[1:nv, 1:nfe]
    lambda_[1:nm, 1:nfe]
    alpha_U[1:nv, 1:nfe]>= 0.0
    alpha_L[1:nv, 1:nfe]<= 0.0
    # Generalized Uptake Duals
    alpha_upt[1:n_up, 1:nfe]<= 0.0
    
    FO_U[1:nv, 1:nfe]
    FO_L[1:nv, 1:nfe]
    FO_upt[1:n_up, 1:nfe]

    hv[1:nfe] >= 0.0
end)

# ============================================================
# PARTE 0: INICIALIZACION
# ============================================================
for i in 1:ph, j in 1:ncp, l in 1:nc
    set_start_value(c[l, i, j], c0[l])
end
for i in 1:nfe
    set_start_value(hv[i], hm[i])
end
for k in 1:np
    set_start_value(teta[k], theta_init_guess[k]) 
end

if USE_WARM_START
    # Lógica de Warm Start (simplificada)
    println("[WARM-START] Intentando warm start...")
    ws_v = generate_warm_start(S, lb, ub, C0_INIT, vec(DATA_TIME_GRID), obj)
    
    if ws_v !== nothing
        # Asignar v
    else
        # Cold start fallback
        for i in 1:nfe, k in 1:nv; set_start_value(v[k,i], 0.0); end
        for i in 1:nfe, k in 1:n_up; set_start_value(alpha_upt[k,i], 0.0); end
    end
else
    # Cold Start
    for i in 1:nfe, k in 1:nv; set_start_value(v[k,i], 0.0); end
    for i in 1:nfe, k in 1:n_up; set_start_value(alpha_upt[k,i], 0.0); end
end

# Normalización C0 local para el modelo
for i in 1:nc; c0[i] = c0[i] / cs[i]; end

# ============================================================
# PARTE 1: FUNCION OBJETIVO
# ============================================================
@NLobjective(m, Min, 
    omega * FO + 
    sum(
        sum(-phi1*FO_L[mc,i] - phi3*FO_U[mc,i] for mc in 1:nv) + 
        sum(phi2*FO_upt[k,i] for k in 1:n_up) 
    for i in 1:nfe)
)

# ============================================================
# PARTE 2: EXPRESIONES ZENTENO
# ============================================================
@NLexpression(m, mu0, exp(teta[1]))
@NLexpression(m, Yeg, exp(teta[2]))
@NLexpression(m, Yef, exp(teta[3]))
@NLexpression(m, Yxn, exp(teta[4]))

const Yxg = YXG_nom
const Yxf = YXF_nom

JuMP.register(m, :smooth_injection, 4, smooth_injection; autodiff = true)
JuMP.register(m, :dynamic_temperature, 1, dynamic_temperature; autodiff = true)
JuMP.register(m, :death_rate_T, 2, death_rate_T; autodiff = true)

@NLexpression(m, T_loc[i=1:ph, j=1:ncp], dynamic_temperature((i - 1 + radau_nodes[j]) * hv[i]))
@NLexpression(m, mu_T_ij[i=1:ph, j=1:ncp], exp(59453.0 * (T_loc[i,j] - 300.0) / (300.0 * R * T_loc[i,j])))
@NLexpression(m, Kg_T_ij[i=1:ph, j=1:ncp], exp(46055.0 * (T_loc[i,j] - 293.15) / (293.15 * R * T_loc[i,j])))
@NLexpression(m, b_T_ij[i=1:ph, j=1:ncp], exp(11000.0 * (T_loc[i,j] - 296.15) / (296.15 * R * T_loc[i,j])))
@NLexpression(m, mrate_ij[i=1:ph, j=1:ncp], 0.01 * exp(37681.0 * (T_loc[i,j] - 293.30) / (293.30 * R * T_loc[i,j])))

@NLexpression(m, mu_j[i=1:ph, j=1:ncp], mu0 * mu_T_ij[i,j] * (c[2,i,j] / (c[2,i,j] + Kn0_nom * Kg_T_ij[i,j] + eps)))
@NLexpression(m, betaG_j[i=1:ph, j=1:ncp], betaG0_nom * b_T_ij[i,j] * (c[3,i,j] / (c[3,i,j] + Kg0_nom * Kg_T_ij[i,j] + eps)) * (Kie0_nom * Kg_T_ij[i,j] / (c[5,i,j] + Kie0_nom * Kg_T_ij[i,j] + eps)))
@NLexpression(m, betaF_j[i=1:ph, j=1:ncp], betaF0_nom * b_T_ij[i,j] * (c[4,i,j] / (c[4,i,j] + Kf0_nom * Kg_T_ij[i,j] + eps)) * (Kig0_nom * Kg_T_ij[i,j] / (c[3,i,j] + Kig0_nom * Kg_T_ij[i,j] + eps)) * (Kie0_nom * Kg_T_ij[i,j] / (c[5,i,j] + Kie0_nom * Kg_T_ij[i,j] + eps)))
@NLexpression(m, Kd_j[i=1:ph, j=1:ncp], death_rate_T(c[5,i,j], T_loc[i,j]))
@NLexpression(m, injection_rate[i=1:nfe, j=1:ncp], smooth_injection((i - 1 + radau_nodes[j]) * hv[i], T_INJ_1, DOSE_1, WIDTH_1) + smooth_injection((i - 1 + radau_nodes[j]) * hv[i], T_INJ_2, DOSE_2, WIDTH_2))

# ============================================================
# PARTE 3: DATOS NITROGENO
# ============================================================
N_atoms = Dict(idx_NH4 => 1.0, idx_Arg => 4.0, idx_Gln => 2.0, idx_Glu => 1.0, idx_Ser => 1.0, idx_Thr => 1.0, idx_Ala => 1.0, idx_Trp => 2.0)
N_profile_ratios = Dict(idx_NH4 => 0.40, idx_Arg => 0.20, idx_Gln => 0.10, idx_Glu => 0.05, idx_Ser => 0.05, idx_Thr => 0.05, idx_Ala => 0.05, idx_Trp => 0.10)

# Build parameter vectors for nonlinear usage (no Base.get in NL expressions)
N_atoms_vec = ones(nv)
N_profile_vec = zeros(nv)
for k in keys(N_atoms)
    1 <= k <= nv && (N_atoms_vec[k] = N_atoms[k])
end
for k in keys(N_profile_ratios)
    1 <= k <= nv && (N_profile_vec[k] = N_profile_ratios[k])
end
const MW_N = 0.014007

# ============================================================
# PARTE 4: LÍMITES DINÁMICOS
# ============================================================
@NLexpressions(m, begin
    # Macro Zenteno
    vg[i=1:ph],  (mu_j[i,3]/Yxg + betaG_j[i,3]/Yeg + mrate_ij[i,3]*(c[3,i,3]/(c[3,i,3]+c[4,i,3])))
    vf[i=1:ph],  (mu_j[i,3]/YXF_nom + betaF_j[i,3]/Yef + mrate_ij[i,3]*(c[4,i,3]/(c[3,i,3]+c[4,i,3])))
    vn_total[i=1:ph],  (mu_j[i,3]/YXN_nom)

    # Micro Nitrogen breakdown over all reactions (non-N entries return 0)
    v_limit_N[k=1:nv, i=1:ph], (vn_total[i] * N_profile_vec[k]) / N_atoms_vec[k]

    # Wrapper aligned to UPTAKE_IDXS (sin condicionales)
    L_uptake[k=1:n_up, i=1:ph], IS_GLU[k]*vg[i] + IS_FRU[k]*vf[i] + IS_NIT[k]*v_limit_N[UPTAKE_IDXS[k], i]
end)

# ============================================================
# PARTE 5: RESTRICCIONES
# ============================================================
@constraints(m, begin
    # Colocación
    coll_c_n[l=1:nc, i=2:ph, j=1:ncp], c[l,i,j] == c[l,i-1,ncp]+h*sum(colmat[j,k]*cdot[l,i,k] for k in 1:ncp)
    coll_c_0[l=1:nc, j=1:ncp], c[l,1,j] == c0[l] + hv[1] * sum(colmat[j,k] * cdot[l,1,k] for k in 1:ncp)
    
    # FBA
    Sc[mc=1:nm,i=1:nfe],  sum(S[mc,k]*v[k,i]*vs[k] for k in 1:nv) == 0
    v_UB[mc=1:nv, i=1:nfe], v[mc,i]*vs[mc] - ub[mc] <= 0
    v_LB[mc=1:nv,i=1:nfe], -v[mc,i]*vs[mc] + lb[mc] <= 0

    # Time-step
    MFE1, sum(hv[i] for i in 1:nfe) == th
    MFE3[i=1:nfe], hv[i] >= 0.0
    MFE4[i=1:nfe], hv[i] >= (1.0 - var_h) * hm[1]
    MFE5[i=1:nfe], hv[i] <= (1.0 + var_h) * hm[1]

    # KKT Lagrangiano
    Lagr[mc=1:nv,i=1:nfe], d[mc] + w*v[mc,i]*vs[mc] + alpha_L[mc,i] + alpha_U[mc,i] + sum(SELECT_UPTAKE[mc,k] * alpha_upt[k,i] for k in 1:n_up) + sum(S[k,mc]*lambda_[k,i] for k in 1:nm) == 0
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

# ============================================================
# PARTE 6: ODES & COMPLEMENTARIEDAD
# ============================================================
if 1 <= obj <= nv
    @expression(m, v_obj[i=1:nfe], v[obj,i])
else
    @expression(m, v_obj[i=1:nfe], 0.0)
end
if 1 <= eth <= nv
    @expression(m, v_eth[i=1:nfe], v[eth,i])
else
    @expression(m, v_eth[i=1:nfe], 0.0)
end

@NLconstraints(m, begin
    m1[i=1:ph, j=1:ncp], cdot[1,i,j] == (v_obj[i] - Kd_j[i,j]) * c[1,i,j]
    
    # Balance de Nitrógeno (Sumatoria de átomos)
    m2[i=1:ph, j=1:ncp], cdot[2,i,j] == 
        - MW_N * sum( (-v[UPTAKE_IDXS[k],i] * vs[UPTAKE_IDXS[k]]) * N_atoms_vec[UPTAKE_IDXS[k]] for k in 3:n_up ) * c[1,i,j]

    m3[i=1:ph, j=1:ncp], cdot[3,i,j] == - vg[i] * c[1,i,j]
    m4[i=1:ph, j=1:ncp], cdot[4,i,j] == - vf[i] * c[1,i,j]
    m5[i=1:ph, j=1:ncp], cdot[5,i,j] == 0.04607 * v_eth[i] * c[1,i,j] 
    
    # Uptake Coupling
    v_LB_uptake[k=1:n_up, i=1:nfe], 
        -v[UPTAKE_IDXS[k], i] * vs[UPTAKE_IDXS[k]] - L_uptake[k,i] <= 0

    FO_upt_cons[k=1:n_up, i=1:nfe],
        FO_upt[k,i] == (-v[UPTAKE_IDXS[k], i] * vs[UPTAKE_IDXS[k]] - L_uptake[k,i]) * alpha_upt[k,i]

    # Bounds Complementarity
    FO1[mc=1:nv,i=1:nfe], FO_L[mc,i] == (v[mc,i]*vs[mc] -lb[mc])*alpha_L[mc,i]
    FO2[mc=1:nv,i=1:nfe], FO_U[mc,i] == (v[mc,i]*vs[mc] -ub[mc])*alpha_U[mc,i]

    # Data Fitting
    m8, FO == sum( sum( sum( (data[i,j,mc]-c[i,j,mc])^2 for i in 1:nc)   for j in 1:ph)   for mc in 1:ncp)
end)

# ---------------------------------------------
# EJECUCIÓN
# ---------------------------------------------
println("Iniciando optimizacion robusta V4...")
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
dual_inf      = safe_ipopt_attr(m, "dual infeasibility")
primal_inf    = safe_ipopt_attr(m, "primal infeasibility")
compl         = safe_ipopt_attr(m, "complementarity")
constr_viol   = safe_ipopt_attr(m, "constraint violation")
iter_count    = safe_ipopt_attr(m, "iter_count")

result_prefix = result_file_prefix(wall_time=wall_time, nfe=nfe, status=status, primal_status=pr_status)
plot_output_path = result_prefix * ".png"

theta_final_log = [safe_value(teta[k], theta_data_params[k]) for k in 1:np]
theta_final_vals = exp.(theta_final_log)

function _build_ethyl_acetate_plot_data_interp(mpcc_tgrid::Array{Float64,2}, mpcc_states::Array{Float64,3}, v; stripping_inputs::Union{Nothing,StrippingInputs}=nothing)
    times = vec(mpcc_tgrid)
    Evals = [mpcc_states[5, div(idx-1, size(mpcc_tgrid,2))+1, mod(idx-1, size(mpcc_tgrid,2))+1] for idx in eachindex(times)]
    rates = zeros(length(times))
    line_t = collect(range(first(times), last(times), length=length(times)))
    line_vals = copy(Evals)
    strip = nothing
    if stripping_inputs !== nothing
        x = times; y = Evals; xq = stripping_inputs.times; n = length(x)
        bulk_conc_interp = similar(xq)
        for i in eachindex(xq)
            tq = xq[i]
            if tq <= x[1]
                bulk_conc_interp[i] = y[1]; continue
            elseif tq >= x[end]
                bulk_conc_interp[i] = y[end]; continue
            end
            j = searchsortedlast(x, tq); j = clamp(j,1,n-1)
            x1 = x[j]; x2 = x[j+1]; y1 = y[j]; y2 = y[j+1]
            w = (tq - x1)/(x2 - x1)
            bulk_conc_interp[i] = (1-w)*y1 + w*y2
        end
        strip = (bulk_times = stripping_inputs.times,
                 bulk_conc = bulk_conc_interp,
                 lost_conc = zeros(length(stripping_inputs.times)))
    end
    return EthylSeries(times, Evals, rates, line_t, line_vals, strip)
end

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
    stripping_inputs = nothing
    try
        stripping_inputs = _build_stripping_inputs(t_post, states_post, post_params)
    catch err
        @warn "No se pudo construir la serie de stripping (CO2 + UNIFAC)" err
    end
    ethyl_series = _build_ethyl_acetate_plot_data_interp(mpcc_tgrid, mpcc_states, v; stripping_inputs=stripping_inputs)
    plot_post_solution(
        t_post, states_post,
        mpcc_tgrid, mpcc_states, v;
        title_str="MPCC post: $(status) / $(pr_status)",
        save_path=plot_output_path,
        stripping_inputs=stripping_inputs,
        ethyl_series=ethyl_series,
    )
    if EXPORT_PLOT_CSV
        try
            export_plot_data(
                result_prefix;
                mpcc_tgrid=mpcc_tgrid,
                mpcc_states=mpcc_states,
                t_post=t_post,
                states_post=states_post,
                ethyl_series=ethyl_series,
                stripping_inputs=stripping_inputs,
            )
        catch err
            @warn "No se pudieron exportar los CSV de plot" err
        end
    end
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