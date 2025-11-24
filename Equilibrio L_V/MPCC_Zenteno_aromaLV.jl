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
using Clapeyron
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

# Perfil de temperatura por defecto (horas, °C): 0-24h 15°C, 24-48h 18°C, 48+ 23°C
const TEMP_PROFILE_STEPS_C = [
    (0.0, 15.0),
    (24.0, 18.0),
    (48.0, 23.0),
]
const TEMP_TRANSITION_WIDTH_H = 0.5 # suavizado en las transiciones para evitar discontinuidades

# ---------------------------------------------
# Termodinámica para stripping de aromas (Clapeyron)
# ---------------------------------------------
const AROMA_SPECIES = ["water", "ethanol", "ethylacetate"]

const UNIFAC_GROUPS_CSV = normpath(joinpath(dirname(pathof(Clapeyron)), "..", "database", "Activity", "UNIFAC", "UNIFAC_groups.csv"))
const ISOAMYL_USER_GROUPS = joinpath(@__DIR__, "isoamyl_acetate_unifac.csv")

function _normalize_unifac_entry(entry::AbstractString)
    return lowercase(strip(String(entry)))
end

function _load_available_unifac_species(db_path::AbstractString)
    if !isfile(db_path)
        @warn "No se encontro la tabla de especies UNIFAC en Clapeyron." path=db_path
        return String[]
    end
    lines = readlines(db_path)
    length(lines) <= 3 && return String[]
    species_list = String[]
    for line in lines[4:end]
        stripped = strip(line)
        isempty(stripped) && continue
        first_col = split(stripped, ','; limit=2)[1]
        cleaned = replace(replace(first_col, '"' => ""), "~|~" => " / ")
        push!(species_list, cleaned)
    end
    return species_list
end

const USER_UNIFAC_SPECIES = _load_available_unifac_species(ISOAMYL_USER_GROUPS)
const AVAILABLE_UNIFAC_SPECIES = vcat(_load_available_unifac_species(UNIFAC_GROUPS_CSV), USER_UNIFAC_SPECIES)

function _build_unifac_model(species::Vector{String})
    synonyms = String[]
    for entry in AVAILABLE_UNIFAC_SPECIES
        parts = split(entry, "/")
        for p in parts
            norm = _normalize_unifac_entry(p)
            push!(synonyms, norm)
            push!(synonyms, replace(norm, " " => ""))
        end
    end
    available = Set(synonyms)
    missing = [s for s in species if !(_normalize_unifac_entry(s) in available)]
    if !isempty(missing)
        @info "UNIFAC no disponible para algunos componentes; se usara idealidad (gamma=1)." faltantes=missing
        return nothing
    end
    try
        group_locs = isfile(ISOAMYL_USER_GROUPS) ? [ISOAMYL_USER_GROUPS] : String[]
        return UNIFAC(species; group_userlocations=group_locs)
    catch err
        @warn "No se pudieron construir parametros UNIFAC; se usara idealidad." exception=err
        return nothing
    end
end

const MODEL_UNIFAC = nothing#_build_unifac_model(AROMA_SPECIES)

const SPECIES_PARAM_LOCATIONS = [
    "properties/molarmass.csv",
    "properties/critical.csv",
    "Correlations/saturation_correlations/dippr101_like.csv",
]

const MANUAL_SPECIES_FALLBACK = Dict(
    "water" => (mw = 18.015, psat = (type = :custom, eval = (T -> exp(23.1964 - 3816.44 / (T - 46.13))))),
    "ethanol" => (mw = 46.07, psat = (type = :custom, eval = (T -> exp(23.8381 - 3803.98 / (T - 41.68))))),
)

function _fetch_species_properties(species::Vector{String})
    n = length(species)
    mw = fill(NaN, n)
    psat = Vector{Union{Nothing, NamedTuple}}(undef, n)
    psat .= nothing
    manual_keys = Set(keys(MANUAL_SPECIES_FALLBACK))
    to_query_idx = [i for i in 1:n if !(lowercase(strip(species[i])) in manual_keys)]
    to_query = species[to_query_idx]
    params = nothing
    if !isempty(to_query)
        try
            params = getparams(
                to_query,
                SPECIES_PARAM_LOCATIONS;
                verbose=false,
                ignore_missing_singleparams=["A","B","C","D","E","Tmin","Tmax"],
            )
        catch err
            @warn "No se pudieron recuperar parametros puros desde Clapeyron." exception=err
        end
    end
    if params !== nothing
        mw_param = get(params, "Mw", nothing)
        if mw_param !== nothing
            for (loc_idx, global_idx) in enumerate(to_query_idx)
                if !mw_param.ismissingvalues[loc_idx]
                    mw[global_idx] = mw_param.values[loc_idx]
                end
            end
        end
        dippr_keys = ["A","B","C","D","E","Tmin","Tmax"]
        if all(k -> haskey(params, k), dippr_keys)
            dippr_data = Dict(k => collect(params[k].values) for k in dippr_keys)
            for (loc_idx, global_idx) in enumerate(to_query_idx)
                if any(params[k].ismissingvalues[loc_idx] for k in dippr_keys)
                    continue
                end
                vals = (dippr_data["A"][loc_idx], dippr_data["B"][loc_idx], dippr_data["C"][loc_idx],
                        dippr_data["D"][loc_idx], dippr_data["E"][loc_idx], dippr_data["Tmin"][loc_idx], dippr_data["Tmax"][loc_idx])
                if all(x -> !isnan(x), vals)
                    psat[global_idx] = (type=:dippr, A=vals[1], B=vals[2], C=vals[3], D=vals[4], E=vals[5], Tmin=vals[6], Tmax=vals[7])
                end
            end
        end
    end
    for (i, comp) in pairs(species)
        name_key = lowercase(strip(comp))
        fallback = get(MANUAL_SPECIES_FALLBACK, name_key, nothing)
        if isnan(mw[i]) && fallback !== nothing
            mw[i] = fallback.mw
        end
        if psat[i] === nothing && fallback !== nothing
            psat[i] = fallback.psat
        end
        if isnan(mw[i])
            @warn "No se encontro masa molar para $(comp); se usa 1.0 g/mol."
            mw[i] = 1.0
        end
        if psat[i] === nothing
            @warn "No hay correlacion psat para $(comp); se aproxima a cero."
            psat[i] = (type=:none,)
        end
    end
    return (mw=mw, psat=psat)
end

const AROMA_SPECIES_PROPS = _fetch_species_properties(AROMA_SPECIES)
const AROMA_MW = AROMA_SPECIES_PROPS.mw

function psat_from_data(T::Float64, idx::Int)
    params = AROMA_SPECIES_PROPS.psat[idx]
    typ = params.type
    if typ === :dippr
        if !(params.Tmin <= T <= params.Tmax)
            @warn "Temperatura fuera del rango DIPPR para $(AROMA_SPECIES[idx])." T=T range=(params.Tmin, params.Tmax)
        end
        return exp(params.A + params.B / T + params.C * log(T) + params.D * T^params.E)
    elseif typ === :custom
        return params.eval(T)
    else
        return 0.0
    end
end

function calculate_partition_coefficient(model, T, x_molar)
    gamma_cap = 1e3
    gamma = ones(length(x_molar))
    if model !== nothing
        try
            ln_gamma = activity_coefficient(model, P_ATM, T, x_molar)
            if any(x -> !isfinite(x), ln_gamma)
                @warn "UNIFAC devolvio NaN/Inf; se usa idealidad." T=T x=x_molar ln_gamma=ln_gamma
            else
                gamma .= clamp.(exp.(ln_gamma), 0.0, gamma_cap)
            end
        catch err
            @warn "Fallo calculo gamma; se usa idealidad." exception=err T=T x=x_molar
        end
    end
    p_sat = [psat_from_data(T, i) for i in eachindex(AROMA_SPECIES)]
    if any(isnan, p_sat)
        @warn "psat NaN detectado; se fuerzan a cero." T=T p_sat=p_sat
        p_sat = map(x -> isnan(x) ? 0.0 : x, p_sat)
    end
    Ki_termo = (gamma .* p_sat) ./ P_ATM
    Ki_termo = map(x -> isfinite(x) ? clamp(x, 0.0, 100.0) : 0.0, Ki_termo)
    rho_L = 1000.0
    MW_mix = sum(x_molar .* AROMA_MW)
    if !isfinite(MW_mix) || MW_mix <= 0
        @warn "MW_mix no finito; se usa valor de respaldo." MW_mix=MW_mix
        MW_mix = 50.0
    end
    H_cc = Ki_termo .* (P_ATM / (R*T)) ./ (rho_L ./ MW_mix)
    return isfinite(H_cc[3]) ? H_cc[3] : 0.0
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
const P_ATM = 101325.0 # Pa
const T_const = try parse(Float64, get(ENV, "T_CONST", "293.15")) catch; 293.15 end
const FERMENTER_VOLUME_L = try parse(Float64, get(ENV, "FERMENTER_VOL_L", "100.0")) catch; 100.0 end
const eps = 1e-9

function _smooth_step(t, t0, width)
    0.5 * (1.0 + tanh((t - t0) / max(width, 1e-6)))
end

function temperature_profile_builder(steps)
    sorted = sort(steps; by=x->x[1])
    # tomamos tres escalones: T1 hasta t1, T2 hasta t2, T3 en adelante
    t1, T1 = sorted[1]
    t2, T2 = sorted[min(2, length(sorted))]
    t3, T3 = sorted[min(3, length(sorted))]
    w = TEMP_TRANSITION_WIDTH_H
    function Tfun(t)
        s1 = _smooth_step(t, t1, w)  # sube de 0 a 1 alrededor de t1
        s2 = _smooth_step(t, t2, w)  # sube de 0 a 1 alrededor de t2
        # mezcla suave: T1 + (T2-T1)*s1 + (T3-T2)*s2
        Tc = T1 + (T2 - T1) * s1 + (T3 - T2) * s2
        return Tc + 273.15
    end
    return Tfun
end

const T_PROFILE = temperature_profile_builder(TEMP_PROFILE_STEPS_C)

death_rate(E) = begin
    Td = -0.0001 * E^3 + 0.0049 * E^2 - 0.1279 * E + 315.89
    s = 0.5 * (1.0 + tanh(0.5 * (T_const - Td)))
    base = Kd0_nom * exp(0.0415 * E + (130000.0 * (T_const - 305.65)) / (305.65 * R * T_const))
    base * s
end
death_rate_T(E, T) = begin
    Td = -0.0001 * E^3 + 0.0049 * E^2 - 0.1279 * E + 315.89
    s = 0.5 * (1.0 + tanh(0.5 * (T - Td)))
    base = Kd0_nom * exp(0.0415 * E + (130000.0 * (T - 305.65)) / (305.65 * R * T))
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
th  = 72.0
h   = th / nfe
ph  = nfe
hm    = fill(h, nfe)'
const HM_REFERENCE = vec(hm)
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

function _build_reduced_axes()
    if !REDUCED_MODE || reduced_sets === nothing
        return collect(1:nv), collect(1:nm)
    end
    A_sets, C_sets, _ = reduced_sets
    K = Int[]
    for i in 1:nfe
        if i <= length(A_sets)
            append!(K, A_sets[i])
        end
        if i <= length(C_sets)
            append!(K, C_sets[i])
        end
    end
    append!(K, (glu, fru))
    K = unique(K)
    sort!(K)
    M = Int[]
    for mc in 1:nm
        for k in K
            if S[mc, k] != 0.0
                push!(M, mc)
                break
            end
        end
    end
    M = unique(M)
    sort!(M)
    return K, M
end

const K_AX, M_AX = _build_reduced_axes()
const FLUX_INDEX_SET = (!REDUCED_MODE || reduced_sets === nothing) ? collect(1:nv) : K_AX
const MET_INDEX_SET = (!REDUCED_MODE || reduced_sets === nothing) ? collect(1:nm) : M_AX

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

function configure_custom_ipopt()
    get(ENV, "USE_DEFAULT_IPOPT", "0") == "1" && return
    function parent_chain(path::String; max_depth::Int=8)
        acc = String[]
        cur = abspath(path)
        for _ in 1:max_depth
            push!(acc, cur)
            parent = dirname(cur)
            parent == cur && break
            cur = parent
        end
        return acc
    end

    root = get(ENV, "IPOPT_DLL_DIR", get(ENV, "PANUA_IPOPT_DIR", get(ENV, "PANUA_IPOPT_ROOT", "")))
    if isempty(strip(root))
        for p in parent_chain(BASE_DIR)
            dirs = filter(d -> startswith(lowercase(basename(d)), "panua-ipopt"), readdir(p; join=true))
            for d in dirs
                if isfile(joinpath(d, "lib", "libipopt.dll")) || isfile(joinpath(d, "libipopt.dll"))
                    root = d
                    ENV["PANUA_IPOPT_ROOT"] = root
                    println("[IPOPT] Auto-detected PANUA_IPOPT_ROOT=", root)
                    break
                end
            end
            !isempty(strip(root)) && break
        end
    end
    if isempty(strip(root))
        return
    end

    cand_lib = isfile(joinpath(root, "lib", "libipopt.dll")) ? joinpath(root, "lib", "libipopt.dll") : joinpath(root, "libipopt.dll")
    isfile(cand_lib) || return
    dll_dir = dirname(cand_lib)
    ENV["JULIA_IPOPT_LIBRARY_PATH"] = cand_lib
    ENV["PATH"] = dll_dir * ";" * get(ENV, "PATH", "")
    ipopt_root_bin = joinpath(root, "bin")
    if isdir(ipopt_root_bin)
        ENV["PATH"] = ipopt_root_bin * ";" * ENV["PATH"]
    end
    if dll_dir != root
        ENV["PATH"] = root * ";" * ENV["PATH"]
    end
    println("[IPOPT] Custom Ipopt library: ", cand_lib)
end
configure_custom_ipopt()

# ---------------------------------------------
# Herramientas de simulacion/plot
# ---------------------------------------------
const STATE_LABELS = ("X", "N", "G", "F", "E")
const STATE_COLORS = (:royalblue, :forestgreen, :firebrick, :darkorange, :purple)
const STATE_MIN_CONC = (
    1e-6,  # X: biomasa no debe anularse numéricamente
    1e-5,  # N: nitrógeno puede agotarse pero mantenemos piso suave
    1e-4,  # G: glucosa
    1e-4,  # F: fructosa
    1e-6,  # E: etanol
)
const FREEZE_DEPLETED_STATES = get(ENV, "FREEZE_DEPLETED_STATES", "1") == "1"
const FREEZE_NITROGEN = get(ENV, "FREEZE_NITROGEN", "1") == "1"
const STATE_FREEZE_ELIGIBLE = (
    false,
    FREEZE_NITROGEN,
    true,
    true,
    false,
)
const STATE_FREEZE_THRESH = (
    1.0e-5,
    5.0e-5,
    5.0e-4,
    5.0e-4,
    1.0e-5,
)
const STATE_FREEZE_REL_FRAC = (
    0.0,
    0.05,
    0.02,
    0.02,
    0.0,
)
const STATE_FREEZE_MIN_CONSEC_FE = 2

struct ZentenoPlotParams
    mu0::Float64
    Yeg::Float64
    Yef::Float64
    Tfun::Function
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
    T = p.Tfun(t)
    mu_T =  exp(59453.0 * (T - 300.0) / (300.0 * R * T))
    Kg_T =  exp(46055.0 * (T - 293.15) / (293.15 * R * T))
    b_T  =  exp(11000.0 * (T - 296.15) / (296.15 * R * T))
    mrate = 0.01 * exp(37681.0 * (T - 293.30) / (293.30 * R * T))
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
    s_sw = 0.5 * (1.0 + tanh(0.5 * (T - Td)))
    Kd_val = Kd0_nom * exp(0.0415 * E + (130000.0 * (T - 305.65)) / (305.65 * R * T)) * s_sw
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

function _build_ethyl_acetate_series(mpcc_tgrid, mpcc_states, v_var)
    prod_series = _build_ethyl_acetate_production_series(mpcc_tgrid, mpcc_states, v_var)
    prod_series === nothing && return nothing
    times = prod_series.times
    rates = prod_series.rates_mmol
    isempty(times) && return nothing
    concentrations = _cumulative_trapezoid(times, rates)
    return (times=times, concentrations=concentrations)
end

function _build_ethyl_acetate_production_series(mpcc_tgrid, mpcc_states, v_var)
    if mpcc_tgrid === nothing || mpcc_states === nothing
        return nothing
    end
    rxn_idx = ETHYL_ACETATE_EX_RXN
    if rxn_idx === nothing
        @warn "No se encontro ninguna reaccion asociada a la fila $(ETHYL_ACETATE_ROW) de S"
        return nothing
    end
    nfe = size(mpcc_states, 2)
    flux_vals = Vector{Float64}(undef, nfe)
    for i in 1:nfe
        flux_var = try
            v_var[rxn_idx, i]
        catch err
            @warn "El MPCC no contiene la reaccion de ethyl acetate en el conjunto de variables" rxn_idx err
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
    return (times=times, rates_mmol=rates, rates_g=rates .* (MOLAR_MASS_ETHYL_ACETATE / 1000.0))
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
                @warn "Concentracion de biomasa no definida al construir la serie de ethyl acetate" collocation=(i, j)
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
        concentrations[idx] = concentrations[idx - 1] + 0.5 * (Float64(values[idx]) + Float64(values[idx - 1])) * dt
    end
    return concentrations
end

function _deduplicate_series(times::Vector{Float64}, values::Vector{Vector{Float64}}; atol::Float64=1e-8)
    isempty(times) && return times, values
    unique_times = Float64[]
    unique_vals = [Float64[] for _ in values]
    last_t = NaN
    for (i, t) in enumerate(times)
        if i == 1 || abs(t - last_t) > atol
            push!(unique_times, t)
            for (arr, v) in zip(unique_vals, values)
                push!(arr, v[i])
            end
            last_t = t
        else
            for (arr, v) in zip(unique_vals, values)
                arr[end] = v[i]
            end
        end
    end
    return unique_times, unique_vals
end

function _build_piecewise_linear(times::Vector{Float64}, values::Vector{Float64}; default::Float64=0.0)
    n = length(times)
    if n == 0
        return (t -> default)
    elseif n == 1
        val = values[1]
        return (t -> val)
    end
    function f(t)
        if t <= times[1]
            return values[1]
        elseif t >= times[end]
            return values[end]
        end
        k = clamp(searchsortedlast(times, t), 1, n-1)
        t0 = times[k]; t1 = times[k+1]
        v0 = values[k]; v1 = values[k+1]
        dt = t1 - t0
        dt == 0.0 && return v0
        return v0 + (v1 - v0) * (t - t0) / dt
    end
    return f
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

function _build_cubic_interpolant(times::Vector{Float64}, values::Vector{Float64})
    n = length(times)
    if n == 0
        return (t -> 0.0)
    elseif n == 1
        val = values[1]
        return (t -> val)
    end
    coeffs = _natural_cubic_coefficients(times, values)
    tmin, tmax = times[1], times[end]
    return t -> _evaluate_natural_cubic(coeffs, clamp(t, tmin, tmax))
end

function _build_state_interpolator(mpcc_tgrid, mpcc_states)
    nt = length(mpcc_tgrid)
    flat_times = vec(mpcc_tgrid)
    perm = sortperm(flat_times)
    sorted_times = Float64.(flat_times[perm])
    state_vals = [Float64.(reshape(mpcc_states[l, :, :], nt))[perm] for l in 1:nc]
    clean_times, clean_vals = _deduplicate_series(sorted_times, state_vals)
    interpolants = [_build_cubic_interpolant(clean_times, clean_vals[l]) for l in 1:nc]
    tspan = (clean_times[1], clean_times[end])
    function state_at(t)
        tt = clamp(t, tspan[1], tspan[2])
        vals = similar(C0_INIT)
        for l in 1:nc
            vals[l] = interpolants[l](tt)
        end
        return vals
    end
    return state_at, tspan
end

function _build_co2_flow_function(state_fun, params::ZentenoPlotParams; V_liq::Float64, tspan::Tuple{Float64,Float64})
    function co2_func(t)
        tt = clamp(t, tspan[1], tspan[2])
        u = state_fun(tt)
        du = zeros(length(u))
        zenteno_ode!(du, u, params, tt)
        mass_CO2_h = max(du[5], 0.0) * V_liq * 0.95
        Tloc = params.Tfun(tt)
        vol_CO2_h = (mass_CO2_h / 44.01) * R * Tloc / P_ATM * 1000.0
        return max(vol_CO2_h, 0.0)
    end
    return co2_func
end

function _build_partition_function(state_fun; activity_model=MODEL_UNIFAC)
    function part_func(t)
        state = state_fun(t)
        E = max(state[5], 0.0)
        total_mass = 1000.0
        w_eth = clamp(E / total_mass, 0.0, 0.8)
        w_water = max(1.0 - w_eth - 1e-6, 1e-6)
        moles = [w_water/18.015, w_eth/46.07, 1e-6/MOLAR_MASS_ETHYL_ACETATE]
        x_molar = moles ./ sum(moles)
        return calculate_partition_coefficient(activity_model, T_PROFILE(t), x_molar)
    end
    return part_func
end

function _build_save_times(prod_times::Vector{Float64}, tspan_states::Tuple{Float64,Float64})
    dense = collect(range(tspan_states[1], tspan_states[2]; length=200))
    return sort(unique(vcat(prod_times, dense)))
end

function _simulate_ethyl_acetate_with_stripping(mpcc_tgrid, mpcc_states, v_var, params::ZentenoPlotParams;
        V_liq::Float64=FERMENTER_VOLUME_L, activity_model=MODEL_UNIFAC)
    prod_series = _build_ethyl_acetate_production_series(mpcc_tgrid, mpcc_states, v_var)
    prod_series === nothing && return nothing
    prod_times = Float64.(prod_series.times)
    prod_rates_g = Float64.(prod_series.rates_mmol) .* (MOLAR_MASS_ETHYL_ACETATE / 1000.0)
    isempty(prod_times) && return nothing
    state_fun, tspan_states = _build_state_interpolator(mpcc_tgrid, mpcc_states)
    prod_func = _build_piecewise_linear(prod_times, prod_rates_g; default=0.0)
    co2_func = _build_co2_flow_function(state_fun, params; V_liq=V_liq, tspan=tspan_states)
    part_func = _build_partition_function(state_fun; activity_model=activity_model)
    tspan = (min(prod_times[1], tspan_states[1]), max(prod_times[end], tspan_states[2]))
    saveat = _build_save_times(prod_times, tspan_states)
    function aroma_ode!(du, u, p, t)
        C = max(u[1], 0.0)
        prod = max(p.prod_func(t), 0.0)
        K = max(p.part_func(t), 0.0)
        Q = max(p.co2_func(t), 0.0)
        loss_rate = (Q / p.V_liq) * K * C
        du[1] = prod - loss_rate
        du[2] = loss_rate * p.V_liq
    end
    ode_params = (prod_func=prod_func, co2_func=co2_func, part_func=part_func, V_liq=V_liq)
    prob = ODEProblem(aroma_ode!, [0.0, 0.0], tspan, ode_params)
    sol = solve(prob, Rodas5(autodiff=false); saveat=saveat, reltol=1e-8, abstol=1e-10, maxiters=1_000_000)
    times = sol.t
    conc = [sol.u[i][1] for i in 1:length(times)]
    lost = [sol.u[i][2] for i in 1:length(times)]
    return (times=times, concentration_gL=conc, lost_mass_g=lost, prod_times=prod_times, prod_rates_g=prod_rates_g)
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

function plot_ethyl_acetate_concentration(mpcc_tgrid, mpcc_states, v_var, params::ZentenoPlotParams; title_str::AbstractString, save_path::AbstractString)
    series = _simulate_ethyl_acetate_with_stripping(mpcc_tgrid, mpcc_states, v_var, params)
    if series === nothing
        @warn "No se pudo construir la serie de concentracion de ethyl acetate"
        return false
    end
    # Serie base del MPCC (sin perdida) en puntos de colocacion
    base_series = _build_ethyl_acetate_series(mpcc_tgrid, mpcc_states, v_var)
    if base_series === nothing
        @warn "No se pudo construir la serie base de ethyl acetate"
        return false
    end

    times_strip = Float64.(series.times)
    conc_strip_mg = Float64.(series.concentration_gL) .* 1000.0
    isempty(times_strip) && return false

    times_base = Float64.(base_series.times)
    conc_base_mg = Float64.(base_series.concentrations) .* MOLAR_MASS_ETHYL_ACETATE # mmol/L * g/mol -> g/L; *1000 -> mg/L
    conc_base_mg .*= 1.0 # placeholder to emphasize units; already mg after factor above

    # Re-evaluar solucion con perdida en los puntos de colocacion
    colloc_times = vec(mpcc_tgrid)
    perm_strip = sortperm(times_strip)
    strip_interp = _build_cubic_interpolant(times_strip[perm_strip], conc_strip_mg[perm_strip])
    conc_strip_colloc = strip_interp.(colloc_times)

    line_t, line_vals = _sample_cubic_spline(times_strip, conc_strip_mg)
    plt = plot(size=(950, 420))
    scatter!(plt, times_base, conc_base_mg;
        color=:darkorange, ms=5, alpha=0.9, label="MPCC sin perdida (mg/L)")
    scatter!(plt, colloc_times, conc_strip_colloc;
        color=:navy, ms=5, alpha=0.9, marker=:diamond, label="Con perdida (mg/L)")
    plot!(plt, line_t, line_vals;
        color=:navy, lw=2.5, label="Interpolado con perdida")
    xlabel!(plt, "tiempo [h]")
    ylabel!(plt, "concentracion ethyl acetate [mg/L]")
    title!(plt, title_str)
    plot!(plt, legend=:topright)
    savefig(plt, save_path)
    println("[PLOT] Guardado ", save_path)
    return true
end

@inline function _state_depleted(l::Int, value::Real)
    abs_thresh = STATE_FREEZE_THRESH[l]
    rel_frac = STATE_FREEZE_REL_FRAC[l]
    rel_cond = rel_frac > 0 && value <= rel_frac * C0_INIT[l]
    return value <= abs_thresh || rel_cond
end

function freeze_phase_from_time(h_lengths::AbstractVector{<:Real}, freeze_time::Real)
    freeze_time <= 0.0 && return 1
    acc = 0.0
    for (idx, len) in enumerate(h_lengths)
        acc += len
        if freeze_time <= acc + 1e-9
            return idx
        end
    end
    return length(h_lengths) + 1
end

function detect_state_activity_from_data(data_vals::Array{Float64,3}, h_lengths::AbstractVector{<:Real})
    phases = fill(nfe + 1, nc)
    times = fill(NaN, nc)
    values = fill(NaN, nc, ncp)
    below = zeros(Int, nc)
    prefix = zeros(Float64, nfe)
    acc = 0.0
    for i in 1:nfe
        prefix[i] = acc
        acc += h_lengths[i]
    end
    for i in 1:nfe
        for l in 1:nc
            STATE_FREEZE_ELIGIBLE[l] || continue
            depleted = false
            for j in 1:ncp
                if _state_depleted(l, data_vals[l, i, j])
                    depleted = true
                    break
                end
            end
            if depleted
                below[l] += 1
                if below[l] == STATE_FREEZE_MIN_CONSEC_FE && phases[l] > nfe
                    phases[l] = i
                    times[l] = prefix[i]
                    @views values[l, :] .= data_vals[l, i, :]
                end
            else
                below[l] = 0
            end
        end
    end
    return phases, times, values
end

function detect_state_activity_from_presim(t_vals::AbstractVector{<:Real}, states_dense::AbstractMatrix{<:Real}, h_lengths::AbstractVector{<:Real})
    phases = fill(nfe + 1, nc)
    times = fill(NaN, nc)
    values = fill(NaN, nc)
    states_cols = size(states_dense, 2)
    for l in 1:nc
        STATE_FREEZE_ELIGIBLE[l] || continue
        freeze_time = nothing
        freeze_value = NaN
        for idx in 1:states_cols
            if _state_depleted(l, states_dense[l, idx])
                freeze_time = t_vals[idx]
                freeze_value = states_dense[l, idx]
                break
            end
        end
        freeze_time === nothing && continue
        phase = freeze_phase_from_time(h_lengths, freeze_time)
        phases[l] = phase
        times[l] = freeze_time
        values[l] = freeze_value
    end
    return phases, times, values
end

function build_state_activity_schedule(data_vals::Array{Float64,3}, h_lengths::AbstractVector{<:Real};
        t_pre_dense=nothing, states_pre_dense=nothing)
    mask = trues(nc, nfe)
    freeze_phase = fill(nfe + 1, nc)
    freeze_time = fill(NaN, nc)
    freeze_sources = fill(:none, nc)
    freeze_data_values = fill(NaN, nc, ncp)
    freeze_presim_values = fill(NaN, nc)

    if t_pre_dense !== nothing && states_pre_dense !== nothing
        phases_pre, times_pre, values_pre = detect_state_activity_from_presim(t_pre_dense, states_pre_dense, h_lengths)
        for l in 1:nc
            if phases_pre[l] < freeze_phase[l]
                freeze_phase[l] = phases_pre[l]
                freeze_time[l] = times_pre[l]
                freeze_sources[l] = :presim
                freeze_presim_values[l] = values_pre[l]
            end
        end
    end

    phases_data, times_data, values_data = detect_state_activity_from_data(data_vals, h_lengths)
    for l in 1:nc
        if phases_data[l] < freeze_phase[l]
            freeze_phase[l] = phases_data[l]
            freeze_time[l] = times_data[l]
            freeze_sources[l] = :data
            @views freeze_data_values[l, :] .= values_data[l, :]
        end
    end

    for l in 1:nc
        if freeze_phase[l] <= nfe
            for ii in freeze_phase[l]:nfe
                mask[l, ii] = false
            end
        end
    end
    return mask, freeze_phase, freeze_time, freeze_sources, freeze_data_values, freeze_presim_values
end

function summarize_freeze_schedule(
        freeze_phase::AbstractVector{<:Integer},
        freeze_time::AbstractVector,
        freeze_sources::AbstractVector{Symbol},
        freeze_data_values::AbstractMatrix,
        freeze_presim_values::AbstractVector)
    for l in 1:nc
        if freeze_phase[l] <= nfe
            approx_time = isfinite(freeze_time[l]) ? freeze_time[l] : sum(HM_REFERENCE[1:freeze_phase[l]-1])
            source = freeze_sources[l]
            state_label = STATE_LABELS[l]
            phase = freeze_phase[l]
            if source == :data
                data_vals = collect(view(freeze_data_values, l, :))
                @info "Estado se congelara tras agotarse" state=state_label phase=phase approx_time_h=approx_time source=source data_values=data_vals
            elseif source == :presim
                @info "Estado se congelara tras agotarse" state=state_label phase=phase approx_time_h=approx_time source=source value=freeze_presim_values[l]
            else
                @info "Estado se congelara tras agotarse" state=state_label phase=phase approx_time_h=approx_time source=source
            end
        end
    end
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
    return joinpath(EXPERIMENT_DIR, "MPCCpost_$(wall_tok)_nfe$(nfe)_$(feas_tok)_$(term_tok)_$(stamp)")
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

    for s in 1:(nsteps-1)
        x = X[1, s]; n = X[2, s]; g = X[3, s]; f = X[4, s]; e = X[5, s]

        Tloc = T_PROFILE((s-1)*dt)
        mu_T_val  = exp(59453.0 * (Tloc - 300.0) / (300.0 * R * Tloc))
        Kg_T_val  = exp(46055.0 * (Tloc - 293.15) / (293.15 * R * Tloc))
        b_T_val   = exp(11000.0 * (Tloc - 296.15) / (296.15 * R * Tloc))
        mrate_val = 0.01 * exp(37681.0 * (Tloc - 293.30) / (293.30 * R * Tloc))

        mu_val = MU0_nom * mu_T_val * (n / (n + Kn0_nom * Kg_T_val + eps))
        betaG_val = betaG0_nom * b_T_val *
            (g / (g + Kg0_nom * Kg_T_val + eps)) *
            (Kie0_nom * Kg_T_val / (e + Kie0_nom * Kg_T_val + eps))
        betaF_val = betaF0_nom * b_T_val *
            (f / (f + Kf0_nom * Kg_T_val + eps)) *
            (Kig0_nom * Kg_T_val / (g + Kig0_nom * Kg_T_val + eps)) *
            (Kie0_nom * Kg_T_val / (e + Kie0_nom * Kg_T_val + eps))

        Td = -0.0001 * e^3 + 0.0049 * e^2 - 0.1279 * e + 315.89
        s_sw = 0.5 * (1 + tanh(0.5 * (Tloc - Td)))
        Kd_val = Kd0_nom * exp(0.0415 * e + (130000.0 * (Tloc - 305.65)) / (305.65 * R * Tloc)) * s_sw

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
    pre_params = ZentenoPlotParams(exp(theta0[1]), exp(theta0[2]), exp(theta0[3]), T_PROFILE)
    local_t_pre, local_states_pre = simulate_zenteno(pre_params)
    global t_pre = local_t_pre
    global states_pre = local_states_pre
catch err
    @warn "No se pudo simular la ODE previa a la optimizacion" err
end

state_activity_mask = trues(nc, nfe)
state_freeze_phase = fill(nfe + 1, nc)
state_freeze_time = fill(NaN, nc)
state_freeze_sources = fill(:none, nc)
state_freeze_data_values = fill(NaN, nc, ncp)
state_freeze_presim_values = fill(NaN, nc)
if FREEZE_DEPLETED_STATES
    state_activity_mask, state_freeze_phase, state_freeze_time,
        state_freeze_sources, state_freeze_data_values, state_freeze_presim_values =
        build_state_activity_schedule(data, HM_REFERENCE;
            t_pre_dense=(t_pre === nothing ? nothing : t_pre),
            states_pre_dense=(states_pre === nothing ? nothing : states_pre))
    summarize_freeze_schedule(
        state_freeze_phase,
        state_freeze_time,
        state_freeze_sources,
        state_freeze_data_values,
        state_freeze_presim_values)
else
    @info "FREEZE_DEPLETED_STATES=0 -> no se congelaran especies agotadas"
end

state_is_active(l::Int, i::Int) = (!FREEZE_DEPLETED_STATES || state_activity_mask[l, i])

# ---------------------------------------------
# Modelo JuMP
# ---------------------------------------------
m = Model(Ipopt.Optimizer)
set_optimizer_attribute(m, "warm_start_init_point", "yes")
set_optimizer_attribute(m, "print_level", 5)
set_optimizer_attribute(m, "tol", 1e-4)
set_optimizer_attribute(m, "acceptable_iter", 5)
set_optimizer_attribute(m, "acceptable_tol", 1e-2)
linear_solver = lowercase(strip(get(ENV, "IPOPT_LINEAR_SOLVER", "")))
if isempty(linear_solver)
    linear_solver = DEFAULT_LINEAR_SOLVER
end
set_optimizer_attribute(m, "linear_solver", linear_solver)

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
    hv[1:nfe]
end)

if !REDUCED_MODE || reduced_sets === nothing
    @variables(m, begin
        v[1:nv, 1:nfe]
        lambda_[1:nm, 1:nfe]
        alpha_U[1:nv, 1:nfe]
        alpha_L[1:nv, 1:nfe]
        FO_U[1:nv, 1:nfe]
        FO_L[1:nv, 1:nfe]
    end)
else
    @variables(m, begin
        v[K_AX, 1:nfe]
        lambda_[M_AX, 1:nfe]
        alpha_U[K_AX, 1:nfe]
        alpha_L[K_AX, 1:nfe]
        FO_U[K_AX, 1:nfe]
        FO_L[K_AX, 1:nfe]
    end)
end

@variables(m, begin
    alpha_upt[1:n_up, 1:nfe]
    FO_upt[1:n_up, 1:nfe]
end)

for k in 1:np
    set_start_value(teta[k], theta0[k])
end

for i in 1:ph, j in 1:ncp, l in 1:nc
    set_start_value(c[l, i, j], max(c0[l], STATE_MIN_CONC[l]))
    set_start_value(cdot[l, i, j], 0.0)
end
for i in 1:nfe
    set_start_value(hv[i], hm[i])
end

for i in 1:nc
    c0[i] = c0[i] / cs[i]
    c0[i] = max(c0[i], STATE_MIN_CONC[i])
end

@NLobjective(m, Min,
    omega * FO +
    sum(
        sum(-phi1 * FO_L[mc, i] - phi3 * FO_U[mc, i] for mc in FLUX_INDEX_SET) +
        phi2 * FO_upt[1, i] + phi2 * FO_upt[2, i]
        for i in 1:nfe
    )
)

const T_STEP1_K = 15.0 + 273.15
const T_STEP2_K = 18.0 + 273.15
const T_STEP3_K = 23.0 + 273.15

@NLexpression(m, t_prefix[i=1:ph], sum(hv[k] for k in 1:(i-1)))
@NLexpression(m, t_colloc[i=1:ph, j=1:ncp], t_prefix[i] + radau_nodes[j] * hv[i])
@NLexpression(m, T_s1[i=1:ph, j=1:ncp],
    0.5 * (1.0 + tanh((t_colloc[i,j] - 24.0) / TEMP_TRANSITION_WIDTH_H)))
@NLexpression(m, T_s2[i=1:ph, j=1:ncp],
    0.5 * (1.0 + tanh((t_colloc[i,j] - 48.0) / TEMP_TRANSITION_WIDTH_H)))
@NLexpression(m, T_loc[i=1:ph, j=1:ncp],
    T_STEP1_K + (T_STEP2_K - T_STEP1_K) * T_s1[i,j] + (T_STEP3_K - T_STEP2_K) * T_s2[i,j])
@NLexpression(m, mu_T[i=1:ph, j=1:ncp],
    exp(59453.0 * (T_loc[i,j] - 300.0) / (300.0 * R * T_loc[i,j])))
@NLexpression(m, Kg_T[i=1:ph, j=1:ncp],
    exp(46055.0 * (T_loc[i,j] - 293.15) / (293.15 * R * T_loc[i,j])))
@NLexpression(m, b_T[i=1:ph, j=1:ncp],
    exp(11000.0 * (T_loc[i,j] - 296.15) / (296.15 * R * T_loc[i,j])))
@NLexpression(m, mrate[i=1:ph, j=1:ncp],
    0.01 * exp(37681.0 * (T_loc[i,j] - 293.30) / (293.30 * R * T_loc[i,j])))

@NLexpression(m, mu0, exp(teta[1]))
@NLexpression(m, Yeg, exp(teta[2]))
@NLexpression(m, Yef, exp(teta[3]))

const Yxn = YXN_nom
const Yxg = YXG_nom
const Yxf = YXF_nom

@NLexpression(m, mu_j[i=1:ph, j=1:ncp],
    mu0 * mu_T[i,j] * (c[2,i,j] / (c[2,i,j] + Kn0_nom * Kg_T[i,j] + eps))
)
@NLexpression(m, betaG_j[i=1:ph, j=1:ncp],
    betaG0_nom * b_T[i,j] *
    (c[3,i,j] / (c[3,i,j] + Kg0_nom * Kg_T[i,j] + eps)) *
    (Kie0_nom * Kg_T[i,j] / (c[5,i,j] + Kie0_nom * Kg_T[i,j] + eps))
)
@NLexpression(m, betaF_j[i=1:ph, j=1:ncp],
    betaF0_nom * b_T[i,j] *
    (c[4,i,j] / (c[4,i,j] + Kf0_nom * Kg_T[i,j] + eps)) *
    (Kig0_nom * Kg_T[i,j] / (c[3,i,j] + Kig0_nom * Kg_T[i,j] + eps)) *
    (Kie0_nom * Kg_T[i,j] / (c[5,i,j] + Kie0_nom * Kg_T[i,j] + eps))
)
@NLexpression(m, phiG_j[i=1:ph, j=1:ncp], c[3,i,j] / (c[3,i,j] + c[4,i,j] + eps))
@NLexpression(m, phiF_j[i=1:ph, j=1:ncp], c[4,i,j] / (c[3,i,j] + c[4,i,j] + eps))
JuMP.register(m, :death_rate_T, 2, death_rate_T; autodiff = true)
@NLexpression(m, Kd_j[i=1:ph, j=1:ncp], death_rate_T(c[5,i,j], T_loc[i,j]))

@constraints(m, begin
    coll_c_n[l=1:nc, i=2:ph, j=1:ncp],
        c[l,i,j] == c[l,i-1,ncp] + hv[i] * sum(colmat[j,k] * cdot[l,i,k] for k in 1:ncp)
    coll_c_0[l=1:nc, j=1:ncp],
        c[l,1,j] == c0[l] + hv[1] * sum(colmat[j,k] * cdot[l,1,k] for k in 1:ncp)

    teta_fix[p=1:np], teta[p] == theta0[p]

    c_LB[l=1:nc, i=1:nfe, j=1:ncp], STATE_MIN_CONC[l] - c[l,i,j] <= 0

    MFE1, sum(hv[i] for i in 1:nfe) == th
    MFE3[i=1:nfe], hv[i] >= 0.0
    MFE4[i=1:nfe], hv[i] >= (1.0 - var_h) * hm[1]
    MFE5[i=1:nfe], hv[i] <= (1.0 + var_h) * hm[1]
    alpha4_LB[mc=1:n_up, i=1:nfe], alpha_upt[mc,i] <= 0
end)

if !REDUCED_MODE || reduced_sets === nothing
    @constraints(m, begin
        Sc[mc=1:nm, i=1:nfe],  sum(S[mc,k] * v[k,i] * vs[k] for k in 1:nv) == 0
        v_UB[mc=1:nv, i=1:nfe], v[mc,i]*vs[mc] - ub[mc] <= 0
        v_LB[mc=1:nv, i=1:nfe], -v[mc,i]*vs[mc] + lb[mc] <= 0
        Lagr[mc=1:nv, i=1:nfe],
            d[mc] + w * v[mc,i] * vs[mc] + alpha_L[mc,i] + alpha_U[mc,i] +
            up[mc] * alpha_upt[1,i] + up2[mc] * alpha_upt[2,i] +
            sum(S[k,mc] * lambda_[k,i] for k in 1:nm) == 0
        alpha1_LB[mc=1:nv, i=1:nfe], alpha_L[mc,i] <= 0
        alpha1_UB[mc=1:nv, i=1:nfe], alpha_U[mc,i] >= 0
    end)
else
    @constraints(m, begin
        v_UB[k=K_AX, i=1:nfe], v[k,i]*vs[k] - ub[k] <= 0
        v_LB[k=K_AX, i=1:nfe], -v[k,i]*vs[k] + lb[k] <= 0
        alpha1_LB[k=K_AX, i=1:nfe], alpha_L[k,i] <= 0
        alpha1_UB[k=K_AX, i=1:nfe], alpha_U[k,i] >= 0
    end)
    A_sets, C_sets, _ = reduced_sets
    for i in 1:nfe
        Ai = (i <= length(A_sets)) ? A_sets[i] : Int[]
        Ci = (i <= length(C_sets)) ? C_sets[i] : Int[]
        Ri = union(Ai, Ci)
        active_mc = Int[]
        for mc in 1:nm
            for k in Ri
                if S[mc, k] != 0.0
                    push!(active_mc, mc)
                    break
                end
            end
        end
        active_mc = unique(active_mc)
        sort!(active_mc)
        for mc in active_mc
            nz_rxn = Int[]
            for k in Ri
                if S[mc, k] != 0.0 && (k in K_AX)
                    push!(nz_rxn, k)
                end
            end
            isempty(nz_rxn) && continue
            @constraint(m, sum(S[mc, k] * v[k,i] * vs[k] for k in nz_rxn) == 0)
        end
    end
    for i in 1:nfe
        Ci = (i <= length(C_sets)) ? C_sets[i] : Int[]
        nonC = setdiff(K_AX, Ci)
        for k in nonC
            @constraint(m, alpha_L[k, i] == 0.0)
            @constraint(m, alpha_U[k, i] == 0.0)
        end
        for k in Ci
            if !(k in K_AX)
                continue
            end
            nz_met = Int[]
            for r in M_AX
                if S[r, k] != 0.0
                    push!(nz_met, r)
                end
            end
            @constraint(m,
                d[k] + w * v[k,i] * vs[k] + alpha_L[k,i] + alpha_U[k,i] +
                up[k] * alpha_upt[1,i] + up2[k] * alpha_upt[2,i] +
                sum(S[r,k] * lambda_[r,i] for r in nz_met) == 0
            )
        end
    end
end

for i in 1:ph, j in 1:ncp
    if state_is_active(1, i)
        @NLconstraint(m, cdot[1,i,j] == (mu_j[i,j] - Kd_j[i,j]) * c[1,i,j])
    else
        @constraint(m, cdot[1,i,j] == 0.0)
    end
    if state_is_active(2, i)
        @NLconstraint(m, cdot[2,i,j] == -(mu_j[i,j] / Yxn) * c[1,i,j])
    else
        @constraint(m, cdot[2,i,j] == 0.0)
    end
    if state_is_active(3, i)
        @NLconstraint(m, cdot[3,i,j] == -((mu_j[i,j] / Yxg) + (betaG_j[i,j] / Yeg) + mrate[i,j] * phiG_j[i,j]) * c[1,i,j])
    else
        @constraint(m, cdot[3,i,j] == 0.0)
    end
    if state_is_active(4, i)
        @NLconstraint(m, cdot[4,i,j] == -((mu_j[i,j] / Yxf) + (betaF_j[i,j] / Yef) + mrate[i,j] * phiF_j[i,j]) * c[1,i,j])
    else
        @constraint(m, cdot[4,i,j] == 0.0)
    end
    if state_is_active(5, i)
        @NLconstraint(m, cdot[5,i,j] == (betaG_j[i,j] + betaF_j[i,j]) * c[1,i,j])
    else
        @constraint(m, cdot[5,i,j] == 0.0)
    end
    @constraint(m, c[1,i,j] >= STATE_MIN_CONC[1])
    @constraint(m, c[2,i,j] >= STATE_MIN_CONC[2])
    @constraint(m, c[3,i,j] >= STATE_MIN_CONC[3])
    @constraint(m, c[4,i,j] >= STATE_MIN_CONC[4])
    @constraint(m, c[5,i,j] >= STATE_MIN_CONC[5])
end

@NLconstraints(m, begin
    FO3_upt[i=1:nfe], FO_upt[1,i] == (-v[glu,i]*vs[glu]) * alpha_upt[1,i]
    FO4_upt[i=1:nfe], FO_upt[2,i] == (-v[fru,i]*vs[fru]) * alpha_upt[2,i]

    FO_def,
        FO == sum((data[l,i,j] - c[l,i,j])^2 for l in MEAS_STATES, i in 1:ph, j in 1:ncp)
end)

if !REDUCED_MODE || reduced_sets === nothing
    @NLconstraints(m, begin
        FO1[mc=1:nv, i=1:nfe], FO_L[mc,i] == (v[mc,i]*vs[mc] - lb[mc]) * alpha_L[mc,i]
        FO2[mc=1:nv, i=1:nfe], FO_U[mc,i] == (v[mc,i]*vs[mc] - ub[mc]) * alpha_U[mc,i]
    end)
else
    @NLconstraints(m, begin
        FO1_red[mc=K_AX, i=1:nfe], FO_L[mc,i] == (v[mc,i]*vs[mc] - lb[mc]) * alpha_L[mc,i]
        FO2_red[mc=K_AX, i=1:nfe], FO_U[mc,i] == (v[mc,i]*vs[mc] - ub[mc]) * alpha_U[mc,i]
    end)
end

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
    post_params = ZentenoPlotParams(exp(mu_log), exp(yeg_log), exp(yef_log), T_PROFILE)
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
    plot_ethyl_acetate_concentration(
        mpcc_tgrid, mpcc_states, v, post_params;
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
)
