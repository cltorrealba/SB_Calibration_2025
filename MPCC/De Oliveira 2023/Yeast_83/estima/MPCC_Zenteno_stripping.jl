#!/usr/bin/env julia
# MPCC_Zenteno_stripping.jl
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
using Clapeyron
using CSV
using HiGHS
const MOI = MathOptInterface

# ---------------------------------------------
# Paths e IO
# ---------------------------------------------
const EXPORT_PLOT_CSV = get(ENV, "EXPORT_PLOT_CSV", "0") == "1"
const USE_WARM_START = get(ENV, "USE_WARM_START", "1") == "1"
const BASE_DIR   = @__DIR__
const TEST_FIX_THETA_M = false       # Paso 7.1: true => fija theta_m=log(0.01) para solo simular
const CALIBRATE_ONLY_THETA_M = true # Paso 7.2: true => solo calibra theta_m; otros teta fijados a data
const DIAG_SIMPLE = get(ENV, "DIAG_SIMPLE", "0") == "1" # Modo diagnóstico sin complementariedad
# Permite activar solo la complementariedad de cota superior (FO_U/alpha_U) en modo diagnóstico
const DIAG_ENABLE_FO_U = get(ENV, "DIAG_ENABLE_FO_U", "0") == "1"
const ESTIMA_DIR = BASE_DIR
const PLOTS_DIR  = joinpath(ESTIMA_DIR, "plots")
isdir(PLOTS_DIR) || mkpath(PLOTS_DIR)

const REDUCED_MODE = get(ENV, "REDUCED_MODE", "0") == "1"
const REDUCED_SETS_PATH = joinpath(BASE_DIR, "julia_deploy", "results", "reduced_sets.jld2")

const ESTIMATE_PARAMS = true

function _sanitize_experiment_name(str::AbstractString)
    clean = strip(str)
    isempty(clean) && return "default"
    return replace(clean, r"[^0-9A-Za-z._-]+" => "_")
end

const EXPERIMENT_TOKEN = _sanitize_experiment_name(get(ENV, "EXPERIMENT", "default"))
const EXPERIMENT_DIR = joinpath(PLOTS_DIR, EXPERIMENT_TOKEN)
isdir(EXPERIMENT_DIR) || mkpath(EXPERIMENT_DIR)
const WARM_START_FILE = joinpath(EXPERIMENT_DIR, "warm_start_seed.jld2")


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


# --- FACTORES DE ESCALA MANUAL ---
const SC_V = 1.0  # escala flujos
const SC_C = 1.0   # escala sustratos altos (G/F)
const SC_X = 1.0     # biomasa/N se mantienen en ~1

# ---------------------------------------------
# Constantes de Penalización MPCC (Faltantes en versión anterior)
# ---------------------------------------------
const phi1 = 1e0
const phi2 = 1e0
const phi3 = 1e0

# --- MANTENIMIENTO ---
const IDX_ATPM = 3414          # Índice de la reacción de mantenimiento

# ---------------------------------------------
# Indices y tamanos del GEM
# ---------------------------------------------

S     = readdlm(joinpath(ESTIMA_DIR, "S_hen.csv"), ',')
lbraw = readdlm(joinpath(ESTIMA_DIR, "lb_hen.csv"), ',')
ubraw = readdlm(joinpath(ESTIMA_DIR, "ub_hen.csv"), ',')
lb    = lbraw isa AbstractVector ? Float64.(lbraw) : Float64.(lbraw[:,1])
ub    = ubraw isa AbstractVector ? Float64.(ubraw) : Float64.(ubraw[:,1])

nm = size(S, 1)
nv = size(S, 2)

# --- IDs de reacciones y metabolitos exportados desde MATLAB ---
const RXN_IDS = readlines(joinpath(ESTIMA_DIR, "rxn_ids.txt"))
const MET_IDS = readlines(joinpath(ESTIMA_DIR, "met_ids.txt"))

const RXN_INDEX = Dict{String, Int}(rxn => i for (i, rxn) in pairs(RXN_IDS))
const MET_INDEX = Dict{String, Int}(met => i for (i, met) in pairs(MET_IDS))

get_rxn(name::AbstractString) = get(RXN_INDEX, String(name)) do
    error("La reacción '" * String(name) * "' no se encontró en RXN_INDEX")
end

get_met(name::AbstractString) = get(MET_INDEX, String(name)) do
    error("El metabolito '" * String(name) * "' no se encontró en MET_INDEX")
end

# --- Reacciones clave (vía IDs) ---
const eth = get_rxn("r_1761")   # intercambio de etanol
const obj = get_rxn("r_4041")   # biomasa pseudoreaction
const glu = get_rxn("r_1714")   # uptake glucosa
const fru = get_rxn("r_1709")   # uptake fructosa
const o2  = get_rxn("r_1992")   # intercambio de oxígeno

# Nitrógeno que ya se usa en las restricciones dinámicas
const idx_NH4 = get_rxn("r_1654")
const idx_Arg = get_rxn("r_1879")
const idx_Gln = get_rxn("r_1891")
const idx_Glu = get_rxn("r_1889")
const idx_Ser = get_rxn("r_1906")
const idx_Thr = get_rxn("r_1911")
const idx_Ala = get_rxn("r_1873")
const idx_Trp = get_rxn("r_1912")

const ACTIVE_N_SOURCE_IDS = [
    "r_1654", # NH4Cl
    "r_1879", # Arg
    "r_1891", # Gln
    "r_1889", # Glu
    "r_1906", # Ser
    "r_1911", # Thr
    "r_1873", # Ala
    "r_1912", # Trp
]

const UNUSED_AA_UPTAKE_IDS = [
    "r_1880", # Asp
    "r_1883", # Cys
    "r_1810", # Gly
    "r_1893", # His
    "r_1897", # Ile
    "r_1899", # Leu
    "r_1900", # Lys
    "r_1902", # Met
    "r_1903", # Phe
    "r_1913", # Tyr
    "r_1914", # Val
]

const PRODUCT_RXN_IDS = [
    "r_1761", # Ethanol exchange
    "r_1808", # Glycerol exchange
    "r_1634", # Acetate exchange
    "r_2056", # Succinate exchange
    "r_1549", # 2,3-butanediol
    "r_1546", # Lactate
    "r_1552", # Malate
    "r_1765", # Ethyl acetate
    "r_1867", # Isobutyl acetate
    "r_1866", # Isobutanol
    "r_1862", # Isoamyl acetate
    "r_1865", # Isoamyl alcohol
]

# Cofactores para válvulas de bypass en anaerobiosis
const METS_ANAEROBIC_BYPASS = [
    "s_3714[c]", # Heme A
    "s_1198[c]", # Heme O
    "s_1203[c]", # Coenzyme Q
    "s_1207[c]", # Coenzyme Q6
    "s_1212[c]", # Demethyl-menaquinone
    "s_0529[c]"  # Calmodulin
]

# --- INYECCIÓN ESTRUCTURAL DE BYPASS ---
println(">>> INYECTANDO REACCIONES BYPASS (Heme/CoQ)...")

n_bypass = length(METS_ANAEROBIC_BYPASS)
S_bypass = zeros(size(S,1), n_bypass)

for (k, met_id) in enumerate(METS_ANAEROBIC_BYPASS)
    if haskey(MET_INDEX, met_id)
        row_idx = get_met(met_id)
        S_bypass[row_idx, k] = 1.0
    else
        clean_id = replace(met_id, "[c]" => "")
        if haskey(MET_INDEX, clean_id)
            S_bypass[get_met(clean_id), k] = 1.0
        end
    end
end

S = hcat(S, S_bypass)
nv_old = nv
nv = size(S, 2)
nm = size(S, 1)

lb = vcat(lb, zeros(n_bypass))
ub = vcat(ub, zeros(n_bypass))

const IDX_BYPASS_START = nv_old + 1
const IDX_BYPASS_END   = nv

function apply_anaerobic_model!(S::AbstractMatrix, lb::AbstractVector, ub::AbstractVector)
    mets_ana = ["s_3714[c]", "s_1198[c]", "s_1203[c]", "s_1207[c]", "s_1212[c]", "s_0529[c]"]
    rxn_cofactor = get_rxn("r_4598")

    for mid in mets_ana
        if haskey(MET_INDEX, mid)
            S[get_met(mid), rxn_cofactor] = 0.0
        elseif haskey(MET_INDEX, replace(mid, "[c]" => ""))
            S[get_met(replace(mid, "[c]" => "")), rxn_cofactor] = 0.0
        end
    end

    lb[get_rxn("r_1992")] = 0.0      # O2
    lb[get_rxn("r_1757")] = -1000.0  # ergosterol
    lb[get_rxn("r_1915")] = -1000.0  # lanosterol
    lb[get_rxn("r_1994")] = -1000.0  # palmitoleate
    lb[get_rxn("r_2106")] = -1000.0  # zymosterol
    lb[get_rxn("r_2134")] = -1000.0  # 14-demethyllanosterol
    lb[get_rxn("r_2137")] = -1000.0  # ergosta-5,7,22,24(28)-tetraen-3beta-ol
    lb[get_rxn("r_2189")] = -1000.0  # oleate

    lb[get_rxn("r_0713")] = 0.0      # OAA-malate shuttle (mito)
    lb[get_rxn("r_0714")] = 0.0      # OAA-malate shuttle (cito)
    ub[get_rxn("r_0487")] = 0.0      # glycerol dehydrogenase

    return nothing
end

function configure_uptake_and_products!(lb::AbstractVector, ub::AbstractVector)
    lb[glu] = -1000.0; ub[glu] = 0.0
    lb[fru] = -1000.0; ub[fru] = 0.0

    for rxn in ACTIVE_N_SOURCE_IDS
        idx = get_rxn(rxn)
        lb[idx] = -1000.0
        ub[idx] = 0.0
    end

    for rxn in UNUSED_AA_UPTAKE_IDS
        idx = get_rxn(rxn)
        lb[idx] = 0.0
        ub[idx] = 0.0
    end

    for rxn in PRODUCT_RXN_IDS
        idx = get_rxn(rxn)
        lb[idx] = 0.0
        ub[idx] = 1000.0
    end

    return nothing
end

# --- CORRECCIÓN DE EMERGENCIA: MANTENIMIENTO FLEXIBLE ---
# Liberamos el límite inferior estricto para gestionarlo con Slack
if length(lb) >= IDX_ATPM
    println(">>> CONFIGURANDO SLACK PARA ATP MAINTENANCE (idx $IDX_ATPM). LB fijado a 0.0.")
    lb[IDX_ATPM] = 0.01
    # Mantenemos ub alto o fijo según tu CSV, lo importante es que lb sea 0
end

println(">>> CONFIGURACIÓN FACULTATIVA: O2 inicial abierto y suplementos disponibles")

# 1. Transporte de O2 (solo uptake; se prohíbe producción)
lb[o2] = -1000.0
ub[o2] = 0.0

# 2. Suplementos anaerobios disponibles cuando se necesiten
lb[get_rxn("r_1757")] = -1000.0  # ergosterol
lb[get_rxn("r_1915")] = -1000.0  # lanosterol
lb[get_rxn("r_1994")] = -1000.0  # palmitoleate
lb[get_rxn("r_2106")] = -1000.0  # zymosterol
lb[get_rxn("r_2134")] = -1000.0  # 14-demethyllanosterol
lb[get_rxn("r_2137")] = -1000.0  # ergosta-5,7,22,24(28)-tetraen-3beta-ol
lb[get_rxn("r_2189")] = -1000.0  # oleate

# 3. No bloquear shuttles (r_0713, r_0714) ni forzar O2=0

configure_uptake_and_products!(lb, ub)

const NITROGEN_SOURCES = [get_rxn(rid) for rid in ACTIVE_N_SOURCE_IDS]
const UPTAKE_IDXS = vcat([glu, fru], NITROGEN_SOURCES)
const n_up = length(UPTAKE_IDXS)
# Selectores constantes para evitar condicionales en NLexpresiones
const IS_GLU = [x == glu ? 1.0 : 0.0 for x in UPTAKE_IDXS]
const IS_FRU = [x == fru ? 1.0 : 0.0 for x in UPTAKE_IDXS]
const IS_NIT = [1.0 - IS_GLU[i] - IS_FRU[i] for i in 1:n_up]
const SELECT_UPTAKE = [Float64(mc == UPTAKE_IDXS[k]) for mc in 1:nv, k in 1:n_up]

# ---------------------------------------------
# Modelo Zenteno (param nominal)
# ---------------------------------------------
const nc = 6 # X, N, G, F, E, O2
const NOISE_SEED = 1234
const DEFAULT_LINEAR_SOLVER = "mumps"

# --- Constantes Físicas y de Bypass ---
const MW_O2   = 31.998   # g/mol
const KO2_MM  = 0.009    # mmol/L     (Original en paper: ~33.1 mg/L, saturación O2 puro)
const VO2_MAX = 0.606    # mmol/gDW/h (Original en paper: 19.4 mg/g/h)
const O2_SAT  = 0.26     # mmol/L saturado
const n_Hill  = 2.3      # Exponente ajustado según Tabla 2 de Cerda-Drago et al. (2016)

# ---------------------------------------------
# Selección de solver lineal (MUMPS por defecto)
# HSL (MA57/MA77/MA86/MA97) requiere HSL_jll instalado
# Pardiso requiere configuración de Panua
# ---------------------------------------------
const ALLOWED_IPOPT_SOLVERS = Set(["mumps", "spral", "pardiso", "ma57", "ma77", "ma86", "ma97"])

function _prepend_to_path!(dir::AbstractString)
    isempty(dir) && return
    if isdir(String(dir))
        path_now = get(ENV, "PATH", "")
        dir_norm = replace(String(dir), '\\' => '/')
        path_norm = replace(path_now, '\\' => '/')
        occursin(lowercase(dir_norm), lowercase(path_norm)) || (ENV["PATH"] = string(String(dir), ";", path_now))
    end
end

function configure_ipopt_env!()
    solver = lowercase(get(ENV, "IPOPT_LINEAR_SOLVER", DEFAULT_LINEAR_SOLVER))
    # Pardiso (Panua) – añadir rutas si se solicita
    if solver == "pardiso"
        haskey(ENV, "PANUA_IPOPT_DIR") && _prepend_to_path!(joinpath(ENV["PANUA_IPOPT_DIR"], "bin"))
        haskey(ENV, "IPOPT_PARDISO_DLL_DIR") && _prepend_to_path!(ENV["IPOPT_PARDISO_DLL_DIR"])
        haskey(ENV, "PANUA_LIC_PATH") && _prepend_to_path!(ENV["PANUA_LIC_PATH"])
        # Mapear hilos si se define PARDISO_NUM_THREADS
        if haskey(ENV, "PARDISO_NUM_THREADS")
            ENV["OMP_NUM_THREADS"] = ENV["PARDISO_NUM_THREADS"]
            ENV["MKL_NUM_THREADS"] = ENV["PARDISO_NUM_THREADS"]
        end
    # HSL (MA57/MA77/MA86/MA97) – Requiere HSL_jll con solvers reales
    elseif solver in ("ma57", "ma77", "ma86", "ma97")
        try
            @eval import HSL_jll
            # Cargar OpenBLAS32 para soporte LP64 (requerido por HSL)
            @eval using LinearAlgebra, OpenBLAS32_jll
            @eval LinearAlgebra.BLAS.lbt_forward(OpenBLAS32_jll.libopenblas)
            @info "HSL_jll detectado; configurando Ipopt para usar HSL" solver HSL_jll.libhsl_path
        catch
            # Silencioso: el fallback a CoinHSL local se maneja en la configuración de Ipopt
        end
    end
    return solver in ALLOWED_IPOPT_SOLVERS ? solver : DEFAULT_LINEAR_SOLVER
end

const SELECTED_IPOPT_SOLVER = configure_ipopt_env!()

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

# --- PERFIL DE TEMPERATURA DINÁMICA ---
const T_BASE = T_const
const T_STEPS = [36.0, 96.0]   # dos eventos de cambio de T (ajusta valores a gusto)
const T_DELTAS = [5.0, 3.0]    # salto asociado a cada evento (el 2do default es neutro)
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

# --- INYECCIONES SUAVIZADAS ---
const SQRT_2PI = sqrt(2*pi)
function smooth_injection(t, t_shot, dose, width=1.0)
    abs(t - t_shot) > 5 * width && return 0.0
    return (dose / (width * SQRT_2PI)) * exp(-0.5 * ((t - t_shot) / width)^2)
end

# SETTINGS PARA ESTIMACION DE PARAMETROS
const np = 5
const IDX_THETA_M = np
const THETA_NAMES = ("mu0", "Yeg", "Yef", "Yxn", "mrate")
const theta_data_params = log.([MU0_nom, YEG_nom, YEF_nom, YXN_nom, 0.01])
const theta_init_guess  = log.([MU0_nom*1.0, YEG_nom*1.0, YEF_nom*1.0, YXN_nom*1.0, 0.01])
LB = log.([1*MU0_nom, 1*YEG_nom, 1*YEF_nom, 1*YXN_nom, 1e-4])
UB = log.([1*MU0_nom, 1*YEG_nom, 1*YEF_nom, 1*YXN_nom, 1e0])

# Variables observadas y ruido aleatorio
const MEAS_STATES = (3, 4, 5) # G, F, E
const NOISE_REL_STD = 0.0
Random.seed!(NOISE_SEED)

# Condiciones iniciales
X0 = 0.5; N0 = 0.14; G0 = 110.0; F0 = 110.0; E0 = 0.0
O2_init = O2_SAT * MW_O2 / 1000.0 # almacenar en g/L
const C0_INIT = [X0, N0, G0, F0, E0, O2_init]
c0 = copy(C0_INIT)

# DISCRETIZACION
nfe = DIAG_SIMPLE ? 12 : 8   
ncp = 3
th  = 120.0
h   = th / nfe
ph  = nfe
hm    = fill(h, nfe)'
var_h = 0.5

colmat = [
    0.19681547722366   -0.06553542585020   0.02377097434822;
    0.39442431473909    0.29207341166523  -0.04154875212600;
    0.37640306270047    0.51248582618842   0.11111111111111
]
const radau_nodes = (0.15505, 0.64495, 1.0)

# OMEGA
w     = 1e-6
omega = 1
d   = zeros(nv); d[obj] = -1.0
cs = ones(nc)
vs = ones(nv)

# PERFIL U DINAMICO
const T_INJ_1 = 0   # horas
const DOSE_1  = 0   # g/L (100 mg/L)
const WIDTH_1 = 5.0    # ancho de pulso (h)
const T_INJ_2 = 0   # segundo pulso (ajusta valores)
const DOSE_2  = 0   # g/L (default neutro)
const WIDTH_2 = 5.0


# ---------------------------------------------
# Configuracion de stripping (CO2 + UNIFAC)
# ---------------------------------------------

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
    rho_L = 1000.0
    rho_G = (101325.0 / (R * T)) * 44.01
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

# ---------------------------------------------
# Herramientas de simulacion/plot
# ---------------------------------------------
const STATE_LABELS = ("X", "N", "G", "F", "E", "O2")
const STATE_COLORS = (:royalblue, :forestgreen, :firebrick, :darkorange, :purple, :cyan)
const STATE_MIN_CONC = (
    1e-6,
    0.0,
    0.0,
    0.0,
    1e-6,
    0.0,
)
struct ZentenoPlotParams
    mu0::Float64
    Yeg::Float64
    Yef::Float64
    Yxn::Float64
    mrate0::Float64
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
    # Clamping defensivo para evitar negativos/overflow en exp
    X = max(u[1], 1e-9)
    N = max(u[2], 0.0)
    G = max(u[3], 0.0)
    F = max(u[4], 0.0)
    E = max(u[5], 0.0)
    O2_gL = max(u[6], 0.0)
    O2_mmol = (O2_gL * 1000.0) / MW_O2

    T_curr = dynamic_temperature(t)
    safe_exp(val) = exp(clamp(val, -700.0, 100.0))

    mu_T =  safe_exp(59453.0 * (T_curr - 300.0) / (300.0 * R * T_curr))
    Kg_T =  safe_exp(46055.0 * (T_curr - 293.15) / (293.15 * R * T_curr))
    b_T  =  safe_exp(11000.0 * (T_curr - 296.15) / (296.15 * R * T_curr))
    mrate = p.mrate0 * safe_exp(37681.0 * (T_curr - 293.30) / (293.30 * R * T_curr))
    denom = G + F + eps
    phiG = G / denom
    phiF = F / denom
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
    term_num = O2_mmol^2.3
    term_den = term_num + (KO2_MM^2.3) + eps
    vO2 = VO2_MAX * (term_num / term_den)
    du[6] = - (MW_O2 / 1000.0) * vO2 * X
    return nothing
end

function simulate_zenteno(params::ZentenoPlotParams; tspan=(0.0, th))
    u0 = copy(C0_INIT)
    # Permitimos leves negativos numéricos; el ODE ya clampea internamente.
    prob = ODEProblem(zenteno_ode!, u0, tspan, params)
    sol = try
        solve(prob, TRBDF2();
              reltol=1e-5, abstol=1e-8,
              saveat=0.1, maxiters=100_000)
    catch err
        @warn "Fallo integracion ODE (simulated Zenteno); se usara perfil constante" err
        nothing
    end
    t_dense = collect(range(tspan[1], tspan[2]; length=2001))
    if sol === nothing || sol.retcode != :Success
        if sol !== nothing && !isempty(sol.t)
            @warn "Integracion incompleta. Retcode: $(sol.retcode). Usando tramo exitoso."
            U = zeros(length(u0), length(t_dense))
            last_u = sol.u[end]
            for (i, t) in enumerate(t_dense)
                if t <= sol.t[end]
                    U[:, i] = sol(t)
                else
                    U[:, i] = last_u
                end
            end
            U .= max.(U, 0.0)
            return t_dense, U
        end
        @warn "Fallo total evaluacion ODE; se usa perfil constante"
        U = reduce(hcat, (u0 for _ in t_dense))
        return t_dense, U
    end

    U = reduce(hcat, (sol(t) for t in t_dense))
    U .= max.(U, 0.0)
    return t_dense, U
end

function co2_vol_flow_from_state(t::Float64, state_vec::AbstractVector{<:Real}, p::ZentenoPlotParams; V_liq::Float64=STRIPPING_LIQ_VOLUME)
    du = zeros(eltype(state_vec), length(state_vec))
    zenteno_ode!(du, state_vec, p, t)
    mass_CO2_h = du[5] * V_liq * 0.95
    T = dynamic_temperature(t)
    vol_CO2_h = (mass_CO2_h / 44.01) * R * T / 101325.0 * 1000.0
    return max(vol_CO2_h, 0.0)
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
    return (times=times, rates=rates, concentrations=concentrations)
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

# ---------------------------------------------
# Warm-start por trayectoria pFBA (dFBA parsimonioso)
# ---------------------------------------------
function _default_lp_factory()
    return () -> begin
        opt = Ipopt.Optimizer()
        # Silenciar y relajar un poco para rapidez en warm-start
        MOI.set(opt, MOI.RawOptimizerAttribute("print_level"), 0)
        MOI.set(opt, MOI.RawOptimizerAttribute("tol"), 1e-4)
        MOI.set(opt, MOI.RawOptimizerAttribute("max_iter"), 200)  # Límite de seguridad
        return opt
    end
end

function generate_warm_start(S::AbstractMatrix, lb_base::AbstractVector, ub_base::AbstractVector, 
                             c0_init::AbstractVector, times::AbstractVector, param_obj_idx::Int;
                             solver_factory::Function=_default_lp_factory())
    
    nv_local = size(S, 2)
    n_steps = length(times)
    
    v_guess = zeros(Float64, nv_local, n_steps)

    model_pfba = Model(solver_factory) 
    try
        set_silent(model_pfba)
    catch
    end
    
    @variable(model_pfba, v_lp[1:nv_local])
    @constraint(model_pfba, mass_bal, S * v_lp .== 0)

    curr_c = copy(c0_init) # [X, N, G, F, E]
    
    println("[WARM-START] Generando trayectoria dinámica pFBA (Min Sum v^2)...")
    
    for i in 1:n_steps
        X, N, G, F, E = curr_c
        T_curr = dynamic_temperature(times[i])
        Kg_T = exp(46055.0 * (T_curr - 293.15) / (293.15 * R * T_curr))
        lim_N = N / (N + Kn0_nom * Kg_T + 1e-6)
        lim_G = G / (G + Kg0_nom * Kg_T + 1e-6)
        lim_F = F / (F + Kf0_nom * Kg_T + 1e-6)

        dt = i < n_steps ? times[i+1] - times[i] : 1.0

        for k in 1:nv_local
            lower = lb_base[k]
            upper = ub_base[k]
            
            if k == glu
                v_kin = lower * lim_G
                v_phys = -G / (max(X, 1e-4) * dt)
                real_lb = max(v_kin, v_phys)
                set_lower_bound(v_lp[k], real_lb)
                set_upper_bound(v_lp[k], 0.0)
            elseif k == fru
                v_kin = lower * lim_F
                v_phys = -F / (max(X, 1e-4) * dt)
                real_lb = max(v_kin, v_phys)
                set_lower_bound(v_lp[k], real_lb)
                set_upper_bound(v_lp[k], 0.0)
            elseif lower < -0.1
                set_lower_bound(v_lp[k], lower * max(lim_N, 0.1))
                set_upper_bound(v_lp[k], upper)
            else
                set_lower_bound(v_lp[k], lower)
                set_upper_bound(v_lp[k], upper)
            end
        end
        
        if N < 0.01
             set_upper_bound(v_lp[param_obj_idx], 0.01)
        else
             set_upper_bound(v_lp[param_obj_idx], ub_base[param_obj_idx] * max(X, 0.1))
        end

        # Paso 1: LP max biomasa
        con_bio_loose = @constraint(model_pfba, v_lp[param_obj_idx] <= 1000.0)
        @objective(model_pfba, Max, v_lp[param_obj_idx])
        # Verificacion de seguridad de cotas
        for k in 1:nv_local
            lb_chk = lower_bound(v_lp[k])
            ub_chk = upper_bound(v_lp[k])
            if lb_chk > ub_chk
                set_lower_bound(v_lp[k], min(lb_chk, ub_chk))
                set_upper_bound(v_lp[k], max(lb_chk, ub_chk))
            end
        end
        # Blindaje contra degeneracy del solver
        solve_success = false
        try
            optimize!(model_pfba)
            st = termination_status(model_pfba)
            if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED
                solve_success = true
            end
        catch
            solve_success = false
        end
        
        if solve_success
            mu_max_step = value(v_lp[param_obj_idx])
            
            delete(model_pfba, con_bio_loose)
            con_bio_fix = @constraint(model_pfba, v_lp[param_obj_idx] >= mu_max_step * 0.999)
            @objective(model_pfba, Min, sum(v_lp[k]^2 for k in 1:nv_local))
            
            qp_success = false
            try
                optimize!(model_pfba)
                st2 = termination_status(model_pfba)
                if st2 == MOI.OPTIMAL || st2 == MOI.LOCALLY_SOLVED
                    qp_success = true
                end
            catch
                qp_success = false
            end
            
            if qp_success && has_values(model_pfba)
                v_val = value.(v_lp)
            elseif has_values(model_pfba)
                v_val = value.(v_lp) # usar LP si QP fallo pero tiene valores
            else
                v_val = nothing
            end

            if v_val !== nothing
                v_guess[:, i] = v_val
                if i < n_steps
                    mu_val = v_val[param_obj_idx]
                    dX = mu_val * X 
                    dG = v_val[glu] * X
                    dF = v_val[fru] * X
                    dN = -(mu_val / YXN_nom) * X
                    dE = v_val[eth] * X 
                    injN = smooth_injection(times[i], T_INJ_1, DOSE_1, WIDTH_1) + smooth_injection(times[i], T_INJ_2, DOSE_2, WIDTH_2)
                    
                    curr_c[1] += dX * dt
                    curr_c[2] += (dN + injN) * dt
                    curr_c[3] += dG * dt
                    curr_c[4] += dF * dt
                    curr_c[5] += dE * dt
                    curr_c .= max.(curr_c, 0.0)
                end
            else
                if i > 1
                    v_guess[:, i] = v_guess[:, i-1]
                end
            end
            
            delete(model_pfba, con_bio_fix)
        else
            if i > 1
                v_guess[:, i] = v_guess[:, i-1]
            end
        end
    end
    
    return v_guess
end

function _piecewise_linear(times::Vector{Float64}, values::Vector{Float64};
        left::Symbol=:hold, right::Symbol=:hold, fallback::Float64=0.0)
    n = length(times)
    fb = fallback
    if n == 0
        return (t -> fb)
    end
    function interp(t)
        if t <= times[1]
            return left === :hold ? values[1] : fb
        elseif t >= times[end]
            return right === :hold ? values[end] : fb
        end
        idx = searchsortedfirst(times, t)
        t0, t1 = times[idx-1], times[idx]
        v0, v1 = values[idx-1], values[idx]
        frac = (t - t0) / max(t1 - t0, eps) # eps es Float64; no llamarlo como funcion
        return v0 + frac * (v1 - v0)
    end
    return interp
end

function _build_stripping_inputs(t_vec::AbstractVector{<:Real}, states_mat::AbstractMatrix{<:Real}, params::ZentenoPlotParams)
    n = length(t_vec)
    n == size(states_mat, 2) || @warn "Dimensiones inconsistentes en estados de stripping" n_states=size(states_mat)
    k_vals = Vector{Float64}(undef, n)
    co2_vals = Vector{Float64}(undef, n)
    for i in 1:n
        state = states_mat[:, i]
        q_co2 = co2_vol_flow_from_state(Float64(t_vec[i]), state, params; V_liq=STRIPPING_LIQ_VOLUME)
        T_loc = dynamic_temperature(Float64(t_vec[i]))
        k_part = partition_coeff_from_state(state[5], T_loc)
        k_vals[i] = max(q_co2 / STRIPPING_LIQ_VOLUME * k_part, 0.0)
        co2_vals[i] = q_co2
    end
    return (times=Float64.(t_vec), k_vals=k_vals, co2_vals=co2_vals)
end

function _simulate_aroma_stripping(series, stripping_inputs)
    prod_rates = Float64.(series.rates) .* (MOLAR_MASS_ETHYL_ACETATE / 1000.0) # g/L/h
    prod_fun_obj = _piecewise_linear(Float64.(series.times), prod_rates; left=:zero, right=:zero, fallback=0.0)
    k_fun_obj = _piecewise_linear(Float64.(stripping_inputs.times), Float64.(stripping_inputs.k_vals); left=:hold, right=:hold, fallback=0.0)
    # Salvaguarda: si _piecewise_linear devolviera un escalar, lo envolvemos capturando su valor.
    if prod_fun_obj isa Function
        prod_fun = prod_fun_obj
    else
        val = prod_fun_obj
        prod_fun = (t -> val)
    end
    if k_fun_obj isa Function
        k_fun = k_fun_obj
    else
        val = k_fun_obj
        k_fun = (t -> val)
    end
    t_start = min(series.times[1], stripping_inputs.times[1])
    t_end = max(series.times[end], stripping_inputs.times[end])
    function stripping_ode!(du, u, p, t)
        k = k_fun(t)
        prod = prod_fun(t)
        du[1] = prod - k * u[1] # concentracion en medio
        du[2] = k * u[1]        # acumulado perdido
    end
    prob = ODEProblem(stripping_ode!, [STRIPPING_AROMA_INIT, 0.0], (t_start, t_end))
    sol = solve(prob, Rodas5(autodiff=false); saveat=0.25, reltol=1e-8, abstol=1e-10, maxiters=1_000_000)
    bulk = first.(sol.u)
    lost = last.(sol.u)
    return (
        bulk_times=Float64.(sol.t),
        bulk_conc=Float64.(bulk),
        lost_times=Float64.(sol.t),
        lost_conc=Float64.(lost),
    )
end

function _build_ethyl_acetate_plot_data(mpcc_tgrid, mpcc_states, v_var; stripping_inputs=nothing)
    series = _build_ethyl_acetate_series(mpcc_tgrid, mpcc_states, v_var)
    if series === nothing
        return nothing
    end
    times = Float64.(series.times)
    concentrations_mmol = Float64.(series.concentrations)
    isempty(times) && return nothing
    concentrations_g = concentrations_mmol .* (MOLAR_MASS_ETHYL_ACETATE / 1000.0)
    line_t, line_vals = _sample_cubic_spline(times, concentrations_g)
    stripping_data = nothing
    if stripping_inputs !== nothing
        try
            stripping_data = _simulate_aroma_stripping(series, stripping_inputs)
        catch err
            @warn "No se pudo simular el stripping de ethyl acetate" err
        end
    end
    return (
        times=times,
        concentrations=concentrations_g,
        line_t=line_t,
        line_vals=line_vals,
        stripping=stripping_data,
        rates=Float64.(series.rates),
    )
end

function _plot_state_series!(ax, state_idx, t_pre, states_pre, t_post, states_post,
        ts_data, data_vals, mpcc_tgrid, mpcc_states; plot_data::Bool=false)
    color = STATE_COLORS[state_idx]
    label_base = STATE_LABELS[state_idx]
    if ESTIMATE_PARAMS && t_pre !== nothing && states_pre !== nothing
        plot!(ax, t_pre, states_pre[state_idx, :];
            color=color, lw=2, linestyle=:dashdot, label="ODE pre $(label_base)")
    end
    if t_post !== nothing && states_post !== nothing
        plot!(ax, t_post, states_post[state_idx, :];
            color=color, lw=3, label="ODE post $(label_base)")
    end
    if mpcc_tgrid !== nothing && mpcc_states !== nothing
        ts_mpcc = vec(mpcc_tgrid)
        ys_mpcc = reshape(mpcc_states[state_idx, :, :], length(ts_mpcc))
        scatter!(ax, ts_mpcc, ys_mpcc;
            color=:purple, ms=5, alpha=0.9, marker=:diamond, label="MPCC $(label_base)")
    end
    if ESTIMATE_PARAMS && plot_data && data_vals !== nothing && !isempty(ts_data)
        ys_data = reshape(data_vals[state_idx, :, :], length(ts_data))
        scatter!(ax, ts_data, ys_data;
            color=color, ms=4, alpha=0.8, marker=:circle, label="Datos exp. $(label_base)")
    end
end

function _plot_temperature_panel!(ax, t_pre, t_post, mpcc_tgrid)
    # Recolectar todos los valores de temperatura para analizar el rango
    all_temps = Float64[]
    
    if ESTIMATE_PARAMS && t_pre !== nothing
        vals = dynamic_temperature.(t_pre)
        append!(all_temps, vals)
        plot!(ax, t_pre, vals;
            color=:royalblue, lw=2, linestyle=:dashdot, label="ODE pre temp")
    end
    if t_post !== nothing
        vals = dynamic_temperature.(t_post)
        append!(all_temps, vals)
        plot!(ax, t_post, vals;
            color=:firebrick, lw=3, label="ODE post temp")
    end
    if mpcc_tgrid !== nothing
        ts_mpcc = vec(mpcc_tgrid)
        vals = dynamic_temperature.(ts_mpcc)
        append!(all_temps, vals)
        scatter!(ax, ts_mpcc, vals;
            color=:purple, marker=:diamond, ms=5, alpha=0.9, label="MPCC temp")
    end
    
    # --- LÓGICA DE CENTRADO DE EJE Y ---
    if !isempty(all_temps)
        T_min, T_max = minimum(all_temps), maximum(all_temps)
        T_span = T_max - T_min
        
        # Si la temperatura es prácticamente constante (rango < 0.5 grados)
        if T_span < 0.5
            T_mid = (T_max + T_min) / 2.0
            # Forzamos un rango de +/- 2 grados para que se vea centrada y elegante
            ylims!(ax, (T_mid - 2.0, T_mid + 2.0))
        else
            # Si varía, dejamos un margen del 10% para que no toque los bordes
            ylims!(ax, (T_min - 0.1*T_span, T_max + 0.1*T_span))
        end
    end
    
    ylabel!(ax, "Temperatura [°C]") # O [K] según prefieras
end

function _plot_ethyl_acetate_panel!(ax, mpcc_tgrid, mpcc_states, v_var, stripping_inputs=nothing; ethyl_series=nothing)
    series = ethyl_series === nothing ? _build_ethyl_acetate_plot_data(mpcc_tgrid, mpcc_states, v_var; stripping_inputs=stripping_inputs) : ethyl_series
    if series === nothing
        plot!(ax, [0.0], [0.0]; color=:gray, lw=1, label="Ethyl acetate sin datos")
    else
        scatter!(ax, series.times, series.concentrations;
            color=:darkorange, ms=5, alpha=0.9, label="Colocaciones MPCC (sin stripping)")
        plot!(ax, series.line_t, series.line_vals;
            color=:navy, lw=2, label="Acumulado sin stripping")
        if series.stripping !== nothing
            plot!(ax, series.stripping.bulk_times, series.stripping.bulk_conc;
                color=:seagreen, lw=2.5, label="Conc. en caldo post stripping")
            plot!(ax, series.stripping.lost_times, series.stripping.lost_conc;
                color=:crimson, lw=2, linestyle=:dash, label="Aroma perdido por stripping")
        end
    end
    ylabel!(ax, "Ethyl acetate [g/L]")
end

function plot_post_solution(t_pre, states_pre, t_post, states_post, tgrid_data, data_vals,
        mpcc_tgrid, mpcc_states, v_var; title_str::AbstractString, save_path::AbstractString, stripping_inputs=nothing, ethyl_series=nothing)
    ts_data = tgrid_data === nothing ? Float64[] : vec(tgrid_data)
    # Layout en filas (una por variable + temp + etil acetato)
    n_panels = nc + 2
    plt = plot(layout=(n_panels, 1), size=(1000, 1900), legend=:topright, dpi=250)

    # Estados (una fila cada uno)
    for s in 1:nc
        _plot_state_series!(plt[s], s, t_pre, states_pre, t_post, states_post,
            ts_data, data_vals, mpcc_tgrid, mpcc_states; plot_data=(s in MEAS_STATES))
        ylabel!(plt[s], STATE_LABELS[s] * " [g/L]")
        if s == 1
            title!(plt[s], title_str)
        end
    end

    # Temperatura
    temp_panel = nc + 1
    _plot_temperature_panel!(plt[temp_panel], t_pre, t_post, mpcc_tgrid)
    ylabel!(plt[temp_panel], "Temp [°C]")

    # Etil acetato + stripping
    eth_panel = nc + 2
    _plot_ethyl_acetate_panel!(plt[eth_panel], mpcc_tgrid, mpcc_states, v_var, stripping_inputs; ethyl_series=ethyl_series)
    ylabel!(plt[eth_panel], "Ethyl acetate [g/L]")
    xlabel!(plt[eth_panel], "Tiempo [h]")

    save_dir = dirname(save_path)
    isdir(save_dir) || mkpath(save_dir)
    savefig(plt, save_path)
    println("[PLOT] Guardado Layout en filas: ", save_path)
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
# Datos sintéticos desde la ODE de Zenteno (con T dinámica e inyección)
# ---------------------------------------------
function _simulate_zenteno_synthetic(; nfe::Int, ncp::Int, th::Float64, c0_vec::Vector{Float64})
    params = ZentenoPlotParams(
        exp(theta_data_params[1]),
        exp(theta_data_params[2]),
        exp(theta_data_params[3]),
        exp(theta_data_params[4]),
        exp(theta_data_params[IDX_THETA_M]),
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
println("[SYNTH] Datos sinteticos listos.")

t_pre = nothing
states_pre = nothing
try
    pre_params = ZentenoPlotParams(
        exp(theta_data_params[1]),
        exp(theta_data_params[2]),
        exp(theta_data_params[3]),
        exp(theta_data_params[4]),
        exp(theta_data_params[IDX_THETA_M]),
    )
    local_t_pre, local_states_pre = simulate_zenteno(pre_params; tspan=(0.0, th))
    global t_pre = local_t_pre
    global states_pre = local_states_pre
catch err
    @warn "No se pudo simular la ODE previa a la optimizacion" err
end

# ---------------------------------------------
# MODELO JuMP
# ---------------------------------------------
m = Model(Ipopt.Optimizer)

# 1. Configuración Básica
set_optimizer_attribute(m, "linear_solver", SELECTED_IPOPT_SOLVER)

# Si se usa HSL, configurar ruta a libhsl
if SELECTED_IPOPT_SOLVER in ("ma57", "ma77", "ma86", "ma97")
    local hsl_configured = false
    try
        # Intentar usar HSL_jll si está disponible
        @eval import HSL_jll
        # Cargar OpenBLAS32 para soporte LP64 (requerido por HSL)
        @eval using LinearAlgebra, OpenBLAS32_jll
        @eval LinearAlgebra.BLAS.lbt_forward(OpenBLAS32_jll.libopenblas)
        set_optimizer_attribute(m, "hsllib", HSL_jll.libhsl_path)
        hsl_configured = true
    catch
        # Fallback: usar CoinHSL binarios locales
        coinhsl_path = raw"C:\COIN_HSL\CoinHSL.v2023.11.17.x86_64-w64-mingw32-libgfortran5\bin\libhsl.dll"
        if isfile(coinhsl_path)
            set_optimizer_attribute(m, "hsllib", coinhsl_path)
            hsl_configured = true
        end
    end
    
    if !hsl_configured
        @error "No se pudo configurar HSL en Ipopt. HSL_jll no disponible y CoinHSL no encontrado."
    end
    
    # Configuración específica para MA77 (más robusto para problemas grandes)
    if SELECTED_IPOPT_SOLVER == "ma77"
        # MA77 usa archivos temporales automáticamente - no configurar ma77_file_base
        set_optimizer_attribute(m, "ma77_print_level", -1)  # Silenciar MA77
        set_optimizer_attribute(m, "ma77_order", "metis")   # Reordenamiento con METIS
        set_optimizer_attribute(m, "ma77_small", 1e-20)     # Umbral para pivotes pequeños
        set_optimizer_attribute(m, "ma77_u", 0.01)          # Tolerancia de pivoteo
    end
end

set_optimizer_attribute(m, "print_level", 5)
if DIAG_SIMPLE
    set_optimizer_attribute(m, "max_iter", 5000)
    set_optimizer_attribute(m, "tol", 1e-2)
    set_optimizer_attribute(m, "acceptable_iter", 5)
    set_optimizer_attribute(m, "acceptable_tol", 1e-1)
    set_optimizer_attribute(m, "acceptable_constr_viol_tol", 1e-1)
else
    set_optimizer_attribute(m, "max_iter", 1000) # Damos más iteraciones por si el modo adaptativo es lento
    # 2. Tolerancia Estricta (El objetivo ideal)
    set_optimizer_attribute(m, "tol", 1e-4)
    # 3. Estrategia de Terminación Aceptable ("Caza-Óptimos")
    # Si el solver se atasca cerca de la solución pero no puede bajar el error dual, se detiene aquí.
    set_optimizer_attribute(m, "acceptable_iter", 5)       # Mantenerse estable 5 iteraciones
    set_optimizer_attribute(m, "acceptable_tol", 1e-1)     # Tolerancia relajada (suficiente para ingeniería)
    set_optimizer_attribute(m, "acceptable_constr_viol_tol", 1e-2) # Violación de restricciones aceptable
    # set_optimizer_attribute(m, "acceptable_dual_inf_tol", 1e10)   # ¡CRÍTICO! Ignora el ruido dual del MPCC
    set_optimizer_attribute(m, "acceptable_compl_inf_tol", 1e-2)   # Tolerancia de complementariedad
end

# 4. Estrategia de Barrera (Mu Strategy) - Anti-Rebote
# "adaptive" es más lento pero mucho más seguro que "monotone" para problemas no convexos.
set_optimizer_attribute(m, "mu_strategy", "adaptive") # monotone adaptive
set_optimizer_attribute(m, "mu_oracle", "quality-function") # Ayuda a elegir mejor el paso adaptativo

# 5. Manejo de Cotas y Escalado
set_optimizer_attribute(m, "bound_relax_factor", 1e-4) # 0.0 para respetar estrictamente c >= 0 (física)
set_optimizer_attribute(m, "honor_original_bounds", "yes")
set_optimizer_attribute(m, "nlp_scaling_method", "gradient-based") # A veces ayuda si los flujos tienen escalas muy distintas

@variables(m, begin
    c[1:nc, 1:ph, 1:ncp] >= 0.0
    cdot[1:nc, 1:ph, 1:ncp]
    teta[1:np]
    v[1:nv, 1:nfe]
    lambda_[1:nm, 1:nfe]
    hv[1:nfe] >= 0.0
end)
const DIAG_ALPHA_U_ACTIVE = (!DIAG_SIMPLE) || DIAG_ENABLE_FO_U

if DIAG_ALPHA_U_ACTIVE
    @variables(m, begin
        alpha_U[1:nv, 1:nfe]>= 0.0
        FO_U[1:nv, 1:nfe]
    end)
end

if !DIAG_SIMPLE
    @variables(m, begin
        alpha_L[1:nv, 1:nfe]<= 0.0
        alpha_upt[1:n_up, 1:nfe]<= 0.0
        FO_L[1:nv, 1:nfe]
        FO_upt[1:n_up, 1:nfe]
    end)
end

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
set_start_value(teta[IDX_THETA_M], log(0.01))

if USE_WARM_START
    # Warm Start: primero intenta cargar semilla guardada; si no hay, se arranca en cero (sin pFBA)
    local seed_loaded = false
    if isfile(WARM_START_FILE)
        try
            JLD2.jldopen(WARM_START_FILE, "r") do f
                if haskey(f, "v_seed")
                    v_seed = read(f, "v_seed")
                    for i in 1:nfe, k in 1:nv
                        set_start_value(v[k, i], v_seed[k, i])
                    end
                end
                if !DIAG_SIMPLE && haskey(f, "alpha_upt_seed")
                    au_seed = read(f, "alpha_upt_seed")
                    for i in 1:nfe, k in 1:n_up
                        set_start_value(alpha_upt[k, i], au_seed[k, i])
                    end
                end
                if haskey(f, "teta_seed")
                    teta_seed = read(f, "teta_seed")
                    for k in 1:min(np, length(teta_seed))
                        set_start_value(teta[k], teta_seed[k])
                    end
                end
                if haskey(f, "hv_seed")
                    hv_seed = read(f, "hv_seed")
                    for i in 1:min(nfe, length(hv_seed))
                        set_start_value(hv[i], hv_seed[i])
                    end
                end
            end
            seed_loaded = true
            println("[WARM-START] Semilla cargada desde ", WARM_START_FILE)
        catch err
            @warn "[WARM-START] No se pudo cargar la semilla guardada; usando pFBA" err
        end
    else
        println("[WARM-START] No hay semilla en ", WARM_START_FILE, "; inicio sin warm start guardado.")
    end
    if !seed_loaded
        println("[WARM-START] Sin semilla guardada; inicio en cero (sin pFBA).")
        for i in 1:nfe, k in 1:nv; set_start_value(v[k,i], 0.0); end
        if !DIAG_SIMPLE
            for i in 1:nfe, k in 1:n_up; set_start_value(alpha_upt[k,i], 0.0); end
        end
    end
else
    # Cold Start
    for i in 1:nfe, k in 1:nv; set_start_value(v[k,i], 0.0); end
    if !DIAG_SIMPLE
        for i in 1:nfe, k in 1:n_up; set_start_value(alpha_upt[k,i], 0.0); end
    end
end

# Normalización C0 local para el modelo
for i in 1:nc; c0[i] = c0[i] / cs[i]; end

# ============================================================
# PARTE 1: FUNCION OBJETIVO
# ============================================================
@NLexpression(m, FO_expr, sum( sum( sum( (data[i,j,mc] - c[i,j,mc])^2 for i in 1:nc) for j in 1:ph) for mc in 1:ncp))

if DIAG_SIMPLE
    @NLobjective(m, Min, omega * FO_expr)
else
    @NLobjective(m, Min,
        omega * FO_expr +
        phi1 * sum(FO_L[mc,i]^2   for mc in 1:nv,  i in 1:nfe) +
        phi3 * sum(FO_U[mc,i]^2   for mc in 1:nv,  i in 1:nfe) +
        phi2 * sum(FO_upt[k,i]^2  for k  in 1:n_up, i in 1:nfe)
    )
end

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
@NLexpression(m, mrate_ij[i=1:ph, j=1:ncp],
    exp(teta[IDX_THETA_M]) *
    exp(37681.0 * (T_loc[i,j] - 293.30) / (293.30 * R * T_loc[i,j])))

@NLexpression(m, mu_j[i=1:ph, j=1:ncp], mu0 * mu_T_ij[i,j] * (c[2,i,j] / (c[2,i,j] + Kn0_nom * Kg_T_ij[i,j] + eps)))
@NLexpression(m, betaG_j[i=1:ph, j=1:ncp], betaG0_nom * b_T_ij[i,j] * (c[3,i,j] / (c[3,i,j] + Kg0_nom * Kg_T_ij[i,j] + eps)) * (Kie0_nom * Kg_T_ij[i,j] / (c[5,i,j] + Kie0_nom * Kg_T_ij[i,j] + eps)))
@NLexpression(m, betaF_j[i=1:ph, j=1:ncp], betaF0_nom * b_T_ij[i,j] * (c[4,i,j] / (c[4,i,j] + Kf0_nom * Kg_T_ij[i,j] + eps)) * (Kig0_nom * Kg_T_ij[i,j] / (c[3,i,j] + Kig0_nom * Kg_T_ij[i,j] + eps)) * (Kie0_nom * Kg_T_ij[i,j] / (c[5,i,j] + Kie0_nom * Kg_T_ij[i,j] + eps)))
@NLexpression(m, Kd_j[i=1:ph, j=1:ncp], death_rate_T(c[5,i,j], T_loc[i,j]))
@NLexpression(m, injection_rate[i=1:nfe, j=1:ncp], smooth_injection((i - 1 + radau_nodes[j]) * hv[i], T_INJ_1, DOSE_1, WIDTH_1) + smooth_injection((i - 1 + radau_nodes[j]) * hv[i], T_INJ_2, DOSE_2, WIDTH_2))

# Oxígeno
@NLexpression(m, O2_conc_mM[i=1:ph, j=1:ncp], (c[6,i,j] * 1000.0) / MW_O2)
@NLexpression(m, v_limit_O2[i=1:ph, j=1:ncp], 
    VO2_MAX * (
        (O2_conc_mM[i,j]^n_Hill) / 
        ( (O2_conc_mM[i,j]^n_Hill) + (KO2_MM^n_Hill) + 1e-9 )
    ))
@NLexpression(m, signal_anaerobic[i=1:nfe], 1.0 / (1.0 + ((c[6,i,3] * 1000.0 / MW_O2) / 0.01)^2))

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
const MW_N   = 0.014007 #g/mmol
const MW_GLU = 0.180156
const MW_FRU = 0.180156
const MW_ETH = 0.046070

# ============================================================
# PARTE 4: LÍMITES DINÁMICOS
# ============================================================
@NLexpressions(m, begin
# 1. Macro Zenteno (Calculados en GRAMOS/gDW/h)    
    vx[i=1:ph],  (mu_j[i,3])
    vg[i=1:ph],  (mu_j[i,3]/Yxg + betaG_j[i,3]/Yeg + mrate_ij[i,3]*(c[3,i,3]/(c[3,i,3]+c[4,i,3])))
    vf[i=1:ph],  (mu_j[i,3]/YXF_nom + betaF_j[i,3]/Yef + mrate_ij[i,3]*(c[4,i,3]/(c[3,i,3]+c[4,i,3])))
    vn[i=1:ph],  (mu_j[i,3]/Yxn)

# 2. Desglose del Nitrógeno (Transformación a mmol de METABOLITO)
    # Fórmula: (gN_total * Ratio) / (MW_N * Atomos) * 1000
    # MW_N convierte gN -> molN
    # N_atoms convierte molN -> molMetabolito
    # 1000 convierte mol -> mmol
    v_limit_N[k=1:nv, i=1:ph], 
        (vn[i] * N_profile_vec[k]) / (N_atoms_vec[k] * MW_N)

# 3. Wrapper L_uptake (Todo en mmol)
    # Convertimos también Glucosa y Fructosa a mmol para que L_uptake sea homogéneo 1/h : g/mmol = mmol/gh
    L_uptake[k=1:n_up, i=1:ph], 
        IS_GLU[k] * (vg[i] / MW_GLU) + 
        IS_FRU[k] * (vf[i] / MW_FRU) + 
        IS_NIT[k] * v_limit_N[UPTAKE_IDXS[k], i]
end)

# ============================================================
# PARTE 5: RESTRICCIONES
# ============================================================
@constraints(m, begin
    # Colocación
    coll_c_0[l=1:nc, j=1:ncp], c[l,1,j] == c0[l] + hv[1] * sum(colmat[j,k] * cdot[l,1,k] for k in 1:ncp)
    coll_c_n[l=1:nc, i=2:ph, j=1:ncp], c[l,i,j] == c[l,i-1,ncp] + hv[i] * sum(colmat[j,k]*cdot[l,i,k] for k in 1:ncp)
    
    # FBA
    Sc[mc=1:nm,i=1:nfe],  sum(S[mc,k]*v[k,i]*vs[k] for k in 1:nv) == 0
    v_UB[mc=1:nv, i=1:nfe], v[mc,i]*vs[mc] - ub[mc] <= 0
    v_LB[mc=1:nv,i=1:nfe], -v[mc,i]*vs[mc] + lb[mc] <= 0

    # Time-step
    MFE1, sum(hv[i] for i in 1:nfe) == th
    MFE3[i=1:nfe], hv[i] >= 0.0
    MFE4[i=1:nfe], hv[i] >= (1.0 - var_h) * hm[1]
    MFE5[i=1:nfe], hv[i] <= (1.0 + var_h) * hm[1]
end)

if !DIAG_SIMPLE
    @constraints(m, begin
        # KKT Lagrangiano
        Lagr[mc=1:nv,i=1:nfe], d[mc] + w*v[mc,i]*vs[mc] + alpha_L[mc,i] + alpha_U[mc,i] + sum(SELECT_UPTAKE[mc,k] * alpha_upt[k,i] for k in 1:n_up) + sum(S[k,mc]*lambda_[k,i] for k in 1:nm) == 0
    end)
end

if ESTIMATE_PARAMS
    if CALIBRATE_ONLY_THETA_M
        @constraints(m, begin
            teta_fix_data[p=1:np-1], teta[p] == theta_data_params[p]
            tetaM_LB, teta[IDX_THETA_M] >= LB[IDX_THETA_M]
            tetaM_UB, teta[IDX_THETA_M] <= UB[IDX_THETA_M]
        end)
    else
        @constraints(m, begin
            teta_LB[p=1:np], teta[p] >= LB[p]
            teta_UB[p=1:np], teta[p] <= UB[p]
        end)
    end
else
    @constraints(m, begin
        teta_fix[p=1:np], teta[p] == theta_data_params[p]
    end)
end

if TEST_FIX_THETA_M
    JuMP.fix(teta[IDX_THETA_M], log(0.01); force=true)
end

# ============================================================
# PARTE 6: ODES & COMPLEMENTARIEDAD
# ============================================================


@NLconstraints(m, begin

# ==========================================
    # 1. ECUACIONES DIFERENCIALES (ODES)
# ==========================================

    m1[i=1:ph, j=1:ncp], cdot[1,i,j] == 
        (v[obj,i] - Kd_j[i,j]) * c[1,i,j]
    m2[i=1:ph, j=1:ncp], cdot[2,i,j] ==
        - MW_N * (
            sum(
                IS_NIT[k] *
                (-v[UPTAKE_IDXS[k], i] * vs[UPTAKE_IDXS[k]]) *
                N_atoms_vec[UPTAKE_IDXS[k]]
                for k = 1:n_up
            )
        ) * c[1,i,j] + injection_rate[i,j]
    m3[i=1:ph, j=1:ncp], cdot[3,i,j] == 
        - MW_GLU * (-v[glu,i]*vs[glu]) * c[1,i,j]
    m4[i=1:ph, j=1:ncp], cdot[4,i,j] == 
        - MW_FRU * (-v[fru,i]*vs[fru]) * c[1,i,j]
    m5[i=1:ph, j=1:ncp], cdot[5,i,j] == 
        MW_ETH * v[eth,i] * c[1,i,j]
    m6[i=1:ph, j=1:ncp], cdot[6,i,j] == 
        - MW_O2 * (-v[o2,i]*vs[o2]) * c[1,i,j]
    
# ==========================================
    # 2. RESTRICCIONES DE ACOPLAMIENTO            
# ==========================================
    # Restricción de Crecimiento (Semi-fijación)
    growth_UB_dyn[i=1:nfe], 
        v[obj, i] * vs[obj] <= vx[i]

    # Consumo de O2 acotado por cinética (Monod)
    v_LB_O2_uptake[i=1:nfe], -v[o2, i] * vs[o2] - v_limit_O2[i,3] <= 0

    # Control de válvulas bypass (Heme/CoQ) según señal anaeróbica
    bypass_ctrl[k=IDX_BYPASS_START:IDX_BYPASS_END, i=1:nfe], v[k, i] * vs[k] <= 1000.0 * signal_anaerobic[i]

    # Uptake Coupling 
    v_LB_uptake[k=1:n_up, i=1:nfe], 
        -v[UPTAKE_IDXS[k], i] * vs[UPTAKE_IDXS[k]] - L_uptake[k,i] <= 0

end)

if !DIAG_SIMPLE
@NLconstraints(m, begin
    # Complementariedad
    FO_upt_cons[k=1:n_up, i=1:nfe],
        FO_upt[k,i] == (-v[UPTAKE_IDXS[k], i] * vs[UPTAKE_IDXS[k]] - L_uptake[k,i]) * alpha_upt[k,i]

    # Bounds Complementarity
    FO1[mc=1:nv,i=1:nfe], FO_L[mc,i] == (v[mc,i]*vs[mc] -lb[mc])*alpha_L[mc,i]
    FO2[mc=1:nv,i=1:nfe], FO_U[mc,i] == (v[mc,i]*vs[mc] -ub[mc])*alpha_U[mc,i]
end)
elseif DIAG_ENABLE_FO_U
@NLconstraints(m, begin
    # Solo complementariedad de cota superior en modo diagnóstico
    FO2_diag[mc=1:nv,i=1:nfe], FO_U[mc,i] == (v[mc,i]*vs[mc] - ub[mc]) * alpha_U[mc,i]
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
println("Objective FO   = ", safe_value(FO_expr))
println("Wall time (s)  = ", wall_time)

# ============================================================
# DIAGNÓSTICO DIMENSIONAL MICRO → MACRO (DEBUG)
# ============================================================
function print_dimensional_diagnostics(i=1, j=1)
    println("\n========== DIMENSIONAL DIAGNOSTICS ==========")

    # ---------------------------
    # 1) Biomasa
    # ---------------------------
    X      = value(c[1,i,j])           # gDW/L
    Xdot   = value(cdot[1,i,j])        # gDW/L/h
    mu     = value(mu_j[i,j])          # 1/h
    vx_loc = value(vx[i])              # debería ser 1/h (Zenteno)

    println("---- BIOMASA ----")
    println("X        (gDW/L)     = ", X)
    println("Xdot     (gDW/L/h)   = ", Xdot)
    println("mu       (1/h)       = ", mu)
    println("mu*X     (gDW/L/h)   = ", mu * X)
    println("vx (definido)       = ", vx_loc)

    # ---------------------------
    # 2) Glucosa
    # ---------------------------
    v_glu = value(v[glu,i])            # mmol/gDW/h
    Gdot  = value(cdot[3,i,j])         # g/L/h

    glu_macro = MW_GLU * (-v_glu) * X  # g/L/h

    println("\n---- GLUCOSA ----")
    println("v_glu (mmol/gDW/h)        = ", v_glu)
    println("GLU macro FBA (g/L/h)    = ", glu_macro)
    println("GLU ODE cdot[3] (g/L/h)  = ", Gdot)

    # ---------------------------
    # 3) Fructosa
    # ---------------------------
    v_fru = value(v[fru,i])            # mmol/gDW/h
    Fdot  = value(cdot[4,i,j])         # g/L/h

    fru_macro = MW_FRU * (-v_fru) * X  # g/L/h

    println("\n---- FRUCTOSA ----")
    println("v_fru (mmol/gDW/h)        = ", v_fru)
    println("FRU macro FBA (g/L/h)    = ", fru_macro)
    println("FRU ODE cdot[4] (g/L/h)  = ", Fdot)

    # ---------------------------
    # 4) Nitrógeno
    # ---------------------------
    vn_loc = value(vn[i])              # gN/gDW/h (según tu definición)
    Ndot   = value(cdot[2,i,j])        # gN/L/h

    N_inner = sum(
        IS_NIT[k] *
        (-value(v[UPTAKE_IDXS[k], i]) * vs[UPTAKE_IDXS[k]]) *
        N_atoms_vec[UPTAKE_IDXS[k]]
        for k in 1:length(UPTAKE_IDXS)
    )                                  # mmol N/gDW/h

    N_macro = MW_N * N_inner * X       # gN/L/h
    inj = value(injection_rate[i,j])

    macro_from_v = MW_N * value(c[1,i,j]) * N_inner

    println("\n---- NITRÓGENO ----")
    println("vn (gN/gDW/h)            = ", vn_loc)
    println("Sum N uptake (mmol/gDW/h)= ", N_inner)
    println("N macro FBA (gN/L/h)     = ", N_macro)
    println("N ODE cdot[2] (gN/L/h)   = ", Ndot)
    println(" -- inner (mmol/gDW/h)   = ", N_inner)
    println(" -- macro_from_v (gN/L/h)= ", macro_from_v)
    println(" -- cdot2 (gN/L/h)       = ", Ndot)
    println(" -- inj (gN/L/h)         = ", inj)
    println(" -- cdot2 - inj          = ", Ndot - inj)

    # ---------------------------
    # 5) Crecimiento FBA vs ODE
    # ---------------------------
    v_bio = value(v[obj,i])            # mmol/gDW/h
    bio_macro = v_bio * X              # mmol/L/h (sin MW)

    println("\n---- CRECIMIENTO ----")
    println("v_biomass (mmol/gDW/h)   = ", v_bio)
    println("v_biomass*X (mmol/L/h)  = ", bio_macro)
    println("mu*X (gDW/L/h)          = ", mu * X)

    println("============================================\n")
end

print_dimensional_diagnostics(1, 3)



objective_val = try
    objective_value(m)
catch
    NaN
end
fo_val = safe_value(FO_expr, NaN)
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

function report_residuals()
    try
        _v(x) = begin
            try
                value.(x; result=1)
            catch
                try
                    value.(x)
                catch
                    fill(NaN, size(x))
                end
            end
        end

        vval  = _v(v)
        cval  = _v(c)
        hvval = _v(hv)
        cdval = _v(cdot)
        lam   = _v(lambda_)
        Llim  = _v(L_uptake)
        vxval = _v(vx)
        alL   = DIAG_SIMPLE ? zeros(nv, nfe) : _v(alpha_L)
        alU   = DIAG_ALPHA_U_ACTIVE ? _v(alpha_U) : zeros(nv, nfe)
        alUp  = DIAG_SIMPLE ? zeros(n_up, nfe) : _v(alpha_upt)

        coll0 = [cval[l,1,j] - (c0[l] + hvval[1] * sum(colmat[j,k] * cdval[l,1,k] for k in 1:ncp))
                 for l in 1:nc, j in 1:ncp]
        colln = [cval[l,i,j] - (cval[l,i-1,ncp] + hvval[i] * sum(colmat[j,k] * cdval[l,i,k] for k in 1:ncp))
                 for l in 1:nc, i in 2:ph, j in 1:ncp]
        fba   = [sum(S[mc,k] * vval[k,i] * vs[k] for k in 1:nv) for mc in 1:nm, i in 1:nfe]
        ubv   = [vval[mc,i] * vs[mc] - ub[mc] for mc in 1:nv, i in 1:nfe]
        lbv   = [-vval[mc,i] * vs[mc] + lb[mc] for mc in 1:nv, i in 1:nfe]
        lagr  = [d[mc] + w * vval[mc,i] * vs[mc] + alL[mc,i] + alU[mc,i] +
                 sum(SELECT_UPTAKE[mc,k] * alUp[k,i] for k in 1:n_up) +
                 sum(S[k,mc] * lam[k,i] for k in 1:nm)
                 for mc in 1:nv, i in 1:nfe]
        grow  = (1 <= obj <= nv) ? [vval[obj,i] * vs[obj] - vxval[i] for i in 1:nfe] : zeros(nfe)
        upt   = [-vval[UPTAKE_IDXS[k], i] * vs[UPTAKE_IDXS[k]] - Llim[k, i] for k in 1:n_up, i in 1:nfe]

        println("[RESID TYPES] coll0=", typeof(coll0), " eltype=", eltype(coll0))
        println("[RESID TYPES] colln=", typeof(colln), " eltype=", eltype(colln))
        println("[RESID TYPES] fba=", typeof(fba), " eltype=", eltype(fba))

        # Tiempos locales para contextualizar
        hv_prefix = vcat(0.0, cumsum(hvval[1:end-1]))
        t_end = cumsum(hvval)
        t_colloc(i, j) = hv_prefix[i] + radau_nodes[j] * hvval[i]

        _finite_summary(name, arr) = begin
            finite_vals = Float64[]
            if arr isa AbstractArray
                for x in arr
                    if x isa Real && isfinite(x)
                        push!(finite_vals, abs(x))
                    end
                end
            else
                if arr isa Real && isfinite(arr)
                    push!(finite_vals, abs(arr))
                end
            end
            if isempty(finite_vals)
                println(@sprintf("%-16s finite=0", name))
            else
                println(@sprintf("%-16s finite=%d max|x|=%.3e", name, length(finite_vals), maximum(finite_vals)))
            end
        end

        _max_with_label(name, arr, label_fn) = begin
            println("[MAX LABEL] name=", name, " typeof=", typeof(name), " arr eltype=", arr isa AbstractArray ? string(eltype(arr)) : string(typeof(arr)))
            max_val = -Inf
            max_idx = nothing
            raw_val = nothing
            if arr isa AbstractArray
                for idx in CartesianIndices(arr)
                    x = arr[idx]
                    x isa Real || continue
                    isfinite(x) || continue
                    ax = abs(x)
                    if ax > max_val
                        max_val = ax
                        max_idx = idx
                        raw_val = x
                    end
                end
            else
                x = arr
                if x isa Real && isfinite(x)
                    max_val = abs(x)
                    max_idx = 1
                    raw_val = x
                end
            end
            if max_idx === nothing
                println(@sprintf("%-16s max=NA (no numeric entries)", name))
                return
            end
            raw_str = raw_val isa Real ? @sprintf("%.3e", raw_val) : string(raw_val)
            println(@sprintf("%-16s max=%.3e %s raw=%s", name, max_val, label_fn(max_idx, raw_val), raw_str))
        end

        rxn_name(mc) = (1 <= mc <= length(RXN_IDS)) ? RXN_IDS[mc] : string(mc)

        println("=== Post-solve residuals ===")
        _finite_summary("colloc0", coll0)
        _max_with_label("colloc0", coll0, (idx, _) -> @sprintf("at t=%.2f l=%d j=%d", radau_nodes[idx.I[2]]*hvval[1], idx.I[1], idx.I[2]))
        _finite_summary("collocN", colln)
        _max_with_label("collocN", colln, (idx, _) -> @sprintf("at t=%.2f l=%d j=%d", t_colloc(idx.I[2], idx.I[3]), idx.I[1], idx.I[3]))
        _finite_summary("FBA", fba)
        _max_with_label("FBA", fba, (idx, _) -> @sprintf("rxn=%s t=%.2f", rxn_name(idx.I[1]), t_end[idx.I[2]]))
        _finite_summary("v UB", ubv)
        _max_with_label("v UB", ubv, (idx, _) -> begin mc = idx.I[1]; i = idx.I[2]; @sprintf("rxn=%s t=%.2f ub=%.3e v=%.3e", rxn_name(mc), t_end[i], ub[mc], vval[mc,i]*vs[mc]) end)
        _finite_summary("v LB", lbv)
        _max_with_label("v LB", lbv, (idx, _) -> begin mc = idx.I[1]; i = idx.I[2]; @sprintf("rxn=%s t=%.2f lb=%.3e v=%.3e", rxn_name(mc), t_end[i], lb[mc], vval[mc,i]*vs[mc]) end)
        _finite_summary("Lagr", lagr)
        _max_with_label("Lagr", lagr, (idx, _) -> @sprintf("rxn=%s t=%.2f", rxn_name(idx.I[1]), t_end[idx.I[2]]))
        _finite_summary("growth_UB", grow)
        _max_with_label("growth_UB", grow, (idx, _) -> @sprintf("t=%.2f", t_end[idx.I[1]]))
        _finite_summary("uptake", upt)
        _max_with_label("uptake", upt, (idx, _) -> begin k = idx.I[1]; i = idx.I[2]; mc = UPTAKE_IDXS[k]; @sprintf("rxn=%s t=%.2f L=%.3e v=%.3e", rxn_name(mc), t_end[i], Llim[k,i], vval[mc,i]*vs[mc]) end)

        if !DIAG_SIMPLE
            foL   = [(vval[mc,i] * vs[mc] - lb[mc]) * alL[mc,i] for mc in 1:nv, i in 1:nfe]
            _max_with_label("FO_L", foL, (idx, _) -> begin mc = idx.I[1]; i = idx.I[2]; @sprintf("rxn=%s t=%.2f", rxn_name(mc), t_end[i]) end)
        end

        if (!DIAG_SIMPLE) || DIAG_ENABLE_FO_U
            foU   = [(vval[mc,i] * vs[mc] - ub[mc]) * alU[mc,i] for mc in 1:nv, i in 1:nfe]
            _max_with_label("FO_U", foU, (idx, _) -> begin mc = idx.I[1]; i = idx.I[2]; @sprintf("rxn=%s t=%.2f", rxn_name(mc), t_end[i]) end)
        end

        if !DIAG_SIMPLE
            fou   = [(-vval[UPTAKE_IDXS[k], i] * vs[UPTAKE_IDXS[k]] - Llim[k, i]) * alUp[k,i] for k in 1:n_up, i in 1:nfe]
            _max_with_label("FO_upt", fou, (idx, _) -> begin k = idx.I[1]; i = idx.I[2]; mc = UPTAKE_IDXS[k]; @sprintf("rxn=%s t=%.2f", rxn_name(mc), t_end[i]) end)
        end

        println(@sprintf("%-16s min=%.3e", "c min", minimum(cval)))

    catch err
        @warn "No se pudieron imprimir residuales" err
    end
end

function report_uptake_details()
    try
        vval  = value.(v)
        Llim  = value.(L_uptake)
        au    = DIAG_SIMPLE ? zeros(n_up, nfe) : value.(alpha_upt)
        upt   = [-vval[UPTAKE_IDXS[k], i] * vs[UPTAKE_IDXS[k]] - Llim[k, i] for k in 1:n_up, i in 1:nfe]
        mx, idx = findmax(abs.(upt))
        k = idx.I[1]; i = idx.I[2]; mc = UPTAKE_IDXS[k]
        println("=== Uptake detail ===")
        println("UPTAKE_IDXS = ", UPTAKE_IDXS)
        println(@sprintf("max uptake viol=%.3e at k=%d (mc=%d) i=%d raw=%.3e L_upt=%.3e v=%.3e alpha_upt=%.3e",
            mx, k, mc, i, upt[k,i], Llim[k,i], vval[mc,i], au[k,i]))
        for kk in 1:min(n_up, 5)
            jmax = min(nfe, 3)
            vals_upt = [upt[kk, j] for j in 1:jmax]
            vals_L   = [Llim[kk, j] for j in 1:jmax]
            vals_v   = [vval[UPTAKE_IDXS[kk], j] for j in 1:jmax]
            println(@sprintf("k=%d mc=%d upt[1..%d]=%s L_upt[1..%d]=%s v[mc,1..%d]=%s",
                kk, UPTAKE_IDXS[kk], jmax, string(vals_upt), jmax, string(vals_L), jmax, string(vals_v)))
        end
    catch err
        @warn "No se pudo imprimir detalle de uptake" err
    end
end
function save_warm_start_seed(path::AbstractString)
    try
        v_seed = zeros(nv, nfe)
        for i in 1:nfe, k in 1:nv
            v_seed[k, i] = safe_value(v[k, i], 0.0)
        end
        alpha_upt_seed = DIAG_SIMPLE ? nothing : zeros(n_up, nfe)
        if !DIAG_SIMPLE
            for i in 1:nfe, k in 1:n_up
                alpha_upt_seed[k, i] = safe_value(alpha_upt[k, i], 0.0)
            end
        end
        hv_seed = [safe_value(hv[i], hm[i]) for i in 1:nfe]
        teta_seed = [safe_value(teta[k], theta_data_params[k]) for k in 1:np]
        if DIAG_SIMPLE
            JLD2.jldsave(path; v_seed=v_seed, hv_seed=hv_seed, teta_seed=teta_seed,
                         omega=omega, phi1=phi1, phi2=phi2, phi3=phi3, timestamp=Dates.now())
        else
            JLD2.jldsave(path; v_seed=v_seed, alpha_upt_seed=alpha_upt_seed, hv_seed=hv_seed, teta_seed=teta_seed,
                         omega=omega, phi1=phi1, phi2=phi2, phi3=phi3, timestamp=Dates.now())
        end
        println("[WARM-START] Semilla guardada en ", path)
    catch err
        @warn "[WARM-START] No se pudo guardar la semilla" err
    end
end

report_residuals()
report_uptake_details()
save_warm_start_seed(WARM_START_FILE)

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
        theta_final_vals[IDX_THETA_M],
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
    ethyl_series = _build_ethyl_acetate_plot_data(mpcc_tgrid, mpcc_states, v; stripping_inputs=stripping_inputs)
    plot_post_solution(
        t_pre, states_pre, t_post, states_post, DATA_TIME_GRID, data,
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
