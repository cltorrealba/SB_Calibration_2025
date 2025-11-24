using Clapeyron
using DifferentialEquations
using Plots
using LinearAlgebra

# ==========================================
# 1. Termodinámica (Clapeyron)
# ==========================================
# Especies: agua, etanol y el aroma (ethyl hexanoate)
species = ["water", "ethanol", "isoamylacetate"]

# Listado de especies UNIFAC para avisar si falta alguna
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
const ISOAMYL_USER_GROUPS = joinpath(@__DIR__, "isoamyl_acetate_unifac.csv")
const USER_UNIFAC_SPECIES = load_available_unifac_species(ISOAMYL_USER_GROUPS)
const AVAILABLE_UNIFAC_SPECIES = vcat(load_available_unifac_species(UNIFAC_GROUPS_CSV), USER_UNIFAC_SPECIES)
# Construimos un set normalizado incluyendo cada sinónimo separado por "/" para evitar falsos faltantes
function _normalize_unifac_entry(entry::AbstractString)
    return lowercase(strip(String(entry)))
end
synonyms = String[]
for entry in AVAILABLE_UNIFAC_SPECIES
    parts = split(entry, "/")
    for p in parts
        norm = _normalize_unifac_entry(p)
        push!(synonyms, norm)
        push!(synonyms, replace(norm, " " => "")) # alias sin espacios (p.ej. isoamylacetate)
    end
end
const AVAILABLE_UNIFAC_SET = Set(synonyms)

function print_available_unifac_species()
    println("\nComponentes UNIFAC (Clapeyron) disponibles: $(length(AVAILABLE_UNIFAC_SPECIES))")
    for comp in AVAILABLE_UNIFAC_SPECIES
        println(" - ", comp)
    end
end
print_available_unifac_species()

function build_activity_model(species)
    missing = [s for s in species if !(_normalize_unifac_entry(s) in AVAILABLE_UNIFAC_SET)]
    if !isempty(missing)
        @info "UNIFAC no disponible para algunos componentes; se usara idealidad (gamma=1)." faltantes=missing
        return nothing
    end
    try
        group_locs = isfile(ISOAMYL_USER_GROUPS) ? [ISOAMYL_USER_GROUPS] : String[]
        return UNIFAC(species; group_userlocations=group_locs)
    catch err
        @warn "No se encontraron todos los parametros UNIFAC; se asumira idealidad (gamma=1)." exception=err
        return nothing
    end
end
model = build_activity_model(species)

# Parámetros puros desde la base de datos + fallback manual
const SPECIES_PARAM_LOCATIONS = [
    "properties/molarmass.csv",
    "properties/critical.csv",
    "Correlations/saturation_correlations/dippr101_like.csv",
]
antoine_mmhg_to_pa(A, B, C, T) = 133.322368 * 10.0^(A - B / ((T - 273.15) + C))
const MANUAL_SPECIES_FALLBACK = Dict(
    "water" => (mw = 18.015, psat = (type = :custom, eval = (T -> exp(23.1964 - 3816.44 / (T - 46.13))))),
    "ethanol" => (mw = 46.07, psat = (type = :custom, eval = (T -> exp(23.8381 - 3803.98 / (T - 41.68))))),
    "ethyl hexanoate" => (mw = 144.21, psat = (type = :custom, eval = (T -> exp(20.7 - 3550.0 / (T - 60.0))))),
    # Fallback aproximado para isoamyl acetate, usando ajuste exp(a - b/T) con datos 25-50 °C (P≈0.5-2.1 kPa)
    "isoamylacetate" => (mw = 130.18, psat = (type = :custom, eval = (T -> exp(24.16 - 5330.0 / T)))),
)

function fetch_species_properties(species::Vector{String})
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
                verbose = false,
                ignore_missing_singleparams = ["A","B","C","D","E","Tmin","Tmax"],
            )
        catch err
            @warn "No se pudieron recuperar los parametros puros desde la base de Clapeyron." exception = err
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
        dippr_keys = ["A", "B", "C", "D", "E", "Tmin", "Tmax"]
        if all(k -> haskey(params, k), dippr_keys)
            dippr_data = Dict(k => collect(params[k].values) for k in dippr_keys)
            for (loc_idx, global_idx) in enumerate(to_query_idx)
                # evitamos usar correlaciones con valores faltantes (Clapeyron rellena con cero)
                if any(params[k].ismissingvalues[loc_idx] for k in dippr_keys)
                    continue
                end
                vals = (dippr_data["A"][loc_idx], dippr_data["B"][loc_idx], dippr_data["C"][loc_idx],
                        dippr_data["D"][loc_idx], dippr_data["E"][loc_idx], dippr_data["Tmin"][loc_idx], dippr_data["Tmax"][loc_idx])
                if all(x -> !isnan(x), vals)
                    psat[global_idx] = (type = :dippr, A = vals[1], B = vals[2], C = vals[3], D = vals[4], E = vals[5], Tmin = vals[6], Tmax = vals[7])
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

const SPECIES_PROPERTIES = fetch_species_properties(species)
const MW = SPECIES_PROPERTIES.mw

function psat_from_data(T::Float64, idx::Int)
    params = SPECIES_PROPERTIES.psat[idx]
    typ = params.type
    if typ === :dippr
        if !(params.Tmin <= T <= params.Tmax)
            @warn "Temperatura fuera del rango de la correlacion DIPPR para $(species[idx])." T = T range = (params.Tmin, params.Tmax)
        end
        return exp(params.A + params.B / T + params.C * log(T) + params.D * T^params.E)
    elseif typ === :custom
        return params.eval(T)
    else
        return 0.0
    end
end

# Constantes físicas
const R = 8.314        # J/(mol K)
const P_atm = 101325.0 # Pa

# ==========================================
# 2. Fermentación (modelo Zenteno), esquema de colocación Radau
# ==========================================
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
const eps = 1e-9

const nc = 5 # X,N,G,F,E
const c0_default = [0.5, 0.14, 110.0, 110.0, 0.0]

colmat = [
    0.19681547722366   -0.06553542585020   0.02377097434822;
    0.39442431473909    0.29207341166523  -0.04154875212600;
    0.37640306270047    0.51248582618842   0.11111111111111
]
const radau_nodes = (0.15505, 0.64495, 1.0)

struct ZentenoParams
    mu0::Float64
    Yeg::Float64
    Yef::Float64
    T_profile::Function
    N_feed::Function
    lag_t50::Float64
    lag_k::Float64
end

function temperature_profile_builder(steps::Vector{Tuple{Float64,Float64}})
    sorted = sort(steps; by = first)
    function Tfun(t)
        last_T = sorted[end][2]
        for k in 1:length(sorted)-1
            t0, T0 = sorted[k]
            t1, T1 = sorted[k+1]
            if t <= t0
                return T0 + 273.15
            elseif t <= t1
                frac = (t - t0) / max(t1 - t0, 1e-6)
                return (T0 + frac * (T1 - T0)) + 273.15
            end
        end
        return last_T + 273.15
    end
    return Tfun
end

smooth_pulse(t, t0, width, amt) = 0.5 * amt * (tanh((t - t0)/max(width/4,1e-6)) - tanh((t - (t0 + width))/max(width/4,1e-6)))

function nitrogen_feed_profile(pulses::Vector{Tuple{Float64,Float64}}; width::Float64=1.0)
    isempty(pulses) && return (t -> 0.0)
    function feed(t)
        acc = 0.0
        for (t0, amt) in pulses
            acc += smooth_pulse(t, t0, width, amt)
        end
        return acc
    end
    return feed
end

function zenteno_rates!(du, u, p::ZentenoParams, t)
    X, N, G, F, E = u
    T = p.T_profile(t)
    mu_T =  exp(59453.0 * (T - 300.0) / (300.0 * R * T))
    Kg_T =  exp(46055.0 * (T - 293.15) / (293.15 * R * T))
    b_T  =  exp(11000.0 * (T - 296.15) / (296.15 * R * T))
    mrate = 0.01 * exp(37681.0 * (T - 293.30) / (293.30 * R * T))
    denom = G + F + eps
    phiG = G / denom
    phiF = F / denom
    lag_factor = 1.0 / (1.0 + exp(-(t - p.lag_t50) / max(p.lag_k, 1e-6))) # fase log retardada
    mu   = p.mu0 * mu_T * lag_factor * (N / (N + Kn0_nom * Kg_T + eps))
    # producción de etanol y mantenimiento escalados por la latencia para amortiguar el arranque de CO2
    betaG = lag_factor * betaG0_nom * b_T * (G / (G + Kg0_nom * Kg_T + eps)) * (Kie0_nom * Kg_T / (E + Kie0_nom * Kg_T + eps))
    betaF = lag_factor * betaF0_nom * b_T * (F / (F + Kf0_nom * Kg_T + eps)) * (Kig0_nom * Kg_T / (G + Kig0_nom * Kg_T + eps)) * (Kie0_nom * Kg_T / (E + Kie0_nom * Kg_T + eps))
    mrate_scaled = mrate * lag_factor
    Td = -0.0001 * E^3 + 0.0049 * E^2 - 0.1279 * E + 315.89
    s = 0.5 * (1.0 + tanh(0.5 * (T - Td)))
    Kd_val = Kd0_nom * exp(0.0415 * E + (130000.0 * (T - 305.65)) / (305.65 * R * T)) * s
    du[1] = (mu - Kd_val) * X
    du[2] = -(mu / YXN_nom) * X + p.N_feed(t)
    du[3] = -((mu / YXG_nom) + (betaG / p.Yeg) + mrate_scaled * phiG) * X
    du[4] = -((mu / YXF_nom) + (betaF / p.Yef) + mrate_scaled * phiF) * X
    du[5] = (betaG + betaF) * X
    return nothing
end

function run_fermentation(; th=120.0, nfe=18, ncp=3, c0=c0_default, T_steps=[(0.0,15.0),(24.0,27.0)], N_pulses=Float64[], pulse_amounts=Float64[], mu0=MU0_nom, Yeg=YEG_nom, Yef=YEF_nom, pulse_width=1.0, lag_t50=8.0, lag_k=2.0)
    h = th / nfe
    hv = fill(h, nfe)
    Tfun = temperature_profile_builder(T_steps)
    pulses = collect(zip(N_pulses, pulse_amounts))
    Nfeed = nitrogen_feed_profile(pulses; width=pulse_width)
    params = ZentenoParams(mu0, Yeg, Yef, Tfun, Nfeed, lag_t50, lag_k)
    prob = ODEProblem(zenteno_rates!, copy(c0), (0.0, th), params)
    sol = solve(prob, Rodas5(); saveat=0.25, reltol=1e-8, abstol=1e-10, maxiters=1_000_000)
    function build_tgrid(lengths::AbstractVector{<:Real})
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
    tgrid = build_tgrid(hv)
    states_grid = Array{Float64}(undef, nc, nfe, ncp)
    for i in 1:nfe, j in 1:ncp
        states_grid[:, i, j] .= sol(tgrid[i, j])
    end
    return (sol=sol, hv=hv, tgrid=tgrid, states_grid=states_grid, params=params, Tfun=Tfun)
end

function co2_vol_flow(t, sol, params; V_liq=100.0)
    u = sol(t)
    du = similar(u)
    zenteno_rates!(du, u, params, t)
    # du[5] es dE/dt (g/L/h); 1 g EtOH -> 0.95 g CO2
    mass_CO2_h = du[5] * V_liq * 0.95
    T = params.T_profile(t)
    vol_CO2_h = (mass_CO2_h / 44.01) * R * T / P_atm * 1000.0
    return max(vol_CO2_h, 0.0)
end

# ==========================================
# 3. Stripping de aroma usando Clapeyron
# ==========================================
function calculate_partition_coefficient(model, T, x_molar)
    gamma = ones(length(x_molar))
    if model !== nothing
        try
            ln_gamma = activity_coefficient(model, P_atm, T, x_molar)
            if any(isnan, ln_gamma)
                @warn "UNIFAC devolvio NaN en gamma; se usa idealidad en su lugar." T=T x=x_molar
            else
                gamma .= exp.(ln_gamma)
            end
        catch err
            @warn "Fallo calculo de gamma; se usa idealidad (gamma=1)." exception=err T=T x=x_molar
        end
    end
    p_sat = [psat_from_data(T, i) for i in eachindex(species)]
    if any(isnan, p_sat)
        @warn "psat NaN detectado; se forzan a cero." T=T p_sat=p_sat
        p_sat = map(x -> isnan(x) ? 0.0 : x, p_sat)
    end
    Ki_termo = (gamma .* p_sat) ./ P_atm
    Ki_termo = map(x -> isfinite(x) ? x : 0.0, Ki_termo)
    rho_L = 1000.0
    rho_G = (P_atm / (R * T)) * 44.01
    MW_mix = sum(x_molar .* MW)
    if !isfinite(MW_mix) || MW_mix <= 0
        @warn "MW_mix no finito; se usa valor de respaldo." MW_mix=MW_mix
        MW_mix = 50.0
    end
    H_cc = Ki_termo .* (P_atm / (R*T)) ./ (rho_L ./ MW_mix)
    return isfinite(H_cc[3]) ? H_cc[3] : 0.0
end

function aroma_ode!(du, u, p, t)
    C_aroma = u[1]
    V_liq = p.V_liq
    T = p.Tfun(t)
    state = p.sol(t)
    E = state[5]
    total_mass = 1000.0
    w_eth = clamp(E / total_mass, 0.0, 0.2)
    w_water = max(1.0 - w_eth - 1e-6, 1e-6)
    moles = [w_water/18.015, w_eth/46.07, 1e-6/144.21]
    x_molar = moles ./ sum(moles)
    K_part = calculate_partition_coefficient(model, T, x_molar)
    Q_CO2 = p.co2_func(t)
    du[1] = - (Q_CO2 / V_liq) * K_part * C_aroma
    if !isfinite(du[1])
        @warn "Derivada de aroma NaN/Inf; se fuerza a cero." t=t Q=Q_CO2 K=K_part C=C_aroma
        du[1] = 0.0
    end
end

# ==========================================
# 4. Secuencia completa: fermentación -> flujo CO2 -> stripping
# ==========================================
function run_workflow(; th=120.0, nfe=18, ncp=3, c0=c0_default, T_steps=[(0.0,15.0),(24.0,27.0)], N_pulses=Float64[], pulse_amounts=Float64[], pulse_width=1.0, aroma_init=10.0, V_liq=100.0, lag_t50=8.0, lag_k=2.0)
    ferm = run_fermentation(th=th, nfe=nfe, ncp=ncp, c0=c0, T_steps=T_steps, N_pulses=N_pulses, pulse_amounts=pulse_amounts, pulse_width=pulse_width, lag_t50=lag_t50, lag_k=lag_k)
    sol = ferm.sol
    params = ferm.params
    co2_func = t -> co2_vol_flow(t, sol, params; V_liq=V_liq)
    aroma_prob = ODEProblem(aroma_ode!, [aroma_init], (0.0, th), (; sol=sol, Tfun=ferm.Tfun, co2_func=co2_func, V_liq=V_liq))
    # Desactivamos autodiff (ForwardDiff) porque la func. de flujo CO2 interpola la solución del primer ODE,
    # y no es diferenciable; así evitamos fallos en Rodas5 por AD.
    aroma_solver = Rodas5(autodiff=false)
    aroma_sol = solve(aroma_prob, aroma_solver; saveat=1.0, reltol=1e-8, abstol=1e-10, maxiters=1_000_000)
    return (ferm=ferm, aroma=aroma_sol, co2_func=co2_func, N_pulses=N_pulses, pulse_amounts=pulse_amounts, T_steps=T_steps)
end

# ==========================================
# 5. Ejecutar y graficar
# ==========================================
ferm_res = run_workflow(
    T_steps=[(0.0,15.0),(30.0,18.0),(60.0,15.0)],
    N_pulses=[24.0, 72.0],
    pulse_amounts=[0.05, 0.05],
    aroma_init=1000.0,
    pulse_width=4.0,
    lag_t50=8.0,
    lag_k=2.0,
)
sol = ferm_res.ferm.sol
aroma_sol = ferm_res.aroma
co2_func = ferm_res.co2_func
Tfun = ferm_res.ferm.Tfun
N_pulses = ferm_res.N_pulses
pulse_amounts = ferm_res.pulse_amounts

t_dense = collect(range(sol.t[1], sol.t[end]; length=400))
state_labels = ["X","N","G","F","E"]
state_mat = reduce(hcat, (sol(t) for t in t_dense))
plots_states = [plot(t_dense, state_mat[i, :]; lw=2, label=false, title=state_labels[i]) for i in 1:5]

t_co2 = collect(range(sol.t[1], sol.t[end]; length=400))
p_co2 = plot(t_co2, [co2_func(t) for t in t_co2]; lw=2, label="Q_CO2 [L/h]", title="Flujo CO2", legend=:topright)

p_temp = plot(t_dense, [Tfun(t)-273.15 for t in t_dense]; lw=2, label="T (°C)", title="Perfil de temperatura")
if !isempty(N_pulses)
    vline!(p_temp, N_pulses; lc=:gray, ls=:dash, label="pulsos N")
end

p_aroma = plot(aroma_sol; label="Aroma (mg/L)", lw=2, title="Perdida de aroma por stripping", xlabel="Tiempo (h)", legend=:topright)

lay = grid(4,2, widths=[0.5,0.5], heights=[0.25,0.25,0.25,0.25])
fig = plot(
    plots_states[1], plots_states[2],
    plots_states[3], plots_states[4],
    plots_states[5], p_co2,
    p_temp, p_aroma;
    layout=lay, size=(1100,1400)
)
if !isempty(N_pulses)
    vline!(fig[2], N_pulses; lc=:gray, ls=:dash, label="pulsos N")
end
display(fig)
fig_path = joinpath(@__DIR__, "pfba_aroma_stripping.png")
try
    savefig(fig, fig_path)
    println("Grafico guardado en: ", fig_path)
catch err
    @warn "No se pudo guardar el grafico generado." exception=err path=fig_path
end
