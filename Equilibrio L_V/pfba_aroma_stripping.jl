using Clapeyron
using DifferentialEquations
using Plots
using LinearAlgebra

# ==========================================
# 1. DEFINICIÓN DEL MODELO TERMODINÁMICO
# ==========================================
# Usamos UNIFAC para la fase líquida.
# Clapeyron buscará automáticamente los grupos funcionales en su base de datos.

species = ["water", "ethanol", "ethyl hexanoate"]
function build_activity_model(species)
    try
        return UNIFAC(species)
    catch err
        @warn "No se encontraron todos los parametros UNIFAC; se asumira idealidad (gamma=1)." exception=err
        return nothing
    end
end
model = build_activity_model(species)

const UNIFAC_GROUPS_CSV = normpath(joinpath(dirname(pathof(Clapeyron)), "..", "database", "Activity", "UNIFAC", "UNIFAC_groups.csv"))

function load_available_unifac_species(db_path::AbstractString)
    if !isfile(db_path)
        @warn "No se encontro la tabla de especies UNIFAC en Clapeyron." path=db_path
        return String[]
    end
    lines = readlines(db_path)
    length(lines) <= 2 && return String[]
    species_list = String[]
    for line in lines[3:end] # saltamos metadata y encabezado
        stripped = strip(line)
        isempty(stripped) && continue
        first_col = split(stripped, ','; limit=2)[1]
        cleaned = replace(replace(first_col, "\"" => ""), "~|~" => " / ")
        push!(species_list, cleaned)
    end
    return species_list
end

const AVAILABLE_UNIFAC_SPECIES = load_available_unifac_species(UNIFAC_GROUPS_CSV)

function print_available_unifac_species(; filter_keyword::Union{Nothing,String}=nothing)
    comps = AVAILABLE_UNIFAC_SPECIES
    label = "Componentes UNIFAC (Clapeyron) disponibles"
    if filter_keyword !== nothing
        comps = filter(name -> occursin(lowercase(filter_keyword), lowercase(name)), comps)
        label *= " filtrados por \"$(filter_keyword)\""
    end
    println("\n$label: $(length(comps)) encontrados.")
    for comp in comps
        println(" - ", comp)
    end
end

print_available_unifac_species()
antoine_mmhg_to_pa(A, B, C, T) = 133.322368 * 10.0^(A - B / ((T - 273.15) + C))

const SPECIES_PARAM_LOCATIONS = [
    "properties/molarmass.csv",
    "properties/critical.csv",
    "Correlations/saturation_correlations/dippr101_like.csv",
]

const MANUAL_SPECIES_FALLBACK = Dict(
    "water" => (
        mw = 18.015,
        psat = (type = :custom, eval = (T -> exp(23.1964 - 3816.44 / (T - 46.13)))),
    ),
    "ethanol" => (
        mw = 46.07,
        psat = (type = :custom, eval = (T -> exp(23.8381 - 3803.98 / (T - 41.68)))),
    ),
    "ethylacetate" => (
        mw = 88.11,
        psat = (type = :custom, eval = (T -> antoine_mmhg_to_pa(7.00474, 1245.951, 226.232, T))),
    ),
    "ethyl hexanoate" => (
        mw = 144.21,
        psat = (type = :custom, eval = (T -> exp(20.7 - 3550.0 / (T - 60.0)))),
    ),
)

function fetch_species_properties(species::Vector{String})
    n = length(species)
    mw = fill(NaN, n)
    psat = Vector{Union{Nothing, NamedTuple}}(undef, n)
    psat .= nothing
    params = nothing
    try
        params = getparams(species, SPECIES_PARAM_LOCATIONS; verbose = false)
    catch err
        @warn "No se pudieron recuperar los parametros puros desde la base de Clapeyron." exception = err
    end
    if params !== nothing
        mw_param = get(params, "Mw", nothing)
        if mw_param !== nothing
            mw .= collect(mw_param.values)
        end
        dippr_keys = ["A", "B", "C", "D", "E", "Tmin", "Tmax"]
        if all(k -> haskey(params, k), dippr_keys)
            dippr_data = Dict(k => collect(params[k].values) for k in dippr_keys)
            for i in 1:n
                vals = (dippr_data["A"][i], dippr_data["B"][i], dippr_data["C"][i],
                        dippr_data["D"][i], dippr_data["E"][i], dippr_data["Tmin"][i], dippr_data["Tmax"][i])
                if all(x -> !isnan(x), vals)
                    psat[i] = (
                        type = :dippr,
                        A = vals[1],
                        B = vals[2],
                        C = vals[3],
                        D = vals[4],
                        E = vals[5],
                        Tmin = vals[6],
                        Tmax = vals[7],
                    )
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

# Constantes fisicas
const R = 8.314        # J/(mol K)
const P_atm = 101325.0 # Pa



# ==========================================
# 2. FUNCIONES AUXILIARES DEL PROCESO
# ==========================================

# Perfil de Etanol en el tiempo (Sigmoide simplificada de fermentación)
# t en horas, devuelve fracción másica aproximada (0 a 12% v/v aprox)
function get_ethanol_mass_frac(t)
    max_eth = 0.10 # 10% en peso final
    k = 0.1
    t_mid = 48.0
    return max_eth / (1 + exp(-k * (t - t_mid)))
end

# Perfil de Temperatura (ej. control de frío que empieza tarde)
function get_temperature(t)
    if t < 24.0
        return 288.15 # 15°C inicial (arranque)
    else
        return 288.15 # 15°C (enfriamiento)
    end
end

# Tasa de Respiración de CO2 (L/h)
# Proporcional a la tasa de producción de etanol (derivada de la sigmoide)
function get_CO2_rate(t, V_liq)
    # Derivada analítica simple de la sigmoide de etanol
    eth_max = 0.10; k = 0.1; t_mid = 48.0
    dEth_dt = (k * eth_max * exp(-k*(t - t_mid))) / (1 + exp(-k*(t - t_mid)))^2
    
    # Estequiometria aprox: 1 g Etanol genera ~0.95 g CO2
    # Asumimos densidad del mosto ~1080 g/L bajando a 990... usamos promedio 1000
    mass_CO2_h = dEth_dt * V_liq * 1000.0 * 0.95 
    
    # Convertir a Volumen (L) usando Ley de Gases Ideales a P_atm y T actual
    T = get_temperature(t)
    vol_CO2_h = (mass_CO2_h / 44.01) * R * T / P_atm * 1000.0 # L/h
    return vol_CO2_h
end

# ==========================================
# 3. CÁLCULO TERMODINÁMICO (CLAPEYRON)
# ==========================================

function calculate_partition_coefficient(model, T, x_molar)
    # 1. Calcular Coeficientes de Actividad (Gamma) usando UNIFAC
    # Clapeyron devuelve el logaritmo natural, así que aplicamos exp
    gamma = if model === nothing
        ones(length(x_molar))
    else
        ln_gamma = activity_coefficient(model, P_atm, T, x_molar)
        exp.(ln_gamma)
    end
    
    # 2. Calcular Presion de Vapor de componentes puros (P_sat)
    # Se utilizan directamente las correlaciones disponibles en la base de Clapeyron

    p_sat = [psat_from_data(T, i) for i in eachindex(species)]
    
    # 3. Calcular Ki termodinámico (y/x) = gamma * Psat / Ptotal
    Ki_termo = (gamma .* p_sat) ./ P_atm
    
    # 4. Convertir Ki (y/x) a Coeficiente de Partición másico m = (C_gas / C_liq)
    # m_i = Ki_termo * (rho_gas / rho_liq) * (MW_liq_mix / MW_i)
    # Esta conversión es CRÍTICA para balances de materia
    
    # Simplificación: Asumimos densidades y PM promedio para la conversión
    rho_L = 1000.0 # g/L
    rho_G = (P_atm / (R * T)) * 44.01 # g/L (asumiendo CO2 puro)
    MW_mix = sum(x_molar .* MW)
    
    # m = (Concentracion en gas mg/L) / (Concentracion en liquido mg/L)
    m = Ki_termo .* (rho_L / rho_G) .* (MW ./ MW_mix) # Factor de corrección dimensional
    
    # Pero para la ecuación de stripping standard dC/dt = -(Q/V)*H*C, 
    # H suele definirse como Cg/Cl.
    # Usamos la definición directa Cg = Ki_termo * (P/RT) / (Cl_molar) ... es complejo.
    # Método directo ingenieril: Henry adimensional H_cc = C_gas [mol/L] / C_liq [mol/L]
    
    H_cc = Ki_termo .* (P_atm / (R*T)) ./ (rho_L ./ MW_mix)
    
    return H_cc[3] # Retornamos solo el del aroma
end


# ==========================================
# 4. SISTEMA DE ECUACIONES DIFERENCIALES
# ==========================================

function fermentation_stripping!(du, u, p, t)
    # u[1] = Concentración de Aroma (mg/L)
    C_aroma = u[1]
    V_liq = 100.0 # Litros (Asumimos constante por simplicidad, o hazlo variable)
    
    # 1. Obtener estado actual
    T_curr = get_temperature(t)
    w_eth = get_ethanol_mass_frac(t)
    w_water = 1.0 - w_eth - 1e-6 # Asumiendo el aroma es traza despreciable para el balance masico mayor
    
    # 2. Convertir fracción másica a molar (necesario para UNIFAC)
    moles = [w_water/MW[1], w_eth/MW[2], 1e-6/MW[3]] # Aroma traza
    total_moles = sum(moles)
    x_molar = moles ./ total_moles
    
    # 3. Llamar a Clapeyron/Termodinámica
    # Calculamos el coeficiente de partición adimensional (Conc Gas / Conc Liq)
    K_part = calculate_partition_coefficient(model, T_curr, x_molar)
    
    # 4. Obtener flujo de gas
    Q_CO2 = get_CO2_rate(t, V_liq) # L/h
    
    # 5. Ecuación diferencial: Stripping
    # dC/dt = - (Q_gas / V_liq) * C_gas
    # Como C_gas = K_part * C_liq
    stripping_rate = - (Q_CO2 / V_liq) * K_part * C_aroma
    
    du[1] = stripping_rate
end

# ==========================================
# 5. EJECUCIÓN Y GRÁFICOS
# ==========================================

# Condiciones iniciales
C_aroma_0 = 1000.0 # mg/L iniciales
u0 = [C_aroma_0]
tspan = (0.0, 120.0) # 120 horas de fermentación

# Resolver
prob = ODEProblem(fermentation_stripping!, u0, tspan)
# Con etil acetato la dinámica se vuelve más rígida, por lo que usamos un integrador stiffness-aware
sol = solve(prob, Rodas5(), saveat=1.0, reltol=1e-6, abstol=1e-8)


# Graficar
p1 = plot(sol, label="Aroma (mg/L)", lw=2, title="Pérdida de Aroma por Stripping", xlabel="Tiempo (h)")
p2 = plot(t->get_ethanol_mass_frac(t)*100, 0, 120, label="% Etanol (v/v aprox)", color=:red, linestyle=:dash)
p3 = plot(t->get_temperature(t)-273.15, 0, 120, label="Temp (°C)", color=:green)

fig = plot(p1, p2, p3, layout=(3,1), size=(600,800))
display(fig)
fig_path = joinpath(@__DIR__, "pfba_aroma_stripping.png")
try
    savefig(fig, fig_path)
    println("Grafico guardado en: ", fig_path)
catch err
    @warn "No se pudo guardar el grafico generado." exception=err path=fig_path
end
