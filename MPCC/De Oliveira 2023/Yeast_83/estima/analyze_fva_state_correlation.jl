#!/usr/bin/env julia
# analyze_fva_state_correlation.jl
#
# Herramienta auxiliar para estudiar la relación entre los estados del modelo
# ODE y las reacciones activas resultantes del FVA reducido. Carga el JSON de
# condiciones usado para el FVA, simula la ODE de Zenteno con los parámetros
# nominales especificados y cruza las muestras de estado con los conjuntos A/C
# guardados en reduced_sets.jld2, exportando un CSV listo para análisis.
#
# Uso:
#   julia --project=. analyze_fva_state_correlation.jl \
#       path/a/condiciones_FVA.json [path/a/reduced_sets.jld2] [salida.csv]
#
# Si no se entregan los últimos argumentos, el script usa los valores dentro
# del JSON (campo fva_pipeline_config.results_path) y genera
# results/fva_state_activation.csv por defecto.

using JSON3
using JLD2
using DifferentialEquations
using Printf

function _sanitize_nonstandard_json(raw::String)
    buf = raw
    rx = r"log\(([^()]+)\)"
    while (m = match(rx, buf)) !== nothing
        expr = m.captures[1]
        val = try
            log(Base.eval(Main, Meta.parse(expr)))
        catch err
            @warn "[ANALYSIS] No se pudo evaluar $expr en log(...); se dejará texto original." err
            break
        end
        buf = replace(buf, m.match => string(val); count=1)
    end
    comment_counter = Ref(0)
    stand_alone = r"(\n\s*)\"([^\"\n:]+)\"\s*,\s*\n"
    buf = replace(buf, stand_alone) do m
        comment_counter[] += 1
        indent = m.captures[1]
        text = m.captures[2]
        return string(indent, "\"_comment", comment_counter[], "\": \"", text, "\",\n")
    end
    return buf
end

function read_json(path::AbstractString)
    isfile(path) || error("No se encontró json de condiciones en $path")
    raw = read(path, String)
    try
        return JSON3.read(raw)
    catch err
        @warn "[ANALYSIS] JSON inválido detectado; intentando sanitizar expresiones no estándar (log(...))." err
        sanitized = _sanitize_nonstandard_json(raw)
        return JSON3.read(sanitized)
    end
end

_as_float(x) = x isa Number ? Float64(x) : parse(Float64, String(x))
_as_int(x) = Int(round(_as_float(x)))

function extract_config(cfg)
    grid = cfg["mpcc_grid"]
    thermo = cfg["thermo_conditions"]
    init = cfg["initial_states"]
    params = cfg["nominal_parameters"]

    return Dict(
        :nfe => _as_int(grid["nfe"]),
        :ncp => _as_int(grid["ncp"]),
        :th  => _as_float(grid["horizon_h"]),
        :T   => _as_float(thermo["T_const_K"]),
        :R   => _as_float(thermo["R"]),
        :c0  => [
            _as_float(init["X0"]),
            _as_float(init["N0"]),
            _as_float(init["G0"]),
            _as_float(init["F0"]),
            _as_float(init["E0"]),
        ],
        :params => params,
    )
end

function simulate_states(conf)
    th = conf[:th]
    nfe = conf[:nfe]
    R = conf[:R]
    T_const = conf[:T]
    p = conf[:params]

    mu0 = _as_float(p["mu0"])
    YXN = _as_float(p["YXN"])
    YXG = _as_float(p["YXG"])
    YXF = _as_float(p["YXF"])
    YEG = _as_float(p["YEG"])
    YEF = _as_float(p["YEF"])
    Kn0 = _as_float(p["Kn0"])
    Kg0 = _as_float(p["Kg0"])
    Kf0 = _as_float(p["Kf0"])
    Kig0 = _as_float(p["Kig0"])
    Kie0 = _as_float(p["Kie0"])
    Kd0 = _as_float(p["Kd0"])
    betaG0 = _as_float(p["betaG0"])
    betaF0 = _as_float(p["betaF0"])

    eps = 1e-9
    mu_T = exp(59453.0 * (T_const - 300.0) / (300.0 * R * T_const))
    Kg_T = exp(46055.0 * (T_const - 293.15) / (293.15 * R * T_const))
    b_T  = exp(11000.0 * (T_const - 296.15) / (296.15 * R * T_const))
    mrate = 0.01 * exp(37681.0 * (T_const - 293.30) / (293.30 * R * T_const))

    function ode!(du, u, _, t)
        X, N, G, F, E = u
        mu_val = mu0 * mu_T * (N / (N + Kn0 * Kg_T + eps))
        betaG = betaG0 * b_T *
            (G / (G + Kg0 * Kg_T + eps)) *
            (Kie0 * Kg_T / (E + Kie0 * Kg_T + eps))
        betaF = betaF0 * b_T *
            (F / (F + Kf0 * Kg_T + eps)) *
            (Kig0 * Kg_T / (G + Kig0 * Kg_T + eps)) *
            (Kie0 * Kg_T / (E + Kie0 * Kg_T + eps))
        Td = -0.0001 * E^3 + 0.0049 * E^2 - 0.1279 * E + 315.89
        s_sw = 0.5 * (1.0 + tanh(0.5 * (T_const - Td)))
        Kd_val = Kd0 * exp(0.0415 * E + (130000.0 * (T_const - 305.65)) / (305.65 * R * T_const)) * s_sw
        denom = G + F + eps
        phiG = G / denom
        phiF = F / denom
        du[1] = (mu_val - Kd_val) * X
        du[2] = -(mu_val / YXN) * X
        du[3] = -((mu_val / YXG) + (betaG / YEG) + mrate * phiG) * X
        du[4] = -((mu_val / YXF) + (betaF / YEF) + mrate * phiF) * X
        du[5] = (betaG + betaF) * X
        return nothing
    end

    prob = ODEProblem(ode!, conf[:c0], (0.0, th))
    sol = solve(prob, Rodas5(); reltol=1e-8, abstol=1e-10)
    fe_times = range(th / nfe, stop=th, length=nfe)
    snapshots = reduce(hcat, (sol(t) for t in fe_times))
    return collect(fe_times), snapshots
end

function load_reduced_sets(path::AbstractString)
    isfile(path) || error("No se encontró reduced_sets en $path")
    return jldopen(path, "r") do f
        (read(f, "A"), read(f, "C"), read(f, "F"))
    end
end

function build_rows(fe_times, states, A_sets, C_sets)
    nfe = size(states, 2)
    rows = Vector{NTuple{10,Any}}()
    for i in 1:nfe
        Xi = states[:, i]
        Ai = (i <= length(A_sets)) ? A_sets[i] : Int[]
        Ci = (i <= length(C_sets)) ? C_sets[i] : Int[]
        base_time = fe_times[i]
        Ai_set = Set(Ai)
        Ci_set = Set(Ci)
        rxn_union = sort!(collect(union(Ai_set, Ci_set)))
        for rxn in rxn_union
            push!(rows, (
                i,
                base_time,
                rxn,
                Int(in(rxn, Ai_set)),
                Int(in(rxn, Ci_set)),
                Xi[1], Xi[2], Xi[3], Xi[4], Xi[5],
            ))
        end
    end
    return rows
end

function write_csv(path::AbstractString, rows)
    isdir(dirname(path)) || mkpath(dirname(path))
    open(path, "w") do io
        println(io, "fe,time_h,reaction_idx,is_active,is_candidate,X,N,G,F,E")
        for r in rows
            @printf(io, "%d,%.6f,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n", r...)
        end
    end
end

function main()
    config_path = length(ARGS) >= 1 ? ARGS[1] :
        joinpath(@__DIR__, "plots", "default", "exp_seed_FVA", "condiciones_FVA_22_11_25.json")
    cfg_json = read_json(config_path)
    conf = extract_config(cfg_json)
    reduced_path = length(ARGS) >= 2 ? ARGS[2] :
        get(cfg_json["fva_pipeline_config"], "results_path", begin
            joinpath(@__DIR__, "julia_deploy", "results", "reduced_sets.jld2")
        end)
    out_csv = length(ARGS) >= 3 ? ARGS[3] :
        joinpath(@__DIR__, "results", "fva_state_activation.csv")

    println("[ANALYSIS] Configuración: nfe=$(conf[:nfe]), horizonte=$(conf[:th]) h, T=$(conf[:T]) K")
    println("[ANALYSIS] Leyendo conjuntos reducidos: $reduced_path")
    A_sets, C_sets, _ = load_reduced_sets(reduced_path)

    println("[ANALYSIS] Simulando ODE Zenteno para obtener estados por FE…")
    fe_times, states = simulate_states(conf)

    println("[ANALYSIS] Construyendo tabla reacción-estado…")
    rows = build_rows(fe_times, states, A_sets, C_sets)
    println("[ANALYSIS] Registros generados: $(length(rows))")

    println("[ANALYSIS] Escribiendo CSV en $out_csv")
    write_csv(out_csv, rows)
    println("[ANALYSIS] Listo.")
end

main()
