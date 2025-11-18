# pfBA/FVA preprocessing (versión mínima, SIN FVA)
# Genera A, C, F por FE usando pFBA en 2 pasos (estilo De Oliveira)
#
# Salida:
#   results/reduced_sets.jld2 con:
#     - A :: Vector{Vector{Int}}
#     - C :: Vector{Vector{Int}}   (aquí C = A)
#     - F :: Vector{Vector{Int}}

using JuMP
using LinearAlgebra
using SparseArrays
using DelimitedFiles
using FileIO, JLD2
using Ipopt

# ----------------------------------------------------
# 1) Rutas y red GSM (alineado con MPCC_Zenteno)
# ----------------------------------------------------
const BASE_DIR    = @__DIR__
const ESTIMA_DIR  = normpath(joinpath(BASE_DIR, ".."))
const RESULTS_DIR = joinpath(BASE_DIR, "results")
isdir(RESULTS_DIR) || mkpath(RESULTS_DIR)

S      = readdlm(joinpath(ESTIMA_DIR, "S.csv"), ',')
lb_raw = readdlm(joinpath(ESTIMA_DIR, "lb.csv"), ',')
ub_raw = readdlm(joinpath(ESTIMA_DIR, "ub.csv"), ',')
lb = lb_raw isa AbstractVector ? copy(lb_raw) : copy(lb_raw[:,1])
ub = ub_raw isa AbstractVector ? copy(ub_raw) : copy(ub_raw[:,1])

nm, nv = size(S)

# Índices clave (mantener en sync con MPCC_Zenteno)
const obj = 3414
const glu = 2588
const fru = 2583

# ----------------------------------------------------
# 2) Tiempo + parámetros nominales Zenteno
# ----------------------------------------------------
th  = 240.0   # horizonte (h)
nfe = 12      # nº FEs
hm  = fill(th/nfe, nfe)

R = 8.314
T_const = try parse(Float64, get(ENV, "T_CONST", "296.15")) catch; 296.15 end

const Pnom = Dict(
    :mu0   => 0.141665,
    :betaG0=> 1.41182,
    :betaF0=> 8.49482,
    :Kn0   => 0.226882,
    :Kg0   => 3.1514,
    :Kf0   => 2.97625,
    :Kig0  => 29.5276,
    :Kie0  => 2.99809,
    :Kd0   => 3.11736e-5,
    :Yxn   => 9.80576,
    :Yxg   => 0.394345,
    :Yxf   => 0.18622,
    :Yeg   => 0.14133,
    :Yef   => 0.96932
)

mu0   = Pnom[:mu0]; Kn0 = Pnom[:Kn0]; Kg0 = Pnom[:Kg0]; Kf0 = Pnom[:Kf0]
Kig0  = Pnom[:Kig0]; Kie0 = Pnom[:Kie0]
Yxg0  = Pnom[:Yxg];  Yxf0 = Pnom[:Yxf]; Yeg0 = Pnom[:Yeg]; Yef0 = Pnom[:Yef]
bG0   = Pnom[:betaG0]; bF0 = Pnom[:betaF0]

mu_T0 = exp(59453.0 * (T_const - 300.0) / (300.0 * R * T_const))
Kg_T0 = exp(46055.0 * (T_const - 293.15) / (293.15 * R * T_const))
b_T0  = exp(11000.0 * (T_const - 296.15) / (296.15 * R * T_const))
mrate0 = 0.01 * exp(37681.0 * (T_const - 293.30) / (293.30 * R * T_const))

# ----------------------------------------------------
# 3) Aproximar rG_fe, rF_fe con ODE Zenteno nominal
# ----------------------------------------------------
const HAVE_DE = Base.find_package("DifferentialEquations") !== nothing
if HAVE_DE
    import DifferentialEquations
end

function build_FE_end_rates()
    X0, N0, G0, F0, E0 = 0.5, 0.14, 110.0, 110.0, 0.0
    c0 = [X0, N0, G0, F0, E0]

    Ge = Vector{Float64}(undef, nfe)
    Fe = similar(Ge); Ne = similar(Ge); Ee = similar(Ge)

    if HAVE_DE
        function rhs!(du,u,p,t)
            X,N,G,F,E = u
            mu    = mu0 * mu_T0 * (N / (N + Kn0 * Kg_T0 + 1e-9))
            betaG = bG0 * b_T0 * (G / (G + Kg0 * Kg_T0 + 1e-9)) *
                    ((Kie0 * Kg_T0) / (E + Kie0 * Kg_T0 + 1e-9))
            betaF = bF0 * b_T0 * (F / (F + Kf0 * Kg_T0 + 1e-9)) *
                    ((Kig0 * Kg_T0) / (G + Kig0 * Kg_T0 + 1e-9)) *
                    ((Kie0 * Kg_T0) / (E + Kie0 * Kg_T0 + 1e-9))
            denom = G + F + 1e-9
            phiG  = G / denom
            phiF  = F / denom

            du[1] =  mu * X
            du[2] = -(mu / Pnom[:Yxn]) * X
            du[3] = -((mu / Yxg0) + (betaG / Yeg0) + mrate0 * phiG) * X
            du[4] = -((mu / Yxf0) + (betaF / Yef0) + mrate0 * phiF) * X
            du[5] =  (betaG + betaF) * X
        end
        prob = DifferentialEquations.ODEProblem(rhs!, c0, (0.0, th))
        sol  = DifferentialEquations.solve(prob,
                                           DifferentialEquations.Tsit5();
                                           reltol=1e-6, abstol=1e-8)
        te = cumsum(fill(th/nfe, nfe))
        Ge .= [sol(t)[3] for t in te]
        Fe .= [sol(t)[4] for t in te]
        Ne .= [sol(t)[2] for t in te]
        Ee .= [sol(t)[5] for t in te]
    else
        println("[PFBA] DifferentialEquations no disponible; usando c0 constante como proxy FE-end.")
        Ge .= G0; Fe .= F0; Ne .= N0; Ee .= E0
    end

    rG_fe = Float64[]; rF_fe = Float64[]
    for i in 1:nfe
        G, F, N, E = Ge[i], Fe[i], Ne[i], Ee[i]
        mu_fe    = mu0 * mu_T0 * (N / (N + Kn0 * Kg_T0 + 1e-9))
        betaG_fe = bG0 * b_T0 * (G / (G + Kg0 * Kg_T0 + 1e-9)) *
                   ((Kie0 * Kg_T0) / (E + Kie0 * Kg_T0 + 1e-9))
        betaF_fe = bF0 * b_T0 * (F / (F + Kf0 * Kg_T0 + 1e-9)) *
                   ((Kig0 * Kg_T0) / (G + Kig0 * Kg_T0 + 1e-9)) *
                   ((Kie0 * Kg_T0) / (E + Kie0 * Kg_T0 + 1e-9))
        denom = G + F + 1e-9
        phiG  = G / denom
        phiF  = F / denom

        push!(rG_fe, (mu_fe / Yxg0) + (betaG_fe / Yeg0) + (mrate0 * phiG))
        push!(rF_fe, (mu_fe / Yxf0) + (betaF_fe / Yef0) + (mrate0 * phiF))
    end
    return rG_fe, rF_fe
end

# ----------------------------------------------------
# 4) pFBA 2 pasos por FE (SIN FVA)
# ----------------------------------------------------
try
    rG_fe, rF_fe = build_FE_end_rates()

    A = Vector{Vector{Int}}(undef, nfe)
    F = Vector{Vector{Int}}(undef, nfe)
    C = Vector{Vector{Int}}(undef, nfe)

    PFBA_EPS = try parse(Float64, get(ENV, "PFBA_EPS", "1e-7")) catch; 1e-7 end

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    try
        set_optimizer_attribute(model, "print_level", 0)
        set_optimizer_attribute(model, "sb", "yes")
    catch
    end

    @variable(model, lb[k] <= v[k=1:nv] <= ub[k])
    @constraint(model, [mc=1:nm], sum(S[mc,k]*v[k] for k in 1:nv) == 0)

    for i in 1:nfe
        println("[PFBA] FE $i / $nfe")

        # Fijar uptakes a las tasas cinéticas FE-end
        JuMP.set_lower_bound(v[glu], -rG_fe[i]); JuMP.set_upper_bound(v[glu], -rG_fe[i])
        JuMP.set_lower_bound(v[fru], -rF_fe[i]); JuMP.set_upper_bound(v[fru], -rF_fe[i])

        # Paso 1: Max crecimiento
        @objective(model, Max, v[obj])
        optimize!(model)
        obj_val = value(v[obj])
        if !isfinite(obj_val)
            error("pFBA: fallo al maximizar objetivo en FE $i")
        end

        # Constraint: mantener obj casi óptimo
        obj_link = @constraint(model, v[obj] >= (1.0 - 1e-6) * obj_val)

        # Paso 2: Min norma L2 del flujo (parsimonia)
        @objective(model, Min, sum(v[k]^2 for k in 1:nv))
        optimize!(model)

        vstar = [value(v[k]) for k in 1:nv]
        Ai = Int[]; Fi = Int[]
        for k in 1:nv
            if !isfinite(vstar[k]) || isnan(vstar[k]) || abs(vstar[k]) < PFBA_EPS
                push!(Fi, k)
            else
                push!(Ai, k)
            end
        end
        Ai = sort(unique(Ai))
        Fi = sort(unique(Fi))
        Ci = copy(Ai)  # sin FVA: candidatos = activos

        A[i] = Ai
        F[i] = Fi
        C[i] = Ci

        # Limpiar constraint y restaurar objetivo para siguiente FE
        JuMP.delete(model, obj_link)
        @objective(model, Max, v[obj])
    end

    outpath = joinpath(RESULTS_DIR, "reduced_sets.jld2")
    save(outpath, "A", A, "F", F, "C", C)
    println("[PFBA] Saved reduced sets to ", outpath)

    using JLD2

    path = joinpath(@__DIR__, "results", "reduced_sets.jld2")

    A_sets, C_sets, F_sets = JLD2.jldopen(path, "r") do f
        (read(f, "A"), read(f, "C"), read(f, "F"))
    end

    for i in 1:length(A_sets)
        Ai, Ci, Fi = A_sets[i], C_sets[i], F_sets[i]
        println("FE $i: |A|=$(length(Ai))  |C|=$(length(Ci))  |F|=$(length(Fi))")
    end

catch err
    @warn "pFBA preprocess failed" err
    rethrow(err)
end

