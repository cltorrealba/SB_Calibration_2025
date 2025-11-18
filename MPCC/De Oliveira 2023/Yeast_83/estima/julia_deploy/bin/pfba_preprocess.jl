# pFBA/FVA preprocessing to build reduced active/candidate sets per FE
# Usage (PowerShell example):
#   $env:REDUCED_MODE="0"; $env:PFBA_EPS="1e-7"; julia --project=. .\pfba_preprocess.jl
# It will write results/reduced_sets.jld2 with variables A, C, F (Vector{Vector{Int}})

using JuMP
using LinearAlgebra
using SparseArrays
using DelimitedFiles
using FileIO, JLD2
using Ipopt

# Optional LP solvers if available
const HAVE_HIGHS = Base.find_package("HiGHS") !== nothing
const HAVE_GLPK  = Base.find_package("GLPK")  !== nothing
const HAVE_CLP   = Base.find_package("Clp")   !== nothing
if HAVE_HIGHS
    import HiGHS
end
if HAVE_GLPK
    import GLPK
end
if HAVE_CLP
    import Clp
end

# Try linear solvers in order
function make_lp_optimizer()
    if HAVE_HIGHS
        return HiGHS.Optimizer
    elseif HAVE_GLPK
        return GLPK.Optimizer
    elseif HAVE_CLP
        return Clp.Optimizer
    else
        # Fallback: Ipopt (not ideal for LP but works as last resort)
        return Ipopt.Optimizer
    end
end

const BASE_DIR = @__DIR__
const ESTIMA_DIR = normpath(joinpath(BASE_DIR, ".."))
const RESULTS_DIR = joinpath(BASE_DIR, "results")
isdir(RESULTS_DIR) || mkpath(RESULTS_DIR)

# Load network
S = readdlm(joinpath(ESTIMA_DIR, "S.csv"), ',')
lb_raw = readdlm(joinpath(ESTIMA_DIR, "lb.csv"), ',')
ub_raw = readdlm(joinpath(ESTIMA_DIR, "ub.csv"), ',')
lb = lb_raw isa AbstractVector ? copy(lb_raw) : copy(lb_raw[:,1])
ub = ub_raw isa AbstractVector ? copy(ub_raw) : copy(ub_raw[:,1])

nm = size(S,1); nv = size(S,2)

# Reaction indices (align with MPCC_Zenteno.jl)
const obj = 3414
const glu = 2588
const fru = 2583

# Grid config (keep aligned)
th = 240.0
nfe = 12
hm = fill(th/nfe, nfe)

# Parameter and temperature (defaults)
R = 8.314
T_const = try parse(Float64, get(ENV, "T_CONST", "296.15")) catch; 296.15 end

# Nominal parameters (copy from MPCC_Zenteno)
const Pnames = (
    :mu0, :betaG0, :betaF0, :Kn0, :Kg0, :Kf0, :Kig0, :Kie0, :Kd0,
    :Yxn, :Yxg, :Yxf, :Yeg, :Yef
)
const Pnom = Dict(
    :mu0=>0.141665, :betaG0=>1.41182, :betaF0=>8.49482, :Kn0=>0.226882,
    :Kg0=>3.1514, :Kf0=>2.97625, :Kig0=>29.5276, :Kie0=>2.99809, :Kd0=>3.11736e-5,
    :Yxn=>9.80576, :Yxg=>0.394345, :Yxf=>0.18622, :Yeg=>0.14133, :Yef=>0.96932
)

# Temperature scalars
mu_T0 = exp(59453.0 * (T_const - 300.0) / (300.0 * R * T_const))
Kg_T0 = exp(46055.0 * (T_const - 293.15) / (293.15 * R * T_const))
b_T0  = exp(11000.0 * (T_const - 296.15) / (296.15 * R * T_const))
mrate0 = 0.01 * exp(37681.0 * (T_const - 293.30) / (293.30 * R * T_const))

mu0   = Pnom[:mu0]; Kn0 = Pnom[:Kn0]; Kg0 = Pnom[:Kg0]; Kf0 = Pnom[:Kf0]
Kig0  = Pnom[:Kig0]; Kie0 = Pnom[:Kie0]
Yxg0 = Pnom[:Yxg]; Yxf0=Pnom[:Yxf]; Yeg0=Pnom[:Yeg]; Yef0=Pnom[:Yef]
bG0   = Pnom[:betaG0]; bF0 = Pnom[:betaF0]

# FE-end state proxy from ODE with nominal params (quick and deterministic)
# ODE support optional
const HAVE_DE = Base.find_package("DifferentialEquations") !== nothing
if HAVE_DE
    import DifferentialEquations
end

try
    X0, N0, G0, F0, E0 = 0.5, 0.14, 110.0, 110.0, 0.0
    c0 = [X0,N0,G0,F0,E0]
    Ge = Vector{Float64}(undef, nfe)
    Fe = similar(Ge); Ne = similar(Ge); Ee = similar(Ge)
    if HAVE_DE
        function rhs!(du,u,p,t)
            X,N,G,F,E = u
            mu   = mu0 * mu_T0 * (N / (N + Kn0 * Kg_T0 + 1e-9))
            betaG = bG0 * b_T0 * (G / (G + Kg0 * Kg_T0 + 1e-9)) * ((Kie0 * Kg_T0) / (E + Kie0 * Kg_T0 + 1e-9))
            betaF = bF0 * b_T0 * (F / (F + Kf0 * Kg_T0 + 1e-9)) * ((Kig0 * Kg_T0) / (G + Kig0 * Kg_T0 + 1e-9)) * ((Kie0 * Kg_T0) / (E + Kie0 * Kg_T0 + 1e-9))
            denom = G + F + 1e-9
            phiG = G / denom; phiF = F / denom
            du[1] = mu * X
            du[2] = -(mu / Pnom[:Yxn]) * X
            du[3] = -((mu / Yxg0) + (betaG / Yeg0) + mrate0 * phiG) * X
            du[4] = -((mu / Yxf0) + (betaF / Yef0) + mrate0 * phiF) * X
            du[5] = (betaG + betaF) * X
        end
        prob = DifferentialEquations.ODEProblem(rhs!, c0, (0.0, th))
        sol = DifferentialEquations.solve(prob, DifferentialEquations.Tsit5(), reltol=1e-6, abstol=1e-8)
        te = cumsum(hm)
        Ge .= [sol(t)[3] for t in te]
        Fe .= [sol(t)[4] for t in te]
        Ne .= [sol(t)[2] for t in te]
        Ee .= [sol(t)[5] for t in te]
    else
        println("[PFBA] DifferentialEquations not available; using constant FE-end proxies from c0")
        Ge .= G0; Fe .= F0; Ne .= N0; Ee .= E0
    end
    rG_fe = Float64[]; rF_fe = Float64[]
    for i in 1:nfe
        G = Ge[i]; F = Fe[i]; N = Ne[i]; E = Ee[i]
        mu_fe = mu0 * mu_T0 * (N / (N + Kn0 * Kg_T0 + 1e-9))
        betaG_fe = bG0 * b_T0 * (G / (G + Kg0 * Kg_T0 + 1e-9)) * ((Kie0 * Kg_T0) / (E + Kie0 * Kg_T0 + 1e-9))
        betaF_fe = bF0 * b_T0 * (F / (F + Kf0 * Kg_T0 + 1e-9)) * ((Kig0 * Kg_T0) / (G + Kig0 * Kg_T0 + 1e-9)) * ((Kie0 * Kg_T0) / (E + Kie0 * Kg_T0 + 1e-9))
        denom = G + F + 1e-9
        phiG = G / denom; phiF = F / denom
        push!(rG_fe, (mu_fe / Yxg0) + (betaG_fe / Yeg0) + (mrate0 * phiG))
        push!(rF_fe, (mu_fe / Yxf0) + (betaF_fe / Yef0) + (mrate0 * phiF))
    end
        # Solve pFBA per FE, then optional FVA refinement on near-zero fluxes
        A = Vector{Vector{Int}}(undef, nfe)
        F = Vector{Vector{Int}}(undef, nfe)
        C = Vector{Vector{Int}}(undef, nfe)
        Opt = make_lp_optimizer()
        opt_is_ipopt = Opt === Ipopt.Optimizer
        FVA_ENABLE = get(ENV, "FVA_ENABLE", "1") == "1"
        FVA_FORCE = get(ENV, "FVA_FORCE", "0") == "1"
        if opt_is_ipopt && FVA_ENABLE && !FVA_FORCE
            @warn "FVA disabled: no LP solver (HiGHS/GLPK/Clp) detected; using Ipopt fallback is unstable for FVA. Set FVA_FORCE=1 to override."
            FVA_ENABLE = false
        end
        FVA_MAG_THRESH = try parse(Float64, get(ENV, "FVA_MAG_THRESH", "1e-5")) catch; 1e-5 end
        FVA_RANGE_EPS = try parse(Float64, get(ENV, "FVA_RANGE_EPS", "1e-6")) catch; 1e-6 end
        FVA_MAXK = try parse(Int, get(ENV, "FVA_MAXK", "400")) catch; 400 end

        # Build a single model and reuse it across FEs to avoid repeated solver initialization
        model = Model(Opt)
        set_silent(model)
        # Extra: try to suppress Ipopt banner if used
        if opt_is_ipopt
            try
                set_optimizer_attribute(model, "print_level", 0)
                set_optimizer_attribute(model, "sb", "yes")
            catch
            end
        end
        @variable(model, lb[k] <= v[k=1:nv] <= ub[k])
        @constraint(model, [mc=1:nm], sum(S[mc,k]*v[k] for k in 1:nv) == 0)
        abs_aux = nothing
        if !opt_is_ipopt
            # Aux variables to linearize |v| objective for LP-based parsimonious step
            @variable(model, abs_aux_var[1:nv] >= 0)
            @constraint(model, [k=1:nv], abs_aux_var[k] >= v[k])
            @constraint(model, [k=1:nv], abs_aux_var[k] >= -v[k])
            abs_aux = abs_aux_var
        end
        # We'll enforce FE-specific uptakes by tightening variable bounds each iteration

        for i in 1:nfe
            # Set per-FE exact bounds for uptake fluxes (lb=ub=value)
            JuMP.set_lower_bound(v[glu], -rG_fe[i]); JuMP.set_upper_bound(v[glu], -rG_fe[i])
            JuMP.set_lower_bound(v[fru], -rF_fe[i]); JuMP.set_upper_bound(v[fru], -rF_fe[i])
            @objective(model, Max, v[obj])
            optimize!(model)
            obj_val = value(v[obj])
            if !isfinite(obj_val)
                error("Failed to compute biomass optimum at FE $i")
            end
            obj_link = @constraint(model, v[obj] >= (1.0 - 1e-6) * obj_val)
            if opt_is_ipopt
                @objective(model, Min, sum(v[k]^2 for k in 1:nv))
            else
                abs_vars = abs_aux === nothing ? error("Missing auxiliary variables for LP norm minimization") : abs_aux
                @objective(model, Min, sum(abs_vars[k] for k in 1:nv))
            end
            optimize!(model)
            vstar = [value(v[k]) for k in 1:nv]
            eps = try parse(Float64, get(ENV, "PFBA_EPS", "1e-7")) catch; 1e-7 end
            Ai = Int[]; Fi = Int[]
            for k in 1:nv
                if !isfinite(vstar[k]) || isnan(vstar[k])
                    push!(Fi, k)
                elseif abs(vstar[k]) < eps
                    push!(Fi, k)
                else
                    push!(Ai, k)
                end
            end
            # Initial candidate set = active set
            Ci = copy(Ai)

            # FVA refinement on near-zero magnitude actives
            if FVA_ENABLE
                to_check = [k for k in Ai if abs(vstar[k]) < FVA_MAG_THRESH]
                if length(to_check) > FVA_MAXK
                    to_check = to_check[1:FVA_MAXK]
                end
                for k in to_check
                    try
                        # minimize v[k]
                    @objective(model, Min, v[k])
                        optimize!(model)
                        vmin = value(v[k])
                        # maximize v[k]
                    @objective(model, Max, v[k])
                        optimize!(model)
                        vmax = value(v[k])
                        if !isfinite(vmin) || !isfinite(vmax)
                            continue
                        end
                        width = vmax - vmin
                        if width <= FVA_RANGE_EPS
                            # Effectively fixed. If interval straddles zero and magnitude tiny, mark as zero-fixed (F)
                            if vmin <= 0.0 <= vmax && abs(vstar[k]) < 10*eps
                                if k ∈ Ci
                                    deleteat!(Ci, findfirst(==(k), Ci))
                                end
                                if k ∈ Ai
                                    deleteat!(Ai, findfirst(==(k), Ai))
                                end
                                push!(Fi, k)
                            else
                                # Keep active but not candidate (remove from C)
                                if k ∈ Ci
                                    deleteat!(Ci, findfirst(==(k), Ci))
                                end
                            end
                        else
                            # Wide range: keep as candidate
                            if !(k ∈ Ci)
                                push!(Ci, k)
                            end
                        end
                    catch fva_err
                        @warn "FVA step failed for reaction $k at FE $i; keeping as candidate" fva_err
                        if !(k ∈ Ci)
                            push!(Ci, k)
                        end
                    end
                end
            end

            # Deduplicate and sort
            Ai = sort(unique(Ai)); Fi = sort(unique(Fi)); Ci = sort(unique(Ci))
            A[i] = Ai; F[i] = Fi; C[i] = Ci
            JuMP.delete(model, obj_link)
            @objective(model, Max, v[obj])
        end
    save(joinpath(RESULTS_DIR, "reduced_sets.jld2"), "A", A, "F", F, "C", C)
    println("[PFBA] Saved reduced sets to ", joinpath(RESULTS_DIR, "reduced_sets.jld2"))
catch err
    @warn "pFBA preprocess failed" err
    exit(1)
end
