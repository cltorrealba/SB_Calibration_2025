#!/usr/bin/env julia
# MPCC_Zenteno_basic.jl
#
# Versión "downgrade" del MPCC de Zenteno:
#  - Estructura de MPCC y acoplamiento con S idénticos a original.jl (DC_dFBA).
#  - Modelo dinámico macroscópico de Zenteno (dependiente de temperatura).
#  - Parámetros estimados: μ₀, Yeg, Yef (en log).
#  - FO (SSE) sólo sobre estados G, F, E.

using JuMP
using Ipopt
using LinearAlgebra
using TickTock
using FileIO, JLD2
using DelimitedFiles

# ---------------------------------------------
# Solver lineal (por entorno) – MUMPS por defecto
# y preparación de PATH para DLLs en Windows
# ---------------------------------------------
const DEFAULT_LINEAR_SOLVER = "mumps"
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
    if solver == "pardiso"
        haskey(ENV, "PANUA_IPOPT_DIR") && _prepend_to_path!(joinpath(ENV["PANUA_IPOPT_DIR"], "bin"))
        haskey(ENV, "IPOPT_PARDISO_DLL_DIR") && _prepend_to_path!(ENV["IPOPT_PARDISO_DLL_DIR"])
        haskey(ENV, "PANUA_LIC_PATH") && _prepend_to_path!(ENV["PANUA_LIC_PATH"])
        if haskey(ENV, "PARDISO_NUM_THREADS")
            ENV["OMP_NUM_THREADS"] = ENV["PARDISO_NUM_THREADS"]
            ENV["MKL_NUM_THREADS"] = ENV["PARDISO_NUM_THREADS"]
        end
    elseif solver in ("ma57", "ma77", "ma86", "ma97")
        try
            @eval import HSL
            @info "HSL.jl detectado; solver lineal HSL disponible" solver
        catch
            @warn "HSL.jl no instalado; solvers HSL no disponibles. Instala con: import Pkg; Pkg.add(\"HSL\")" solver
        end
    end
    return solver in ALLOWED_IPOPT_SOLVERS ? solver : DEFAULT_LINEAR_SOLVER
end

const SELECTED_IPOPT_SOLVER = configure_ipopt_env!()

# ---------------------------------------------
# Paths e IO
# ---------------------------------------------
const BASE_DIR   = @__DIR__
const ESTIMA_DIR = normpath(joinpath(BASE_DIR, ".."))

# Datos experimentales (3D: estados × fases × puntos de colocación)
data_path = joinpath(ESTIMA_DIR, "data.jld2")
@assert isfile(data_path) "No se encontró data.jld2 en $data_path"
data = FileIO.load(data_path, "data")

# Matriz estequiométrica y cotas de flujos
S     = readdlm(joinpath(ESTIMA_DIR, "S.csv"), ',')
lbraw = readdlm(joinpath(ESTIMA_DIR, "lb.csv"), ',')
ubraw = readdlm(joinpath(ESTIMA_DIR, "ub.csv"), ',')
lb    = lbraw isa AbstractVector ? copy(lbraw) : copy(lbraw[:,1])
ub    = ubraw isa AbstractVector ? copy(ubraw) : copy(ubraw[:,1])

# ---------------------------------------------
# Índices y tamaños del modelo GEM
# ---------------------------------------------
nm = size(S, 1)          # número de metabolitos
nv = size(S, 2)          # número de reacciones

# Índices yeastGEM (Yeast 8.3)
const eth = 2630   # reacción etanol
const obj = 3414   # reacción crecimiento / objetivo
const glu = 2588   # uptake glucosa
const fru = 2583   # uptake fructosa
const o2  = 2816   # uptake oxígeno
const ATP = 3415   # reacción ATP

# Fijar O2 y ATP
if 1 <= o2 <= nv
    lb[o2] = 0.0
    ub[o2] = 0.0
end
if 1 <= ATP <= nv
    lb[ATP] = 0.0
end

# ---------------------------------------------
# Parámetros del modelo macroscópico Zenteno
# ---------------------------------------------
# Estados: X, N, G, F, E
const nc = 5

# Nominales (no estimados) tomados de MPCC_Zenteno.py
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

# Parámetros a estimar: μ0, Yeg, Yef (en log)
const np = 3
θ0 = log.([MU0_nom, YEG_nom, YEF_nom])
LB = log.([0.5*MU0_nom, 0.5*YEG_nom, 0.5*YEF_nom])
UB = log.([2.0*MU0_nom, 2.0*YEG_nom, 2.0*YEF_nom])

# ---------------------------------------------
# Condiciones iniciales (X,N,G,F,E)
# ---------------------------------------------
X0 = 0.5
N0 = 0.14
G0 = 110.0
F0 = 110.0
E0 = 0.0
c0 = [X0, N0, G0, F0, E0]

# ---------------------------------------------
# Parámetros de integración / colocación
# ---------------------------------------------
nfe = 12          # elementos finitos (fases)
ncp = 3           # puntos de colocación (Radau-3)
th  = 22.0        # horizonte temporal (h)
h   = th / nfe
ph  = nfe         # horizonte de predicción (fases)

hm    = h * ones(nfe)'
var_h = 1.0       # tiempo variable (como en original.jl)

# ---------------------------------------------
# Pesos y MPCC pFBA (idéntico a original.jl)
# ---------------------------------------------
w    = 1e-20      # peso ||v|| en Lagrange (estacionariedad)
omega = 1e2       # peso SSE (FO) en objetivo
phi1  = 1.0
phi2  = 1.0
phi3  = 1.0

d   = zeros(nv); d[obj] = -1.0
up  = zeros(nv); up[glu] = 1.0
up2 = zeros(nv); up2[fru] = 1.0
const n_up = 2

# Escalado de estados (dejamos 1.0 como original)
cs = ones(nc)

# Escalado de flujos (dejamos 1.0 como original)
vs = ones(nv)

# ---------------------------------------------
# Índices de estados medidos para FO (SSE)
# Sólo G (3), F (4), E (5)
# ---------------------------------------------
const MEAS_STATES = (3, 4, 5)

# ---------------------------------------------
# Colocación Radau-3 (idéntico a original)
# ---------------------------------------------
colmat = [
    0.19681547722366   -0.06553542585020   0.02377097434822;
    0.39442431473909    0.29207341166523  -0.04154875212600;
    0.37640306270047    0.51248582618842   0.11111111111111
]

# ---------------------------------------------
# Modelo JuMP
# ---------------------------------------------
m = Model(Ipopt.Optimizer)
set_optimizer_attribute(m, "warm_start_init_point", "yes")
set_optimizer_attribute(m, "print_level", 5)
set_optimizer_attribute(m, "tol", 1e-4)
set_optimizer_attribute(m, "acceptable_iter", 5)
set_optimizer_attribute(m, "acceptable_tol", 1e-2)
set_optimizer_attribute(m, "linear_solver", SELECTED_IPOPT_SOLVER)

if haskey(ENV, "J_IPOPT_MAX_ITER")
    try
        maxit = parse(Int, ENV["J_IPOPT_MAX_ITER"])
        set_optimizer_attribute(m, "max_iter", maxit)
        @info "Ipopt max_iter set from ENV[J_IPOPT_MAX_ITER]" maxit
    catch err
        @warn "Failed to parse ENV[J_IPOPT_MAX_ITER]; ignoring" error=err
    end
end

# Variables
@variables(m, begin
    c[1:nc, 1:ph, 1:ncp]        # X,N,G,F,E
    cdot[1:nc, 1:ph, 1:ncp]
    FO                          # SSE
    teta[1:np]                  # [log(mu0), log(Yeg), log(Yef)]
    v[1:nv, 1:nfe]              # flujos
    lambda_[1:nm, 1:nfe]        # multiplicadores estequiométricos
    alpha_U[1:nv, 1:nfe]
    alpha_L[1:nv, 1:nfe]
    alpha_upt[1:n_up, 1:nfe]
    FO_U[1:nv, 1:nfe]
    FO_L[1:nv, 1:nfe]
    FO_upt[1:n_up, 1:nfe]
    hv[1:nfe]
end)

# Start values básicos: estados planos y hv = h
for i in 1:ph, j in 1:ncp
    for l in 1:nc
        set_start_value(c[l, i, j], c0[l])
        set_start_value(cdot[l, i, j], 0.0)
    end
end
for i in 1:nfe
    set_start_value(hv[i], hm[i])
end

# Escalar c0 (como original)
for i in 1:nc
    c0[i] = c0[i] / cs[i]
end

# ---------------------------------------------
# Objetivo: SSE (FO) + penalización complementaria lineal (como original)
# ---------------------------------------------
@NLobjective(m, Min,
    omega * FO +
    sum(
        sum(-phi1 * FO_L[mc, i] - phi3 * FO_U[mc, i] for mc in 1:nv) +
        phi2 * FO_upt[1, i] + phi2 * FO_upt[2, i]
        for i in 1:nfe
    )
)

# ---------------------------------------------
# Cinética Zenteno (dependiente de T)
# ---------------------------------------------
const R = 8.314
const T_const = try parse(Float64, get(ENV, "T_CONST", "293.15")) catch; 293.15 end
const eps = 1e-9

@NLexpression(m, mu_T,
    exp(59453.0 * (T_const - 300.0) / (300.0 * R * T_const))
)
@NLexpression(m, Kg_T,
    exp(46055.0 * (T_const - 293.15) / (293.15 * R * T_const))
)
@NLexpression(m, b_T,
    exp(11000.0 * (T_const - 296.15) / (296.15 * R * T_const))
)
@NLexpression(m, mrate,
    0.01 * exp(37681.0 * (T_const - 293.30) / (293.30 * R * T_const))
)

# Parámetros en espacio real
@NLexpression(m, mu0,  exp(teta[1]))
@NLexpression(m, Yeg,  exp(teta[2]))
@NLexpression(m, Yef,  exp(teta[3]))

# NO estimamos estos: los dejamos fijos
const Yxn = YXN_nom
const Yxg = YXG_nom
const Yxf = YXF_nom

# Kinetics pointwise (i,j)
@NLexpression(m, mu_j[i=1:ph, j=1:ncp],
    mu0 * mu_T * ( c[2,i,j] / (c[2,i,j] + Kn0_nom * Kg_T + eps) )
)
@NLexpression(m, betaG_j[i=1:ph, j=1:ncp],
    betaG0_nom * b_T *
    ( c[3,i,j] / (c[3,i,j] + Kg0_nom * Kg_T + eps) ) *
    ( Kie0_nom * Kg_T / (c[5,i,j] + Kie0_nom * Kg_T + eps) )
)
@NLexpression(m, betaF_j[i=1:ph, j=1:ncp],
    betaF0_nom * b_T *
    ( c[4,i,j] / (c[4,i,j] + Kf0_nom * Kg_T + eps) ) *
    ( Kig0_nom * Kg_T / (c[3,i,j] + Kig0_nom * Kg_T + eps) ) *
    ( Kie0_nom * Kg_T / (c[5,i,j] + Kie0_nom * Kg_T + eps) )
)

@NLexpression(m, phiG_j[i=1:ph, j=1:ncp],
    c[3,i,j] / (c[3,i,j] + c[4,i,j] + eps)
)
@NLexpression(m, phiF_j[i=1:ph, j=1:ncp],
    c[4,i,j] / (c[3,i,j] + c[4,i,j] + eps)
)

@NLexpression(m, Kd_j[i=1:ph, j=1:ncp],
    begin
        E_ij = c[5,i,j]
        Td_ij = -0.0001 * E_ij^3 + 0.0049 * E_ij^2 - 0.1279 * E_ij + 315.89
        s = 0.5 * (1.0 + tanh(0.5 * (T_const - Td_ij)))
        base = Kd0_nom * exp(0.0415 * E_ij + (130000.0 * (T_const - 305.65)) / (305.65 * R * T_const))
        base * s
    end
)

# ---------------------------------------------
# Restricciones lineales / pFBA / MPCC
# ---------------------------------------------
@constraints(m, begin
    # Collocación (como original, pero con hv[i])
    coll_c_n[l=1:nc, i=2:ph, j=1:ncp],
        c[l,i,j] == c[l,i-1,ncp] + hv[i] * sum(colmat[j,k] * cdot[l,i,k] for k in 1:ncp)
    coll_c_0[l=1:nc, j=1:ncp],
        c[l,1,j] == c0[l] + hv[1] * sum(colmat[j,k] * cdot[l,1,k] for k in 1:ncp)

    # Parámetros (bounds en teta)
    teta_LB[p=1:np], teta[p] >= LB[p]
    teta_UB[p=1:np], teta[p] <= UB[p]

    # Balances estequiométricos y cotas de flujos
    Sc[mc=1:nm, i=1:nfe],  sum(S[mc,k] * v[k,i] * vs[k] for k in 1:nv) == 0
    v_UB[mc=1:nv, i=1:nfe], v[mc,i]*vs[mc] - ub[mc] <= 0
    v_LB[mc=1:nv, i=1:nfe], -v[mc,i]*vs[mc] + lb[mc] <= 0

    # No negatividad de estados
    c_LB[l=1:nc, i=1:nfe, j=1:ncp], -c[l,i,j] <= 0

    # Longitud de elementos finitos variable
    MFE1, sum(hv[i] for i in 1:nfe) == th
    MFE3[i=1:nfe], hv[i] >= 0.0
    MFE4[i=1:nfe], hv[i] >= (1.0 - var_h) * hm[1]
    MFE5[i=1:nfe], hv[i] <= (1.0 + var_h) * hm[1]

    # pFBA KKT (estacionariedad)
    Lagr[mc=1:nv, i=1:nfe],
        d[mc] + w * v[mc,i] * vs[mc] + alpha_L[mc,i] + alpha_U[mc,i] +
        up[mc] * alpha_upt[1,i] + up2[mc] * alpha_upt[2,i] +
        sum(S[k,mc] * lambda_[k,i] for k in 1:nm) == 0

    alpha1_LB[mc=1:nv, i=1:nfe], alpha_L[mc,i] <= 0
    alpha4_LB[mc=1:n_up, i=1:nfe], alpha_upt[mc,i] <= 0
    alpha1_UB[mc=1:nv, i=1:nfe], alpha_U[mc,i] >= 0
end)

# ---------------------------------------------
# Restricciones no lineales (ODE Zenteno + FO + acoples MPCC)
# ---------------------------------------------
@NLconstraints(m, begin
    # ODEs Zenteno (X,N,G,F,E)
    dX[i=1:ph, j=1:ncp],
        cdot[1,i,j] == (mu_j[i,j] - Kd_j[i,j]) * c[1,i,j]
    dN[i=1:ph, j=1:ncp],
        cdot[2,i,j] == -(mu_j[i,j] / Yxn) * c[1,i,j]
    dG[i=1:ph, j=1:ncp],
        cdot[3,i,j] == -((mu_j[i,j] / Yxg) + (betaG_j[i,j] / Yeg) + mrate * phiG_j[i,j]) * c[1,i,j]
    dF[i=1:ph, j=1:ncp],
        cdot[4,i,j] == -((mu_j[i,j] / Yxf) + (betaF_j[i,j] / Yef) + mrate * phiF_j[i,j]) * c[1,i,j]
    dE[i=1:ph, j=1:ncp],
        cdot[5,i,j] == (betaG_j[i,j] + betaF_j[i,j]) * c[1,i,j]

    # Complementariedad MPCC (idéntica a original)
    FO1[mc=1:nv, i=1:nfe],
        FO_L[mc,i] == (v[mc,i]*vs[mc] - lb[mc]) * alpha_L[mc,i]
    FO2[mc=1:nv, i=1:nfe],
        FO_U[mc,i] == (v[mc,i]*vs[mc] - ub[mc]) * alpha_U[mc,i]

    # Amarres de uptake (análogos a original pero sin vg/vz explícitos)
    # Aquí podrías en el futuro ligar v[glu], v[fru] a tasas macroscópicas;
    # por ahora los dejamos como variables libres (FO_upt sólo penaliza α_upt).
    FO3_upt[i=1:nfe],
        FO_upt[1,i] == (-v[glu,i]*vs[glu]) * alpha_upt[1,i]
    FO4_upt[i=1:nfe],
        FO_upt[2,i] == (-v[fru,i]*vs[fru]) * alpha_upt[2,i]

    # FO = SSE sólo en G (3), F (4), E (5)
    FO_def,
        FO == sum(
            (data[l, i, j] - c[l, i, j])^2
            for l in MEAS_STATES, i in 1:ph, j in 1:ncp
        )
end)

# ---------------------------------------------
# Resolver
# ---------------------------------------------
tick()
optimize!(m)
tock()

println("Solver status = ", termination_status(m))
println("Primal status = ", primal_status(m))
println("Objective FO   = ", value(FO))

