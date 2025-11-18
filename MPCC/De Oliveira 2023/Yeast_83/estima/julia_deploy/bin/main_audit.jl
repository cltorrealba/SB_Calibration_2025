###############################################
# Project: DC_dFBA (Yeast_83) — AUDIT VERSION
# Purpose: Build JuMP model and print linear vs nonlinear
#          constraint summary using MOI, without solving.
###############################################

import Pkg
# Activate the project pinned to this file's directory to avoid path issues
Pkg.activate(@__DIR__)
# Ensure dependencies from Project/Manifest are available (no-ops if already installed)
try
    Pkg.instantiate()
    Pkg.precompile()
catch err
    @warn "Pkg.instantiate/precompile failed; continuing. Run these manually if packages are missing." error=err
end

using JuMP
using Ipopt
using LinearAlgebra
using FileIO, JLD2
using DelimitedFiles
using MathOptInterface
const MOI = MathOptInterface

# ---------------------------
# Data loading (same as main)
# ---------------------------
# Experimental data (used in SSE)
data = FileIO.load("data.jld2","data");

# Stoichiometric matrix and bounds
S = readdlm("S.csv", ',');
vlb2 = readdlm("lb.csv", ',');
vlb = vlb2[:,1]
vub2 = readdlm("ub.csv", ',');
vub = vub2[:,1]

# Dimensions and special indices
nm = size(S,1)
nv = size(S,2)
# Yeast 8.3 reaction indices
eth = 2630
obj = 3414
glu = 2588
o2  = 2816
ATP = 3415
xyl = 2592

# Anaerobic / ATP adjustments
vlb[o2] = 0.0
vub[o2] = 0.0
vlb[ATP] = 0.0

# ---------------------------
# Model fixed parameters
# ---------------------------
nc  = 4
nfe = 12
ncp = 3
th  = 22.0
h   = th/nfe
ph  = nfe
np  = 5

hm   = (h*ones(nfe))'
var_h = 1.0

w     = 1e-20
omega = 1e2
phi1  = 1.0
phi2  = 1.0
phi3  = 1.0

d = zeros(nv); d[obj] = -1.0
up = zeros(nv);  up[glu] = 1.0
up2 = zeros(nv); up2[xyl] = 1.0
n_up = 2

# Initial conditions
x0 = 0.2; g0 = 4.0; z0 = 2.0; e0 = 0.0
c0 = [x0,g0,z0,e0]

# Param bounds (log scale)
teta0 = log.([7.5, 1.0, 35.0, 15.00, 0.5])
UB    = log.([8.0, 1.13, 33.0, 15.85, 0.6])
LB    = log.([7.0, 0.9, 31.0, 13.85, 0.4])

# Scaling
cs = fill(1.0, nc)
vs = fill(1.0, nv)

# Collocation matrices (Radau)
colmat = [0.19681547722366  -0.06553542585020 0.02377097434822;
          0.39442431473909   0.29207341166523 -0.04154875212600;
          0.37640306270047   0.51248582618842 0.11111111111111]
# radau nodes are unused here

# ---------------------------
# Build model
# ---------------------------
m = Model(Ipopt.Optimizer)
set_optimizer_attribute(m, "warm_start_init_point", "yes")
set_optimizer_attribute(m, "print_level", 5)
set_optimizer_attribute(m, "tol", 1e-4)
set_optimizer_attribute(m, "acceptable_iter", 5)
set_optimizer_attribute(m, "acceptable_tol", 1e-2)
set_optimizer_attribute(m, "linear_solver", "mumps")

@variables(m, begin
    c[1:nc, 1:ph, 1:ncp]
    cdot[1:nc, 1:ph, 1:ncp]
    FO
    teta[1:np]
    v[1:nv, 1:nfe]
    lambda[1:nm, 1:nfe]
    alpha_U[1:nv, 1:nfe]
    alpha_L[1:nv, 1:nfe]
    alpha_upt[1:n_up,1:nfe]
    FO_U[1:nv, 1:nfe]
    FO_L[1:nv, 1:nfe]
    FO_upt[1:n_up,1:nfe]
    hv[1:nfe]
end)

# Starts
for i in 1:ph, j in 1:ncp
    set_start_value(c[1,i,j], c0[1])
    set_start_value(c[2,i,j], c0[2])
    set_start_value(c[3,i,j], c0[3])
    set_start_value(c[4,i,j], c0[4])
end
for i in 1:nfe
    set_start_value(hv[i], hm[i])
end
# Scale c0 (same as main)
for i in 1:nc
    c0[i] = c0[i]/cs[i]
end

# Objective (linear in FO and FO_*):
@NLobjective(m, Min, omega*FO + sum( sum(-phi1*FO_L[mc,i] + -phi3*FO_U[mc,i] for mc in 1:nv) + phi2*FO_upt[1,i] + phi2*FO_upt[2,i] for i in 1:nfe))

# Nonlinear expressions (vg, vz)
@NLexpressions(m, begin
    vg[i=1:ph], exp(teta[1])*(  (c[2,i,3])/(exp(teta[2])+(c[2,i,3]) ) )
    vz[i=1:ph], exp(teta[3])*(  (c[3,i,3])/(exp(teta[4])+(c[3,i,3]) ))*(1/ (1 + (c[2,i,3]/exp(teta[5]))  )  )
end)

# Linear constraints
@constraints(m, begin
    coll_c_n[l=1:nc, i=2:ph, j=1:ncp], c[l,i,j] == c[l,i-1,ncp]+h*sum(colmat[j,k]*cdot[l,i,k] for k in 1:ncp)
    coll_c_0[l=1:nc, i=1, j=1:ncp], c[l,i,j] == c0[l] + h*sum(colmat[j,k]*cdot[l,i,k] for k in 1:ncp)
    teta_LB[p=1:np], teta[p] >= LB[p]
    teta_UB[p=1:np], teta[p] <= UB[p]
    Sc[mc=1:nm,i=1:nfe],  sum(S[mc,k]*v[k,i]*vs[k] for k in 1:nv) == 0
    v_UB[mc=1:nv,i=1:nfe], v[mc,i]*vs[mc] - vub[mc]  <= 0
    v_LB[mc=1:nv,i=1:nfe], -v[mc,i]*vs[mc]  + vlb[mc] <= 0
    c_LB[mc=1:nc,i=1:nfe, j=1:ncp], -c[mc,i,j]  <= 0
    MFE1, sum(hv[i] for i in 1:nfe) == th
    MFE3[i=1:nfe], hv[i]  >= 0.0
    MFE4[i=1:nfe], hv[i]  >= (1-var_h)*hm[1]
    MFE5[i=1:nfe], hv[i]  <= (1+var_h)*hm[1]
    Lagr[mc=1:nv,i=1:nfe], +d[mc] + w*v[mc,i]*vs[mc]  + alpha_L[mc,i] + alpha_U[mc,i] + up[mc]*alpha_upt[1,i] + up2[mc]*alpha_upt[2,i]  + sum(S[k,mc]*lambda[k,i] for k in 1:nm) == 0
    alpha1_LB[mc=1:nv,i=1:nfe], alpha_L[mc,i] <= 0
    alpha4_LB[mc=1:n_up,i=1:nfe], alpha_upt[mc,i] <= 0
    alpha1_UB[mc=1:nv,i=1:nfe], alpha_U[mc,i] >= 0
end)

# Nonlinear constraints
@NLconstraints(m, begin
    m1[i=1:ph, j=1:ncp], cdot[1,i,j] == v[obj,i]*c[1,i,j]
    m2[i=1:ph, j=1:ncp], cdot[2,i,j] == - 0.180156*vg[i]*c[1,i,j]
    m3[i=1:ph, j=1:ncp], cdot[3,i,j] == - 0.15013*vz[i]*c[1,i,j]
    m4[i=1:ph, j=1:ncp], cdot[4,i,j] ==  0.04607*v[eth,i]*c[1,i,j]
    v_LB_g[i=1:nfe], -v[glu,i]*vs[glu]  - vg[i] <= 0
    v_LB_z[i=1:nfe], -v[xyl,i]*vs[xyl]  - vz[i] <= 0
    FO1[mc=1:nv,i=1:nfe], FO_L[mc,i] == (v[mc,i]*vs[mc] -vlb[mc])*alpha_L[mc,i]
    FO2[mc=1:nv,i=1:nfe], FO_U[mc,i] == (v[mc,i]*vs[mc] -vub[mc])*alpha_U[mc,i]
    FO3_upt[i=1:nfe], FO_upt[1,i] == (-v[glu,i]*vs[glu] -vg[i])*alpha_upt[1,i]
    FO4_upt[i=1:nfe], FO_upt[2,i] == (-v[xyl,i]*vs[xyl] -vz[i])*alpha_upt[2,i]
    m8, FO == sum( sum( sum( (data[i,j,mc]-c[i,j,mc])^2 for i in 1:nc)   for j in 1:ph)   for mc in 1:ncp)
end)

# ---------------------------
# Audit report (constraint types)
# ---------------------------
println("[AUDIT] Constraint types and counts:")
for (F,S) in JuMP.list_of_constraint_types(m)
    n = JuMP.num_constraints(m, F, S)
    println("  - ", F, " in ", S, ": ", n)
end

println("[AUDIT] Named groups summary (counts):")
# Linear groups
println("  - coll_c_n : ", length(coll_c_n))
println("  - coll_c_0 : ", length(coll_c_0))
println("  - teta_LB  : ", length(teta_LB))
println("  - teta_UB  : ", length(teta_UB))
println("  - Sc       : ", length(Sc))
println("  - v_UB     : ", length(v_UB))
println("  - v_LB     : ", length(v_LB))
println("  - c_LB     : ", length(c_LB))
println("  - MFE1     : 1")
println("  - MFE3     : ", length(MFE3))
println("  - MFE4     : ", length(MFE4))
println("  - MFE5     : ", length(MFE5))
println("  - Lagr     : ", length(Lagr))
println("  - alpha1_LB: ", length(alpha1_LB))
println("  - alpha4_LB: ", length(alpha4_LB))
println("  - alpha1_UB: ", length(alpha1_UB))
# Nonlinear groups
println("  - m1       : ", length(m1))
println("  - m2       : ", length(m2))
println("  - m3       : ", length(m3))
println("  - m4       : ", length(m4))
println("  - v_LB_g   : ", length(v_LB_g))
println("  - v_LB_z   : ", length(v_LB_z))
println("  - FO1      : ", length(FO1))
println("  - FO2      : ", length(FO2))
println("  - FO3_upt  : ", length(FO3_upt))
println("  - FO4_upt  : ", length(FO4_upt))
println("  - m8       : 1")

println("[AUDIT] Done. (No solve performed)")
