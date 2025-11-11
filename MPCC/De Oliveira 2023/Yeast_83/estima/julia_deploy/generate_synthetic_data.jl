# Synthetic data generation for Zenteno ODE
# - Reads parameter set from zenteno_parameters.xlsx (set==5)
# - Integrates Zenteno ODE at T=293.0 K, no nutrient injections
# - Samples at MPCC collocation FE-end times and adds 10% Gaussian noise
# - Saves outputs to:
#   * synthetic_data.jld2: keys "states" (nc x nt), "time" (nt)
#   * data.jld2: key "data" (nc x nfe x ncp) filled at j=ncp with noisy samples; others 0

using XLSX
using FileIO, JLD2
using DifferentialEquations
using Distributions

const BASE_DIR = @__DIR__
const XLSX_PATH = joinpath(BASE_DIR, "zenteno_parameters.xlsx")

# Problem sizes and indices
const nc = 5  # X,N,G,F,E
const ncp = 3
# FE grid must be consistent with MPCC script
const th = 240.0
const nfe = 12
const hm = fill(th / nfe, nfe)

# Initial conditions from MPCC
const c0 = [0.5, 0.14, 110.0, 110.0, 0.0]

"""
    load_params_set(path; set_value=5, sheet_index=1)
Robust loader using XLSX.getdata for the given worksheet. Avoids internal field assumptions.
Headers taken from first row, normalized (lowercase, strip). Missing/empty headers ignored.
"""
function load_params_set(path::String; set_value::Int=5, sheet_index::Int=1)
    XLSX.openxlsx(path) do xf
        sheetnames = XLSX.sheetnames(xf)
        sheet_index < 1 && (sheet_index = 1)  # clamp to first sheet
        sheet_index > length(sheetnames) && error("sheet_index=$(sheet_index) exceeds worksheet count=$(length(sheetnames))")
        ws = xf[sheetnames[sheet_index]]
        raw = XLSX.getdata(ws)  # Matrix{Any}
        nrows = size(raw,1); ncols = size(raw,2)
        nrows < 2 && error("Worksheet has fewer than 2 rows")
        # Headers
        raw_headers = Vector{String}(undef, ncols)
        for j in 1:ncols
            cell = raw[1,j]
            if cell === nothing || cell isa Missing
                raw_headers[j] = ""
            else
                raw_headers[j] = lowercase(strip(string(cell)))
            end
        end
        colmap = Dict{String,Int}()
        for j in 1:ncols
            h = raw_headers[j]
            if !isempty(h)
                colmap[h] = j
            end
        end
        set_col = get(colmap, "set", 0)
        set_col == 0 && error("Column 'set' not found in header row")
        target_row = 0
        for i in 2:nrows
            v = raw[i,set_col]
            (v === nothing || v isa Missing) && continue
            sval = try parse(Int, string(v)) catch; nothing end
            if sval === set_value
                target_row = i
                break
            end
        end
        target_row == 0 && error("No row with set=$(set_value) found in sheet $(sheetnames[sheet_index])")
        param_names = (
            :mu0, :betaG0, :betaF0, :Kn0, :Kg0, :Kf0, :Kig0, :Kie0, :Kd0,
            :Yxn, :Yxg, :Yxf, :Yeg, :Yef
        )
        p = Dict{Symbol,Float64}()
        for nm in param_names
            cname = lowercase(string(nm))
            col = get(colmap, cname, 0)
            col == 0 && error("Column '$cname' not found for parameter $nm")
            valcell = raw[target_row, col]
            (valcell === nothing || valcell isa Missing) && error("Missing value for $nm in row $target_row (set=$(set_value))")
            p[nm] = parse(Float64, string(valcell))
        end
        return p
    end
end

# RHS for Zenteno ODE
tmp_R = 8.314
function zenteno_rhs!(du,u,p,t)
    X,N,G,F,E = u
    T = p[:T]
    mu0=p[:mu0]; betaG0=p[:betaG0]; betaF0=p[:betaF0]; Kn0=p[:Kn0]; Kg0=p[:Kg0]; Kf0=p[:Kf0];
    Kig0=p[:Kig0]; Kie0=p[:Kie0]; Kd0=p[:Kd0]; Yxn=p[:Yxn]; Yxg=p[:Yxg]; Yxf=p[:Yxf]; Yeg=p[:Yeg]; Yef=p[:Yef]
    mu_T  = exp(59453.0 * (T - 300.0) / (300.0 * tmp_R * T))
    Kg_T  = exp(46055.0 * (T - 293.15) / (293.15 * tmp_R * T))
    b_T   = exp(11000.0 * (T - 296.15) / (296.15 * tmp_R * T))
    mrate = 0.01 * exp(37681.0 * (T - 293.30) / (293.30 * tmp_R * T))
    mu    = mu0 * mu_T * (N / (N + Kn0 * Kg_T + 1e-9))
    betaG = betaG0 * b_T * (G / (G + Kg0 * Kg_T + 1e-9)) * ((Kie0 * Kg_T) / (E + Kie0 * Kg_T + 1e-9))
    betaF = betaF0 * b_T * (F / (F + Kf0 * Kg_T + 1e-9)) * ((Kig0 * Kg_T) / (G + Kig0 * Kg_T + 1e-9)) * ((Kie0 * Kg_T) / (E + Kie0 * Kg_T + 1e-9))
    Td   = -0.0001 * E^3 + 0.0049 * E^2 - 0.1279 * E + 315.89
    sw   = 0.5 * (1.0 + tanh(0.5 * (T - Td)))
    Kd   = Kd0 * exp(0.0415 * E + (130000.0 * (T - 305.65)) / (305.65 * tmp_R * T)) * sw
    denom = G + F + 1e-9
    phiG = G / denom
    phiF = F / denom
    du[1] = (mu - Kd) * X
    du[2] = -(mu / Yxn) * X
    du[3] = -((mu / Yxg) + (betaG / Yeg) + mrate * phiG) * X
    du[4] = -((mu / Yxf) + (betaF / Yef) + mrate * phiF) * X
    du[5] = (betaG + betaF) * X
    return nothing
end

# Main
set_value = try parse(Int, get(ENV, "PARAM_SET", "5")) catch; 5 end
sheet_index = try parse(Int, get(ENV, "SHEET_INDEX", "1")) catch; 1 end
params = load_params_set(XLSX_PATH; set_value=set_value, sheet_index=sheet_index)
params[:T] = 293.0  # fixed synthesis temperature
u0 = copy(c0)
prob = DifferentialEquations.ODEProblem(zenteno_rhs!, u0, (0.0, th), params)
sol = DifferentialEquations.solve(prob, DifferentialEquations.Tsit5(), reltol=1e-6, abstol=1e-8)

# Sample at FE-end times
t_nodes = cumsum(hm)
Y = zeros(nc, length(t_nodes))
for (k,t) in enumerate(t_nodes)
    u = sol(t)
    Y[1,k] = u[1]; Y[2,k] = u[2]; Y[3,k] = u[3]; Y[4,k] = u[4]; Y[5,k] = u[5]
end

# Add 10% Gaussian noise (element-wise): y_noisy = y * (1 + 0.1*N(0,1))
normals = rand(Normal(0,1), size(Y)...)
Y_noisy = Y .* (1 .+ 0.1 .* normals)

# Save synthetic_data.jld2
FileIO.save(joinpath(BASE_DIR, "synthetic_data.jld2"), "states", Y_noisy, "time", t_nodes)

# Save data.jld2 in (nc x nfe x ncp) with j=ncp filled by noisy samples (others zero)
D = zeros(nc, nfe, ncp)
for i in 1:nfe
    for l in 1:nc
        D[l, i, ncp] = Y_noisy[l, i]
    end
end
FileIO.save(joinpath(BASE_DIR, "data.jld2"), "data", D)

println("[GEN] synthetic_data.jld2 and data.jld2 generated from set=$(set_value) sheet=$(sheet_index) at T=293K")
