#!/usr/bin/env julia
# Compare two (or three) MPCC_Zenteno metrics outcome files and emit a summary.
# Usage examples (PowerShell):
#   $env:RESULTS_DIR="..\results"; julia --project=. .\compare_runs.jl
#   (Optionally pre-set BASE_METRICS / SEEDED_METRICS / SEEDED_FULL_METRICS)
# If env vars not provided, script auto-detects latest files by pattern.

using Dates
using Printf
using Glob
using FileIO

# Use a non-conflicting name to avoid clashes when included by other scripts
CMP_RESULTS_DIR = get(ENV, "RESULTS_DIR", joinpath(pwd(), "results"))
if !isdir(CMP_RESULTS_DIR)
    error("RESULTS_DIR not found: " * CMP_RESULTS_DIR)
end

"""
Parse a metrics file produced by MPCC_Zenteno.jl into a Dict{String,Float64}.
Accepts lines of the form KEY=VALUE or 'KEY VALUE'. Ignores unparsable lines.
"""
function parse_metrics(path::String)
    D = Dict{String,Float64}()
    for ln in eachline(path)
        ln = strip(ln)
        isempty(ln) && continue
        if occursin("=", ln)
            parts = split(ln, "=")
            length(parts) >= 2 || continue
            k = strip(parts[1])
            vraw = strip(join(parts[2:end], "="))
        else
            parts = split(ln)
            length(parts) >= 2 || continue
            k = parts[1]
            vraw = parts[end]
        end
        try
            v = parse(Float64, replace(vraw, ","=>""))
            D[k] = v
        catch
        end
    end
    return D
end

function latest_by_prefix(prefix::String)
    pats = collect(filter(x -> occursin(prefix, basename(x)), glob("*.txt", CMP_RESULTS_DIR)))
    isempty(pats) && return nothing
    sort(pats)[end]
end

"""
Return the latest file whose name contains `prefix` and whose parsed metrics Dict has all `require_keys`.
Falls back to latest_by_prefix(prefix) if none match.
"""
function latest_with_keys(prefix::String, require_keys::Vector{String})
    candidates = collect(filter(x -> occursin(prefix, basename(x)), glob("*.txt", CMP_RESULTS_DIR)))
    isempty(candidates) && return nothing
    # sort by modified time ascending
    sort!(candidates, by = x -> stat(x).mtime)
    last_match = nothing
    for p in candidates
        D = parse_metrics(p)
        ok = all(k -> haskey(D, k), require_keys)
        if ok
            last_match = p
        end
    end
    last_match === nothing && return candidates[end]
    return last_match
end

base_path   = get(ENV, "BASE_METRICS",   latest_with_keys("zenteno_metrics_baseline", ["SSE", "PEN", "REG", "OBJ"]))
seed_stage_path = get(ENV, "SEEDED_METRICS", latest_with_keys("zenteno_metrics_hom_s3", ["SSE", "PEN"]))
# Use the proper prefix for the seeded full-run metrics
seed_full_path  = get(ENV, "SEEDED_FULL_METRICS", latest_with_keys("zenteno_metrics_seeded", ["SSE", "PEN", "REG", "OBJ"]))

println("[CMP] Using baseline file: ", base_path)
println("[CMP] Using seeded stage-3 file: ", seed_stage_path)
println("[CMP] Using seeded full-run file: ", seed_full_path)

base_path === nothing && error("No baseline metrics file detected.")
seed_stage_path === nothing && println("[CMP] Warning: no stage-3 metrics detected; homotopy seed comparison will be partial.")
seed_full_path === nothing && println("[CMP] Warning: no seeded full-run metrics detected.")

Db = parse_metrics(base_path)
Ds3 = seed_stage_path === nothing ? Dict{String,Float64}() : parse_metrics(seed_stage_path)
Df  = seed_full_path === nothing ? Dict{String,Float64}() : parse_metrics(seed_full_path)

function pct_improve(old::Float64, new::Float64; larger_is_better::Bool=false)
    if isnan(old) || isnan(new) || old == 0.0
        return NaN
    end
    if larger_is_better
        return 100.0 * (new - old)/abs(old)
    else
        return 100.0 * (old - new)/abs(old)
    end
end

keys_interest = [
    "SSE", "PEN", "REG", "OBJ", "comp_max", "stationarity_residual", "comp_sum"
]

out_lines = String[]
push!(out_lines, "Comparison report generated: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
push!(out_lines, "Baseline: $(base_path)")
seed_stage_path !== nothing && push!(out_lines, "Seed Stage-3: $(seed_stage_path)")
seed_full_path !== nothing && push!(out_lines, "Seed Full-run: $(seed_full_path)")
push!(out_lines, "")
push!(out_lines, @sprintf("%-30s %15s %15s %15s %12s %12s", "Metric", "Baseline", "Seed_s3", "Seed_full", "%Imp_s3", "%Imp_full"))
push!(out_lines, repeat("-", 110))

for k in keys_interest
    b = get(Db, k, NaN)
    s3 = get(Ds3, k, NaN)
    sf = get(Df,  k, NaN)
    imp_s3 = pct_improve(b, s3)
    imp_sf = pct_improve(b, sf)
    push!(out_lines, @sprintf("%-30s %15.6e %15.6e %15.6e %12.2f %12.2f", k, b, s3, sf, imp_s3, imp_sf))
end

# Additional derived ratios (PEN/SSE) if available
if haskey(Db, "PEN") && haskey(Db, "SSE")
    pen_sse_b = Db["PEN"]/Db["SSE"]
    pen_sse_s3 = haskey(Ds3, "PEN") && haskey(Ds3, "SSE") ? Ds3["PEN"]/Ds3["SSE"] : NaN
    pen_sse_sf = haskey(Df, "PEN") && haskey(Df, "SSE") ? Df["PEN"]/Df["SSE"] : NaN
    push!(out_lines, "")
    push!(out_lines, @sprintf("%-30s %15.6e %15.6e %15.6e", "PEN/SSE ratio", pen_sse_b, pen_sse_s3, pen_sse_sf))
end

cmp_path = joinpath(CMP_RESULTS_DIR, "zenteno_comparison_" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS") * ".txt")
open(cmp_path, "w") do io
    for ln in out_lines
        println(io, ln)
    end
end
println("[CMP] Saved comparison report: ", cmp_path)
