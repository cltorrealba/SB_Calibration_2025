#!/usr/bin/env julia
# Module 9 — Comparator before vs after
# Reads two metrics files (or resolves by tags) and reports deltas and a decision verdict.

using Dates
using Printf

struct Metrics
    path::String
    tag::String
    wall_s::Float64
    SSE::Float64
    comp_max::Float64
    stat_inf::Float64
    viol_pr::Float64
end

function parse_metrics_file(path::String)
    tag = ""
    wall_s = NaN
    sse = NaN
    comp_max = NaN
    stat_inf = NaN
    viol_pr = NaN
    open(path, "r") do io
        for ln in eachline(io)
            if startswith(ln, "tag=")
                tag = strip(split(ln, "=")[2])
            elseif startswith(ln, "wall_s=")
                wall_s = try parse(Float64, split(ln, "=")[2]) catch; NaN end
            elseif startswith(ln, "SSE=")
                sse = try parse(Float64, split(ln, "=")[2]) catch; NaN end
            elseif startswith(ln, "comp_max=")
                comp_max = try parse(Float64, split(ln, "=")[2]) catch; NaN end
            elseif startswith(ln, "stationarity_residual=")
                stat_inf = try parse(Float64, split(ln, "=")[2]) catch; NaN end
            elseif startswith(ln, "viol_primal=")
                viol_pr = try parse(Float64, split(ln, "=")[2]) catch; NaN end
            end
        end
    end
    return Metrics(path, tag, wall_s, sse, comp_max, stat_inf, viol_pr)
end

function latest_metrics_for_tag(results_dir::AbstractString, tag::AbstractString)
    # pick newest file matching zenteno_metrics_<tag>_*.txt
    prefix = "zenteno_metrics_" * tag * "_"
    files = [f for f in readdir(results_dir) if startswith(f, prefix) && endswith(f, ".txt")]
    isempty(files) && error("No metrics files for tag='" * tag * "' in " * results_dir)
    files_sorted = sort(files; by=f->mtime(joinpath(results_dir, f)), rev=true)
    return joinpath(results_dir, first(files_sorted))
end

function decide(before::Metrics, after::Metrics)
    d_wall = after.wall_s - before.wall_s
    d_sse  = after.SSE - before.SSE
    d_comp = after.comp_max - before.comp_max
    d_stat = after.stat_inf - before.stat_inf
    d_viol = after.viol_pr - before.viol_pr
    # Acceptance rule:
    # Δwall_s < 0 and SSE not worse by >1e-5 relative and comp_max/stat_inf not worse.
    rel_tol = 1e-5
    sse_ok = !(isfinite(before.SSE) && isfinite(after.SSE)) || (after.SSE <= before.SSE * (1 + rel_tol))
    comp_ok = !(isfinite(before.comp_max) && isfinite(after.comp_max)) || (after.comp_max <= before.comp_max)
    stat_ok = !(isfinite(before.stat_inf) && isfinite(after.stat_inf)) || (after.stat_inf <= before.stat_inf)
    viol_ok = !(isfinite(before.viol_pr) && isfinite(after.viol_pr)) || (after.viol_pr <= before.viol_pr)
    wall_ok = isfinite(d_wall) && (d_wall < 0.0)
    accept = wall_ok && sse_ok && comp_ok && stat_ok && viol_ok
    return (accept, d_wall, d_sse, d_comp, d_stat, d_viol)
end

function main()
    results_dir = joinpath(@__DIR__, "results")
    pathA = nothing
    pathB = nothing
    if length(ARGS) >= 2
        pathA = ARGS[1]; pathB = ARGS[2]
        @info "Comparing files" pathA pathB
    else
        cmp = get(ENV, "COMPARE", "")
        if isempty(cmp)
            println("Usage:\n  julia compare_metrics.jl <before_metrics.txt> <after_metrics.txt>\n  # or\n  COMPARE=tagA,tagB julia compare_metrics.jl")
            return
        end
        tags = split(cmp, ",")
        @assert length(tags) == 2 "COMPARE must be 'tagA,tagB'"
        tagA, tagB = strip.(tags)
        pathA = latest_metrics_for_tag(results_dir, tagA)
        pathB = latest_metrics_for_tag(results_dir, tagB)
        @info "Comparing latest metrics for tags" tagA pathA tagB pathB
    end
    before = parse_metrics_file(pathA)
    after  = parse_metrics_file(pathB)
    accept, d_wall, d_sse, d_comp, d_stat, d_viol = decide(before, after)
    function fmt(x)
        return isfinite(x) ? @sprintf("%.6e", x) : "NaN"
    end
    println("\n=== Module 9 Comparator ===")
    println("before=", before.path)
    println(" after=", after.path)
    println(@sprintf("Δwall_s=%.3f s (before=%.3f, after=%.3f)", d_wall, before.wall_s, after.wall_s))
    println("ΔSSE=", fmt(d_sse), " (before=", fmt(before.SSE), ", after=", fmt(after.SSE), ")")
    println("Δcomp_max=", fmt(d_comp), " (before=", fmt(before.comp_max), ", after=", fmt(after.comp_max), ")")
    println("Δstat_inf=", fmt(d_stat), " (before=", fmt(before.stat_inf), ", after=", fmt(after.stat_inf), ")")
    println("Δviol_primal=", fmt(d_viol), " (before=", fmt(before.viol_pr), ", after=", fmt(after.viol_pr), ")")
    println()
    println(accept ? "VEREDICTO: ACEPTAR (mejor tiempo y sin empeorar métricas)" : "VEREDICTO: RECHAZAR (no cumple regla)")
end

main()
