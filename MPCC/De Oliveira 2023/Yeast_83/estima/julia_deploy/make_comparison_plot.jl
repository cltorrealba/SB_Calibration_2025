#!/usr/bin/env julia
using Printf
using Plots

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

base_dir = @__DIR__
results_dir = joinpath(base_dir, "results")
expA = joinpath(results_dir, "experiment1_no_reduc_0a3")
expB = joinpath(results_dir, "experiment1_reduc_0a3")

# Get latest seeded metrics in each folder
function latest_seeded(dir)
    files = [joinpath(dir, f) for f in readdir(dir) if endswith(f, ".txt") && occursin("zenteno_metrics_seeded_", f)]
    isempty(files) && return nothing
    sort(files)[end]
end

function latest_report(dir)
    files = [joinpath(dir, f) for f in readdir(dir) if endswith(f, ".txt") && occursin("zenteno_estimation_report_", f)]
    isempty(files) && return nothing
    sort(files)[end]
end

function metrics_from_dir(dir)
    mfile = latest_seeded(dir)
    if mfile !== nothing
        return parse_metrics(mfile)
    end
    # Fallback to estimation report
    rfile = latest_report(dir)
    rfile === nothing && error("No metrics or report found in " * dir)
    Dr = parse_metrics(rfile)
    # Compute comp_max if FO_* available
    foL = get(Dr, "FO_L_max", NaN)
    foU = get(Dr, "FO_U_max", NaN)
    fou = get(Dr, "FO_upt_max", NaN)
    Dr["comp_max"] = maximum([foL, foU, fou])
    return Dr
end

Da = metrics_from_dir(expA)
Db = metrics_from_dir(expB)

metrics = ["SSE", "PEN", "OBJ", "comp_max"]
Avals = [get(Da, m, NaN) for m in metrics]
Bvals = [get(Db, m, NaN) for m in metrics]

labels = ["no_reduc (seeded)", "reduc (seeded)"]
# Use GR backend (default) and set size
gr()
default(size=(1000,500))
bar(metrics, hcat(Avals, Bvals), bar_position=:dodge,
    label=labels, title="Seeded full-run comparison: no_reduc vs reduc",
    xlabel="Metric", yscale=:log10, legend=:topright)
annotate!(1, Avals[1], text(@sprintf("%.2e", Avals[1]), 8))
annotate!(1, Bvals[1], text(@sprintf("%.2e", Bvals[1]), 8))
annotate!(2, Avals[2], text(@sprintf("%.2e", Avals[2]), 8))
annotate!(2, Bvals[2], text(@sprintf("%.2e", Bvals[2]), 8))
annotate!(3, Avals[3], text(@sprintf("%.2e", Avals[3]), 8))
annotate!(3, Bvals[3], text(@sprintf("%.2e", Bvals[3]), 8))
annotate!(4, Avals[4], text(@sprintf("%.2e", Avals[4]), 8))
annotate!(4, Bvals[4], text(@sprintf("%.2e", Bvals[4]), 8))

out = joinpath(results_dir, "comparison_seeded_no_reduc_vs_reduc.png")
savefig(out)
println("[PLOT] Saved ", out)
# Also copy into each experiment folder for convenience
cp(out, joinpath(expA, basename(out)); force=true)
cp(out, joinpath(expB, basename(out)); force=true)
println("[PLOT] Copied into both experiment folders")
