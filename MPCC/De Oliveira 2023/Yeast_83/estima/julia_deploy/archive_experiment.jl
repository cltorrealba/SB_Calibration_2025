#!/usr/bin/env julia
# archive_experiment.jl
# Move/copy artifacts for experiment 2 (reduced seed) into results/experiment1_reduc_0a3

using FileIO

const BASE_DIR = @__DIR__
const RESULTS = joinpath(BASE_DIR, "results")
const DST = joinpath(RESULTS, "experiment1_reduc_0a3")

isdir(DST) || mkpath(DST)

function safe_cp(src)
    if isfile(src)
        cp(src, joinpath(DST, basename(src)); force=true)
        println("[ARCHIVE] Copied ", basename(src))
    else
        println("[ARCHIVE] Skip copy (missing): ", basename(src))
    end
end

function safe_mv(src)
    if isfile(src)
        mv(src, joinpath(DST, basename(src)); force=true)
        println("[ARCHIVE] Moved ", basename(src))
    else
        println("[ARCHIVE] Skip move (missing): ", basename(src))
    end
end

# Keep a copy of reduced sets at root for future runs
safe_cp(joinpath(RESULTS, "reduced_sets.jld2"))

# Baseline (second experiment) + pre ODE plot
safe_mv(joinpath(RESULTS, "zenteno_metrics_baseline_20251112-181243.txt"))
safe_mv(joinpath(RESULTS, "zenteno_pre_ode_vs_data_20251112-181249.png"))

# Homotopy seed stages and plot
safe_mv(joinpath(RESULTS, "zenteno_metrics_hom_s1_20251112-180005.txt"))
safe_mv(joinpath(RESULTS, "zenteno_metrics_hom_s2_20251112-180318.txt"))
safe_mv(joinpath(RESULTS, "zenteno_metrics_hom_s3_20251112-180930.txt"))
safe_mv(joinpath(RESULTS, "zenteno_post_ode_vs_data_mpcc_20251112-180934.png"))

# Checkpoints
safe_mv(joinpath(RESULTS, "zenteno_handoff_full_checkpoint.jld2"))
safe_mv(joinpath(RESULTS, "zenteno_seed_checkpoint.jld2"))

# Seeded full-run metrics and report/plot
safe_mv(joinpath(RESULTS, "zenteno_metrics_seeded_20251112-181953.txt"))
safe_mv(joinpath(RESULTS, "zenteno_estimation_report_20251112-181948.txt"))
safe_mv(joinpath(RESULTS, "zenteno_post_ode_vs_data_mpcc_20251112-181953.png"))

# Estimation report for stage-3 and relax summary
safe_mv(joinpath(RESULTS, "zenteno_estimation_report_20251112-180932.txt"))
safe_mv(joinpath(RESULTS, "zenteno_relax_summary_20251112-181948.txt"))

# Comparison report
safe_mv(joinpath(RESULTS, "zenteno_comparison_20251112-182600.txt"))

println("[ARCHIVE] Done -> ", DST)
