#!/usr/bin/env julia
# experiment_pipeline.jl
# Orchestrates baseline, seed-generation homotopy, and seeded full runs for MPCC_Zenteno.
# Usage (PowerShell examples):
#   julia --project=. .\experiment_pipeline.jl baseline
#   julia --project=. .\experiment_pipeline.jl seed
#   julia --project=. .\experiment_pipeline.jl seed_run
#   julia --project=. .\experiment_pipeline.jl compare

using Dates
using Printf
using FileIO
using Glob

const BASE_DIR = @__DIR__
const RESULTS_DIR = joinpath(BASE_DIR, "results")
isdir(RESULTS_DIR) || mkpath(RESULTS_DIR)

function latest(pattern::String)
    files = filter(f -> occursin(pattern, basename(f)), glob("*.txt", RESULTS_DIR))
    isempty(files) && return nothing
    return sort(files)[end]
end

"""
mode_seed_reduced
Staged homotopy identical to mode_seed but with REDUCED_MODE=1 to build a warm-start
from the reduced model; checkpoint reused by full model in seed_run.
"""
function mode_seed_reduced()
    # Auto-generate reduced sets if missing to ensure REDUCED_MODE actually reduces the model
    reduced_sets_path = joinpath(RESULTS_DIR, "reduced_sets.jld2")
    if !isfile(reduced_sets_path)
        println("[SEED_REDUCED] reduced_sets.jld2 not found – invoking pfba_preprocess.jl to build A/F/C sets")
        # Allow tweaking eps via ENV; fall back to default if unset
        pfba_env = Dict(
            "PFBA_EPS" => get(ENV, "PFBA_EPS", "1e-7"),
            # Enable FVA only if user hasn\"t disabled it
            "FVA_ENABLE" => get(ENV, "FVA_ENABLE", "1"),
            "FVA_FORCE" => get(ENV, "FVA_FORCE", "0"),
        )
        # Set ENV and include pfba_preprocess directly
        for (k,v) in pfba_env
            ENV[k] = v
            println("[ENV] ", k, "=", v)
        end
        include("pfba_preprocess.jl")
        if isfile(reduced_sets_path)
            println("[SEED_REDUCED] Generated reduced sets at ", reduced_sets_path)
        else
            println("[SEED_REDUCED] Warning: pfba_preprocess did not produce reduced_sets.jld2; proceeding with full model")
        end
    else
        println("[SEED_REDUCED] Using existing reduced sets at ", reduced_sets_path)
    end
    wall = get(ENV, "SEED_REDUCED_WALL_TIME", "720")
    env = Dict(
        "HOMOTOPY" => "1",
        "HOM_PHI" => get(ENV, "SEED_REDUCED_HOM_PHI", "1e-2,1e-1,1"),
        "HOM_W" => get(ENV, "SEED_REDUCED_HOM_W", "1e-4,1e-5,1e-6"),
        "HOM_WEIGHTS" => get(ENV, "SEED_REDUCED_HOM_WEIGHTS", "1,1,2"),
        "WALL_TIME" => wall,
        "INIT_FROM_ODE" => get(ENV, "SEED_REDUCED_INIT_FROM_ODE", "1"),
        "INIT_DUAL_FE" => get(ENV, "SEED_REDUCED_INIT_DUAL_FE", "1"),
        "HANDOFF_FULL" => "1",
        "HANDOFF_STAGE" => get(ENV, "SEED_REDUCED_HANDOFF_STAGE", "3"),
        "REDUCED_MODE" => "1"
    )
    run_cmd(env; tag="seed_reduced")
    ck = joinpath(RESULTS_DIR, "zenteno_handoff_full_checkpoint.jld2")
    if isfile(ck)
        new_ck = joinpath(RESULTS_DIR, "zenteno_seed_checkpoint.jld2")
        cp(ck, new_ck; force=true)
        println("[SEED_REDUCED] Copied checkpoint to ", new_ck)
    else
        println("[SEED_REDUCED] Warning: checkpoint not found at ", ck)
    end
end

function run_cmd(env::Dict{String,String}; tag::String)
    println("[RUN] Starting mode='", tag, "' @ ", Dates.now())
    # Build environment prefix for PowerShell-like invocation (here we set ENV inline in-process)
    for (k,v) in env
        ENV[k] = v
        println("[ENV] ", k, "=", v)
    end
    include("MPCC_Zenteno.jl")
    println("[RUN] Completed mode='", tag, "' @ ", Dates.now())
end

function write_baseline_metrics()
    rep = latest("zenteno_estimation_report_")
    rep === nothing && (println("[BASE] No estimation report found to derive baseline metrics"); return)
    # Parse needed metrics
    SSE = PEN = REG = OBJ = FO_L_max = FO_U_max = FO_upt_max = NaN
    for ln in eachline(joinpath(RESULTS_DIR, basename(rep)))
        if startswith(ln, "SSE="); SSE = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
        if startswith(ln, "PEN="); PEN = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
        if startswith(ln, "REG="); REG = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
        if startswith(ln, "OBJ="); OBJ = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
        if startswith(ln, "FO_L_max="); FO_L_max = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
        if startswith(ln, "FO_U_max="); FO_U_max = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
        if startswith(ln, "FO_upt_max="); FO_upt_max = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
    end
    comp_max = maximum([FO_L_max, FO_U_max, FO_upt_max])
    metrics_path = joinpath(RESULTS_DIR, "zenteno_metrics_baseline_" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS") * ".txt")
    open(metrics_path, "w") do io
        println(io, "tag=baseline")
        println(io, @sprintf("SSE=%.6e", SSE))
        println(io, @sprintf("PEN=%.6e", PEN))
        println(io, @sprintf("REG=%.6e", REG))
        println(io, @sprintf("OBJ=%.6e", OBJ))
        println(io, @sprintf("comp_max=%.6e", comp_max))
        println(io, @sprintf("stationarity_residual=%.6e", NaN))
        println(io, @sprintf("comp_sum=%.6e", NaN))
    end
    println("[BASE] Saved baseline metrics: ", metrics_path)
end

function mode_baseline()
    env = Dict(
        "HOMOTOPY" => "0",
        "INIT_FROM_CHECKPOINT" => "0",
        "INIT_FROM_ODE" => "0",
        "INIT_DUAL_FE" => "0",
        "REDUCED_MODE" => "0",
        "WALL_TIME" => "360"
    )
    run_cmd(env; tag="baseline")
    write_baseline_metrics()
end

function mode_seed()
    # 3-stage homotopy 180/180/360 via weights 1,1,2 and total WALL_TIME=720
    wall = get(ENV, "SEED_WALL_TIME", "720")
    env = Dict(
        "HOMOTOPY" => "1",
        "HOM_PHI" => get(ENV, "SEED_HOM_PHI", "1e-2,1e-1,1"),
        "HOM_W" => get(ENV, "SEED_HOM_W", "1e-4,1e-5,1e-6"),
        "HOM_WEIGHTS" => get(ENV, "SEED_HOM_WEIGHTS", "1,1,2"),
        "WALL_TIME" => wall,
        "INIT_FROM_ODE" => get(ENV, "SEED_INIT_FROM_ODE", "1"),
        "INIT_DUAL_FE" => get(ENV, "SEED_INIT_DUAL_FE", "1"),
        "HANDOFF_FULL" => "1",
        "HANDOFF_STAGE" => get(ENV, "SEED_HANDOFF_STAGE", "3"),
        "REDUCED_MODE" => "0",
        "FROZEN_BOUNDS" => get(ENV, "FROZEN_BOUNDS", "0"),
        "FROZEN_REL_WIDTH" => get(ENV, "FROZEN_REL_WIDTH", "0.05"),
        "P_FROZEN_mu0" => get(ENV, "P_FROZEN_mu0", "0.3006891")
    )
    run_cmd(env; tag="seed")
    # Rename checkpoint for clarity
    ck = joinpath(RESULTS_DIR, "zenteno_handoff_full_checkpoint.jld2")
    if isfile(ck)
        new_ck = joinpath(RESULTS_DIR, "zenteno_seed_checkpoint.jld2")
        cp(ck, new_ck; force=true)
        println("[SEED] Copied checkpoint to ", new_ck)
    else
        println("[SEED] Warning: checkpoint not found at ", ck)
    end
end

function mode_seed_run()
    # Use produced seed checkpoint; single-stage solve 360s
    ck = joinpath(RESULTS_DIR, "zenteno_seed_checkpoint.jld2")
    if !isfile(ck)
        error("Seed checkpoint missing: $(ck). Run 'seed' mode first.")
    end
    env = Dict(
        "HOMOTOPY" => "0",
        "INIT_FROM_CHECKPOINT" => "1",
        "CHECKPOINT_PATH" => ck,
        "INIT_FROM_ODE" => "0",
        "INIT_DUAL_FE" => "0",
        "REDUCED_MODE" => "0",
        "WALL_TIME" => "360"
    )
    run_cmd(env; tag="seed_run")
    # Derive seeded metrics similar to baseline
    rep = latest("zenteno_estimation_report_")
    if rep === nothing
        println("[SEED_RUN] No estimation report found; metrics skipped")
        return
    end
    SSE = PEN = REG = OBJ = FO_L_max = FO_U_max = FO_upt_max = NaN
    for ln in eachline(joinpath(RESULTS_DIR, basename(rep)))
        if startswith(ln, "SSE="); SSE = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
        if startswith(ln, "PEN="); PEN = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
        if startswith(ln, "REG="); REG = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
        if startswith(ln, "OBJ="); OBJ = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
        if startswith(ln, "FO_L_max="); FO_L_max = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
        if startswith(ln, "FO_U_max="); FO_U_max = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
        if startswith(ln, "FO_upt_max="); FO_upt_max = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
    end
    comp_max = maximum([FO_L_max, FO_U_max, FO_upt_max])
    metrics_path = joinpath(RESULTS_DIR, "zenteno_metrics_seeded_" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS") * ".txt")
    open(metrics_path, "w") do io
        println(io, "tag=seeded")
        println(io, @sprintf("SSE=%.6e", SSE))
        println(io, @sprintf("PEN=%.6e", PEN))
        println(io, @sprintf("REG=%.6e", REG))
        println(io, @sprintf("OBJ=%.6e", OBJ))
        println(io, @sprintf("comp_max=%.6e", comp_max))
        println(io, @sprintf("stationarity_residual=%.6e", NaN))
        println(io, @sprintf("comp_sum=%.6e", NaN))
    end
    println("[SEED_RUN] Saved seeded metrics: ", metrics_path)
end

function mode_compare()
    include("compare_runs.jl")
end

function main()
    mode = length(ARGS) >= 1 ? ARGS[1] : "baseline"
    if mode == "baseline"
        mode_baseline()
    elseif mode == "seed"
        mode_seed()
    elseif mode == "seed_reduced"
        mode_seed_reduced()
    elseif mode == "seed_frozen"
        # Use frozen bounds inside seeding (activate via ENV overrides)
        ENV["FROZEN_BOUNDS"] = "1"
        mode_seed()
    elseif mode == "seed_run"
        mode_seed_run()
    elseif mode == "seed_run_frozen"
        # Seeded run keeping frozen bounds active
        ck = joinpath(RESULTS_DIR, "zenteno_seed_checkpoint.jld2")
        if !isfile(ck)
            error("Seed checkpoint missing: $(ck). Run 'seed_frozen' first.")
        end
        wall = get(ENV, "SEED_RUN_WALL_TIME", "360")
        env = Dict(
            "HOMOTOPY" => "0",
            "INIT_FROM_CHECKPOINT" => "1",
            "CHECKPOINT_PATH" => ck,
            "INIT_FROM_ODE" => "0",
            "INIT_DUAL_FE" => "0",
            "REDUCED_MODE" => "0",
            "WALL_TIME" => wall,
            "FROZEN_BOUNDS" => get(ENV, "FROZEN_BOUNDS", "1"),
            "FROZEN_REL_WIDTH" => get(ENV, "FROZEN_REL_WIDTH", "0.05"),
            "P_FROZEN_mu0" => get(ENV, "P_FROZEN_mu0", "0.3006891")
        )
        run_cmd(env; tag="seed_run_frozen")
        rep = latest("zenteno_estimation_report_")
        if rep !== nothing
            SSE = PEN = REG = OBJ = FO_L_max = FO_U_max = FO_upt_max = NaN
            for ln in eachline(joinpath(RESULTS_DIR, basename(rep)))
                if startswith(ln, "SSE="); SSE = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
                if startswith(ln, "PEN="); PEN = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
                if startswith(ln, "REG="); REG = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
                if startswith(ln, "OBJ="); OBJ = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
                if startswith(ln, "FO_L_max="); FO_L_max = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
                if startswith(ln, "FO_U_max="); FO_U_max = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
                if startswith(ln, "FO_upt_max="); FO_upt_max = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
            end
            comp_max = maximum([FO_L_max, FO_U_max, FO_upt_max])
            metrics_path = joinpath(RESULTS_DIR, "zenteno_metrics_seeded_" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS") * ".txt")
            open(metrics_path, "w") do io
                println(io, "tag=seeded_frozen")
                println(io, @sprintf("SSE=%.6e", SSE))
                println(io, @sprintf("PEN=%.6e", PEN))
                println(io, @sprintf("REG=%.6e", REG))
                println(io, @sprintf("OBJ=%.6e", OBJ))
                println(io, @sprintf("comp_max=%.6e", comp_max))
                println(io, @sprintf("stationarity_residual=%.6e", NaN))
                println(io, @sprintf("comp_sum=%.6e", NaN))
            end
            println("[SEED_RUN_FROZEN] Saved seeded metrics: ", metrics_path)
        else
            println("[SEED_RUN_FROZEN] No estimation report found; metrics skipped")
        end
    elseif mode == "compare"
        mode_compare()
    else
        error("Unknown mode $(mode). Use one of: baseline, seed, seed_run, compare")
    end
end

main()
