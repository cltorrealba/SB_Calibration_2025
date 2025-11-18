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
# Honor EXPERIMENT env to route outputs into results/<experiment>
const RESULTS_DIR_BASE = joinpath(BASE_DIR, "results")
const EXPERIMENT_NAME = get(ENV, "EXPERIMENT", "")
const RESULTS_DIR = isempty(EXPERIMENT_NAME) ? RESULTS_DIR_BASE : joinpath(RESULTS_DIR_BASE, EXPERIMENT_NAME)
isdir(RESULTS_DIR) || mkpath(RESULTS_DIR)

# Optionally redirect Ipopt to a custom library (e.g., Pardiso-enabled Ipopt)
function maybe_configure_custom_ipopt()
    # Accept either a direct DLL directory or a root folder containing a lib/ subdir
    root = get(ENV, "IPOPT_DLL_DIR", get(ENV, "PANUA_IPOPT_DIR", get(ENV, "PANUA_IPOPT_ROOT", "")))
    isempty(root) && return
    # Try as provided
    cand1 = joinpath(root, "libipopt.dll")
    # Try typical layout root/lib/libipopt.dll
    cand2 = joinpath(root, "lib", "libipopt.dll")
    dll_dir = ""
    libpath = ""
    if isfile(cand1)
        dll_dir = root; libpath = cand1
    elseif isfile(cand2)
        dll_dir = joinpath(root, "lib"); libpath = cand2
    else
        println("[IPOPT] Warning: libipopt.dll not found under ", root)
        return
    end
    # Point Ipopt.jl to the custom library and ensure dependent DLLs are discoverable
    ENV["JULIA_IPOPT_LIBRARY_PATH"] = libpath
    # Prepend Ipopt DLL directory to PATH for dependencies (e.g., libiomp5md.dll)
    # Keep Ipopt's runtime first to avoid mismatched OpenMP DLLs being picked up before it.
    ENV["PATH"] = dll_dir * ";" * get(ENV, "PATH", "")
    # If user provided an explicit Pardiso DLL directory, prepend that too
    pard_dir = get(ENV, "IPOPT_PARDISO_DLL_DIR", get(ENV, "PARDISO_DLL_DIR", ""))
    if !isempty(pard_dir) && isdir(pard_dir)
        # Append Pardiso dir after Ipopt dir to prioritize Ipopt's OpenMP first
        ENV["PATH"] = get(ENV, "PATH", "") * ";" * pard_dir
        println("[IPOPT] Added Pardiso DLL dir to PATH (append): ", pard_dir)
    end
    println("[IPOPT] Using custom Ipopt library: ", libpath)
    # Silence verbose Pardiso license banner and repeated checks if supported by runtime
    if isempty(strip(get(ENV, "PARDISOLICMESSAGE", "")))
        # According to Panua docs, setting PARDISOLICMESSAGE=1 suppresses the banner
        ENV["PARDISOLICMESSAGE"] = "1"
    end
    # If user didn't pick a linear solver, default to Pardiso in this mode
    if isempty(get(ENV, "IPOPT_LINEAR_SOLVER", ""))
        ENV["IPOPT_LINEAR_SOLVER"] = "pardiso"
        println("[IPOPT] Defaulting linear_solver=pardiso (override with IPOPT_LINEAR_SOLVER)")
    end
    # Preflight: if Pardiso is requested, ensure libpardiso.dll is discoverable; otherwise, fall back to MUMPS
    let solver = lowercase(strip(get(ENV, "IPOPT_LINEAR_SOLVER", "")))
        if solver == "pardiso"
            # Common locations to probe
            candidates = String[
                joinpath(dll_dir, "libpardiso.dll"),
                joinpath(root, "libpardiso.dll"),
                joinpath(root, "lib", "libpardiso.dll")
            ]
            if !isempty(pard_dir)
                push!(candidates, joinpath(pard_dir, "libpardiso.dll"))
            end
            has_pardiso = any(isfile, candidates)
            if !has_pardiso
                println("[IPOPT] Warning: Pardiso requested but libpardiso.dll was not found under ", root)
                println("[IPOPT] Falling back to linear_solver=mumps. Place libpardiso.dll next to libipopt.dll or in PATH to enable Pardiso.")
                ENV["IPOPT_LINEAR_SOLVER"] = "mumps"
            end
        end
    end
end

# Optionally configure threading for Pardiso/MKL via a single knob
function maybe_configure_threads()
    local t = strip(get(ENV, "PARDISO_NUM_THREADS", ""))
    isempty(t) && return
    # Only set if user hasn't pinned them already
    if isempty(strip(get(ENV, "OMP_NUM_THREADS", "")))
        ENV["OMP_NUM_THREADS"] = t
        println("[IPOPT] OMP_NUM_THREADS=", t)
    end
    if isempty(strip(get(ENV, "MKL_NUM_THREADS", "")))
        ENV["MKL_NUM_THREADS"] = t
        println("[IPOPT] MKL_NUM_THREADS=", t)
    end
    # Prevent MKL from changing threads dynamically
    if isempty(strip(get(ENV, "MKL_DYNAMIC", "")))
        ENV["MKL_DYNAMIC"] = "FALSE"
        println("[IPOPT] MKL_DYNAMIC=FALSE")
    end
end

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
    # Configure custom Ipopt (if requested) and threading before loading MPCC_Zenteno (which imports Ipopt)
    maybe_configure_custom_ipopt()
    maybe_configure_threads()
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
        "P_FROZEN_mu0" => get(ENV, "P_FROZEN_mu0", "0.3006891"),
        # Default BV hybrid: ON for S1–S2, OFF for S3; uptake-only; mild bounds for uptakes
        "BV_ON" => get(ENV, "BV_ON", "1"),
        "BV_SCOPE" => get(ENV, "BV_SCOPE", "uptake"),
        "DV_MAX_GLU" => get(ENV, "DV_MAX_GLU", "0.5"),
        "DV_MAX_FRU" => get(ENV, "DV_MAX_FRU", "0.5"),
        "DV_MAX_COMMON" => get(ENV, "DV_MAX_COMMON", "50.0"),
        # Explicit 3-stage mask by default (matches default HOM lists here)
        "BV_PHASES" => get(ENV, "BV_PHASES", "1,1,0")
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
    wall = get(ENV, "SEED_RUN_WALL_TIME", "360")
    # Allow optional single-stage homotopy with fixed phi/w via env overrides.
    # By default, keep HOMOTOPY=0 (direct solve). To force fixed phi/w, export:
    #   SEED_RUN_HOMOTOPY=1; SEED_RUN_HOM_PHI="<phi>"; SEED_RUN_HOM_W="<w>"
    env = Dict(
        "HOMOTOPY" => get(ENV, "SEED_RUN_HOMOTOPY", "0"),
        "INIT_FROM_CHECKPOINT" => "1",
        "CHECKPOINT_PATH" => ck,
        "INIT_FROM_ODE" => "0",
        "INIT_DUAL_FE" => "0",
        "REDUCED_MODE" => "0",
        "WALL_TIME" => wall,
        # If a caller sets HOMOTOPY=1 for seed_run, pass through single values
        # for phi and w when provided. Lists of length 1 behave as a direct solve.
        "HOM_PHI" => get(ENV, "SEED_RUN_HOM_PHI", get(ENV, "HOM_PHI", "")),
        "HOM_W"   => get(ENV, "SEED_RUN_HOM_W",   get(ENV, "HOM_W",   ""))
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
        if startswith(ln, "PEN="); PEN = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
        if startswith(ln, "REG="); REG = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
        if startswith(ln, "OBJ="); OBJ = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
        if startswith(ln, "FO_L_max="); FO_L_max = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
        if startswith(ln, "FO_U_max="); FO_U_max = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
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

"""
mode_init_only
Solución simple (sin homotopía) por 360s para generar un checkpoint de inicialización.
Permite controlar INIT_FROM_ODE/INIT_DUAL_FE vía ENV; siempre activa HANDOFF_FULL para guardar warm-start.
Además, escribe un archivo de métricas similar a seed_run.
"""
function mode_init_only()
    env = Dict(
        "HOMOTOPY" => "0",
        "INIT_FROM_CHECKPOINT" => "0",
        # Permite override externo; por defecto inicia desde ODE con duales factibles
        "INIT_FROM_ODE" => get(ENV, "INIT_FROM_ODE", "1"),
        "INIT_DUAL_FE" => get(ENV, "INIT_DUAL_FE", "1"),
        # Forzar guardado de warm-start al final
        "HANDOFF_FULL" => "1",
        "HANDOFF_STAGE" => get(ENV, "HANDOFF_STAGE", "3"),
        "REDUCED_MODE" => "0",
        "WALL_TIME" => "360",
        # Permite ensayos con/sin BV, por defecto respeta ENV actual
        "BV_ON" => get(ENV, "BV_ON", get(ENV, "INIT_BV_ON", "0")),
        "BV_SCOPE" => get(ENV, "BV_SCOPE", get(ENV, "INIT_BV_SCOPE", "uptake")),
    )
    run_cmd(env; tag="init_only")
    # Copiar y renombrar checkpoint a seed estándar
    ck = joinpath(RESULTS_DIR, "zenteno_handoff_full_checkpoint.jld2")
    if isfile(ck)
        new_ck = joinpath(RESULTS_DIR, "zenteno_seed_checkpoint.jld2")
        cp(ck, new_ck; force=true)
        println("[INIT_ONLY] Copied checkpoint to ", new_ck)
    else
        println("[INIT_ONLY] Warning: checkpoint not found at ", ck)
    end
    # Escribir métricas derivadas del estimation_report
    rep = latest("zenteno_estimation_report_")
    if rep === nothing
        println("[INIT_ONLY] No estimation report found; metrics skipped")
        return
    end
    SSE = PEN = REG = OBJ = FO_L_max = FO_U_max = FO_upt_max = NaN
    for ln in eachline(joinpath(RESULTS_DIR, basename(rep)))
        if startswith(ln, "PEN="); PEN = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
        if startswith(ln, "REG="); REG = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
        if startswith(ln, "OBJ="); OBJ = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
        if startswith(ln, "FO_L_max="); FO_L_max = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
        if startswith(ln, "FO_U_max="); FO_U_max = try parse(Float64, split(ln, "=")[2]) catch; NaN end; end
    end
    comp_max = maximum([FO_L_max, FO_U_max, FO_upt_max])
    metrics_path = joinpath(RESULTS_DIR, "zenteno_metrics_init_only_" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS") * ".txt")
    open(metrics_path, "w") do io
        println(io, "tag=init_only")
        println(io, @sprintf("SSE=%.6e", SSE))
        println(io, @sprintf("PEN=%.6e", PEN))
        println(io, @sprintf("REG=%.6e", REG))
        println(io, @sprintf("OBJ=%.6e", OBJ))
        println(io, @sprintf("comp_max=%.6e", comp_max))
        println(io, @sprintf("stationarity_residual=%.6e", NaN))
        println(io, @sprintf("comp_sum=%.6e", NaN))
    end
    println("[INIT_ONLY] Saved init-only metrics: ", metrics_path)
end

function mode_compare()
    include("compare_runs.jl")
end

# Short bounded-variation trial (Module 4) to assess impact of BV_ON constraints.
# Will attempt to use an existing seed checkpoint if present; otherwise will initialize from ODE.
function mode_bv_trial()
    ck = joinpath(RESULTS_DIR, "zenteno_seed_checkpoint.jld2")
    init_from_checkpoint = isfile(ck) ? "1" : "0"
    env = Dict(
        "HOMOTOPY" => "0",
        "INIT_FROM_CHECKPOINT" => init_from_checkpoint,
        "CHECKPOINT_PATH" => ck,
        "INIT_FROM_ODE" => init_from_checkpoint == "1" ? "0" : "1",
        "INIT_DUAL_FE" => "0",
        "WALL_TIME" => get(ENV, "BV_TRIAL_WALL_TIME", "60"),
        "BV_SCOPE" => get(ENV, "BV_SCOPE", "all"),
        "DV_MAX_GLU" => get(ENV, "DV_MAX_GLU", "0.5"),
        "DV_MAX_FRU" => get(ENV, "DV_MAX_FRU", "0.5"),
        "DV_MAX_COMMON" => get(ENV, "DV_MAX_COMMON", "50.0"),
        "BV_RXN_SET" => get(ENV, "BV_RXN_SET", ""),
        "IPOPT_MUMPS_MEM_PERCENT" => get(ENV, "IPOPT_MUMPS_MEM_PERCENT", "")
    )
    run_cmd(env; tag="bv_trial")
    rep = latest("zenteno_estimation_report_")
    if rep === nothing
        println("[BV_TRIAL] No estimation report found; metrics skipped")
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
    metrics_path = joinpath(RESULTS_DIR, "zenteno_metrics_bv_trial_" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS") * ".txt")
    open(metrics_path, "w") do io
        println(io, "tag=bv_trial")
        println(io, @sprintf("SSE=%.6e", SSE))
        println(io, @sprintf("PEN=%.6e", PEN))
        println(io, @sprintf("REG=%.6e", REG))
        println(io, @sprintf("OBJ=%.6e", OBJ))
        println(io, @sprintf("comp_max=%.6e", comp_max))
        println(io, @sprintf("stationarity_residual=%.6e", NaN))
        println(io, @sprintf("comp_sum=%.6e", NaN))
    end
    println("[BV_TRIAL] Saved BV trial metrics: ", metrics_path)
end

function main()
    mode = length(ARGS) >= 1 ? ARGS[1] : "baseline"
    if mode == "baseline"
        mode_baseline()
    elseif mode == "seed"
        # Safety: prevent accidental carry-over of FROZEN_BOUNDS from previous frozen runs
        if get(ENV, "FROZEN_BOUNDS", "0") == "1"
            println("[WARN] Detected FROZEN_BOUNDS=1 but mode='seed'. Resetting to 0 (use 'seed_frozen' explicitly to enable).")
            ENV["FROZEN_BOUNDS"] = "0"
        end
        mode_seed()
    elseif mode == "seed180"
        # 3-stage homotopy compressed into 180s (weights 1,1,1).
        for (k,v) in Dict(
            "SEED_WALL_TIME" => "180",
            "SEED_HOM_PHI" => get(ENV, "SEED_HOM_PHI", "1e-2,1e-1,1"),
            "SEED_HOM_W" => get(ENV, "SEED_HOM_W", "1e-4,1e-5,1e-6"),
            "SEED_HOM_WEIGHTS" => "1,1,1",
            "FROZEN_BOUNDS" => "0",
            # Default hybrid BV (uptake-only)
            "BV_ON" => get(ENV, "BV_ON", "1"),
            "BV_SCOPE" => get(ENV, "BV_SCOPE", "uptake"),
            "DV_MAX_GLU" => get(ENV, "DV_MAX_GLU", "0.5"),
            "DV_MAX_FRU" => get(ENV, "DV_MAX_FRU", "0.5"),
            "DV_MAX_COMMON" => get(ENV, "DV_MAX_COMMON", "50.0"),
            "BV_PHASES" => get(ENV, "BV_PHASES", "1,1,0")
        )
            ENV[k] = v
        end
        println("[SEED180] Running 180s multi-stage homotopy with default hybrid BV (S1–S2 on, S3 off)")
        mode_seed()
    elseif mode == "seed180_bv"
        # 3-stage homotopy 180s with BV uptake constraints.
        for (k,v) in Dict(
            "SEED_WALL_TIME" => "180",
            "SEED_HOM_PHI" => get(ENV, "SEED_HOM_PHI", "1e-2,1e-1,1"),
            "SEED_HOM_W" => get(ENV, "SEED_HOM_W", "1e-4,1e-5,1e-6"),
            "SEED_HOM_WEIGHTS" => "1,1,1",
            "FROZEN_BOUNDS" => "0",
            "BV_ON" => "1",
            "BV_SCOPE" => get(ENV, "BV_SCOPE", "uptake"),
            "DV_MAX_GLU" => get(ENV, "DV_MAX_GLU", "0.5"),
            "DV_MAX_FRU" => get(ENV, "DV_MAX_FRU", "0.5"),
            "DV_MAX_COMMON" => get(ENV, "DV_MAX_COMMON", "50.0"),
            "BV_PHASES" => get(ENV, "BV_PHASES", "1,1,0")
        )
            ENV[k] = v
        end
        println("[SEED180_BV] Running 180s multi-stage homotopy with BV constraints (scope=$(ENV["BV_SCOPE"]))")
        mode_seed()
    elseif mode == "seed180_cf"
        # Coarse->Fine pipeline with total wall < direct 180s (default: 60s coarse, 100s fine)
        cf = get(ENV, "COARSE_TO_FINE", "6,12")
        parts = split(cf, [',',';',' '])
        if length(parts) < 2
            error("COARSE_TO_FINE must be like '6,12'")
        end
        coarse_nfe = parse(Int, strip(parts[1])); fine_nfe = parse(Int, strip(parts[2]))
        # Coarse stage
        local t0 = Dates.now()
        for (k,v) in Dict(
            "NFE" => string(coarse_nfe),
            "SEED_WALL_TIME" => get(ENV, "COARSE_WALL_TIME", "60"),
            "SEED_HOM_PHI" => get(ENV, "SEED_HOM_PHI", "1e-2,1e-1,1"),
            "SEED_HOM_W" => get(ENV, "SEED_HOM_W", "1e-4,1e-5,1e-6"),
            "SEED_HOM_WEIGHTS" => "1,1,1",
            # BV hybrid on coarse as well
            "BV_ON" => get(ENV, "BV_ON", "1"),
            "BV_SCOPE" => get(ENV, "BV_SCOPE", "uptake"),
            "DV_MAX_GLU" => get(ENV, "DV_MAX_GLU", "0.5"),
            "DV_MAX_FRU" => get(ENV, "DV_MAX_FRU", "0.5"),
            "DV_MAX_COMMON" => get(ENV, "DV_MAX_COMMON", "50.0"),
            "BV_PHASES" => get(ENV, "BV_PHASES", "1,1,0")
        )
            ENV[k] = v
        end
        println("[CF] Coarse stage: NFE=$(ENV["NFE"]) wall=$(ENV["SEED_WALL_TIME"])s")
        mode_seed()
        ck_coarse = joinpath(RESULTS_DIR, "zenteno_handoff_full_checkpoint.jld2")
        if !isfile(ck_coarse)
            error("Coarse checkpoint missing: " * ck_coarse)
        end
        # Fine stage
        for (k,v) in Dict(
            "NFE" => string(fine_nfe),
            "SEED_WALL_TIME" => get(ENV, "FINE_WALL_TIME", "100"),
            "SEED_HOM_PHI" => get(ENV, "SEED_HOM_PHI", "1e-2,1e-1,1"),
            "SEED_HOM_W" => get(ENV, "SEED_HOM_W", "1e-4,1e-5,1e-6"),
            "SEED_HOM_WEIGHTS" => "1,1,1",
            "BV_ON" => get(ENV, "BV_ON", "1"),
            "BV_SCOPE" => get(ENV, "BV_SCOPE", "uptake"),
            "DV_MAX_GLU" => get(ENV, "DV_MAX_GLU", "0.5"),
            "DV_MAX_FRU" => get(ENV, "DV_MAX_FRU", "0.5"),
            "DV_MAX_COMMON" => get(ENV, "DV_MAX_COMMON", "50.0"),
            "BV_PHASES" => get(ENV, "BV_PHASES", "1,1,0"),
            # coarse→fine mapping flags
            "INIT_FROM_COARSE_CHECKPOINT" => "1",
            "CHECKPOINT_PATH" => ck_coarse,
            # trim overhead on fine stage
            "SKIP_PRE_ODE" => get(ENV, "SKIP_PRE_ODE_FINE", "1"),
            # Final fine stage: por defecto queremos ver el plot completo a nfe=fine_nfe.
            # Cambiamos el default de SKIP_PLOTS_FINE a "0" para que se ejecuten las gráficas post-optimización.
            # Si el usuario desea seguir ocultándolas puede exportar SKIP_PLOTS_FINE=1.
            "SKIP_PLOTS" => get(ENV, "SKIP_PLOTS_FINE", "0"),
            "BASELINE_WRITE" => get(ENV, "BASELINE_WRITE_FINE", "0"),
            # prefer preserving warm start instead of re-running ODE seeding
            "SEED_INIT_FROM_ODE" => get(ENV, "SEED_INIT_FROM_ODE_FINE", "0")
        )
            ENV[k] = v
        end
        println("[CF] Fine stage: NFE=$(ENV["NFE"]) wall=$(ENV["SEED_WALL_TIME"])s with coarse→fine mapping")
        mode_seed()
        local t1 = Dates.now()
        local dt = convert(Int, Dates.value(t1 - t0) ÷ 1000)
        println("[CF] Total wall coarse→fine ≈ ", dt, " s")
        # Save quick comparison vs direct fine 180s
        # Run direct fine 180s (or use existing) only if explicitly requested via ENV CF_COMPARE
        if get(ENV, "CF_COMPARE", "0") == "1"
            for (k,v) in Dict("NFE"=>string(fine_nfe)) ENV[k]=v; end
            println("[CF] Running direct fine 180s for comparison…")
            for (k,v) in Dict("SEED_WALL_TIME"=>"180") ENV[k]=v; end
            mode_seed()
        end
    elseif mode == "seed_cf"
        # Alias to seed180_cf
        ARGS[1] = "seed180_cf"
        main()
    elseif mode == "seed180_bv12"
        # Explicit hybrid BV: S1–S2 ON, S3 OFF (uptake-only), 180s
        for (k,v) in Dict(
            "SEED_WALL_TIME" => "180",
            "SEED_HOM_PHI" => get(ENV, "SEED_HOM_PHI", "1e-2,1e-1,1"),
            "SEED_HOM_W" => get(ENV, "SEED_HOM_W", "1e-4,1e-5,1e-6"),
            "SEED_HOM_WEIGHTS" => "1,1,1",
            "FROZEN_BOUNDS" => "0",
            "BV_ON" => "1",
            "BV_SCOPE" => get(ENV, "BV_SCOPE", "uptake"),
            "DV_MAX_GLU" => get(ENV, "DV_MAX_GLU", "0.5"),
            "DV_MAX_FRU" => get(ENV, "DV_MAX_FRU", "0.5"),
            "DV_MAX_COMMON" => get(ENV, "DV_MAX_COMMON", "50.0"),
            "BV_PHASES" => "1,1,0"
        )
            ENV[k] = v
        end
        println("[SEED180_BV12] Running 180s homotopy with BV in S1–S2 and off in S3 (uptake-only)")
        mode_seed()
    elseif mode == "seed_reduced"
        mode_seed_reduced()
    elseif mode == "seed_frozen"
        # Use frozen bounds inside seeding (activate via ENV overrides)
        ENV["FROZEN_BOUNDS"] = "1"
        mode_seed()
    elseif mode == "init_only"
        mode_init_only()
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
    elseif mode == "bv_trial"
        mode_bv_trial()
        elseif mode == "multistart"
            # Multi-start over estimable parameters in log-space.
            K = try parse(Int, get(ENV, "MULTISTART", "0")) catch; 0 end
            if K <= 1
                println("[MULTISTART] MULTISTART<=1; running single base mode. Set MULTISTART=3 (por ejemplo).")
                base_mode = get(ENV, "MULTISTART_BASE_MODE", "seed180_cf")
                ARGS[1] = base_mode
                main()
                return
            end
            base_mode = get(ENV, "MULTISTART_BASE_MODE", "seed180_cf")
            println("[MULTISTART] Ejecutando K=", K, " intentos; base_mode=", base_mode)
            # Replicate parameter metadata (must mirror MPCC_Zenteno for bounds logic)
            Pnom = Dict(
                :mu0=>0.141665, :betaG0=>1.41182, :betaF0=>8.49482, :Kn0=>0.226882,
                :Kg0=>3.1514, :Kf0=>2.97625, :Kig0=>29.5276, :Kie0=>2.99809, :Kd0=>3.11736e-5,
                :Yxn=>9.80576, :Yxg=>0.394345, :Yxf=>0.18622, :Yeg=>0.14133, :Yef=>0.96932
            )
            # Parse EST_PARAMS (same as MPCC)
            est_params_raw = get(ENV, "EST_PARAMS", "mu0")
            est_syms = Symbol.(filter(!isempty, split(est_params_raw, [',',';',' '])))
            est_syms = [s for s in est_syms if haskey(Pnom, s)]
            if isempty(est_syms)
                println("[MULTISTART] EST_PARAMS vacío o inválido; nada que optimizar.")
                return
            end
            # Build bounds in log-space with optional overrides via EST_RANGES
            # Syntax: EST_RANGES="mu0:0.4,2.5;kcat_glc:0.2,8" (multiplicative factors relative to Pnom)
            ranges_raw = get(ENV, "EST_RANGES", "")
            range_lo = Dict{Symbol,Float64}(); range_hi = Dict{Symbol,Float64}()
            if !isempty(strip(ranges_raw))
                for tok in filter(!isempty, split(ranges_raw, [';','\n']))
                    parts = split(tok, ':')
                    if length(parts) == 2
                        sname = Symbol(strip(parts[1]))
                        if haskey(Pnom, sname)
                            try
                                lr = split(parts[2], [',','/',' '])
                                if length(lr) >= 2
                                    range_lo[sname] = parse(Float64, strip(lr[1]))
                                    range_hi[sname] = parse(Float64, strip(lr[2]))
                                end
                            catch err
                                println("[MULTISTART] WARNING: failed to parse EST_RANGES token '", tok, "': ", err)
                            end
                        end
                    end
                end
            end
            LB = Dict{Symbol,Float64}(); UB = Dict{Symbol,Float64}()
            for s in est_syms
                local lo = haskey(range_lo, s) ? range_lo[s] : (s == :mu0 ? 0.5 : 0.1)
                local hi = haskey(range_hi, s) ? range_hi[s] : (s == :mu0 ? 2.0 : 10.0)
                LB[s] = log(max(1e-12, lo * Pnom[s]))
                UB[s] = log(max(1e-12, hi * Pnom[s]))
            end
            # Early-stop tunables (improved defaults unless user overrides via ENV):
            # Si el usuario NO define MULTISTART_EARLY_REL usamos 5e-3 (más exigente que 1e-3) para abortar antes.
            # Si el usuario NO define MULTISTART_MAX_NOIMPROVE usamos 1 (un solo intento sin mejora suficiente).
            early_rel = try
                let v = get(ENV, "MULTISTART_EARLY_REL", "")
                    isempty(v) ? 5e-3 : parse(Float64, v)
                end
            catch; 5e-3 end
            max_noimprove = try
                let v = get(ENV, "MULTISTART_MAX_NOIMPROVE", "")
                    isempty(v) ? 1 : parse(Int, v)
                end
            catch; 1 end
            println("[MULTISTART] early_rel=", early_rel, " max_noimprove=", max_noimprove, " (override con MULTISTART_EARLY_REL / MULTISTART_MAX_NOIMPROVE)")
            # Storage (include checkpoint path + real-space starts)
            records = Vector{NamedTuple{(:start_id,:SSE,:OBJ,:wall_s,:starts_log,:starts_real,:checkpoint)}}()
            best_SSE = Inf
            no_improve = 0
            t_run0 = Dates.now()
            for k in 1:K
                # Early stop condition
                if no_improve >= max_noimprove
                    println("[MULTISTART] Early stop: ", max_noimprove, " intentos consecutivos sin mejora relativa >", early_rel, " en SSE.")
                    break
                end
                # Sample uniform log-space
                starts_log = Dict{Symbol,Float64}()
                for s in est_syms
                    starts_log[s] = LB[s] + rand()*(UB[s]-LB[s])
                end
                ENV["EST_STARTS"] = join(string.(starts_log[s] for s in est_syms), ",")
                println("[MULTISTART] start_id=", k, " overrides EST_STARTS=", ENV["EST_STARTS"])
                # Also populate per-parameter real-space overrides (optional; used by TETA_START_ logic)
                for s in est_syms
                    ENV["TETA_START_" * String(s)] = string(exp(starts_log[s]))
                end
                # Run base mode once with overrides
                t0 = Dates.now()
                ARGS[1] = base_mode
                main()  # calls underlying seed180_cf etc.
                t1 = Dates.now()
                wall_s = convert(Int, Dates.value(t1 - t0) ÷ 1000)
                # Parse latest estimation report for SSE & OBJ
                rep = latest("zenteno_estimation_report_")
                SSE = OBJ = NaN
                if rep !== nothing
                    for ln in eachline(rep)
                        if startswith(ln, "SSE=")
                            SSE = try parse(Float64, split(ln, "=")[2]) catch; NaN end
                        elseif startswith(ln, "OBJ=")
                            OBJ = try parse(Float64, split(ln, "=")[2]) catch; NaN end
                        elseif startswith(ln, "OBJ_eff=")
                            OBJ = try parse(Float64, split(ln, "=")[2]) catch; OBJ end
                        elseif startswith(ln, "OBJ_comp=") && !isfinite(OBJ)
                            OBJ = try parse(Float64, split(ln, "=")[2]) catch; OBJ end
                        end
                    end
                else
                    println("[MULTISTART] WARNING: no estimation report found; SSE/OBJ NaN")
                end
                # Capture checkpoint produced by base mode and snapshot it uniquely
                ck_candidates = ["zenteno_seed_checkpoint.jld2", "zenteno_handoff_full_checkpoint.jld2"]
                ck_found = ""
                for ckname in ck_candidates
                    ckpath = joinpath(RESULTS_DIR, ckname)
                    if isfile(ckpath)
                        ck_found = ckpath
                        break
                    end
                end
                unique_ck = ""
                if ck_found != ""
                    unique_ck = joinpath(RESULTS_DIR, @sprintf("multistart_checkpoint_%03d.jld2", k))
                    try
                        cp(ck_found, unique_ck; force=true)
                        println("[MULTISTART] Snapshot checkpoint -> ", unique_ck)
                    catch err
                        println("[MULTISTART] WARNING: failed to copy checkpoint: ", err)
                        unique_ck = ""
                    end
                else
                    println("[MULTISTART] No checkpoint file detected for attempt ", k)
                end
                push!(records, (
                    start_id=k,
                    SSE=SSE,
                    OBJ=OBJ,
                    wall_s=wall_s,
                    starts_log=join(string.(starts_log[s] for s in est_syms), ","),
                    starts_real=join(string.(exp(starts_log[s]) for s in est_syms), ","),
                    checkpoint=unique_ck
                ))
                if isfinite(SSE)
                    if SSE < best_SSE * (1 - early_rel)
                        best_SSE = SSE
                        no_improve = 0
                    else
                        no_improve += 1
                    end
                else
                    no_improve += 1
                end
            end
            # Write summary CSV
            if !isempty(records)
                summ_path = joinpath(RESULTS_DIR, "zenteno_multistart_summary_" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS") * ".csv")
                open(summ_path, "w") do io
                    println(io, "start_id,SSE,OBJ,wall_s,starts_log,starts_real,checkpoint")
                    for r in records
                        @printf(io, "%d,%.6e,%.6e,%d,%s,%s,%s\n", r.start_id, r.SSE, r.OBJ, r.wall_s, r.starts_log, r.starts_real, r.checkpoint)
                    end
                end
                println("[MULTISTART] Saved summary ", summ_path)
                # Select best by SSE then OBJ fallback
                sorted = sort(records; lt=(a,b)->begin
                    as = isfinite(a.SSE) ? a.SSE : Inf
                    bs = isfinite(b.SSE) ? b.SSE : Inf
                    as == bs ? (isfinite(a.OBJ) ? a.OBJ : Inf) < (isfinite(b.OBJ) ? b.OBJ : Inf) : as < bs
                end)
                best = first(sorted)
                best_path = joinpath(RESULTS_DIR, "zenteno_multistart_best_starts.txt")
                open(best_path, "w") do io
                    println(io, "# Best multi-start (by SSE then OBJ)")
                    println(io, @sprintf("SSE=%.6e OBJ=%.6e wall_s=%d", best.SSE, best.OBJ, best.wall_s))
                    println(io, "EST_PARAMS=", est_syms)
                    println(io, "EST_STARTS(log)=", best.starts_log)
                    println(io, "EST_STARTS(real)=", best.starts_real)
                    println(io, "# Reutilizar (log): export EST_STARTS=", best.starts_log)
                    println(io, "# Reutilizar (real -> convertir a log si se requiere): ", best.starts_real)
                    if !isempty(best.checkpoint)
                        println(io, "BEST_CHECKPOINT=", best.checkpoint)
                    end
                end
                println("[MULTISTART] Best start recorded at ", best_path)
                if !isempty(best.checkpoint) && isfile(best.checkpoint)
                    out_ck = joinpath(RESULTS_DIR, "zenteno_multistart_best_checkpoint.jld2")
                    try
                        cp(best.checkpoint, out_ck; force=true)
                        println("[MULTISTART] Copied best checkpoint -> ", out_ck)
                    catch err
                        println("[MULTISTART] WARNING: failed to copy best checkpoint: ", err)
                    end
                else
                    println("[MULTISTART] No checkpoint associated with best start; nothing to copy.")
                end
            else
                println("[MULTISTART] No records; nothing saved.")
            end
    else
        error("Unknown mode $(mode). Use one of: baseline, seed, seed_run, init_only, compare")
    end
end

main()
