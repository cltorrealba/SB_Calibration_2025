#!/usr/bin/env julia
# experiment_pipeline_fresh.jl
# Fresh experiment pipeline for parameter estimation (mu0, Yeg, Yef)
# Clean implementation with clear experimental design
#
# Usage:
#   julia --project=. .\experiment_pipeline_fresh.jl seed
#   julia --project=. .\experiment_pipeline_fresh.jl multistart
#   julia --project=. .\experiment_pipeline_fresh.jl compare

using Dates
using Printf
using FileIO
using Glob

# ============================================================================
# Configuration
# ============================================================================

const BASE_DIR = @__DIR__
const RESULTS_DIR_BASE = abspath(joinpath(BASE_DIR, "results"))
# Default experiment override if ENV["EXPERIMENT"] not set
const EXPERIMENT_NAME = let v = get(ENV, "EXPERIMENT", ""); isempty(strip(v)) ? "exp_undefined" : v end
const RESULTS_DIR = abspath(joinpath(RESULTS_DIR_BASE, EXPERIMENT_NAME))
isdir(RESULTS_DIR) || mkpath(RESULTS_DIR)

# ============================================================================
# Pardiso configuration (IPARM / MTYPE) – tuned for MPCC / KKT casi singular
# ============================================================================

function configure_pardiso_defaults()
    # Solo tiene efecto si realmente estamos usando Pardiso
    solver = lowercase(strip(get(ENV, "IPOPT_LINEAR_SOLVER", "")))
    !(solver in ("pardiso", "pardisomkl")) && return

    # Helper local: solo setea si el usuario no lo definió antes
    set_default!(name::AbstractString, value::AbstractString) =
        isempty(strip(get(ENV, name, ""))) && (ENV[name] = value)

    # Tipo de matriz: real simétrica indefinida (KKT de MPCC)
    # Ipopt ya usa MTYPE = -2 internamente, pero lo dejamos explícito por si
    # el wrapper Panua lee esto.
    set_default!("PARDISO_MTYPE", "-2")

    # IPARM según índices del manual (0-based)
    set_default!("PARDISO_IPARM_0",  "1")   # activar iparm manual
    set_default!("PARDISO_IPARM_1",  "2")   # reordenamiento METIS automático
    set_default!("PARDISO_IPARM_4",  "0")   # no precondensation
    set_default!("PARDISO_IPARM_7",  "2")   # iter refinement (2 pasos)
    set_default!("PARDISO_IPARM_9",  "13")  # pivot perturbation recommended
    set_default!("PARDISO_IPARM_10", "1")   # scaling automático
    set_default!("PARDISO_IPARM_12", "1")   # matching (super importante en MPCC)
    set_default!("PARDISO_IPARM_18", "-1")  # pivot threshold automático
    set_default!("PARDISO_IPARM_24", "1")   # precision check
    set_default!("PARDISO_IPARM_26", "1")   # iterative refinement seguro
    set_default!("PARDISO_IPARM_34", "1")   # low-rank updates (KKT estable)
    set_default!("PARDISO_IPARM_59", "1")   # strong pivoting (KKT casi singular)

    println("[IPOPT] Pardiso defaults set (MTYPE=-2, IPARM[0,1,4,7,9,10,12,18,24,26,34,59])")
end

# ============================================================================
# Ipopt Configuration
# ============================================================================

function configure_custom_ipopt()
    # --- NUEVO: permitir forzar la instalación por defecto ---
    if get(ENV, "USE_DEFAULT_IPOPT", "0") == "1"
        println("[IPOPT] USE_DEFAULT_IPOPT=1 → skipping custom Ipopt; using default Ipopt")
        return
    end
    # Helper: generate parent chain up to project root (Windows safe)
    function parent_chain(path::String; max_depth::Int=8)
        acc = String[]
        cur = abspath(path)
        for i in 1:max_depth
            push!(acc, cur)
            parent = dirname(cur)
            parent == cur && break
            cur = parent
        end
        return acc
    end

    # Resolve explicit ENV first (user overrides win)
    root = get(ENV, "IPOPT_DLL_DIR", get(ENV, "PANUA_IPOPT_DIR", get(ENV, "PANUA_IPOPT_ROOT", "")))
    if isempty(strip(root))
        for p in parent_chain(BASE_DIR)
            dirs = filter(d -> startswith(lowercase(basename(d)), "panua-ipopt"), readdir(p; join=true))
            for d in dirs
                if isfile(joinpath(d, "lib", "libipopt.dll")) || isfile(joinpath(d, "libipopt.dll"))
                    root = d
                    ENV["PANUA_IPOPT_ROOT"] = root
                    println("[IPOPT] Auto-detected PANUA_IPOPT_ROOT=", root)
                    break
                end
            end
            !isempty(strip(root)) && break
        end
    end
    if isempty(strip(root))
        println("[IPOPT] No custom Ipopt root detected; using default Ipopt.")
        return
    end

    # Locate libipopt.dll (prefer lib subdir)
    cand_lib = isfile(joinpath(root, "lib", "libipopt.dll")) ? joinpath(root, "lib", "libipopt.dll") : joinpath(root, "libipopt.dll")
    if !isfile(cand_lib)
        println("[IPOPT] Warning: libipopt.dll not found under ", root)
        return
    end
    dll_dir = dirname(cand_lib)
    ENV["JULIA_IPOPT_LIBRARY_PATH"] = cand_lib
    # Prepend ipopt lib dir; also prepend ipopt root & bin if present to satisfy secondary deps
    ENV["PATH"] = dll_dir * ";" * get(ENV, "PATH", "")
    ipopt_root_bin = joinpath(root, "bin")
    if isdir(ipopt_root_bin)
        ENV["PATH"] = ipopt_root_bin * ";" * ENV["PATH"]
    end
    if dll_dir != root
        ENV["PATH"] = root * ";" * ENV["PATH"]
    end

    # Detect Pardiso directory AFTER Ipopt root (if not provided)
    pard_dir = get(ENV, "IPOPT_PARDISO_DLL_DIR", get(ENV, "PARDISO_DLL_DIR", ""))
    if isempty(strip(pard_dir))
        for p in parent_chain(BASE_DIR)
            dirs = filter(d -> startswith(lowercase(basename(d)), "panua-pardiso"), readdir(p; join=true))
            for d in dirs
                if isfile(joinpath(d, "lib", "libpardiso.dll")) || isfile(joinpath(d, "libpardiso.dll"))
                    pard_dir = d
                    ENV["PARDISO_DLL_DIR"] = pard_dir
                    println("[IPOPT] Auto-detected PARDISO_DLL_DIR=", pard_dir)
                    break
                end
            end
            !isempty(strip(pard_dir)) && break
        end
    end
    # Prepend Pardiso lib directory explicitly (must contain libpardiso.dll)
    if !isempty(strip(pard_dir)) && isdir(pard_dir)
        pard_lib_dir = isdir(joinpath(pard_dir, "lib")) ? joinpath(pard_dir, "lib") : pard_dir
        if isfile(joinpath(pard_lib_dir, "libpardiso.dll"))
            # Prepend pardiso lib, plus its bin if present
            ENV["PATH"] = pard_lib_dir * ";" * get(ENV, "PATH", "")
            pard_root_bin = joinpath(pard_dir, "bin")
            if isdir(pard_root_bin)
                ENV["PATH"] = pard_root_bin * ";" * ENV["PATH"]
            end
            if pard_lib_dir != pard_dir
                ENV["PATH"] = pard_dir * ";" * ENV["PATH"]
            end
            println("[IPOPT] Prepended Pardiso directories to PATH: ", pard_dir)
            # Try to locate Panua license file panua.lic and point runtime to it
            if isempty(strip(get(ENV, "PARDISO_LIC_PATH", ""))) && isempty(strip(get(ENV, "PARDISO_LICENSE_FILE", "")))
                lic_path = ""
                lic_file = ""
                # Candidates: pard_dir, repo-level panua-licenses, current BASE_DIR, project root
                candidates_dirs = String[]
                push!(candidates_dirs, pard_dir)
                for p in parent_chain(BASE_DIR)
                    push!(candidates_dirs, p)
                    push!(candidates_dirs, joinpath(p, "panua-licenses"))
                end
                for d in candidates_dirs
                    f = joinpath(d, "panua.lic")
                    if isfile(f)
                        lic_path = d
                        lic_file = f
                        break
                    end
                end
                if !isempty(lic_path)
                    ENV["PARDISO_LIC_PATH"] = lic_path
                    ENV["PARDISO_LICENSE_FILE"] = lic_file
                    println("[IPOPT] Found panua.lic at ", lic_file)
                else
                    println("[IPOPT] WARNING: panua.lic not found. Place it under panua-licenses/ or set PARDISO_LIC_PATH.")
                end
            end
        else
            println("[IPOPT] Pardiso directory found but libpardiso.dll missing: ", pard_lib_dir)
        end
    else
        println("[IPOPT] Pardiso directory not detected (optional).")
    end

    # Silence banner
    if isempty(strip(get(ENV, "PARDISOLICMESSAGE", "")))
        ENV["PARDISOLICMESSAGE"] = "1"
    end

    # Decide solver only if not user-set
    if isempty(get(ENV, "IPOPT_LINEAR_SOLVER", ""))
        ENV["IPOPT_LINEAR_SOLVER"] = "pardiso"
    end
    solver = lowercase(strip(get(ENV, "IPOPT_LINEAR_SOLVER", "")))

    # Verify pardiso availability if requested
    if solver == "pardiso"
        pard_candidates = String[
            joinpath(dll_dir, "libpardiso.dll"),
            joinpath(root, "lib", "libpardiso.dll"),
            joinpath(root, "libpardiso.dll")
        ]
        if !isempty(strip(pard_dir))
            push!(pard_candidates, joinpath(pard_dir, "lib", "libpardiso.dll"))
            push!(pard_candidates, joinpath(pard_dir, "libpardiso.dll"))
        end
        has_pardiso = any(isfile, pard_candidates)
        println("[IPOPT] Pardiso candidate files:")
        for c in pard_candidates
            println("         ", c, " exists=", isfile(c))
        end
        if has_pardiso
            println("[IPOPT] linear_solver=pardiso (DLL confirmed)")
        else
            println("[IPOPT] libpardiso.dll not found; switching to mumps")
            ENV["IPOPT_LINEAR_SOLVER"] = "mumps"
            solver = "mumps"
        end
    else
        println("[IPOPT] linear_solver=", solver)
    end

    # Aplicar tuning por defecto de Pardiso (solo si sigue activo)
    configure_pardiso_defaults()

    # Final debug summary
    println("[IPOPT] Using custom Ipopt library: ", cand_lib)
    println("[IPOPT] Active solver: ", solver)
    println("[IPOPT] PATH head entries:")
    for (i, seg) in enumerate(split(get(ENV, "PATH", ""), ';')[1:min(6, length(split(get(ENV, "PATH", ""), ';')))])
        println(@sprintf("         [%d] %s", i, seg))
    end
end

function configure_threads()
    # Unified control: PARDISO_NUM_THREADS (fallbacks to JULIA_NUM_THREADS if set)
    t = strip(get(ENV, "PARDISO_NUM_THREADS", get(ENV, "JULIA_NUM_THREADS", "")))
    isempty(t) && return

    if isempty(strip(get(ENV, "OMP_NUM_THREADS", "")))
        ENV["OMP_NUM_THREADS"] = t
    end
    if isempty(strip(get(ENV, "MKL_NUM_THREADS", "")))
        ENV["MKL_NUM_THREADS"] = t
    end
    if isempty(strip(get(ENV, "MKL_DYNAMIC", "")))
        ENV["MKL_DYNAMIC"] = "FALSE"
    end
    println("[IPOPT] Threads configured: OMP_NUM_THREADS=", ENV["OMP_NUM_THREADS"], ", MKL_NUM_THREADS=", ENV["MKL_NUM_THREADS"]) 
end

# Force Ipopt to print linear solver choice early for debugging
function debug_print_ipopt_solver()
    solver = get(ENV, "IPOPT_LINEAR_SOLVER", "(unset)")
    println("[DEBUG] IPOPT_LINEAR_SOLVER=", solver)
    println("[DEBUG] JULIA_IPOPT_LIBRARY_PATH=", get(ENV, "JULIA_IPOPT_LIBRARY_PATH", "(unset)"))
end

# ============================================================================
# Utility Functions
# ============================================================================

function latest(pattern::String)
    files = filter(f -> occursin(pattern, basename(f)), glob("*.txt", RESULTS_DIR))
    isempty(files) && return nothing
    return sort(files)[end]
end

function run_mpcc(env::Dict{String,T}; tag::String) where {T<:AbstractString}
    println("[RUN] Starting mode='", tag, "' @ ", Dates.now())

    # *** PARCHE 1: asegurar que MPCC use el mismo RESULTS_DIR que el pipeline ***
    ENV["EXPERIMENT"] = EXPERIMENT_NAME

    for (k, v) in env
        ENV[k] = String(v)   # nos aseguramos de guardar siempre String en ENV
        println("[ENV] ", k, "=", v)
    end
    configure_custom_ipopt()
    configure_threads()
    debug_print_ipopt_solver()
    include("MPCC_Zenteno_v2.jl")
    println("[RUN] Completed mode='", tag, "' @ ", Dates.now())
end


using Dates

function write_metrics(tag::String)
    # 1) localizar el último z_est_*.txt generado por MPCC_Zenteno
    summ_path = latest("z_est_")
    if summ_path === nothing
        @warn "[METRICS] No z_est_*.txt found; skipping metrics for tag=$(tag)"
        return nothing
    end

    lines = collect(eachline(summ_path))

    # Helper genérico para parsear "clave=valor" con varias posibles claves
    function parse_metric(lines::Vector{String}, keys::Vector{String})
        for line in lines
            for key in keys
                if startswith(line, key)
                    parts = split(line, "=")
                    if length(parts) == 2
                        vstr = strip(parts[2])
                        try
                            return parse(Float64, vstr)
                        catch
                            # seguimos buscando
                        end
                    end
                end
            end
        end
        return NaN
    end

    # 2) métricas básicas
    SSE = parse_metric(lines, ["SSE="])
    OBJ = parse_metric(lines, ["OBJ_eff=", "OBJ_comp=", "OBJ_raw=", "OBJ="])

    if !isfinite(SSE)
        @warn "[METRICS] SSE could not be parsed from $(summ_path); SSE=NaN"
    end
    if !isfinite(OBJ)
        @warn "[METRICS] OBJ could not be parsed from $(summ_path); OBJ=NaN"
    end

    # 3) métricas de complementariedad / KKT (ajusta los prefijos si en tu z_est tienen otros nombres)
    comp_max  = parse_metric(lines, ["comp_max=", "COMP_MAX=", "Phi_comp_max="])
    comp_L1   = parse_metric(lines, ["comp_L1=", "COMP_L1=", "Phi_comp_L1="])
    kkt_feas  = parse_metric(lines, ["eta_feas=", "KKT_feas=", "kkt_feas="])
    kkt_stat  = parse_metric(lines, ["eta_stat=", "KKT_stat=", "kkt_stat="])

    # 4) escribir un archivo de métricas por iteración (report por run)
    ts = Dates.format(Dates.now(), "yyyymmdd-HHMMSS")
    fname = "zmet_$(tag)_$(ts).txt"
    metrics_path = joinpath(RESULTS_DIR, fname)

    if Sys.iswindows() && length(metrics_path) > 240
        fname = "zmet_$(tag).txt"
        metrics_path = joinpath(RESULTS_DIR, fname)
        if Sys.iswindows() && length(metrics_path) > 240
            fname = "zmet.txt"
            metrics_path = joinpath(RESULTS_DIR, fname)
        end
    end

    mkpath(dirname(metrics_path))

    open(metrics_path, "w") do io
        println(io, "# metrics for tag=$(tag)")
        println(io, "summary_source=", summ_path)
        @printf(io, "SSE=%.6e\n", SSE)
        @printf(io, "OBJ=%.6e\n", OBJ)
        @printf(io, "comp_max=%.6e\n", comp_max)
        @printf(io, "comp_L1=%.6e\n", comp_L1)
        @printf(io, "kkt_feas=%.6e\n", kkt_feas)
        @printf(io, "kkt_stat=%.6e\n", kkt_stat)
    end

    println("[METRICS] Saved ", metrics_path)

    # devolvemos todo en un NamedTuple para que el multistart lo meta al CSV
    return (SSE = SSE,
            OBJ = OBJ,
            comp_max = comp_max,
            comp_L1 = comp_L1,
            kkt_feas = kkt_feas,
            kkt_stat = kkt_stat)
end

# ============================================================================
# Experimental Modes
# ============================================================================

"""
Seed generation with 3-stage homotopy
- Total time: 1800s (30 minutes)
- Homotopy stages: φ=[1e-2, 1e-1, 1] and w=[1e-4, 1e-5, 1e-6]
- BV constraints: ENABLED (uptake scope)
- Primal initialization: ODE
- Dual initialization: Feasible estimates
"""
function mode_seed()
    println("="^80)
    println("[SEED] Starting seed generation with 1800s homotopy + BV constraints")
    println("="^80)
    
    env = Dict(
        # Homotopy configuration
        "HOMOTOPY" => "1",
        "HOM_PHI" => get(ENV, "HOM_PHI", "1e-2,1e-1,1"),
        "HOM_W" => get(ENV, "HOM_W", "1e-4,1e-5,1e-6"),
        "HOM_WEIGHTS" => get(ENV, "HOM_WEIGHTS", "1,1,2"),  # 450s, 450s, 900s
        "WALL_TIME" => "600",
        "HV_ADAPTIVE" => "0",
        
        # Initialization
        "INIT_FROM_ODE" => "1",
        "INIT_DUAL_FE" => "1",
        "INIT_FROM_CHECKPOINT" => "0",
        
        # BV constraints (ON for seed)
        "BV_ON" => "1",
        "BV_SCOPE" => get(ENV, "BV_SCOPE", "uptake"),
        "BV_PHASES" => get(ENV, "BV_PHASES", "1,1,0"),  # S1-S2 on, S3 off
        "DV_MAX_GLU" => get(ENV, "DV_MAX_GLU", "0.5"),
        "DV_MAX_FRU" => get(ENV, "DV_MAX_FRU", "0.5"),
        "DV_MAX_COMMON" => get(ENV, "DV_MAX_COMMON", "50.0"),
        
        # Checkpoint handling
        "HANDOFF_FULL" => "1",
        "HANDOFF_STAGE" => "3",
        
        # Other settings
        "REDUCED_MODE" => "0",
        "FROZEN_BOUNDS" => "0"
    )
    
    run_mpcc(env; tag="seed")
    
    # Rename checkpoint
    ck = joinpath(RESULTS_DIR, "zenteno_handoff_full_checkpoint.jld2")
    if isfile(ck)
        new_ck = joinpath(RESULTS_DIR, "zenteno_seed_checkpoint.jld2")
        cp(ck, new_ck; force=true)
        println("[SEED] Saved checkpoint: ", new_ck)
    else
        println("[SEED] Warning: checkpoint not found")
    end
    
    write_metrics("seed")
    println("[SEED] Seed generation completed")

    # Si SEED_POLISH=1 → lanzar automáticamente modo_polish()
    if get(ENV, "SEED_POLISH", "0") == "1"
        println("\n[SEED] SEED_POLISH=1 → launching polishing run")
        mode_polish()
    end
end


"""
Multistart exploration
- Runs: K attempts (set via MULTISTART env variable)
- Time per run: 600s (10 minutes)
- BV constraints: DISABLED (as requested)
- Initialization: Random perturbations in log-space
- Parameters: mu0, Yeg, Yef (configurable via EST_PARAMS)
"""
function mode_multistart()
    println("="^80)
    println("[MULTISTART] Starting multistart exploration")
    println("="^80)
    
    K = try parse(Int, get(ENV, "MULTISTART", "10")) catch; 10 end
    println("[MULTISTART] Number of attempts: K=", K)
    
    # Parameter metadata
    Pnom = Dict(
        :mu0 => 0.141665,
        :betaG0 => 1.41182,
        :betaF0 => 8.49482,
        :Kn0 => 0.226882,
        :Kg0 => 3.1514,
        :Kf0 => 2.97625,
        :Kig0 => 29.5276,
        :Kie0 => 2.99809,
        :Kd0 => 3.11736e-5,
        :Yxn => 9.80576,
        :Yxg => 0.394345,
        :Yxf => 0.18622,
        :Yeg => 0.14133,
        :Yef => 0.96932
    )
    
    # Parse estimable parameters
    est_params_raw = get(ENV, "EST_PARAMS", "mu0,Yeg,Yef")
    est_syms = Symbol.(filter(!isempty, split(est_params_raw, [',',';',' '])))
    est_syms = [s for s in est_syms if haskey(Pnom, s)]
    
    if isempty(est_syms)
        println("[MULTISTART] ERROR: No valid parameters to estimate")
        return
    end
    
    println("[MULTISTART] Estimating parameters: ", est_syms)
    
    # Build log-space bounds
    ranges_raw = get(ENV, "EST_RANGES", "")
    range_lo = Dict{Symbol,Float64}()
    range_hi = Dict{Symbol,Float64}()
    
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
                        println("[MULTISTART] WARNING: failed to parse EST_RANGES token '", tok, "'")
                    end
                end
            end
        end
    end
    
    # Default ranges
    LB = Dict{Symbol,Float64}()
    UB = Dict{Symbol,Float64}()
    for s in est_syms
        lo = haskey(range_lo, s) ? range_lo[s] : (s == :mu0 ? 0.5 : 0.1)
        hi = haskey(range_hi, s) ? range_hi[s] : (s == :mu0 ? 2.0 : 10.0)
        LB[s] = log(max(1e-12, lo * Pnom[s]))
        UB[s] = log(max(1e-12, hi * Pnom[s]))
    end
    
    println("[MULTISTART] Parameter ranges (multiplicative):")
    for s in est_syms
        lo_mult = haskey(range_lo, s) ? range_lo[s] : (s == :mu0 ? 0.5 : 0.1)
        hi_mult = haskey(range_hi, s) ? range_hi[s] : (s == :mu0 ? 2.0 : 10.0)
        println("  ", s, ": [", lo_mult, ", ", hi_mult, "] × ", Pnom[s])
    end
    
    # Early stopping configuration
    early_rel = try parse(Float64, get(ENV, "MULTISTART_EARLY_REL", "5e-3")) catch; 5e-3 end
    max_noimprove = try parse(Int, get(ENV, "MULTISTART_MAX_NOIMPROVE", "3")) catch; 3 end
    println("[MULTISTART] Early stopping: rel_tol=", early_rel, ", max_no_improve=", max_noimprove)
    
    # Storage
    records = NamedTuple[]  # dejamos que el tipo se infiera, así es más fácil ampliar campos
    best_SSE = Inf
    no_improve = 0
    
    for k in 1:K
        # Early stopping check
        if no_improve >= max_noimprove
            println("[MULTISTART] Early stop: ", no_improve, " consecutive attempts without improvement")
            break
        end
        
        println("\n", "-"^80)
        println("[MULTISTART] Attempt ", k, " / ", K)
        println("-"^80)
        
        # Sample random starting points in log-space
        starts_log = Dict{Symbol,Float64}()
        for s in est_syms
            starts_log[s] = LB[s] + rand() * (UB[s] - LB[s])
        end
        
        # Set environment variables
        ENV["EST_STARTS"] = join(string.(starts_log[s] for s in est_syms), ",")
        println("[MULTISTART] EST_STARTS (log)=", ENV["EST_STARTS"])
        
        # Also set per-parameter real-space values
        starts_real = Dict{Symbol,Float64}()
        for s in est_syms
            starts_real[s] = exp(starts_log[s])
            ENV["TETA_START_" * String(s)] = string(starts_real[s])
        end
        println("[MULTISTART] EST_STARTS (real)=", join(string.(starts_real[s] for s in est_syms), ","))
        
        # Run optimization (600s, BV OFF)
        t0 = Dates.now()
        
        # Optional: reuse seed checkpoint for warm start of state vars while still randomizing parameters.
        # Enable via MULTISTART_USE_SEED=1
        seed_ck = joinpath(RESULTS_DIR, "zenteno_seed_checkpoint.jld2")
        fallback_ck = joinpath(RESULTS_DIR, "zenteno_handoff_full_checkpoint.jld2")
        use_seed = get(ENV, "MULTISTART_USE_SEED", "0") == "1"
        ck_to_use = if use_seed && isfile(seed_ck)
            seed_ck
        elseif use_seed && isfile(fallback_ck)
            fallback_ck
        else
            ""
        end

        env = Dict(
            # No homotopy for multistart
            "HOMOTOPY" => "0",
            "WALL_TIME" => "360",
            
            # Initialization from random starts
            "INIT_FROM_CHECKPOINT" => "0",
            "INIT_FROM_ODE" => "1",
            "INIT_DUAL_FE" => "1",
            
            # BV constraints OFF (as requested)
            "BV_ON" => "0",
            
            # Other settings
            "REDUCED_MODE" => "0",
            "FROZEN_BOUNDS" => "0",
            "SKIP_PLOTS" => "0"
        )

        if use_seed && ck_to_use != ""
            env["INIT_FROM_CHECKPOINT"] = "1"
            env["CHECKPOINT_PATH"] = ck_to_use
            println("[MULTISTART] Warm start from seed checkpoint: ", ck_to_use)
        else
            env["INIT_FROM_CHECKPOINT"] = get(env, "INIT_FROM_CHECKPOINT", "0")  # ensure key present
            use_seed && println("[MULTISTART] Seed checkpoint requested but not found; proceeding without warm start")
        end
        
        run_mpcc(env; tag="multistart_$(k)")
        
        t1 = Dates.now()
        wall_s = convert(Int, Dates.value(t1 - t0) ÷ 1000)
        
        # Parse results (robust to missing metrics)
        metrics_result = write_metrics("multistart_$(k)")
        if metrics_result === nothing
            println("[MULTISTART] WARNING: write_metrics returned nothing; assigning NaN to all metrics")
            SSE      = NaN
            OBJ      = NaN
            comp_max = NaN
            comp_L1  = NaN
            kkt_feas = NaN
            kkt_stat = NaN
        else
            SSE      = metrics_result.SSE
            OBJ      = metrics_result.OBJ
            comp_max = metrics_result.comp_max
            comp_L1  = metrics_result.comp_L1
            kkt_feas = metrics_result.kkt_feas
            kkt_stat = metrics_result.kkt_stat
        end
        
        # Capture checkpoint
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
        if !isempty(ck_found)
            unique_ck = joinpath(RESULTS_DIR, @sprintf("multistart_checkpoint_%03d.jld2", k))
            try
                cp(ck_found, unique_ck; force=true)
                println("[MULTISTART] Saved checkpoint: ", unique_ck)
            catch err
                println("[MULTISTART] WARNING: checkpoint copy failed: ", err)
                unique_ck = ""
            end
        end
        
        # Record results
        push!(records, (
            start_id   = k,
            SSE        = SSE,
            OBJ        = OBJ,
            comp_max   = comp_max,
            comp_L1    = comp_L1,
            kkt_feas   = kkt_feas,
            kkt_stat   = kkt_stat,
            wall_s     = wall_s,
            starts_log = join(string.(starts_log[s] for s in est_syms), ","),
            starts_real = join(string.(starts_real[s] for s in est_syms), ","),
            checkpoint = unique_ck
        ))

        
        # Update best and convergence tracking
        if isfinite(SSE)
            if SSE < best_SSE * (1 - early_rel)
                improvement = 100 * (best_SSE - SSE) / best_SSE
                println("[MULTISTART] New best SSE: ", @sprintf("%.6e", SSE), " (improvement: ", @sprintf("%.2f%%", improvement), ")")
                best_SSE = SSE
                no_improve = 0
            else
                no_improve += 1
                println("[MULTISTART] No significant improvement (", no_improve, "/", max_noimprove, ")")
            end
        else
            println("[MULTISTART] Invalid SSE (NaN/Inf); skipping early-stop counter")
            # Importante: NO incrementamos no_improve si no hay SSE válido
            # no_improve += 1
        end
    end
    
    # Write summary
    if !isempty(records)
        # *** PARCHE 2: usar nombres de archivo cortos para evitar límite de ruta en Windows ***
        ts = Dates.format(Dates.now(), "yyyymmdd-HHMMSS")
        fname = "ms_" * ts * ".csv"
        summ_path = joinpath(RESULTS_DIR, fname)

        # Extra seguro: si aun así la ruta es demasiado larga en Windows, usar un nombre mínimo
        if Sys.iswindows() && length(summ_path) > 240
            fname = "ms.csv"
            summ_path = joinpath(RESULTS_DIR, fname)
        end

        mkpath(dirname(summ_path))
        open(summ_path, "w") do io
            println(io, "start_id,SSE,OBJ,comp_max,comp_L1,kkt_feas,kkt_stat,wall_s,starts_log,starts_real,checkpoint")
            for r in records
                @printf(io, "%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%d,%s,%s,%s\n",
                    r.start_id,
                    r.SSE,
                    r.OBJ,
                    r.comp_max,
                    r.comp_L1,
                    r.kkt_feas,
                    r.kkt_stat,
                    r.wall_s,
                    r.starts_log,
                    r.starts_real,
                    r.checkpoint)
            end
        end  # ← cierra el `do io`

        println("\n[MULTISTART] Saved summary: ", summ_path)
        
        # Find and report best
        sorted = sort(records; by = r -> (isfinite(r.SSE) ? r.SSE : Inf,
                                          isfinite(r.OBJ) ? r.OBJ : Inf))
        best = first(sorted)
        
        best_path = joinpath(RESULTS_DIR, "ms_best.txt")
        open(best_path, "w") do io
            println(io, "# Best multistart result")
            println(io, "# Generated: ", Dates.now())
            println(io, @sprintf("SSE=%.6e", best.SSE))
            println(io, @sprintf("OBJ=%.6e", best.OBJ))
            println(io, @sprintf("wall_s=%d", best.wall_s))
            println(io, "EST_PARAMS=", join(string.(est_syms), ","))
            println(io, "EST_STARTS_LOG=", best.starts_log)
            println(io, "EST_STARTS_REAL=", best.starts_real)
            if !isempty(best.checkpoint)
                println(io, "CHECKPOINT=", best.checkpoint)
            end
        end
        println("[MULTISTART] Saved best result: ", best_path)
        
        # Copy best checkpoint
        if !isempty(best.checkpoint) && isfile(best.checkpoint)
            out_ck = joinpath(RESULTS_DIR, "zenteno_multistart_best_checkpoint.jld2")
            try
                cp(best.checkpoint, out_ck; force=true)
                println("[MULTISTART] Copied best checkpoint: ", out_ck)
            catch err
                println("[MULTISTART] WARNING: failed to copy best checkpoint: ", err)
            end
        end
        
        println("\n", "="^80)
        println("[MULTISTART] Summary:")
        println("  Total attempts: ", length(records))
        println("  Best SSE: ", @sprintf("%.6e", best.SSE))
        println("  Best OBJ: ", @sprintf("%.6e", best.OBJ))
        println("  Best starts (real): ", best.starts_real)
        println("="^80)
    else
        println("[MULTISTART] No records generated")
    end
end  # cierra function


"""
Baseline run (simple solve without homotopy or special initialization)
"""
function mode_baseline()
    println("="^80)
    println("[BASELINE] Starting baseline run")
    println("="^80)
    
    env = Dict(
        "HOMOTOPY" => "1",
        "INIT_FROM_CHECKPOINT" => "0",
        "INIT_FROM_ODE" => "1",
        "INIT_DUAL_FE" => "1",
        "REDUCED_MODE" => "0",   # <- cambia esto a 1
        "BV_ON" => "0",
        "WALL_TIME" => "600",
        "SKIP_PLOTS" => "0"
        
    )
    
    run_mpcc(env; tag="baseline")
    write_metrics("baseline")
end

"""
Compare results from different runs
"""
function mode_compare()
    println("="^80)
    println("[COMPARE] Comparing experimental results")
    println("="^80)
    
    if isfile(joinpath(BASE_DIR, "compare_runs.jl"))
        include("compare_runs.jl")
    else
        println("[COMPARE] compare_runs.jl not found")
    end
end

"""
Polishing run desde el checkpoint de seed
- HOMOTOPY: OFF
- Arranca desde zenteno_seed_checkpoint.jld2 (o handoff_full si no existe)
- Objetivo: cerrar complementaridad y alinear mejor ODE vs colocación
- Tiempo controlado vía POLISH_WALL_TIME (por defecto 1200 s)
"""
function mode_polish()
    println("="^80)
    println("[POLISH] Starting polishing run from seed checkpoint")
    println("="^80)

    # Opción 1: override explícito (por ej. checkpoint óptimo de multistart)
    override_ck = strip(get(ENV, "CHECKPOINT_OVERRIDE", ""))

    # Opción 2: lógica antigua (seed / handoff_full)
    seed_ck     = joinpath(RESULTS_DIR, "zenteno_seed_checkpoint.jld2")
    fallback_ck = joinpath(RESULTS_DIR, "zenteno_handoff_full_checkpoint.jld2")

    ck_to_use =
        if !isempty(override_ck) && isfile(override_ck)
            println("[POLISH] Using CHECKPOINT_OVERRIDE = ", override_ck)
            override_ck
        elseif isfile(seed_ck)
            seed_ck
        elseif isfile(fallback_ck)
            fallback_ck
        else
            println("[POLISH] ERROR: no checkpoint found in ", RESULTS_DIR)
            println("         Expected: ", override_ck)
            println("         or:       ", seed_ck)
            println("         or:       ", fallback_ck)
            return
        end

    println("[POLISH] Using checkpoint: ", ck_to_use)

    # Config de pulido (todo overrideable por ENV)
    env = Dict(
        "HOMOTOPY"            => "0",
        "INIT_FROM_CHECKPOINT" => "1",
        "CHECKPOINT_PATH"     => ck_to_use,
        "INIT_FROM_ODE"       => "1",
        "INIT_DUAL_FE"        => "1",
        # BV_ON por defecto apagado en pulido, pero se puede forzar con BV_ON_POLISH
        "BV_ON"               => get(ENV, "BV_ON_POLISH", get(ENV, "BV_ON", "0")),
        "REDUCED_MODE"        => "0",
        "FROZEN_BOUNDS"       => "0",
        "SKIP_PLOTS"          => get(ENV, "POLISH_SKIP_PLOTS", "0"),
        # tiempo total de pulido (segundos)
        "WALL_TIME"           => get(ENV, "POLISH_WALL_TIME", "1200")
    )

    run_mpcc(env; tag="polish")

    # Guardar checkpoint refinado (si se generó uno nuevo)
    ck_final = joinpath(RESULTS_DIR, "zenteno_handoff_full_checkpoint.jld2")
    if isfile(ck_final)
        out_ck = joinpath(RESULTS_DIR, "zenteno_polish_checkpoint.jld2")
        try
            cp(ck_final, out_ck; force = true)
            println("[POLISH] Saved checkpoint: ", out_ck)
        catch err
            println("[POLISH] WARNING: failed to copy checkpoint: ", err)
        end
    end

    write_metrics("polish")
    println("[POLISH] Polishing run completed")
end

# ============================================================================
# Main Entry Point
# ============================================================================

function main()
    mode = length(ARGS) >= 1 ? ARGS[1] : "seed"
    
    println("\n")
    println("╔" * "═"^78 * "╗")
    println("║" * " "^20 * "EXPERIMENT PIPELINE - FRESH" * " "^31 * "║")
    println("║" * " "^78 * "║")
    println("║  Mode: " * rpad(mode, 68) * "║")
    println("║  Experiment: " * rpad(EXPERIMENT_NAME, 62) * "║")
    println("║  Results dir: " * rpad(basename(RESULTS_DIR), 61) * "║")
    println("╚" * "═"^78 * "╝")
    println("\n")
    
    if mode == "seed"
        mode_seed()
    elseif mode == "multistart"
        mode_multistart()
    elseif mode == "baseline"
        mode_baseline()
    elseif mode == "polish"
        mode_polish()
    elseif mode == "compare"
        mode_compare()
    else
        println("ERROR: Unknown mode '", mode, "'")
        println("\nAvailable modes:")
        println("  seed       - Generate seed solution (1800s, 3-stage homotopy, BV ON)")
        println("  multistart - Multistart exploration (600s per run, BV OFF)")
        println("  baseline   - Simple baseline run (360s)")
        println("  compare    - Compare results from different runs")
        error("Invalid mode")
    end
    
    println("\n[PIPELINE] Completed successfully @ ", Dates.now())
end

# Execute main
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
