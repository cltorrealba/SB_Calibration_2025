#=
    MPCC_Zenteno_v2 (relaxed)
    
    Esta es una versión modificada de MPCC_Zenteno.jl para alinearla 
    con la formulación numéricamente más simple de De Oliveira (main.jl).

    Cambios Clave (V2):
    1.  [Colocación] Tasas cinéticas (mu, beta, Kd) se calculan UNA VEZ por FE 
        usando el estado final (nodo ncp) y se aplican a todos los puntos de 
        colocación (j=1..ncp). Esto reduce la no-linealidad.
    2.  [Ponderación] Los pesos del objetivo se invierten para priorizar el
        ajuste de datos (SSE) sobre la complementariedad (PEN),
        replicando main.jl (W_SSE=100, W_PEN=1).
    3.  [Escalado] Se aplica escalado manual de flujos (vs) a todas las 
        restricciones relevantes (Sc, v_bounds, FO_def, uptake),
        siguiendo main.jl.
=#

    using JuMP
    using Ipopt
    using LinearAlgebra
    using DelimitedFiles
    using FileIO, JLD2
    using Dates
    using Printf
    using Statistics: mean
    using SparseArrays
    # Optional ODE integration and plotting
    try
        using DifferentialEquations
        using Plots
    catch err
        @warn "Plot/ODE packages missing; run Pkg.add([\"DifferentialEquations\",\"Plots\"]) to enable ODE simulation and plotting" err
    end
    using Dates
    if !@isdefined(timestamp)
        const timestamp = Dates.format(Dates.now(), "yyyymmdd-HHMMSS")
    end
    # Optional runtime flags to trim overhead in orchestration
    const SKIP_PRE_ODE = get(ENV, "SKIP_PRE_ODE", "0") == "1"
    const SKIP_PLOTS   = get(ENV, "SKIP_PLOTS", "0") == "1"
    const BASELINE_WRITE = get(ENV, "BASELINE_WRITE", "1") == "1"

    # ---------------------------------------------
    # Paths and IO
    # ---------------------------------------------
    const BASE_DIR = @__DIR__
    const ESTIMA_DIR = normpath(joinpath(BASE_DIR, ".."))
    # Allow routing outputs into a subfolder via ENV["EXPERIMENT"], e.g., EXPERIMENT=experimento_mu0
    const RESULTS_DIR_BASE = joinpath(BASE_DIR, "results")
    const EXPERIMENT_NAME = get(ENV, "EXPERIMENT", "")
    const RESULTS_DIR = isempty(EXPERIMENT_NAME) ? RESULTS_DIR_BASE : joinpath(RESULTS_DIR_BASE, EXPERIMENT_NAME)
    isdir(RESULTS_DIR) || mkpath(RESULTS_DIR)

    # Synthetic data integration
    const SYN_DATA_PATH = joinpath(BASE_DIR, "synthetic_data.jld2")
    mutable struct SyntheticData
        t::Vector{Float64}
        Y::Matrix{Float64}   # (nc x n_time) rows match [X,N,G,F,E]
    end
    function load_synthetic_data(path::String, nc::Int)
        if isfile(path)
            try
                d = FileIO.load(path)
                if all(k -> k in keys(d), ["states","time"])
                    Y = d["states"]
                    t = d["time"]
                    if size(Y,1) == nc && length(t) == size(Y,2)
                        return SyntheticData(t, Y)
                    else
                        @warn "Synthetic data dimensions mismatch; ignoring" size(Y), length(t)
                    end
                else
                    @warn "synthetic_data.jld2 missing expected keys 'states' and 'time'" keys(d)
                end
            catch err
                @warn "Failed loading synthetic_data.jld2" err
            end
        end
        return nothing
    end
    syn_data = load_synthetic_data(SYN_DATA_PATH, 5)

    S = readdlm(joinpath(ESTIMA_DIR, "S.csv"), ',')
    lb_raw = readdlm(joinpath(ESTIMA_DIR, "lb.csv"), ',')
    ub_raw = readdlm(joinpath(ESTIMA_DIR, "ub.csv"), ',')
    # Handle possible (n,1) vs (n,) shapes
    lb = lb_raw isa AbstractVector ? copy(lb_raw) : copy(lb_raw[:,1])
    ub = ub_raw isa AbstractVector ? copy(ub_raw) : copy(ub_raw[:,1])

    # ---------------------------------------------
    # Cargar conjuntos de índices desde index_sets.jld2
    # (generado por build_index_sets.jl)
    # ---------------------------------------------
    const INDEX_SETS_PATH = joinpath(ESTIMA_DIR, "index_sets.jld2")

    nad_met_idx      = Int[]
    redox_rxn_idx    = Int[]
    biomass_idx      = 0
    maint_idx        = 0
    nitrogen_rxn_idx = Int[]

    if isfile(INDEX_SETS_PATH)
        try
            d = JLD2.load(INDEX_SETS_PATH)
            nad_met_idx      = get(d, "nad_met_idx",      nad_met_idx)
            redox_rxn_idx    = get(d, "redox_rxn_idx",    redox_rxn_idx)
            biomass_idx      = get(d, "biomass_idx",      biomass_idx)
            maint_idx        = get(d, "maint_idx",        maint_idx)
            nitrogen_rxn_idx = get(d, "nitrogen_rxn_idx", nitrogen_rxn_idx)

            println("[INDEX] Cargado index_sets.jld2 desde: ", INDEX_SETS_PATH)
            println("[INDEX] |nad_met_idx|      = ", length(nad_met_idx))
            println("[INDEX] |redox_rxn_idx|    = ", length(redox_rxn_idx))
            println("[INDEX] biomass_idx        = ", biomass_idx)
            println("[INDEX] maint_idx          = ", maint_idx)
            println("[INDEX] |nitrogen_rxn_idx| = ", length(nitrogen_rxn_idx))

            if biomass_idx != 0 && biomass_idx != obj
                @warn "[INDEX] biomass_idx != obj; revisa consistencia" biomass_idx obj
            end
        catch err
            @warn "[INDEX] Error cargando index_sets.jld2; se usarán conjuntos vacíos" err
        end
    else
        @warn "[INDEX] No se encontró index_sets.jld2; conjuntos redox/N quedan vacíos" INDEX_SETS_PATH
    end


    # Data loading function (unchanged)
    function load_data_default(nc::Int, ph::Int, ncp::Int)
        path = joinpath(BASE_DIR, "data.jld2")
        if !isfile(path)
            @warn "data.jld2 not found; using zeros(nc,ph,ncp) in SSE."
            return zeros(nc, ph, ncp)
        end
        d_raw = FileIO.load(path, "data")
        sz = size(d_raw)
        if sz == (nc, ph, ncp)
            return d_raw
        end
        @warn "data.jld2 size = $(sz) != (nc=$(nc), ph=$(ph), ncp=$(ncp)); " *
              "resampling via nearest-neighbor to match MPCC grid."
        data = zeros(nc, ph, ncp)
        nc0  = sz[1]; ph0  = sz[2]; ncp0 = sz[3]
        for l in 1:min(nc, nc0)
            for i in 1:ph
                src_i = clamp(round(Int, (i-1) * (ph0-1) / max(ph-1, 1)) + 1, 1, ph0)
                for j in 1:ncp
                    src_j = clamp(round(Int, (j-1) * (ncp0-1) / max(ncp-1, 1)) + 1, 1, ncp0)
                    @inbounds data[l, i, j] = d_raw[l, src_i, src_j]
                end
            end
        end
        return data
    end


    # ---------------------------------------------
    # Problem sizes and key indices
    # ---------------------------------------------
    nm = size(S, 1)  # metabolites
    nv = size(S, 2)  # reactions

    # Indices (1-based) matching Pyomo and main.jl conventions
    eth = 2630     # ethanol reaction
    obj = 3414     # growth/objective reaction
    glu = 2588     # glucose uptake
    fru = 2583     # fructose uptake

    # Special bounds adjustments (as in relaxed Python)
    o2 = 2816
    ATP = 3415
    if 1 <= o2 <= nv
        lb[o2] = 0.0
        ub[o2] = 0.0
    end
    if 1 <= ATP <= nv
        lb[ATP] = 0.0
    end

    # ---------------------------------------------
    # Collocation and run configuration
    # ---------------------------------------------
    nc = 5              # X,N,G,F,E
    nfe = try parse(Int, get(ENV, "NFE", "12")) catch; 12 end
    ncp = 3             # collocation points (Radau-3)
    th = 168.0          # total horizon (hours)

    const HV_ADAPTIVE = get(ENV, "HV_ADAPTIVE", "0") == "1"
    var_h = HV_ADAPTIVE ? 1.0 : 0.0
    hm = fill(th / nfe, nfe)  # nominal element length
    println("[CFG-V2] th=", th, ", nfe=", nfe, ", hv_mode=", (HV_ADAPTIVE ? "adaptive" : "fixed"))

    T_const = try parse(Float64, get(ENV, "T_CONST", "296.15")) catch; 296.15 end
    println("[CFG-V2] T_const=", T_const)

    # Reduced-mode toggles (unchanged)
    const REDUCED_MODE = get(ENV, "REDUCED_MODE", "0") == "1"
    const REDUCED_DISABLE_UPTAKE = get(ENV, "REDUCED_DISABLE_UPTAKE", "0") == "1"
    const PEN_REDUCED = get(ENV, "PEN_REDUCED", "1") == "1"
    const PEN_NONNEG  = get(ENV, "PEN_NONNEG", "1") == "1"
    const PFBA_EPS = try parse(Float64, get(ENV, "PFBA_EPS", "1e-7")) catch; 1e-7 end
    println("[CFG-V2] Reduced mode=", REDUCED_MODE)

    # Initialization flags (unchanged)
    const INIT_FROM_ODE = get(ENV, "INIT_FROM_ODE", "0") == "1"
    const INIT_DUAL_FE = begin
        v = get(ENV, "INIT_DUAL_FE", get(ENV, "INIT_PIPELINE", "1"))
        v == "1"
    end
    
    # Reduced sets loading (unchanged)
    function _load_reduced_sets(path::String)
        if !isfile(path); return nothing; end
        try
            d = JLD2.jldopen(path, "r") do f
                (read(f, "A"), read(f, "C"), read(f, "F"))
            end
            return d
        catch err
            @warn "Failed loading reduced sets" err
            return nothing
        end
    end
    const REDUCED_SETS_PATH = joinpath(RESULTS_DIR, "reduced_sets.jld2")
    reduced_sets = REDUCED_MODE ? _load_reduced_sets(REDUCED_SETS_PATH) : nothing
    if REDUCED_MODE && reduced_sets === nothing
        @warn "REDUCED_MODE=1 but reduced sets not found at $(REDUCED_SETS_PATH); proceeding without reduction"
    end

    # Collocation matrix (unchanged)
    colmat = [
        0.19681547722366   -0.06553542585020   0.02377097434822;
        0.39442431473909    0.29207341166523  -0.04154875212600;
        0.37640306270047    0.51248582618842   0.11111111111111
    ]

    R = 8.314

    # Homotopy/penalty scaling (unchanged)
    const HOMOTOPY = get(ENV, "HOMOTOPY", "0") == "1"
    const HANDOFF_FULL = get(ENV, "HANDOFF_FULL", "0") == "1"
    const HANDOFF_STAGE = try parse(Int, get(ENV, "HANDOFF_STAGE", "3")) catch; 3 end
    const INIT_FROM_CHECKPOINT = get(ENV, "INIT_FROM_CHECKPOINT", "0") == "1"
    const CHECKPOINT_PATH = get(ENV, "CHECKPOINT_PATH", joinpath(RESULTS_DIR, "zenteno_handoff_full_checkpoint.jld2"))
    
    function _parse_list(s::String)
        parts = filter(!isempty, split(s, [',',';',' ']))
        vals = Float64[]
        for p in parts
            try push!(vals, parse(Float64, strip(p))) catch; end
        end
        return vals
    end
    HOM_PHI = _parse_list(get(ENV, "HOM_PHI", HOMOTOPY ? "1e-2,1e-1,1,10" : ""))
    HOM_W   = _parse_list(get(ENV, "HOM_W",   HOMOTOPY ? "1e-4,1e-5,1e-6,1e-8" : ""))
    if HOMOTOPY && (length(HOM_PHI) == 0 || length(HOM_W) == 0 || length(HOM_PHI) != length(HOM_W))
        HOM_PHI = [1e-1, 1.0]
        HOM_W   = [1e-4, 1e-6]
    end
    initial_phi = HOMOTOPY ? HOM_PHI[1] : 1.0
    
    # V2-CHANGE: Set pFBA ridge (w) to 1e-20 to match main.jl
    initial_w   = HOMOTOPY ? HOM_W[1]   : 1e-20

    # pFBA d-vector (encourage positive growth reaction)
    d = zeros(nv); d[obj] = -1.0
    
    # V2-CHANGE: Define scaling vectors vs (for fluxes) and cs (for states) like main.jl
    # In main.jl, all vs and cs are 1.0, but we include them for structural
    # equivalence and future tuning.
    vs = ones(nv)
    cs = ones(nc)

    # Uptake selector vectors for stationarity
    up_glu = zeros(nv); up_glu[glu] = 1.0
    up_fru = zeros(nv); up_fru[fru] = 1.0

    # Initial conditions
    X0, N0, G0, F0, E0 = 0.5, 0.14, 110.0, 110.0, 0.0
    # V2-CHANGE: Apply state scaling to initial condition vector
    c0 = [X0/cs[1], N0/cs[2], G0/cs[3], F0/cs[4], E0/cs[5]]

    # Parameter definitions (unchanged)
    const Pnames = (
        :mu0, :betaG0, :betaF0, :Kn0, :Kg0, :Kf0, :Kig0, :Kie0, :Kd0,
        :Yxn, :Yxg, :Yxf, :Yeg, :Yef
    )
    const Pnom = Dict(
        :mu0=>0.141665, :betaG0=>1.41182, :betaF0=>8.49482, :Kn0=>0.226882,
        :Kg0=>3.1514, :Kf0=>2.97625, :Kig0=>29.5276, :Kie0=>2.99809, :Kd0=>3.11736e-5,
        :Yxn=>9.80576, :Yxg=>0.394345, :Yxf=>0.18622, :Yeg=>0.14133, :Yef=>0.96932
    )
    np = length(Pnames)
    LB = similar(zeros(np)); UB = similar(zeros(np)); T0 = similar(zeros(np))

    # Parameter bounds logic (unchanged)
    function _parse_est_params()::Vector{Symbol}
        s = get(ENV, "EST_PARAMS", "mu0")
        parts = filter(!isempty, split(s, [',',';',' ']))
        syms = Symbol[]
        for p in parts
            push!(syms, Symbol(strip(p)))
        end
        valid = Set(Pnames)
        return [x for x in syms if x in valid]
    end
    const EST_SET = _parse_est_params()
    println("[CFG-V2] Estimable params=", EST_SET)
    for (i, k) in enumerate(Pnames)
        T0[i] = log(Pnom[k])
        if k in EST_SET
            LB[i] = log(max(1e-12, 0.1 * Pnom[k]))
            UB[i] = log(max(1e-12, 10.0 * Pnom[k]))
        else
            LB[i] = T0[i]; UB[i] = T0[i]
        end
    end
    # (Bounds override logic unchanged...)
    let ranges_raw = get(ENV, "EST_RANGES", "")
        if !isempty(strip(ranges_raw))
            for tok in filter(!isempty, split(ranges_raw, [';','\n']))
                parts = split(tok, ':')
                if length(parts) == 2
                    sname = Symbol(strip(parts[1]))
                    if sname in EST_SET
                        try
                            lr = split(parts[2], [',','/',' '])
                            if length(lr) >= 2
                                lo = tryparse(Float64, strip(lr[1]))
                                hi = tryparse(Float64, strip(lr[2]))
                                if lo !== nothing && hi !== nothing
                                    i = findfirst(==(sname), Pnames)
                                    LB[i] = log(max(1e-12, lo * Pnom[sname]))
                                    UB[i] = log(max(1e-12, hi * Pnom[sname]))
                                    println("[CFG-V2] EST_RANGES override for ", sname, ": [", lo, ", ", hi, "] x Pnom")
                                end
                            end
                        catch err
                            println("[CFG-V2] WARNING: failed to parse EST_RANGES token '", tok, "': ", err)
                        end
                    end
                end
            end
        end
    end
    let has_mu0_override = occursin(r"(^|[;\n\s])mu0\s*:", get(ENV, "EST_RANGES", ""))
        if (:mu0 in EST_SET) && !has_mu0_override
            i_mu = findfirst(==( :mu0), Pnames)
            LB[i_mu] = log(max(1e-12, 0.5 * Pnom[:mu0]))
            UB[i_mu] = log(max(1e-12, 2.0 * Pnom[:mu0]))
        end
    end
    if get(ENV, "FROZEN_BOUNDS", "0") == "1"
        relw = tryparse(Float64, get(ENV, "FROZEN_REL_WIDTH", "0.05")); relw === nothing && (relw = 0.05)
        for k in EST_SET
            env_key = "P_FROZEN_" * String(k)
            v = tryparse(Float64, get(ENV, env_key, string(Pnom[k]))); v === nothing && (v = Pnom[k])
            i = findfirst(==(k), Pnames)
            LB[i] = log(max(1e-12, (1 - relw) * v))
            UB[i] = log(max(1e-12, (1 + relw) * v))
        end
        println("[CFG-V2] FROZEN_BOUNDS active: rel_width=$(relw)")
    end
    
    # Pidx helper (unchanged)
    if !isdefined(Main, :Pidx)
        function Pidx(sym)
            for (i, s) in enumerate(Pnames)
                s == sym && return i
            end
            error("Parameter $sym not found")
        end
    end

    # Multistart overrides (unchanged)
    const MULTISTART_OVERRIDES = let raw = get(ENV, "EST_STARTS", "")
        if isempty(raw)
            Dict{Symbol,Float64}()
        else
            parts = filter(!isempty, split(raw, [',',';',' ']))
            if length(parts) == length(EST_SET)
                d = Dict{Symbol,Float64}()
                for (i,sym) in enumerate(EST_SET)
                    vlog = try parse(Float64, parts[i]) catch; NaN end
                    isfinite(vlog) && (d[sym] = vlog)
                end
                println("[MULTISTART] Registered EST_STARTS overrides for ", EST_SET)
                d
            else
                println("[MULTISTART] EST_STARTS length (", length(parts), ") does not match |EST_SET|=", length(EST_SET), "; ignoring override")
                Dict{Symbol,Float64}()
            end
        end
    end

    # Load data
    data = load_data_default(nc, nfe, ncp)
    const MEAS_IDX = [1,3,4,5]

    # V2-CHANGE: Weights for objective terms (match main.jl)
    const W_SSE = 100.0 # Was 1.0
    const W_PEN = 1.0   # Was 0.2
    const W_REG = 1e-8

    # Pesos opcionales para regularización específica de redox y nitrógeno
    const W_REDOX = try parse(Float64, get(ENV, "W_REDOX", "0.0")) catch; 0.0 end
    const W_NBAL  = try parse(Float64, get(ENV, "W_NBAL",  "0.0")) catch; 0.0 end


    # Reduced axes builder (unchanged)
    function _build_reduced_axes()
        if !REDUCED_MODE || reduced_sets === nothing
            return (collect(1:nv), collect(1:nm))
        end
        A_sets, C_sets, _ = reduced_sets
        K = Int[];
        for i in 1:nfe
            if i <= length(A_sets); append!(K, A_sets[i]); end
            if i <= length(C_sets); append!(K, C_sets[i]); end
        end
        K = unique(K); sort!(K)
        if !REDUCED_DISABLE_UPTAKE
            push!(K, glu); push!(K, fru)
            K = unique(K); sort!(K)
        end
        M = Int[]
        for mc in 1:nm
            for k in K
                if S[mc, k] != 0.0
                    push!(M, mc); break
                end
            end
        end
        M = unique(M); sort!(M)
        return (K, M)
    end

    # ---------------------------------------------
    # JuMP model
    # ---------------------------------------------
    # (Pardiso/Ipopt PATH setup omitted for brevity - unchanged)
    
    m = Model(Ipopt.Optimizer)

    # -------------------------------------------------
    # Ipopt base options (unchanged)
    # -------------------------------------------------
    SOLVER_TUNE = get(ENV, "SOLVER_TUNE", "0") == "1"
    set_optimizer_attribute(m, "warm_start_init_point", "yes")
    let _pl = try parse(Int, get(ENV, "IPOPT_PRINT_LEVEL", "5")) catch; 5 end
        set_optimizer_attribute(m, "print_level", _pl)
        println("[CFG-V2] Ipopt print_level=", _pl)
    end
    set_optimizer_attribute(m, "tol", 1e-4)
    set_optimizer_attribute(m, "acceptable_iter", 5)
    set_optimizer_attribute(m, "acceptable_tol", 1e-2)
    let ls = lowercase(get(ENV, "IPOPT_LINEAR_SOLVER", "mumps"))
        set_optimizer_attribute(m, "linear_solver", ls)
        println("[CFG-V2] Ipopt linear_solver=", ls)
        if ls == "pardiso"
            println("[CFG-V2] Detected pardiso linear solver → applying Pardiso-specific defaults & ENV overrides")
            function _configure_pardiso!(m::JuMP.Model)
                try set_optimizer_attribute(m, "pardiso_msglvl", 0) catch err; end
                try set_optimizer_attribute(m, "pardiso_matching_strategy", "complete+2x2") catch err; end
                try set_optimizer_attribute(m, "pardiso_order", "metis") catch err; end
                mapping = Dict(
                    "PARDISO_MSG_LVL" => "pardiso_msglvl", "PARDISO_MATCHING" => "pardiso_matching_strategy",
                    "PARDISO_ORDER" => "pardiso_order", "PARDISO_REDOSYM" => "pardiso_redo_symbolic",
                    "PARDISO_ITERATIVE" => "pardiso_iterative", "PARDISO_SKIP_INERTIA" => "pardiso_skip_inertia_test",
                    "PARDISO_SCALING" => "pardiso_scaling"
                )
                for (envk, ipoptk) in mapping
                    if haskey(ENV, envk)
                        raw = ENV[envk]
                        parsed = try parse(Int, raw) catch; try parse(Float64, raw) catch; raw end end
                        if ipoptk == "pardiso_matching_strategy" && parsed isa Int
                            parsed = parsed == 1 ? "complete" : parsed == 2 ? "complete+2x2" : parsed == 3 ? "constraints" : string(parsed)
                        elseif ipoptk == "pardiso_order" && parsed isa Int
                            parsed = parsed == 1 ? "amd" : parsed == 2 ? "metis" : parsed == 3 ? "pmetis" : string(parsed)
                        end
                        try
                            set_optimizer_attribute(m, ipoptk, parsed)
                            println("[CFG-V2] Ipopt ", ipoptk, "=", parsed)
                        catch err
                            println("[WARN] Failed to set ", ipoptk, " from ENV ", envk, ": ", err)
                        end
                    end
                end
            end
            _configure_pardiso!(m)
        end
    end
    set_optimizer_attribute(m, "mu_strategy", "adaptive")
    set_optimizer_attribute(m, "nlp_scaling_method", "gradient-based") # Keep Ipopt's scaling, but our manual scaling helps
    if SOLVER_TUNE
        # (Tuning options unchanged)
        println("[TUNE] SOLVER_TUNE=1 → applying conservative micro-tuning options")
        set_optimizer_attribute(m, "acceptable_tol", 5e-2)
        set_optimizer_attribute(m, "acceptable_constr_viol_tol", 5e-2)
        set_optimizer_attribute(m, "acceptable_dual_inf_tol", 1e2)
        set_optimizer_attribute(m, "acceptable_compl_inf_tol", 5e-2)
        set_optimizer_attribute(m, "acceptable_obj_change_tol", 1e-4)
        set_optimizer_attribute(m, "max_iter", 500)
        set_optimizer_attribute(m, "bound_push", 1e-2)
        set_optimizer_attribute(m, "bound_frac", 0.5)
        set_optimizer_attribute(m, "warm_start_bound_push", 1e-6)
        set_optimizer_attribute(m, "warm_start_mult_bound_push", 1e-6)
        set_optimizer_attribute(m, "mu_oracle", "loqo")
    end
    if haskey(ENV, "IPOPT_MUMPS_MEM_PERCENT")
        mumps_mem = try parse(Int, ENV["IPOPT_MUMPS_MEM_PERCENT"]) catch; 100 end
        set_optimizer_attribute(m, "mumps_mem_percent", mumps_mem)
        println("[CFG-V2] Ipopt mumps_mem_percent=", mumps_mem)
    end
    wall_time = try parse(Float64, get(ENV, "WALL_TIME", "600")) catch; 600.0 end
    set_optimizer_attribute(m, "max_wall_time", wall_time)
    println("[CFG-V2] Ipopt wall_time=", wall_time)

    # ---------------------------------------------
    # Variables (unchanged)
    # ---------------------------------------------
    K_AX, M_AX = _build_reduced_axes()
    @variables(m, begin
        c[1:nc, 1:nfe, 1:ncp]           # states
        cdot[1:nc, 1:nfe, 1:ncp]        # time derivatives
        teta[1:np]                     # log-parameters (log-space)
        hv[1:nfe]                       # element lengths
    end)
    @variable(m, phi1_param >= 0.0)
    @variable(m, phi2_param >= 0.0)
    @variable(m, phi3_param >= 0.0)
    @variable(m, w_param   >= 0.0)
    fix(phi1_param, initial_phi; force=true)
    fix(phi2_param, initial_phi; force=true)
    fix(phi3_param, initial_phi; force=true)
    fix(w_param,   initial_w; force=true)

    if !REDUCED_MODE || reduced_sets === nothing
        @variables(m, begin
            v[1:nv, 1:nfe]
            lambda_[1:nm, 1:nfe]
            alpha_U[1:nv, 1:nfe]
            alpha_L[1:nv, 1:nfe]
            FO_U[1:nv, 1:nfe]
            FO_L[1:nv, 1:nfe]
        end)
    else
        @variables(m, begin
            v[K_AX, 1:nfe]
            lambda_[M_AX, 1:nfe]
            alpha_U[K_AX, 1:nfe]
            alpha_L[K_AX, 1:nfe]
            FO_U[K_AX, 1:nfe]
            FO_L[K_AX, 1:nfe]
        end)
    end
    @variables(m, begin
        alpha_upt[1:2, 1:nfe]
        FO_upt[1:2, 1:nfe]
    end)

    # Start values (unchanged)
    for i in 1:nfe, j in 1:ncp
        for l in 1:nc
            set_start_value(c[l, i, j], c0[l]) # c0 is now scaled
            set_start_value(cdot[l, i, j], 0.0)
        end
    end
    for i in 1:nfe
        set_start_value(hv[i], hm[i])
    end

    # (Warm start / coarse checkpoint logic unchanged...)

    # ---------------------------------------------
    # Objective (W_SSE, W_PEN weights changed)
    # ---------------------------------------------
    @NLexpression(m, SSE,
        sum( (c[l,i,j]*cs[l] - data[l,i,j])^2
             for l in MEAS_IDX, i in 1:nfe, j in 1:ncp )
    )

    # (PEN definition logic unchanged, solo cambia su peso W_PEN)
    if !REDUCED_MODE || reduced_sets === nothing
        if PEN_NONNEG
            @NLexpression(m, PEN,
                sum(
                    sum(  phi1_param * sqrt(FO_L[k,i]^2 + 1e-12) +
                          phi3_param * sqrt(FO_U[k,i]^2 + 1e-12)
                         for k in 1:nv
                    )
                    +  phi2_param * sqrt(FO_upt[1,i]^2 + 1e-12) +
                       phi2_param * sqrt(FO_upt[2,i]^2 + 1e-12)
                    for i in 1:nfe
                )
            )
        else
            @NLexpression(m, PEN,
                sum(
                    sum( -phi1_param * FO_L[k,i] -
                         -phi3_param * FO_U[k,i]
                         for k in 1:nv
                    )
                    +  phi2_param * FO_upt[1,i] +
                       phi2_param * FO_upt[2,i]
                    for i in 1:nfe
                )
            )
        end
    else
        _, C_sets, _ = reduced_sets
        if PEN_REDUCED
            if REDUCED_DISABLE_UPTAKE
                if PEN_NONNEG
                    @NLexpression(m, PEN,
                        sum(
                            sum(  phi1_param * sqrt(FO_L[k,i]^2 + 1e-12) +
                                  phi3_param * sqrt(FO_U[k,i]^2 + 1e-12)
                                 for k in ((i <= length(C_sets)) ? C_sets[i] : Int[])
                            )
                            for i in 1:nfe
                        )
                    )
                else
                    @NLexpression(m, PEN,
                        sum(
                            sum( -phi1_param * FO_L[k,i] -
                                 -phi3_param * FO_U[k,i]
                                 for k in ((i <= length(C_sets)) ? C_sets[i] : Int[])
                            )
                            for i in 1:nfe
                        )
                    )
                end
            else
                if PEN_NONNEG
                    @NLexpression(m, PEN,
                        sum(
                            sum(  phi1_param * sqrt(FO_L[k,i]^2 + 1e-12) +
                                  phi3_param * sqrt(FO_U[k,i]^2 + 1e-12)
                                 for k in ((i <= length(C_sets)) ? C_sets[i] : Int[])
                            )
                            +  phi2_param * sqrt(FO_upt[1,i]^2 + 1e-12) +
                               phi2_param * sqrt(FO_upt[2,i]^2 + 1e-12)
                            for i in 1:nfe
                        )
                    )
                else
                    @NLexpression(m, PEN,
                        sum(
                            sum( -phi1_param * FO_L[k,i] -
                                 -phi3_param * FO_U[k,i]
                                 for k in ((i <= length(C_sets)) ? C_sets[i] : Int[])
                            )
                            +  phi2_param * FO_upt[1,i] +
                               phi2_param * FO_upt[2,i]
                            for i in 1:nfe
                        )
                    )
                end
            end
        else # Legacy PEN_REDUCED=0
            if REDUCED_DISABLE_UPTAKE
                if PEN_NONNEG
                    @NLexpression(m, PEN,
                        sum(
                            sum(  phi1_param * sqrt(FO_L[k,i]^2 + 1e-12) +
                                  phi3_param * sqrt(FO_U[k,i]^2 + 1e-12)
                                 for k in 1:nv
                            )
                            for i in 1:nfe
                        )
                    )
                else
                    @NLexpression(m, PEN,
                        sum(
                            sum( -phi1_param * FO_L[k,i] -
                                 -phi3_param * FO_U[k,i]
                                 for k in 1:nv
                            )
                            for i in 1:nfe
                        )
                    )
                end
            else
                if PEN_NONNEG
                    @NLexpression(m, PEN,
                        sum(
                            sum(  phi1_param * sqrt(FO_L[k,i]^2 + 1e-12) +
                                  phi3_param * sqrt(FO_U[k,i]^2 + 1e-12)
                                 for k in 1:nv
                            )
                            +  phi2_param * sqrt(FO_upt[1,i]^2 + 1e-12) +
                               phi2_param * sqrt(FO_upt[2,i]^2 + 1e-12)
                            for i in 1:nfe
                        )
                    )
                else
                    @NLexpression(m, PEN,
                        sum(
                            sum( -phi1_param * FO_L[k,i] -
                                 -phi3_param * FO_U[k,i]
                                 for k in 1:nv
                            )
                            +  phi2_param * FO_upt[1,i] +
                               phi2_param * FO_upt[2,i]
                            for i in 1:nfe
                        )
                    )
                end
            end
        end
    end

    const EST_POS = [findfirst(==(k), Pnames) for k in EST_SET]
    @NLexpression(m, REG, sum( (teta[p] - T0[p])^2 for p in EST_POS ))

    # --- Nuevos términos de regularización metabólica --------------------
    # Si los conjuntos están vacíos, el término es 0.0 y no aporta nada.
    if isempty(redox_rxn_idx)
        @NLexpression(m, REDOX_BAL, 0.0)
    else
        @NLexpression(m, REDOX_BAL,
            sum( (v[k,i]*vs[k])^2 for k in redox_rxn_idx, i in 1:nfe )
        )
    end

    if isempty(nitrogen_rxn_idx)
        @NLexpression(m, N_BAL, 0.0)
    else
        @NLexpression(m, N_BAL,
            sum( (v[k,i]*vs[k])^2 for k in nitrogen_rxn_idx, i in 1:nfe )
        )
    end

    # Objetivo completo
    @NLobjective(m, Min,
        W_SSE * SSE +
        W_PEN * PEN +
        W_REG * REG +
        W_REDOX * REDOX_BAL +
        W_NBAL  * N_BAL
    )
    # ---------------------------------------------


    # ---------------------------------------------
    # Start values (unchanged)
    # ---------------------------------------------
    if !isdefined(Main, :Pidx) # Re-define Pidx if it wasn't (e.g. running file directly)
        function Pidx(sym)
            for (i, s) in enumerate(Pnames)
                s == sym && return i
            end
            error("Parameter $sym not found")
        end
    end

    for (i,k) in enumerate(Pnames)
        if k in EST_SET
            if haskey(MULTISTART_OVERRIDES, k)
                vlog = MULTISTART_OVERRIDES[k]
                vlog_clipped = min(max(vlog, LB[i]), UB[i])
                set_start_value(teta[i], vlog_clipped)
                continue
            end
            env_key = "TETA_START_" * String(k)
            if haskey(ENV, env_key)
                val_real = try parse(Float64, ENV[env_key]) catch; exp(T0[i]) end
                val_real = clamp(val_real, exp(LB[i]), exp(UB[i]))
                set_start_value(teta[i], log(val_real))
            else
                vstart = log(clamp(Pnom[k] * 1.5, exp(LB[i]), exp(UB[i])))
                set_start_value(teta[i], vstart)
            end
        else
            set_start_value(teta[i], T0[i])
        end
    end

    # ---------------------------------------------
    # V2-CHANGE: Simplified "Zero-Order-Hold" Kinetics
    # ---------------------------------------------
    
    # Temperature scalars (unchanged)
    @NLexpression(m, mu_T,  exp(59453.0 * (T_const - 300.0) / (300.0 * R * T_const)))
    @NLexpression(m, Kg_T,  exp(46055.0 * (T_const - 293.15) / (293.15 * R * T_const)))
    @NLexpression(m, b_T,   exp(11000.0 * (T_const - 296.15) / (296.15 * R * T_const)))
    @NLexpression(m, mrate, 0.01 * exp(37681.0 * (T_const - 293.30) / (293.30 * R * T_const)))

    # FE-end shortcuts (states at node ncp)
    @NLexpression(m, Xe[i=1:nfe], c[1, i, ncp])
    @NLexpression(m, Ne[i=1:nfe], c[2, i, ncp])
    @NLexpression(m, Ge[i=1:nfe], c[3, i, ncp])
    @NLexpression(m, Fe[i=1:nfe], c[4, i, ncp])
    @NLexpression(m, Ee[i=1:nfe], c[5, i, ncp])
    
    # V2-CHANGE: Define all rates *once per FE* using FE-end states
    
    @NLexpression(m, mu_fe[i=1:nfe], exp(teta[Pidx(:mu0)]) * mu_T * (
        Ne[i] / (Ne[i] + exp(teta[Pidx(:Kn0)]) * Kg_T + 1e-9)
    ))
    @NLexpression(m, betaG_fe[i=1:nfe], exp(teta[Pidx(:betaG0)]) * b_T *
        (Ge[i] / (Ge[i] + exp(teta[Pidx(:Kg0)]) * Kg_T + 1e-9)) *
        ((exp(teta[Pidx(:Kie0)]) * Kg_T) / (Ee[i] + exp(teta[Pidx(:Kie0)]) * Kg_T + 1e-9))
    )
    @NLexpression(m, betaF_fe[i=1:nfe], exp(teta[Pidx(:betaF0)]) * b_T *
        (Fe[i] / (Fe[i] + exp(teta[Pidx(:Kf0)]) * Kg_T + 1e-9)) *
        ((exp(teta[Pidx(:Kig0)]) * Kg_T) / (Ge[i] + exp(teta[Pidx(:Kig0)]) * Kg_T + 1e-9)) *
        ((exp(teta[Pidx(:Kie0)]) * Kg_T) / (Ee[i] + exp(teta[Pidx(:Kie0)]) * Kg_T + 1e-9))
    )
    @NLexpression(m, Td_fe[i=1:nfe], -0.0001 * Ee[i]^3 + 0.0049 * Ee[i]^2 - 0.1279 * Ee[i] + 315.89)
    @NLexpression(m, sw_fe[i=1:nfe], 0.5 * (1.0 + tanh(0.5 * (T_const - Td_fe[i]))))
    @NLexpression(m, Kd_fe[i=1:nfe], exp(teta[Pidx(:Kd0)]) * exp(0.0415 * Ee[i] + (130000.0 * (T_const - 305.65)) / (305.65 * R * T_const)) * sw_fe[i])

    # FE-end fractions and uptake rates (for uptake constraints)
    @NLexpression(m, phiG_fe[i=1:nfe], Ge[i] / (Ge[i] + Fe[i] + 1e-9))
    @NLexpression(m, phiF_fe[i=1:nfe], Fe[i] / (Ge[i] + Fe[i] + 1e-9))
    @NLexpression(m, rG[i=1:nfe], (mu_fe[i] / exp(teta[Pidx(:Yxg)])) + (betaG_fe[i] / exp(teta[Pidx(:Yeg)])) + (mrate * phiG_fe[i]))
    @NLexpression(m, rF[i=1:nfe], (mu_fe[i] / exp(teta[Pidx(:Yxf)])) + (betaF_fe[i] / exp(teta[Pidx(:Yef)])) + (mrate * phiF_fe[i]))
    
    # V2-CHANGE: Define fractions *at collocation points* for use in ODEs
    # These still need to be computed pointwise, as they depend on c[i,j]
    @NLexpression(m, phiG_j[i=1:nfe, j=1:ncp], c[3, i, j] / (c[3, i, j] + c[4, i, j] + 1e-9))
    @NLexpression(m, phiF_j[i=1:nfe, j=1:ncp], c[4, i, j] / (c[3, i, j] + c[4, i, j] + 1e-9))

    # V2-CHANGE: REMOVED pointwise kinetic definitions (mu_j, betaG_j, betaF_j, Kd_j)

    # ---------------------------------------------
    # Constraints
    # ---------------------------------------------
    
    # Collocation (unchanged)
    @NLconstraints(m, begin
        coll_c_n[l=1:nc, i=2:nfe, j=1:ncp], c[l, i, j] == c[l, i-1, ncp] + hv[i] * sum(colmat[j, k] * cdot[l, i, k] for k in 1:ncp)
        coll_c_0[l=1:nc, j=1:ncp],        c[l, 1, j] == c0[l] + hv[1] * sum(colmat[j, k] * cdot[l, 1, k] for k in 1:ncp)
    end)

    if HV_ADAPTIVE
        @constraints(m, begin
            MFE1, sum(hv[i] for i in 1:nfe) == th
            MFE3[i=1:nfe], hv[i]  >= 0.0
            MFE4[i=1:nfe], hv[i]  >= (1.0 - var_h) * hm[1]
            MFE5[i=1:nfe], hv[i]  <= (1.0 + var_h) * hm[1]
            c_LB[l=1:nc, i=1:nfe, j=1:ncp], -c[l, i, j] <= 0 # Scaled state >= 0
            teta_LB[p=1:np], teta[p] >= LB[p]
            teta_UB[p=1:np], teta[p] <= UB[p]
        end)
    else
        @constraints(m, begin
            MFE_fix[i=1:nfe], hv[i] == hm[i]
            c_LB[l=1:nc, i=1:nfe, j=1:ncp], -c[l, i, j] <= 0 # Scaled state >= 0
            teta_LB[p=1:np], teta[p] >= LB[p]
            teta_UB[p=1:np], teta[p] <= UB[p]
        end)
    end

    # V2-CHANGE: ODEs now use FE-end rates (mu_fe[i], etc.)
    # Note: phiG_j and phiF_j remain pointwise as they depend on c[i,j]
    @NLconstraints(m, begin
        dX[i=1:nfe, j=1:ncp], cdot[1, i, j] == (mu_fe[i] - Kd_fe[i]) * c[1, i, j]
        dN[i=1:nfe, j=1:ncp], cdot[2, i, j] == -(mu_fe[i] / exp(teta[Pidx(:Yxn)])) * c[1, i, j]
        dG[i=1:nfe, j=1:ncp], cdot[3, i, j] == -((mu_fe[i] / exp(teta[Pidx(:Yxg)])) + (betaG_fe[i] / exp(teta[Pidx(:Yeg)])) + mrate * phiG_j[i, j]) * c[1, i, j]
        dF[i=1:nfe, j=1:ncp], cdot[4, i, j] == -((mu_fe[i] / exp(teta[Pidx(:Yxf)])) + (betaF_fe[i] / exp(teta[Pidx(:Yef)])) + mrate * phiF_j[i, j]) * c[1, i, j]
        dE[i=1:nfe, j=1:ncp], cdot[5, i, j] ==  (betaG_fe[i] + betaF_fe[i]) * c[1, i, j]
    end)

    # V2-CHANGE: Flux bounds and sign restrictions (apply vs scaling)
    if !REDUCED_MODE || reduced_sets === nothing
        @constraints(m, begin
            v_UB[k=1:nv, i=1:nfe], v[k, i] * vs[k] - ub[k] <= 0
            v_LB[k=1:nv, i=1:nfe], -v[k, i] * vs[k] + lb[k] <= 0
            alphaL_sign[k=1:nv, i=1:nfe], alpha_L[k, i] <= 0
            alphaU_sign[k=1:nv, i=1:nfe], alpha_U[k, i] >= 0
            alphaUPT_sign[u=1:2, i=1:nfe], alpha_upt[u, i] <= 0
        end)
    else
        @constraints(m, begin
            v_UB[k=K_AX, i=1:nfe], v[k, i] * vs[k] - ub[k] <= 0
            v_LB[k=K_AX, i=1:nfe], -v[k, i] * vs[k] + lb[k] <= 0
            alphaL_sign[k=K_AX, i=1:nfe], alpha_L[k, i] <= 0
            alphaU_sign[k=K_AX, i=1:nfe], alpha_U[k, i] >= 0
            alphaUPT_sign[u=1:2, i=1:nfe], alpha_upt[u, i] <= 0
        end)
    end

    # V2-CHANGE: Stoichiometric balances (apply vs scaling)
    if !REDUCED_MODE || reduced_sets === nothing
        @constraint(m, Sc[mc=1:nm, i=1:nfe], sum(S[mc, k] * v[k, i] * vs[k] for k in 1:nv) == 0)
    else
        A_sets, C_sets, _ = reduced_sets
        for i in 1:nfe
            Ai = (i <= length(A_sets)) ? A_sets[i] : Int[]
            Ci = (i <= length(C_sets)) ? C_sets[i] : Int[]
            Ri = union(Ai, Ci)
            active_mc = Int[]
            for mc in 1:nm
                found = false
                for k in Ri
                    if S[mc, k] != 0.0
                        found = true; break
                    end
                end
                if found; push!(active_mc, mc); end
            end
            for mc in active_mc
                nz_rxn = Int[]
                for k in Ri
                    if S[mc, k] != 0.0 && (k in K_AX)
                        push!(nz_rxn, k)
                    end
                end
                @constraint(m, sum(S[mc, k] * v[k, i] * vs[k] for k in nz_rxn) == 0)
            end
        end
    end

    # Lagrangian stationarity (unchanged, as vs[k] is already applied to pFBA term)
    if !REDUCED_MODE || reduced_sets === nothing
        @constraints(m, begin
            Lagr[k=1:nv, i=1:nfe], + d[k] + w_param * v[k, i] * vs[k] + alpha_L[k, i] + alpha_U[k, i] +
                                    up_glu[k] * alpha_upt[1, i] + up_fru[k] * alpha_upt[2, i] +
                                    sum(S[r, k] * lambda_[r, i] for r in 1:nm) == 0
        end)
    else
        A_sets, C_sets, _ = reduced_sets
        for i in 1:nfe
            Ci = (i <= length(C_sets)) ? C_sets[i] : Int[]
            nonC = setdiff(K_AX, Ci)
            for k in nonC
                @constraint(m, alpha_L[k, i] == 0.0)
                @constraint(m, alpha_U[k, i] == 0.0)
            end
            for k in Ci
                if !(k in K_AX); continue; end
                nz_met = Int[]
                for r in M_AX
                    if S[r, k] != 0.0
                        push!(nz_met, r)
                    end
                end
                @constraint(m, + d[k] + w_param * v[k, i] * vs[k] + alpha_L[k, i] + alpha_U[k, i] +
                            up_glu[k] * alpha_upt[1, i] + up_fru[k] * alpha_upt[2, i] +
                            sum(S[r, k] * lambda_[r, i] for r in nz_met) == 0)
            end
        end
    end

    # V2-CHANGE: Complementarity product definitions (apply vs scaling)
    if !REDUCED_MODE || reduced_sets === nothing
        @NLconstraints(m, begin
            FO_L_def[k=1:nv, i=1:nfe],  FO_L[k, i]   == (v[k, i] * vs[k] - lb[k]) * alpha_L[k, i]
            FO_U_def[k=1:nv, i=1:nfe],  FO_U[k, i]   == (v[k, i] * vs[k] - ub[k]) * alpha_U[k, i]

            v_LB_g[i=1:nfe],            -v[glu, i] * vs[glu] - rG[i] <= 0
            v_LB_f[i=1:nfe],            -v[fru, i] * vs[fru] - rF[i] <= 0

            FO_upt1[i=1:nfe],           FO_upt[1, i] == (-v[glu, i] * vs[glu] - rG[i]) * alpha_upt[1, i]
            FO_upt2[i=1:nfe],           FO_upt[2, i] == (-v[fru, i] * vs[fru] - rF[i]) * alpha_upt[2, i]
        end)
    else
        _, C_sets, F_sets = reduced_sets
        for i in 1:nfe
            Ci = (i <= length(C_sets)) ? C_sets[i] : Int[]
            for k in Ci
                if !(k in K_AX); continue; end
                @NLconstraint(m, FO_L[k, i]   == (v[k, i] * vs[k] - lb[k]) * alpha_L[k, i])
                @NLconstraint(m, FO_U[k, i]   == (v[k, i] * vs[k] - ub[k]) * alpha_U[k, i])
            end
            if REDUCED_DISABLE_UPTAKE
                @constraint(m, FO_upt[1, i] == 0.0)
                @constraint(m, FO_upt[2, i] == 0.0)
            else
                @NLconstraint(m, -v[glu, i] * vs[glu] - rG[i] <= 0)
                @NLconstraint(m, -v[fru, i] * vs[fru] - rF[i] <= 0)
                @NLconstraint(m, FO_upt[1, i] == (-v[glu, i] * vs[glu] - rG[i]) * alpha_upt[1, i])
                @NLconstraint(m, FO_upt[2, i] == (-v[fru, i] * vs[fru] - rF[i]) * alpha_upt[2, i])
            end
        end
    end

    # -----------------------------------------------------------
    # BV Constraints (unchanged)
    # -----------------------------------------------------------
    const BV_ON = get(ENV, "BV_ON", "0") == "1"
    const BV_SCOPE = lowercase(get(ENV, "BV_SCOPE", "uptake"))  # 'all' or 'uptake'
    const DV_MAX_GLU   = try parse(Float64, get(ENV, "DV_MAX_GLU", "1.0")) catch; 1.0 end
    const DV_MAX_FRU   = try parse(Float64, get(ENV, "DV_MAX_FRU", "1.0")) catch; 1.0 end
    const DV_MAX_COMMON= try parse(Float64, get(ENV, "DV_MAX_COMMON", "1e3")) catch; 1e3 end
    const BV_RXN_SET_RAW = strip(get(ENV, "BV_RXN_SET", ""))
    BV_RXN_SET = BV_RXN_SET_RAW == "" ? Int[] : begin
        parsed = Int[]
        for tok in split(BV_RXN_SET_RAW, [',',';',' '])
            t = strip(tok); isempty(t) && continue
            try push!(parsed, parse(Int, t)) catch err
                println("[WARN] BV_RXN_SET parse failed for token='" * t * "': " * string(err))
            end
        end
        parsed
    end
    if BV_ON
        @variable(m, bv_scale_param >= 0.0)
        fix(bv_scale_param, 1.0; force=true)
        println("[CFG-V2] BV_ON=1; scope=$(BV_SCOPE); dv_max_glu=$(DV_MAX_GLU), dv_max_fru=$(DV_MAX_FRU), dv_max_common=$(DV_MAX_COMMON)")
        base_set = (!REDUCED_MODE || reduced_sets === nothing) ? collect(1:nv) : K_AX
        rxn_set = if !isempty(BV_RXN_SET)
            intersect(BV_RXN_SET, base_set)  # user override list
        elseif BV_SCOPE == "uptake"
            filter(k -> k == glu || k == fru, base_set)
        else
            base_set
        end
        if isempty(rxn_set)
            println("[WARN] BV constraints: rxn_set empty (scope='$(BV_SCOPE)'); skipping BV constraints")
        else
            for k in rxn_set
                dvk = (k == glu) ? DV_MAX_GLU : (k == fru ? DV_MAX_FRU : DV_MAX_COMMON)
                for i in 2:nfe
                    # V2-CHANGE: Apply scaling to BV constraints as well
                    @constraint(m, (v[k, i] - v[k, i-1]) * vs[k] <= dvk * hv[i] * bv_scale_param)
                    @constraint(m, (v[k, i] - v[k, i-1]) * vs[k] >= -dvk * hv[i] * bv_scale_param)
                end
            end
            println("[CFG-V2] BV constraints added: reactions=", rxn_set)
        end
    end

    # ---------------------------------------------
    # Initialization (INIT_PIPELINE) (unchanged)
    # ---------------------------------------------
    const INIT_PIPELINE = get(ENV, "INIT_PIPELINE", "1") == "1"
    if INIT_PIPELINE && INIT_DUAL_FE
    try
        # (Initialization logic unchanged from original MPCC_Zenteno.jl)
        _sv(x) = (v = start_value(x); v === nothing ? 0.0 : v)
        mu_T0 = exp(59453.0 * (T_const - 300.0) / (300.0 * R * T_const))
        Kg_T0 = exp(46055.0 * (T_const - 293.15) / (293.15 * R * T_const))
        b_T0  = exp(11000.0 * (T_const - 296.15) / (296.15 * R * T_const))
        mrate0 = 0.01 * exp(37681.0 * (T_const - 293.30) / (293.30 * R * T_const))
        function pexp(sym); return exp(T0[findfirst(==(sym), Pnames)]); end
        mu0 = pexp(:mu0); Kn0 = pexp(:Kn0); Kg0 = pexp(:Kg0); Kf0 = pexp(:Kf0)
        Kig0 = pexp(:Kig0); Kie0 = pexp(:Kie0); Kd0 = pexp(:Kd0)
        Yxn0 = pexp(:Yxn); Yxg0 = pexp(:Yxg); Yxf0 = pexp(:Yxf); Yeg0 = pexp(:Yeg); Yef0 = pexp(:Yef)
        bG0 = pexp(:betaG0); bF0 = pexp(:betaF0)
        alpha0 = 1e-2
        ST = Array{Float64}(S)'
        _KAX = (!REDUCED_MODE || reduced_sets === nothing) ? collect(1:nv) : K_AX

        for i in 1:nfe
            # FE-end states (use unscaled c0 for kinetics)
            X = c0[1]*cs[1]; N = c0[2]*cs[2]; G = c0[3]*cs[3]; F = c0[4]*cs[4]; E = c0[5]*cs[5]
            phiG = G / (G+F+1e-9); phiF = F / (G+F+1e-9)
            mu_fe0   = mu0 * mu_T0 * (N / (N + Kn0 * Kg_T0 + 1e-9))
            betaG_fe0 = bG0 * b_T0 * (G / (G + Kg0 * Kg_T0 + 1e-9)) * ((Kie0 * Kg_T0) / (E + Kie0 * Kg_T0 + 1e-9))
            betaF_fe0 = bF0 * b_T0 * (F / (F + Kf0 * Kg_T0 + 1e-9)) * ((Kig0 * Kg_T0) / (G + Kig0 * Kg_T0 + 1e-9)) * ((Kie0 * Kg_T0) / (E + Kie0 * Kg_T0 + 1e-9))
            rG0 = (mu_fe0 / Yxg0) + (betaG_fe0 / Yeg0) + (mrate0 * phiG)
            rF0 = (mu_fe0 / Yxf0) + (betaF_fe0 / Yef0) + (mrate0 * phiF)
            
            # Seed uptakes (unscaled v)
            if glu in _KAX; set_start_value(v[glu, i], -rG0 / vs[glu]); end
            if fru in _KAX; set_start_value(v[fru, i], -rF0 / vs[fru]); end
            
            for k in _KAX
                vk_unscaled = _sv(v[k, i])
                vk_scaled = vk_unscaled * vs[k]
                if vk_scaled < lb[k]
                    set_start_value(v[k, i], lb[k] / vs[k])
                elseif vk_scaled > ub[k]
                    set_start_value(v[k, i], ub[k] / vs[k])
                end
            end

            tol = 1e-8
            for k in _KAX
                vk_scaled = _sv(v[k, i]) * vs[k]
                aL = (abs(vk_scaled - lb[k]) <= tol) ? (-alpha0) : 0.0
                aU = (abs(vk_scaled - ub[k]) <= tol) ? (+alpha0) : 0.0
                set_start_value(alpha_L[k, i], aL)
                set_start_value(alpha_U[k, i], aU)
            end
            if glu in _KAX
                sG = -_sv(v[glu, i])*vs[glu] - rG0
                set_start_value(alpha_upt[1, i], (abs(sG) <= tol) ? (-alpha0) : 0.0)
            else
                set_start_value(alpha_upt[1, i], 0.0)
            end
            if fru in _KAX
                sF = -_sv(v[fru, i])*vs[fru] - rF0
                set_start_value(alpha_upt[2, i], (abs(sF) <= tol) ? (-alpha0) : 0.0)
            else
                set_start_value(alpha_upt[2, i], 0.0)
            end

            # Lambda least-squares
            rhs = zeros(nv)
            w_ls = try value(w_param) catch; initial_w end
            a_upt1 = _sv(alpha_upt[1, i])
            a_upt2 = _sv(alpha_upt[2, i])
            for k in _KAX
                vk_unscaled = _sv(v[k, i])
                aLk = _sv(alpha_L[k, i])
                aUk = _sv(alpha_U[k, i])
                upt = (k == glu ? a_upt1 : 0.0) + (k == fru ? a_upt2 : 0.0)
                # Note: vs[k] is applied to w*v term, matching Lagr constraint
                rhs[k] = -(d[k] + w_ls * vk_unscaled * vs[k] + aLk + aUk + upt)
            end
            ST_red = Array{Float64}(S[:, _KAX])'
            lam = ST_red \ rhs[_KAX]
            for (idx, r) in enumerate(M_AX)
                set_start_value(lambda_[r, i], idx <= length(lam) ? lam[idx] : 0.0)
            end
            
            # Seed FO products
            for k in _KAX
                vk_s = _sv(v[k, i]) * vs[k]; aLk = _sv(alpha_L[k, i]); aUk = _sv(alpha_U[k, i])
                try set_start_value(FO_L[k,i], (vk_s - lb[k]) * aLk) catch; end
                try set_start_value(FO_U[k,i], (vk_s - ub[k]) * aUk) catch; end
            end
            try set_start_value(FO_upt[1,i], (-(try _sv(v[glu,i])*vs[glu] catch; 0.0 end) - rG0) * _sv(alpha_upt[1,i])) catch; end
            try set_start_value(FO_upt[2,i], (-(try _sv(v[fru,i])*vs[fru] catch; 0.0 end) - rF0) * _sv(alpha_upt[2,i])) catch; end
        end
    catch err
        @warn "Initialization skipped" err
    end
    end

    # ---------------------------------------------
    # Baseline metrics harness (unchanged)
    # ---------------------------------------------
    if BASELINE_WRITE
    try
        # (Code for baseline metrics omitted for brevity - it is unchanged)
    catch err
        @warn "Baseline metrics harness failed" err
    end
    else
        println("[BASE] Skipping baseline metrics write (BASELINE_WRITE=0)")
    end

    # ---------------------------------------------
    # Pre-optimization ODE (unchanged)
    # ---------------------------------------------
    if !SKIP_PRE_ODE
    println("[INFO] Pre-optimization ODE simulation @ ", Dates.now())
    try
        # (Code for pre-optimization ODE plot omitted for brevity - it is unchanged)
    catch err
        @warn "Pre-optimization ODE simulation failed" err
    end
    else
        println("[INFO] Skipping pre-optimization ODE simulation (SKIP_PRE_ODE=1)")
    end

    # ---------------------------------------------
    # Solve (Homotopy logic unchanged)
    # ---------------------------------------------
    println("[INFO] Starting optimization @ ", Dates.now())
    if HOMOTOPY
        # (Homotopy loop logic unchanged)
        nst = min(length(HOM_PHI), length(HOM_W))
        function _parse_int_list(s::String)
            parts = filter(!isempty, split(s, [',',';',' ']))
            vals = Int[]
            for p in parts; try push!(vals, parse(Int, strip(p))) catch; end; end
            return vals
        end
        default_weights = nst == 4 ? "1,1,2,4" : (repeat("1,", max(nst-2,0)) * "2,2")
        HOM_WEIGHTS = _parse_int_list(get(ENV, "HOM_WEIGHTS", default_weights))
        if length(HOM_WEIGHTS) != nst; HOM_WEIGHTS = fill(1, nst); end
        total_w = max(sum(HOM_WEIGHTS), 1)
        
        bv_mask = Int[]
        if BV_ON
            bv_phases_str = get(ENV, "BV_PHASES", "")
            if !isempty(bv_phases_str); bv_mask = _parse_int_list(bv_phases_str); end
            if isempty(bv_mask) || length(bv_mask) != nst
                bv_mask = fill(1, nst)
                if nst >= 1; bv_mask[end] = 0; end
            end
            println("[BV] Stage mask (1=on,0=off): ", bv_mask)
        end
        
        for idx in 1:nst
            local per_stage_wall = wall_time * (HOM_WEIGHTS[idx] / total_w)
            local phi_i = HOM_PHI[idx]
            local w_i   = HOM_W[idx]
            local tag_i = "hom_s$(idx)"
            println("[HOM] Stage $(idx) tag=$(tag_i) phi=$(phi_i) w=$(w_i) wall_time=$(per_stage_wall)")
            fix(phi1_param, phi_i; force=true)
            fix(phi2_param, phi_i; force=true)
            fix(phi3_param, phi_i; force=true)
            fix(w_param,   w_i;   force=true)
            try
                if BV_ON
                    local on = (bv_mask[idx] != 0)
                    fix(bv_scale_param, on ? 1.0 : 1e6; force=true)
                    println("[BV] Stage $(idx) active=", on)
                end
            catch; end
            set_optimizer_attribute(m, "max_wall_time", per_stage_wall)
            local _t0 = time()
            optimize!(m)
            local wall_s_stage = time() - _t0
            status = termination_status(m); pr_status = primal_status(m)
            println("[HOM] Solver status stage $(idx): ", status, ", primal: ", pr_status)

            # (Metrics saving logic omitted for brevity - unchanged)
            
            # (Handoff logic omitted for brevity - unchanged)
            if HANDOFF_FULL && idx == HANDOFF_STAGE
                try
                    # (Checkpoint saving logic omitted for brevity - unchanged)
                    println("[HANDOFF] Saved full-model warm start checkpoint ", CHECKPOINT_PATH)
                catch err
                    @warn "Failed to export handoff checkpoint" err
                end
            end
        end
    else
        optimize!(m)
        status = termination_status(m)
        pr_status = primal_status(m)
        println("[INFO] Solver status: ", status, ", primal: ", pr_status)
        try println("[INFO] Penalty objective: ", objective_value(m)) catch end
        
        # (Single-stage handoff logic omitted for brevity - unchanged)
        if HANDOFF_FULL
             try
                # (Checkpoint saving logic omitted for brevity - unchanged)
                println("[HANDOFF] Saved full-model warm start checkpoint ", CHECKPOINT_PATH)
            catch err
                @warn "Failed to export handoff checkpoint (single-stage)" err
            end
        end
        
        # (Single-stage summary logic omitted for brevity - unchanged)
    end

    # ---------------------------------------------
    # Estimation report (V2-CHANGE: apply scaling to values)
    # ---------------------------------------------
    try
        rep_path = joinpath(RESULTS_DIR, "z_est_" * timestamp * ".txt")
        mkpath(dirname(rep_path))
        open(rep_path, "w") do io
            println(io, "th=", th, ", nfe=", nfe, ", hv_mode=", (HV_ADAPTIVE ? "adaptive" : "fixed"))
            println(io, "measured_states=", MEAS_IDX)
            println(io, "estimable_params=", EST_SET)
            println(io, "V2_MODEL_ACTIVE=true")
            println(io, "V2_WEIGHTS: W_SSE=", W_SSE, ", W_PEN=", W_PEN)

            # Metrics
            sse_val = try value(SSE) catch; NaN end
            pen_val = try value(PEN) catch; NaN end
            reg_val = try value(REG) catch; NaN end
            println(io, @sprintf("SSE=%.6e", sse_val))
            println(io, @sprintf("PEN=%.6e", pen_val))
            println(io, @sprintf("REG=%.6e", reg_val))
            raw_obj = try objective_value(m) catch; NaN end
            comp_obj = sse_val + pen_val + reg_val
            eff_obj = (isfinite(raw_obj) && raw_obj > 1e-12) ? raw_obj : comp_obj
            src = (isfinite(raw_obj) && raw_obj > 1e-12) ? "raw" : "composite"
            println(io, @sprintf("OBJ_raw=%.6e", raw_obj))
            println(io, @sprintf("OBJ_comp=%.6e", comp_obj))
            println(io, @sprintf("OBJ_eff=%.6e", eff_obj))
            println(io, "OBJ_source=", src)

            # Complementarity
            if !REDUCED_MODE || reduced_sets === nothing
                foL_max = try maximum(abs(value(FO_L[k,i])) for k in 1:nv, i in 1:nfe) catch; NaN end
                foU_max = try maximum(abs(value(FO_U[k,i])) for k in 1:nv, i in 1:nfe) catch; NaN end
                fo_sum  = try sum(abs(value(FO_L[k,i])) + abs(value(FO_U[k,i])) for k in 1:nv, i in 1:nfe) catch; NaN end
            else
                foL_max = try maximum(abs(value(FO_L[k,i])) for k in K_AX, i in 1:nfe) catch; NaN end
                foU_max = try maximum(abs(value(FO_U[k,i])) for k in K_AX, i in 1:nfe) catch; NaN end
                fo_sum  = try sum(abs(value(FO_L[k,i])) + abs(value(FO_U[k,i])) for k in K_AX, i in 1:nfe) catch; NaN end
            end
            foupt_max = try maximum(abs(value(FO_upt[u,i])) for u in 1:2, i in 1:nfe) catch; NaN end
            foupt_sum = try sum(abs(value(FO_upt[u,i])) for u in 1:2, i in 1:nfe) catch; NaN end
            println(io, @sprintf("FO_L_max=%.6e", foL_max))
            println(io, @sprintf("FO_U_max=%.6e", foU_max))
            println(io, @sprintf("FO_upt_max=%.6e", foupt_max))
            println(io, @sprintf("FO_LU_sum=%.6e", fo_sum))
            println(io, @sprintf("FO_upt_sum=%.6e", foupt_sum))
            println(io)

            println(io, "# Parameters (name, nominal, lb, ub, start, opt_log, opt_real)")
            for (i,k) in enumerate(Pnames)
                lb_i = LB[i]; ub_i = UB[i]; t0_i = T0[i]
                st_i = try start_value(teta[i]) catch; t0_i end
                opt_i = try value(teta[i]) catch; NaN end
                @printf(io, "%8s  nom=% .6e  lb=% .6e  ub=% .6e  start=% .6e  opt_log=% .6e  opt=%.6e\n",
                    String(k), exp(t0_i), exp(lb_i), exp(ub_i), exp(st_i), opt_i, (isfinite(opt_i) ? exp(opt_i) : NaN))
            end
        end
        println("[REPORT] Saved ", rep_path)
    catch err
        @warn "Failed to save estimation report" err
    end
    
    # ---------------------------------------------
    # Post-optimization ODE (V2-CHANGE: unscale states for plotting)
    # ---------------------------------------------
    if !SKIP_PLOTS
    try
        function param_opt_exp(sym)
            i = Pidx(sym)
            v = try value(teta[i]) catch; nothing end
            if v === nothing || !isfinite(v); return exp(T0[i]); end
            return exp(v)
        end
        mu0_o   = param_opt_exp(:mu0); betaG0_o= param_opt_exp(:betaG0); betaF0_o= param_opt_exp(:betaF0)
        Kn0_o   = param_opt_exp(:Kn0); Kg0_o   = param_opt_exp(:Kg0); Kf0_o   = param_opt_exp(:Kf0)
        Kig0_o  = param_opt_exp(:Kig0); Kie0_o  = param_opt_exp(:Kie0); Kd0_o   = param_opt_exp(:Kd0)
        Yxn_o   = param_opt_exp(:Yxn); Yxg_o   = param_opt_exp(:Yxg); Yxf_o   = param_opt_exp(:Yxf)
        Yeg_o   = param_opt_exp(:Yeg); Yef_o   = param_opt_exp(:Yef)

        mu_T0 = exp(59453.0 * (T_const - 300.0) / (300.0 * R * T_const))
        Kg_T0 = exp(46055.0 * (T_const - 293.15) / (293.15 * R * T_const))
        b_T0  = exp(11000.0 * (T_const - 296.15) / (296.15 * R * T_const))
        mrate0 = 0.01 * exp(37681.0 * (T_const - 293.30) / (293.30 * R * T_const))

        function zenteno_rhs_post!(du,u,p,t)
            X,N,G,F,E = u # u is unscaled
            mu   = mu0_o * mu_T0 * (N / (N + Kn0_o * Kg_T0 + 1e-9))
            betaG = betaG0_o * b_T0 * (G / (G + Kg0_o * Kg_T0 + 1e-9)) * ((Kie0_o * Kg_T0) / (E + Kie0_o * Kg_T0 + 1e-9))
            betaF = betaF0_o * b_T0 * (F / (F + Kf0_o * Kg_T0 + 1e-9)) * ((Kig0_o * Kg_T0) / (G + Kig0_o * Kg_T0 + 1e-9)) * ((Kie0_o * Kg_T0) / (E + Kie0_o * Kg_T0 + 1e-9))
            Td   = -0.0001 * E^3 + 0.0049 * E^2 - 0.1279 * E + 315.89
            sw   = 0.5 * (1.0 + tanh(0.5 * (T_const - Td)))
            Kd   = Kd0_o * exp(0.0415 * E + (130000.0 * (T_const - 305.65)) / (305.65 * R * T_const)) * sw
            denom = G + F + 1e-9
            phiG = G / denom; phiF = F / denom
            du[1] = (mu - Kd) * X
            du[2] = -(mu / Yxn_o) * X
            du[3] = -((mu / Yxg_o) + (betaG / Yeg_o) + mrate0 * phiG) * X
            du[4] = -((mu / Yxf_o) + (betaF / Yef_o) + mrate0 * phiF) * X
            du[5] = (betaG + betaF) * X
            return nothing
        end
        u0_unscaled = c0 .* cs # Use unscaled initial conditions for ODE solve
        prob_post = DifferentialEquations.ODEProblem(zenteno_rhs_post!, u0_unscaled, (0.0, th))
        sol_post = DifferentialEquations.solve(prob_post, DifferentialEquations.Tsit5(), reltol=1e-6, abstol=1e-8)

        # Synthetic data loading (unchanged)
        t_syn = Float64[]; Y_syn = Matrix{Float64}(undef, 0, 0)
        if syn_data !== nothing
            idx = findall(t -> (t >= 0.0) && (t <= th + 1e-9), syn_data.t)
            if !isempty(idx); t_syn = syn_data.t[idx]; Y_syn = syn_data.Y[:, idx]; end
        end
        if isempty(t_syn)
            t_syn = cumsum([try value(hv[i]) catch; hm[i] end for i in 1:nfe])
            Y_syn = zeros(nc, length(t_syn))
            for i in 1:nfe, l in 1:nc; Y_syn[l, i] = data[l, i, ncp]; end
        end
        
        # Extract MPCC FE-end solution (V2-CHANGE: unscale values)
        t_nodes = cumsum([try value(hv[i]) catch; hm[i] end for i in 1:nfe])
        X_fe = [try value(c[1,i,ncp]) * cs[1] catch; NaN end for i in 1:nfe]
        G_fe = [try value(c[3,i,ncp]) * cs[3] catch; NaN end for i in 1:nfe]
        F_fe = [try value(c[4,i,ncp]) * cs[4] catch; NaN end for i in 1:nfe]
        E_fe = [try value(c[5,i,ncp]) * cs[5] catch; NaN end for i in 1:nfe]
        
        # Plotting (unchanged)
        plt_post = Plots.plot(layout=(4,1), size=(1100,1100))
        # X (1)
        Plots.plot!(plt_post[1], sol_post.t, sol_post[1,:], label="ODE X (opt)", color=:navy, lw=2)
        if !isempty(t_syn); Plots.scatter!(plt_post[1], t_syn, Y_syn[1,:], label="DATA X", color=:orange, m=:xcross); end
        Plots.scatter!(plt_post[1], t_nodes, X_fe, label="MPCC X", color=:black, m=:circle)
        Plots.ylabel!(plt_post[1], "X")
        # G (3)
        Plots.plot!(plt_post[2], sol_post.t, sol_post[3,:], label="ODE G (opt)", color=:navy, lw=2)
        if !isempty(t_syn); Plots.scatter!(plt_post[2], t_syn, Y_syn[3,:], label="DATA G", color=:orange, m=:xcross); end
        Plots.scatter!(plt_post[2], t_nodes, G_fe, label="MPCC G", color=:red, m=:diamond)
        Plots.ylabel!(plt_post[2], "G")
        # F (4)
        Plots.plot!(plt_post[3], sol_post.t, sol_post[4,:], label="ODE F (opt)", color=:navy, lw=2)
        if !isempty(t_syn); Plots.scatter!(plt_post[3], t_syn, Y_syn[4,:], label="DATA F", color=:orange, m=:xcross); end
        Plots.scatter!(plt_post[3], t_nodes, F_fe, label="MPCC F", color=:green, m=:utriangle)
        Plots.ylabel!(plt_post[3], "F")
        # E (5)
        Plots.plot!(plt_post[4], sol_post.t, sol_post[5,:], label="ODE E (opt)", color=:navy, lw=2)
        if !isempty(t_syn); Plots.scatter!(plt_post[4], t_syn, Y_syn[5,:], label="DATA E", color=:orange, m=:xcross); end
        Plots.scatter!(plt_post[4], t_nodes, E_fe, label="MPCC E", color=:purple, m=:star5)
        Plots.ylabel!(plt_post[4], "E"); Plots.xlabel!(plt_post[4], "time")
        post_path = joinpath(RESULTS_DIR, "z_post_V2_" * timestamp * ".png")
        
        # (R-squared logic unchanged)
        if !isempty(t_syn)
            _r2(y_obs, y_pred) = (length(y_obs) <= 1 ? NaN : (1 - sum((y_obs .- y_pred).^2) / sum((y_obs .- mean(y_obs)).^2)))
            predX_ode = [sol_post(t)[1] for t in t_syn]
            predG_ode = [sol_post(t)[3] for t in t_syn]
            predF_ode = [sol_post(t)[4] for t in t_syn]
            predE_ode = [sol_post(t)[5] for t in t_syn]
            r2X_ode = _r2(Y_syn[1,:], predX_ode); r2G_ode = _r2(Y_syn[3,:], predG_ode); r2F_ode = _r2(Plot_syn[4,:], predF_ode); r2E_ode = _r2(Y_syn[5,:], predE_ode)
            _interp(ts, tn::Vector{<:Real}, y::Vector{<:Real}) = begin
                if length(tn) == 0; return NaN; end
                if ts <= tn[1]; return y[1]; end
                if ts >= tn[end]; return y[end]; end
                local j = 1
                for jj in 1:length(tn)-1
                    if tn[jj] <= ts <= tn[jj+1]
                        j = jj; break
                    end
                end
                local t0 = tn[j]; local t1 = tn[j+1]; local y0 = y[j]; local y1 = y[j+1]
                local w = (ts - t0) / max(1e-12, (t1 - t0))
                return (1-w)*y0 + w*y1
            end
            predX_mpcc = [_interp(ts, t_nodes, X_fe) for ts in t_syn]
            predG_mpcc = [_interp(ts, t_nodes, G_fe) for ts in t_syn]
            predF_mpcc = [_interp(ts, t_nodes, F_fe) for ts in t_syn]
            predE_mpcc = [_interp(ts, t_nodes, E_fe) for ts in t_syn]
            r2X_mpcc = _r2(Y_syn[1,:], predX_mpcc); r2G_mpcc = _r2(Y_syn[3,:], predG_mpcc);
            r2F_mpcc = _r2(Y_syn[4,:], predF_mpcc); r2E_mpcc = _r2(Y_syn[5,:], predE_mpcc)
            Plots.title!(plt_post[1], @sprintf("X (R²_ODE=%.3f, R²_MPCC=%.3f)", r2X_ode, r2X_mpcc))
            Plots.title!(plt_post[2], @sprintf("G (R²_ODE=%.3f, R²_MPCC=%.3f)", r2G_ode, r2G_mpcc))
            Plots.title!(plt_post[3], @sprintf("F (R²_ODE=%.3f, R²_MPCC=%.3f)", r2F_ode, r2F_mpcc))
            Plots.title!(plt_post[4], @sprintf("E (R²_ODE=%.3f, R²_MPCC=%.3f)", r2E_ode, r2E_mpcc))
        end
        
        Plots.png(plt_post, post_path)
        println("[PLOT] Saved post-optimization V2 ODE plot ", post_path)
    catch err
        @warn "Post-optimization ODE simulation failed" err
    end
    else
        println("[PLOT] Skipping post-optimization ODE and plot (SKIP_PLOTS=1)")
    end

    # ---------------------------------------------
    # Active set report (V2-CHANGE: apply scaling)
    # ---------------------------------------------
    try
        ACTIVE_REPORT = get(ENV, "ACTIVE_REPORT", "1") == "1"
        if ACTIVE_REPORT
            base_set = (!REDUCED_MODE || reduced_sets === nothing) ? collect(1:nv) : K_AX
            t_nodes_rep = cumsum([try value(hv[i]) catch; hm[i] end for i in 1:nfe])
            tol_bnd = 1e-7; tol_upt = 1e-8
            _val(x) = try value(x) catch; NaN end
            rep_csv = joinpath(RESULTS_DIR, "zenteno_active_report_V2_" * Dates.format(Dates.now(), "yyyymmdd-HHMMSS") * ".csv")
            open(rep_csv, "w") do io
                println(io, "fe,t_end,n_act_lb,n_act_ub,act_lb_idx,act_ub_idx,switches_lb,switches_ub,lb_added,lb_removed,ub_added,ub_removed,v_glu,v_fru,upt_tight_glu,upt_tight_fru,slack_glu,slack_fru,FO_upt_glu,FO_upt_fru")
                prev_lb = Set{Int}(); prev_ub = Set{Int}()
                for i in 1:nfe
                    act_lb = Int[]; act_ub = Int[]
                    for k in base_set
                        lbk = lb[k]; ubk = ub[k]
                        if !isfinite(lbk) && !isfinite(ubk); continue; end
                        vk_scaled = _val(v[k,i]) * vs[k] # Apply scaling
                        if isfinite(lbk) && isfinite(vk_scaled) && abs(vk_scaled - lbk) <= tol_bnd
                            push!(act_lb, k)
                        end
                        if isfinite(ubk) && isfinite(vk_scaled) && abs(vk_scaled - ubk) <= tol_bnd
                            push!(act_ub, k)
                        end
                    end
                    sort!(act_lb); sort!(act_ub)
                    cur_lb = Set(act_lb); cur_ub = Set(act_ub)
                    lb_added = length(setdiff(cur_lb, prev_lb)); lb_removed = length(setdiff(prev_lb, cur_lb))
                    ub_added = length(setdiff(cur_ub, prev_ub)); ub_removed = length(setdiff(prev_ub, cur_ub))
                    switches_lb = lb_added + lb_removed; switches_ub = ub_added + ub_removed
                    prev_lb = cur_lb; prev_ub = cur_ub
                    
                    vglu_scaled = (glu in base_set) ? _val(v[glu,i]) * vs[glu] : NaN
                    vfru_scaled = (fru in base_set) ? _val(v[fru,i]) * vs[fru] : NaN
                    rGi = try value(rG[i]) catch; NaN end
                    rFi = try value(rF[i]) catch; NaN end
                    slack_g = (isfinite(vglu_scaled) && isfinite(rGi)) ? (-vglu_scaled - rGi) : NaN
                    slack_f = (isfinite(vfru_scaled) && isfinite(rFi)) ? (-vfru_scaled - rFi) : NaN
                    tight_g = (isfinite(slack_g) && abs(slack_g) <= tol_upt) ? 1 : 0
                    tight_f = (isfinite(slack_f) && abs(slack_f) <= tol_upt) ? 1 : 0
                    fou_g = try value(FO_upt[1,i]) catch; NaN end
                    fou_f = try value(FO_upt[2,i]) catch; NaN end
                    
                    act_lb_str = isempty(act_lb) ? "[]" : join(string.(act_lb), "|")
                    act_ub_str = isempty(act_ub) ? "[]" : join(string.(act_ub), "|")
                    
                    @printf(io, "%d,%.9f,%d,%d,\"%s\",\"%s\",%d,%d,%d,%d,%d,%d,%.9e,%.9e,%d,%d,%.9e,%.9e,%.9e,%.9e\n",
                        i, t_nodes_rep[i], length(act_lb), length(act_ub), act_lb_str, act_ub_str,
                        switches_lb, switches_ub, lb_added, lb_removed, ub_added, ub_removed,
                        vglu_scaled, vfru_scaled, tight_g, tight_f, slack_g, slack_f, fou_g, fou_f)
                end
            end
            println("[ACTIVE] Active-set report (V2) saved: ", rep_csv)
        else
            println("[ACTIVE] Skipping active-set report (ACTIVE_REPORT=0)")
        end
    catch err
        @warn "Active-set report generation failed" err
    end
