#!/usr/bin/env julia
# build_reduced_sets_basic.jl
#
# Helper para ejecutar el generador de reduced sets (pfba_preprocess_v2.jl)
# directamente desde la carpeta principal de estimación. Esto crea/actualiza
# julia_deploy/results/reduced_sets.jld2, archivo requerido si REDUCED_MODE=1.

const ESTIMA_DIR = @__DIR__
const PFBA_SCRIPT = joinpath(ESTIMA_DIR, "julia_deploy", "pfba_preprocess_v2.jl")

if !isfile(PFBA_SCRIPT)
    error("No se encontró pfba_preprocess_v2.jl en ", PFBA_SCRIPT)
end

println("[REDUCED] Ejecutando generador de reduced sets: ", PFBA_SCRIPT)
include(PFBA_SCRIPT)
