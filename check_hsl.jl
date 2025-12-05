using HSL_jll

function LIBHSL_isfunctional()
    @ccall libhsl.LIBHSL_isfunctional()::Bool
end

isfunctional = LIBHSL_isfunctional()
println("HSL_jll funcional: $isfunctional")

if isfunctional
    println("✓ Tienes la versión completa de HSL_jll con los solvers reales")
    println("Ruta libhsl: ", HSL_jll.libhsl_path)
else
    println("✗ Tienes la versión dummy de HSL_jll (sin solvers)")
    println("Necesitas descargar e instalar la versión completa de HSL_jll.jl")
end
