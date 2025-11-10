using Pkg
Pkg.activate(".")
# Add ODE and plotting packages if missing
for pkg in ["DifferentialEquations", "Plots"]
    if !(pkg in keys(Pkg.project().dependencies))
        println("[INSTALL] Adding package: ", pkg)
        Pkg.add(pkg)
    else
        println("[SKIP] Package already present: ", pkg)
    end
end
Pkg.precompile()
Pkg.status()
