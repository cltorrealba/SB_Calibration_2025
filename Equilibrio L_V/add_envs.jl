using Pkg
Pkg.activate(@__DIR__)
Pkg.add(["Clapeyron", "DifferentialEquations", "Plots", "ComponentArrays"])
Pkg.precompile()
