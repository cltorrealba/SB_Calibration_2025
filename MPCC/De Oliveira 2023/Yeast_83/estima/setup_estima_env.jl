using Pkg
Pkg.activate(@__DIR__)
Pkg.add(["JuMP","Ipopt","TickTock","Plots","FileIO","JLD2","DifferentialEquations"])
Pkg.precompile()
