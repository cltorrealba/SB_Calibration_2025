using Pkg
Pkg.activate(".")
Pkg.add(["JuMP","Ipopt","FileIO","JLD2"]) 
Pkg.precompile()
Pkg.status()
