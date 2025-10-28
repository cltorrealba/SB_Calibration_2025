# Export data from data.jld2 (Julia JLD2) to a long CSV usable by the Pyomo script
# Output: data_long.csv with columns: state,fe,cp,value (1-based indices)

using Pkg
# Activate local environment (this folder) and ensure dependencies
Pkg.activate(@__DIR__)
try
    # If the environment is not instantiated, this will fetch deps from Project.toml
    Pkg.instantiate()
catch err
    @warn "Pkg.instantiate() failed, attempting to add dependencies directly" err
    for pkg in ["FileIO", "JLD2", "CSV", "DataFrames"]
        try
            Pkg.add(pkg)
        catch e
            @warn "Failed adding package" pkg error=e
        end
    end
end

using FileIO
using JLD2
using CSV
using DataFrames

function main()
    estima_dir = normpath(joinpath(@__DIR__, ".."))
    jld_path = joinpath(estima_dir, "data.jld2")
    if !isfile(jld_path)
        error("data.jld2 not found at: $(jld_path)")
    end

    data = FileIO.load(jld_path, "data")
    # Expect data to be a 3D array indexed as (state, fe, cp)
    nd = ndims(data)
    nd == 3 || error("Expected 3D array 'data' in data.jld2, got ndims=$(nd)")

    nc, ph, ncp = size(data)
    rows = DataFrame(state=Int[], fe=Int[], cp=Int[], value=Float64[])
    for c in 1:nc
        for i in 1:ph
            for j in 1:ncp
                push!(rows, (c, i, j, Float64(data[c, i, j])))
            end
        end
    end

    out_path = joinpath(@__DIR__, "data_long.csv")
    CSV.write(out_path, rows)
    println("Wrote: ", out_path)
end

main()
