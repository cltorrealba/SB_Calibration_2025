using Clapeyron
params = getparams(["water"],["properties/critical.csv"])
println(keys(params))
println(params["Mw"].values)
