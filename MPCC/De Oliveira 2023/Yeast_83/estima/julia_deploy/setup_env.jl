using Pkg
Pkg.activate(".")
# Ensure required packages for MPCC_Zenteno estimation workflow and synthetic data generation.
packages = [
	"JuMP", "Ipopt", "FileIO", "JLD2", "DifferentialEquations", "Plots", "XLSX", "Distributions"
]
for pkg in packages
	try
		Pkg.add(pkg)
	catch err
		@warn "Failed adding package" pkg err
	end
end
Pkg.precompile()
Pkg.status()
println("[SETUP] Environment ready. Headless plotting: set ENV[\"GKSwstype\"]=\"nul\" if needed.")
