using Pkg
Pkg.activate(".")
# Ensure required packages for MPCC_Zenteno estimation workflow and synthetic data generation.
core_packages = [
	"JuMP", "Ipopt", "FileIO", "JLD2", "DifferentialEquations", "Plots", "XLSX", "Distributions", "Glob"
]
# LP solvers for pFBA/FVA
lp_packages = ["HiGHS", "GLPK"]

println("[SETUP] Adding core packages…")
for pkg in core_packages
	try
		Pkg.add(pkg)
	catch err
		@warn "Failed adding package" pkg err
	end
end

println("[SETUP] Adding LP solvers (HiGHS/GLPK) for FVA…")
for pkg in lp_packages
	try
		Pkg.add(pkg)
	catch err
		@warn "Failed adding LP solver" pkg err
	end
end

println("[SETUP] Precompiling… this can take a few minutes on first run (including Glob)")
Pkg.precompile()
Pkg.status()
println("[SETUP] Environment ready. Headless plotting: set ENV[\"GKSwstype\"]=\"nul\" if needed.")
