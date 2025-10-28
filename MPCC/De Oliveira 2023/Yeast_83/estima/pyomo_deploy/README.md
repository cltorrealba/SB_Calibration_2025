# Pyomo deployment for Yeast_83/estima (MPCC with IPOPT)

This folder provides a Pyomo+IPOPT implementation that mirrors the Julia/JuMP model in `estima/main.jl` (dFBA + parameter estimation via MPCC with complementarity penalties and Radau collocation).

## Contents

- `run_pyomo_model.py` — builds and solves the Pyomo model, then exports results to `./results`.
- `export_data_to_csv.jl` — helper to export the experimental data from `../data.jld2` to `./data_long.csv` (long format).
- `requirements.txt` — minimal Python dependencies for the Pyomo runner (solver binary not included).
- `results/` — created on first run; contains CSV outputs and FO value.

## 1) Export the experimental data

The Julia model reads `data.jld2`. Pyomo cannot read JLD2, so export once to CSV.

From this folder, run (with Julia available):

```powershell
# PowerShell
julia .\export_data_to_csv.jl
```

This creates `data_long.csv` with columns: `state,fe,cp,value` (1-based indices). The shape must match `(nc=4, ph=12, ncp=3)`.

## 2) Install Python dependencies

Use conda (recommended on Windows) or pip. Examples:

```powershell
# Using conda-forge
conda install -c conda-forge pyomo numpy pandas
# Install IPOPT binary (one of the following)
conda install -c conda-forge ipopt
# or use your existing ipopt.exe and pass its path to the script
```

If using pip:

```powershell
pip install -r .\requirements.txt
# You still need an Ipopt executable on PATH or pass its full path when running.
```

Notes:
- On Windows, Ipopt is typically provided via conda-forge (`ipopt`) or a manually downloaded `ipopt.exe`.
- If `ipopt` is not on PATH, pass its full path as an argument to the script (see below).

## 3) Run the Pyomo model

```powershell
# If ipopt is on PATH
python .\run_pyomo_model.py

# If you need to specify the full path to ipopt.exe
python .\run_pyomo_model.py "C:\\full\\path\\to\\ipopt.exe"
```

The script will:
- Read `..\\..\\S.csv`, `..\\..\\lb.csv`, `..\\..\\ub.csv` (from `Yeast_83` folder)
- Read `data_long.csv` (exported in step 1)
- Build the MPCC with:
  - States (X,G,Z,E) with Radau-3 collocation
  - Stationary FBA per FE with KKT penalty terms
  - Parameter estimation for 5 kinetic params (log-scale bounds)
- Solve with IPOPT
- Write results to `./results`:
  - `c_values.csv` — long format (state,fe,cp,value)
  - `v_values.csv` — long format (rxn,fe,value)
  - `teta_values.csv` — parameters in log/original scales
  - `FO.txt` — SSE residual

## Model parity vs Julia

- Time grid: `nfe=12`, `ncp=3`, horizon `th=22` (h=22/12), `ph=nfe`.
- Indices (1-based) kept identical: objective=`3414`, ethanol=`2630`, glucose=`2588`, oxygen=`2816` (fixed 0), ATP=`3415` (lb=0), xylose=`2592`.
- Uptakes:
  - `vg = exp(t1) * G / (exp(t2)+G)`
  - `vz = exp(t3) * Z / (exp(t4)+Z) * 1/(1 + G/exp(t5))`
- KKT penalty terms reproduce the Julia signs in the objective.

## Troubleshooting

- "Missing experimental data" — run `export_data_to_csv.jl` to generate `data_long.csv`.
- "Cannot find ipopt" — install Ipopt with conda or pass the full path to `ipopt.exe`.
- Memory errors when loading `S.csv` — ensure you’re on 64-bit Python and have enough RAM; `S` is ~ (2666 x 3928).
- Slow solves — turn down `print_level`, or start with fewer FEs for debugging.

## License / Attribution

This is a functional translation of the JuMP model in `estima/main.jl` to Pyomo for the Yeast 8.3 case, intended for internal use in SB_Calibration_2025.
