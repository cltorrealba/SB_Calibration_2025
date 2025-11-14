# Multistart guide (Yeast_83/estima/julia_deploy)

This pipeline provides a coarse→fine seeded solve and a multistart wrapper over estimable parameters.

## Quick start

- Run a single solve (coarse→fine default):
  - `julia --project=. experiment_pipeline.jl seed180_cf`
- Run multistart with K attempts on parameters in log-space:
  - `MULTISTART=10 EST_PARAMS=mu0,kcat_glc; julia --project=. experiment_pipeline.jl multistart`

## Key env variables

- MULTISTART: integer number of randomized starts (K). If K≤1, runs base mode once.
- MULTISTART_BASE_MODE: base pipeline per start (default: `seed180_cf`).
- EST_PARAMS: comma/semicolon/space separated list of parameter symbols to randomize (log-space).
- EST_STARTS: comma-separated log values applied inside a single run (used internally by multistart; you can reuse the best row).
- TETA_START_<param>: real-space per-parameter overrides; multistart sets these for convenience.
- MULTISTART_EARLY_REL: early-stop relative improvement threshold on SSE between best and next (default: 5e-3 if unset).
- MULTISTART_MAX_NOIMPROVE: stop after this many consecutive non-improving attempts (default: 1 if unset).
- EST_RANGES: per-parameter multiplicative ranges relative to nominal values, format:
  - `EST_RANGES="mu0:0.5,2.0;kcat_glc:0.2,8"` → bounds become [lo×Pnom, hi×Pnom] in real space, mapped to log space for sampling.

## Outputs

- results/zenteno_multistart_summary_*.csv: per-attempt summary with columns
  - start_id, SSE, OBJ, wall_s, starts_log, starts_real, checkpoint
- results/zenteno_multistart_best_starts.txt: best row summary and reuse hints; if available, BEST_CHECKPOINT is copied to `zenteno_multistart_best_checkpoint.jld2`.
- Estimation report: `results/zenteno_estimation_report_*.txt` now includes appended lines with R² diagnostics
  - R2_ODE_{X,G,F,E} and R2_MPCC_{X,G,F,E} (MPCC uses linear interpolation between collocation nodes).

## Notes

- Final plots are enabled by default on the fine stage (`SKIP_PLOTS_FINE=0`). Override with `SKIP_PLOTS_FINE=1` to skip.
- BV hybrid constraints are ON in early homotopy stages by default (uptake-only), OFF on the last stage.
