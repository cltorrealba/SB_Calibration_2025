# Model Changes

Date: 2025-10-09

This document summarizes model and objective changes applied after reviewing recent calibration results (`mats/test_1/pbest_checkpoint.json`).

## 1) Sugar-depletion timing penalty in objective

Problem: Optimizer tended to consume sugars too early, dropping G+F to ~0 before the experimental end, especially due to slow final kinetics.

Change: The objective now adds an optional penalty that compares the simulated time at which total sugar $S(t)=G(t)+F(t)$ falls below a threshold vs. the target depletion time inferred from measurements.

- New parameters (CLI and internal):
  - `--sugar-penalty` (float, default 0): penalty weight.
  - `--sugar-threshold` (float, default 1.0 g/L): threshold for "all consumed".
- Implementation details:
  - In `objective.sse_for_experiments_real`, we compute `t_zero_sim` (first simulated time where S<=threshold). The target time is estimated from `SugarTotal_exp` if present, else from `Glucose+Fructose`, otherwise defaults to the last measured time.
  - The penalty term is `( (t_target - t_zero_sim) / max(1, t_target) )^2 * sugar_penalty`.
  - This encourages sugar depletion to align with the end of fermentation.

## 2) Lag-phase mechanism (temperature-dependent)

Problem: A clear initial latency (lag) exists in cold fermentations, not captured by the base kinetics.

Change: Introduced an optional lag-phase multiplier that scales all reaction rates during initial hours; colder temperatures increase lag duration. The lag timescale can now be estimated as part of the parameter vector.

- New parameters (CLI and simulator):
  - `--lag-mode` (none|exp|logistic, default none)
  - `--lag-tau-h` (float, default 12.0): base lag timescale at 20°C
  - `--lag-sensitivity` (float, default 0.06): temperature sensitivity used as `tau = lag_tau_h * exp(lag_sensitivity*(20 - Tc))`
  - `--lag-floor` (float, default 0.0): minimum multiplier to avoid total shutdown
- Implementation details:
  - In `model/zenteno.simulate_on_grid`, a `_lag_multiplier(t)` is computed from the selected mode.
  - The ODE RHS and Jacobian are scaled by this multiplier during the early phase; as `m(t)→1`, dynamics recover the original kinetics.
  - Estimation: when calibrating, the parameter vector now includes a 15th parameter `lag_tau_h` (hours @ 20°C). Bounds default to `[2.0, 96.0]`. Older Excel p0 (14 params) are automatically extended with a default value (12.0 h) for calibration.

## 3) YAN offset correction

Problem: Systematic offset suspected in YAN measurements due to analyzer calibration.

Change: Apply an offset of 20 mg/L subtracted from all YAN measured values, floored at 0.

- New parameter: `--yan-offset-mgl` (float, default 20.0)
- Implementation details:
  - In `objective.sse_for_experiments_real`, we subtract the offset from `YAN` (mg/L), clamp to 0, then convert to g/L for loss calculation.

## Backward compatibility and defaults

- All new behaviors are optional or have defaults matching previous behavior: sugar penalty is 0 (off), lag-mode is `none`, and YAN offset defaults to 20 mg/L per request (can be set to 0 to disable).
- Base ordering of the original 14 kinetic/yield parameters is preserved. New lag parameters are appended at the end: first `lag_tau_h` (15th), then `lag_sensitivity` (16th) when enabled/estimated. Units of existing parameters remain unchanged.

## Notes and Future Considerations

- The lag-phase implementation is deliberately simple and non-invasive; if needed, we can restrict the lag scaling to growth-associated terms only (mu, betaG/F) rather than all rates.
- The sugar-depletion penalty currently uses a fixed threshold; consider making it assay-specific (e.g., from density) if needed.
- The YAN offset is now also applied (optional) in preview plots for consistency with the objective; raw vs. corrected overlays can be added later if comparative visualization is desired.

---

### Update (2025-10-10): Estimable lag_sensitivity (16th parameter) & preview enhancements

Rationale: After introducing `lag_tau_h` as a 15th parameter, residual misfit in early cold-fermentation phases suggested the need to differentiate base lag duration from its temperature sensitivity. We therefore exposed `lag_sensitivity` as a separate estimable parameter.

Key changes:

1. Parameter vector extension
  - Original core parameters: 0..13 (14 total, unchanged order)
  - 15th parameter (`index 14`): `lag_tau_h` (hours @ 20°C), bounds `[2.0, 96.0]`
  - 16th parameter (`index 15`): `lag_sensitivity`, bounds `[0.0, 0.25]`, default `0.06`
  - Auto-extension logic: if a loaded checkpoint / p0 has length 14 → append default `lag_tau_h`; if length 15 → append default `lag_sensitivity`.

2. CLI additions & behavior
  - `--lag-sensitivity` flag added to calibration CLI for fixed runs (when not estimating) or for p0 seeding.
  - Bounds arrays extended automatically when estimation of sensitivity is active.
  - Preview CLI: `--ignore-lag-param` allows simulating legacy (no-lag) behavior even if checkpoint includes lag parameters (useful for ablation comparisons).
  - Environment overrides (`SB_LAG_MODE`, `SB_LAG_TAU_H`, `SB_LAG_SENS`, `SB_LAG_FLOOR`) respected by the preview tool for quick exploratory tweaking without altering checkpoints.

3. Objective / simulator interplay
  - Simulator interprets both lag parameters when length >= 16; if only `lag_tau_h` present, a default sensitivity is injected for backward compatibility.
  - Sensitivity applies an exponential temperature modulation: `tau(Tc) = lag_tau_h * exp(lag_sensitivity * (20 - Tc))`.

4. YAN offset visualization
  - Preview plotting layer now optionally applies the same `--yan-offset-mgl` correction before plotting and during `x0` derivation to maintain consistency with calibration objective transformations.

5. Checkpoint compatibility guidance
  - Legacy checkpoints (14 params): seamlessly upgraded to 15 or 16 parameter vectors on load.
  - Intermediate checkpoints (15 params with only `lag_tau_h`): auto-extended to include `lag_sensitivity` with default value unless explicitly fixed by user.
  - To reproduce historical fits precisely, record both the numeric vector and its interpreted parameter names (now emitted in calibration summaries JSON).

6. Recommended practice
  - When first enabling `lag_sensitivity`, consider bounding it narrowly (e.g. `[0.02, 0.12]`) and inspect correlation with `lag_tau_h` (reported by covariance approximations if local polishing is performed) before widening to the global `[0.0, 0.25]` range.

Future considerations:
  - Potential identifiability diagnostics (e.g., computing condition numbers of local Hessian / Fisher approximations) to warn if `lag_tau_h` and `lag_sensitivity` become strongly collinear.
  - Option to selectively apply lag scaling only to growth-associated fluxes rather than all reaction rates, reducing parameter cross-talk with yield terms.

