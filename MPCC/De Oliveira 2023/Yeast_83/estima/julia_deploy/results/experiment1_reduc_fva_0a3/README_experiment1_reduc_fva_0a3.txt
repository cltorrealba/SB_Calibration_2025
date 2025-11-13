experiment1_reduc_fva_0a3 (HiGHS-based FVA redo)

Scope
- Regenerate reduced sets using pFBA + FVA with HiGHS
  - PFBA_EPS=1e-7, FVA_ENABLE=1, FVA_FORCE=0
- Rerun homotopy seed (180s/180s/360s)
- Rerun seeded full model (360s)
- Refresh comparison against baseline

Key artifacts (timestamps 2025-11-13)
- Baseline: zenteno_metrics_baseline_20251113-092518.txt
- Homotopy S3: zenteno_metrics_hom_s3_20251113-092255.txt
- Seeded full: zenteno_metrics_seeded_20251113-093214.txt
- Reports/plots: zenteno_estimation_report_20251113-093211.txt, zenteno_post_ode_vs_data_mpcc_20251113-093214.png
- Comparison: zenteno_comparison_20251113-093421.txt
- Reduced sets: reduced_sets.jld2

Headline metrics
- Baseline OBJ=2.548405e+05 (SSE0=2.548405e+05, PEN0=9.429600e-02, REG0=1.644020e-01)
- Hom_S3 obj=3.6320665096e+05, comp_max=4.150779e+01, SSE=2.548405e+05, PEN=5.418308e+05
- Seeded OBJ=5.655221e+05 (SSE=2.563243e+05, PEN=1.545989e+06, REG=8.712845e-02), comp_max=4.026590e+01
- Solver status (both runs): TIME_LIMIT with primal NEARLY_FEASIBLE_POINT (as budgeted)

Notes
- LP backend: HiGHS (via setup_env.jl); FVA sets were regenerated before seeding.
- Warm start from homotopy handoff was used for the seeded run.
- Plots saved pre/post ODE vs data for both homotopy S3 and seeded runs.
- See comparison txt for detailed deltas vs baseline.

Folder contents curated for archival and reproducibility.
