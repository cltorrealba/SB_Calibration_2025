from dataclasses import dataclass


@dataclass
class CalibrationConfig:
    mode: str = "multistart"  # or 'de'
    n_starts: int = 8
    local_maxiter: int = 200
    patience_starts: int = 6
    patience_evals: int = 2000
    min_improvement_rel: float = 1e-3
    # objective options
    sse_balance: str = "per_assay"  # "per_assay" | "per_point" | "none"
    sse_resample_dt_h: float | None = None
    # solver tolerances
    rtol: float = 1e-6
    atol_x: float = 1e-3
    atol_n: float = 1e-2
    atol_g: float = 1e-2
    atol_f: float = 1e-2
    atol_e: float = 1e-3

    def atol_vec(self):
        return [self.atol_x, self.atol_n, self.atol_g, self.atol_f, self.atol_e]
