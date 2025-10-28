import pandas as pd
import numpy as np
from sb_calibration.preprocess import sugar_density as sd


def test_fit_density_model_smoke():
    # synthetic data: density = 0.8 + 0.001*x - 1e-6*x^2
    x = np.linspace(0, 200, 20)
    y = 0.8 + 0.001 * x - 1e-6 * x**2
    df = pd.DataFrame({"sugar_pct": x, "density_g_mL": y})
    pipeline, coefs = sd.fit_density_model(df, "sugar_pct", "density_g_mL", degree=3)
    assert pipeline is not None
    assert coefs.shape[0] >= 1
