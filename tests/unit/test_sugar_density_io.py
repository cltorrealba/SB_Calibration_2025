import pandas as pd
import numpy as np
from tempfile import NamedTemporaryFile
from sb_calibration.preprocess import sugar_density as sd


def test_run_from_csv_and_save(tmp_path):
    x = np.linspace(0, 100, 11)
    y = 0.8 + 0.001 * x - 1e-6 * x**2
    df = pd.DataFrame({"sugar_pct": x, "density_g_mL": y})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    coeffs_path = tmp_path / "coeffs.txt"
    pipeline, coefs = sd.run_from_csv(str(csv_path), str(coeffs_path), degree=3)
    assert pipeline is not None
    assert coeffs_path.exists()
    content = coeffs_path.read_text()
    assert len(content.strip().splitlines()) == len(coefs)
