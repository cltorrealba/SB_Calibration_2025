from sb_calibration.cli.calibrate_cli import run_calibration
import os


def test_run_calibration_creates_checkpoint(tmp_path):
    out = tmp_path / "pbest_checkpoint.npz"
    pbest, score = run_calibration(None, out_path=str(out))
    assert out.exists()
    # load and check keys
    import numpy as np
    data = np.load(str(out))
    assert "pbest" in data and "score" in data
