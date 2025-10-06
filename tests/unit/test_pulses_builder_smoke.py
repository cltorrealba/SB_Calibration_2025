import pandas as pd
from sb_calibration.calibration.pulses import build_pulses_from_chem


def test_build_pulses_from_chem_basic():
    df = pd.DataFrame({
        "Código": ["SB005-A", "SB005-B", "SB006-A", "SB006-B"],
        "YAN": [200.0, 250.0, 100.0, 90.0],
        "time_h": [0.0, 1.0, 0.0, 0.5],
    })
    pulses = build_pulses_from_chem(df)
    assert "SB005" in pulses and "SB006" in pulses
    # SB005: delta 50 mg/L -> 0.05 g/L
    p5 = [p for p in pulses["SB005"] if abs(p[0] - 1.0) < 1e-9]
    assert len(p5) == 1 and abs(p5[0][1] - 0.05) < 1e-9
    # SB006: decrease (100 -> 90) shouldn't create negative pulse; expect 0 at t=0.5
    p6 = [p for p in pulses["SB006"] if abs(p[0] - 0.5) < 1e-9]
    assert len(p6) == 1 and abs(p6[0][1] - 0.0) < 1e-12
