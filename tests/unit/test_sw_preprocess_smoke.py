import pandas as pd
from sb_calibration.preprocess import sw_preprocess as swp
from sb_calibration.preprocess import calibration_preprocess as cp


def test_sw_preprocess_imports_and_basic_flow():
    # Create a minimal BDD-like DataFrame
    df = pd.DataFrame({
        "Ensayo": ["SB001", "SB001"],
        "ID Análisis 1": ["Concentration", "Concentration"],
        "Valor 1": [12.0, 14.0],
        "ID Análisis 2": ["Viability", "Viability"],
        "Valor 2": [11.0, 13.0],
    })

    # Use the core calibration_preprocess functions directly (no file IO)
    wide = cp.extract_assay(df, "SB001")
    assert not wide.empty
    processed = cp.process_one_assay(wide, "SB001")
    assert "viable_gL_adj" in processed.columns

    # Verify compatibility wrapper is importable and exposes expected callables
    assert callable(swp.process_multiple)
    assert callable(swp.build_and_export_matrices)
