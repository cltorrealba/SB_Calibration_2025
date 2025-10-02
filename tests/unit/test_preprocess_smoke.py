import pandas as pd

from sb_calibration.preprocess import calibration_preprocess as prep


def test_build_calibration_matrices_smoke():
    # Build a minimal BDD-like table
    df = pd.DataFrame({
        "Ensayo": ["SB001", "SB001", "SB001"],
        "ID Análisis 1": ["Concentration", "Concentration", "Concentration"],
        "Valor 1": [10.0, 20.0, 30.0],
        "ID Análisis 2": ["Viability", "Viability", "Viability"],
        "Valor 2": [9.0, 18.0, 27.0],
    })

    wide = prep.extract_assay(df, "SB001")
    assert not wide.empty
    processed = prep.process_one_assay(wide, "SB001")
    assert "viable_gL_adj" in processed.columns

    results = {"SB001": processed}
    mats = prep.build_calibration_matrices(results)
    assert "SB001" in mats
    mat = mats["SB001"]
    assert "time_h" in mat.columns and "biomass_viable_gL" in mat.columns
