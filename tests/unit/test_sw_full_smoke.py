import pandas as pd
from sb_calibration.preprocess import sw_full as swf


def test_process_multiple_from_df_and_infer():
    df = pd.DataFrame({
        "Ensayo": ["SB001", "SB001"],
        "ID Análisis 1": ["Concentration", "Concentration"],
        "Valor 1": [10.0, 20.0],
        "ID Análisis 2": ["Viability", "Viability"],
        "Valor 2": [9.0, 18.0],
    })

    results = swf.process_multiple_from_df(df)
    assert "SB001" in results
    proc = results["SB001"]
    assert "viable_gL_adj" in proc.columns

    inoc = swf.infer_inoculum_gL_from_concentration(10.0)
    assert inoc >= 0.0
