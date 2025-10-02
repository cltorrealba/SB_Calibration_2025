import numpy as np
from src.sb_calibration.model import zenteno


def test_rk4_smoke():
    p = np.ones(14) * 0.5
    x0 = zenteno.DEFAULT_X0.copy()
    temps_c = [20.0, 20.0, 20.0]
    pulses = [(24.0, 0.05)]
    tf = 48.0
    t_proc, t, x, T_profile, Nadd = zenteno.simulate_process_time(p, x0, temps_c, pulses, tf=tf, n=48)
    # Sanity checks
    assert t.shape[0] == 49
    assert x.shape[0] == 49
    assert x.shape[1] == 5
    assert np.isfinite(x).all()
