import numpy as np
from szpiro_fenchel.spectra import spectrum_In

def test_min_positive():
    _, lam_min_pos, lam_max = spectrum_In(6)
    assert lam_min_pos > 0
    assert lam_max >= lam_min_pos
