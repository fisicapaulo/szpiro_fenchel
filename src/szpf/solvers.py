import numpy as np
from numpy.linalg import svd

def svd_solve(A, b, rcond=1e-10):
    U, s, Vt = svd(A, full_matrices=False)
    tol = rcond * s.max() if s.size else 0.0
    s_inv = np.array([1/x if x > tol else 0.0 for x in s])
    x = Vt.T @ (s_inv * (U.T @ b))
    return x, s

def condition_number_from_s(s, rcond=1e-10):
    if s.size == 0:
        return float('nan')
    s_eff = s[s > rcond * s.max()]
    if s_eff.size == 0:
        return float('inf')
    return float(s_eff.max() / s_eff.min())
