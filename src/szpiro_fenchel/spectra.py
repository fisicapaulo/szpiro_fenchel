import numpy as np

def laplacian_cycle(n: int):
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, (i-1) % n] = 1.0
        A[i, (i+1) % n] = 1.0
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    return L

def renorm_In(L: np.ndarray, n: int):
    return (n**2) * L

def spectrum_In(n: int):
    """
    lambda_k = 4 sin^2(pi k / n), k=0,...,n-1
    Retorna (lambdas_ordenados, lambda_min_pos, lambda_max).
    """
    k = np.arange(n)
    lambdas = 4.0 * (np.sin(np.pi * k / n))**2
    lambdas.sort()
    lam_min_pos = lambdas[1] if n >= 3 else (lambdas[-1] if n == 2 else 0.0)
    lam_max = lambdas[-1]
    return lambdas, lam_min_pos, lam_max
