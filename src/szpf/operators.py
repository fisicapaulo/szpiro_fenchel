import numpy as np

def metric_M(n, diag=1.0):
    return np.eye(n) * diag

def projection_Pi(M):
    n = M.shape[0]
    one = np.ones((n,1))
    alpha = (M @ one) / (one.T @ M @ one)
    Pi = np.eye(n) - one @ alpha.T
    return Pi

def laplacian_Q_from_adj(A):
    d = np.sum(A, axis=1)
    return np.diag(d) - A  # Laplaciano combinatório

def compress_L(Q, M):
    Pi = projection_Pi(M)
    return Pi @ (-Q) @ Pi  # L efetivo em 1^⊥

def whiten(L, M):
    # M SPD: Lw = M^{-1/2} L M^{-1/2}
    # Para M diagonal: Cholesky é simples
    C = np.linalg.cholesky(M)        # M = C^T C
    Minv_half = np.linalg.inv(C).T   # M^{-1/2} = C^{-T}
    return Minv_half @ L @ Minv_half
