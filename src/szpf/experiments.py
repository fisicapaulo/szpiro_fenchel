import numpy as np
import networkx as nx
from .operators import metric_M, laplacian_Q_from_adj, compress_L, whiten
from .solvers import svd_solve, condition_number_from_s

def cycle_run(n=10, renorm=True, rcond=1e-10):
    G = nx.cycle_graph(n)
    A = nx.to_numpy_array(G)
    Q = laplacian_Q_from_adj(A)
    M = metric_M(n, diag=1.0)
    L = compress_L(Q, M)
    if renorm:
        L = (n**2) * L
    Lw = whiten(L, M)
    # Exemplo de b em 1^⊥: diferença local
    b = np.zeros(n); b[0]=1; b[1]=-1
    # Resolver via SVD (modo constante é filtrado por tolerância)
    x, s = svd_solve(Lw, b, rcond=rcond)
    kappa = condition_number_from_s(s, rcond=rcond)
    resid = np.linalg.norm(Lw @ x - b) / (np.linalg.norm(b) + 1e-16)
    return dict(n=n, kappa=kappa, resid=resid, svals=s)
