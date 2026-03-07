import numpy as np
from numpy.linalg import svd

def commutator(A, X):
    return A @ X - X @ A

def commutant_dimension(Delta, Omega):
    n = Delta.shape[0]

    # unknown matrix X flattened
    # Build linear system M vec(X) = 0

    def build_constraint(A):
        M = np.zeros((n*n, n*n))
        for i in range(n):
            for j in range(n):
                E = np.zeros((n,n))
                E[i,j] = 1.0
                C = commutator(A, E)
                M[:, i*n + j] = C.flatten()
        return M

    M1 = build_constraint(Delta)
    M2 = build_constraint(Omega)

    M = np.vstack([M1, M2])

    # SVD to find nullspace
    u, s, vh = svd(M)
    tol = 1e-8
    null_dim = np.sum(s < tol)
    return null_dim