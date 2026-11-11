import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    A = np.asarray(A)

    M, N = A.shape

    result = np.empty((N, M), dtype=A.dtype)

    for i in range(M):
        for j in range(N):
            result[j, i] = A[i, j]

    return result