import numpy as np

from numpy.typing import NDArray


def build_matmul_tensor(n: int, m: int, p: int) -> NDArray[np.float64]:
    """
    Returns the tensor for (n x m, m x p) matrix multiplication.

    Args:
        n: int
        m: int
        p: int

    Returns:
        The 3-dimensional tensor of shape (nm, mp, np).
    """
    A = np.zeros((n * m, m * p, n * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A[m * i + j, p * j + k, i * p + k] = 1
    return A
