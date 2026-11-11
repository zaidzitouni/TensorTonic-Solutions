import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Both inputs must be one-dimensional.")

    if x.size == 0 or y.size == 0:
        raise ValueError("Input vectors must be non-empty.")

    if x.size != y.size:
        raise ValueError("Input vectors must have equal lengths.")

    result = np.dot(x, y)

    return float(result)