import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Both inputs must be one-dimensional.")

    if a.size == 0 or b.size == 0:
        raise ValueError("Input vectors must be non-empty.")

    if a.size != b.size:
        raise ValueError("Input vectors must have equal lengths.")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    dot_product = np.dot(a, b)

    cosine = dot_product / (norm_a * norm_b)

    return float(cosine)