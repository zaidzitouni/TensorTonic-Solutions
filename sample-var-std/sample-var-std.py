import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with sample variance and sample standard deviation.
    """
    x = np.asarray(x, dtype=float)

    if x.size < 2:
        raise ValueError("At least two observations are required.")

    centered = x - np.mean(x)

    var = np.sum(centered ** 2) / (x.size - 1)

    std = np.sqrt(var)

    return {
        "variance": float(var),
        "standard_deviation": float(std)
    }