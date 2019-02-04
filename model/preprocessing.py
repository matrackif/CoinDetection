import numpy as np


def normalize(x_data: np.ndarray):
    tmp = x_data.astype(np.float64)
    tmp /= 255.0  # scale between 0 and 1
    """
    std = np.std(tmp, axis=(0, 1, 2))  # Global RGB std
    mean = np.mean(tmp, axis=(0, 1, 2))  # Global RGB mean
    print('Global RGB mean for dataset', mean)
    print('Global RGB STD for dataset', std)
    return (tmp - mean) / std
    """
    return tmp

