import numpy as np


def matching_dissimilarity(a, b):
    return np.sum(a != b)
