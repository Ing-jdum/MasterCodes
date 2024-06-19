import numpy as np

from carga_datos import *

Xc = np.array([["a", 1, "c", "x"],
               ["b", 2, "c", "y"],
               ["c", 1, "d", "x"],
               ["a", 2, "d", "z"],
               ["c", 1, "e", "y"],
               ["c", 2, "f", "y"]])


def codifica_one_hot(X: np.ndarray) -> np.ndarray:
    result = []
    for i in range(X.shape[1]):
        clases = np.unique(X[:, i])
        for cls in clases:
            arr = np.where(X[:, i] == cls, 1, 0)
            result.append(arr)
    return np.array(result).T

