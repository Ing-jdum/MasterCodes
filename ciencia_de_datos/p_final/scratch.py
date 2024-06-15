from typing import Tuple
import numpy as np


def particion_entr_prueba(x: np.ndarray, y: np.ndarray, test: float = 0.20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    A function to obtain a stratified train/test split.

    :param x: Training data
    :param y:  Target Data
    :param test: Test cluster proportion of the total
    :return: X_train, y_train, X_test, y_test
    """
    # obtener las particiones de estratificación
    unique_classes, counts = np.unique(y, return_counts=True)

    # calcular la proporción de cada clase para test
    test_counts = np.floor(counts * test).astype(int)

    train_indices, test_indices = [], []
    for class_label, test_count in zip(unique_classes, test_counts):
        # Toma los indices de la clase en cuestion
        class_indices = np.where(y == class_label)[0]
        np.random.shuffle(class_indices)
        # obten los indices hasta la cantidad necesaria para test
        test_indices.extend(class_indices[:test_count])
        # los otros ponlo en train
        train_indices.extend(class_indices[test_count:])

    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]


