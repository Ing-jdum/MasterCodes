from typing import Tuple
import numpy as np
from carga_datos import *
from ciencia_de_datos.p_final.scratch import ClasificadorNoEntrenado


class NormalizadorNoAjustado(Exception):
    pass


class NormalizadorStandard:
    """
    Implementación de la normalización estándar (standardization) que traslada y escala cada característica
    para que tenga media 0 y desviación típica 1.
    """

    def __init__(self):
        """
        Inicializa el normalizador estándar.

        Attributes:
        - mean_ (numpy.ndarray or None): Media de cada característica. None antes de ajustar.
        - std_ (numpy.ndarray or None): Desviación estándar de cada característica. None antes de ajustar.
        - adjusted (bool): Indica si el normalizador ha sido ajustado o no.
        """
        self.mean_ = None
        self.std_ = None
        self.adjusted = False

    def ajusta(self, X: np.ndarray):
        """
        Calcula las medias y desviaciones estándar de las características de X necesarias para la normalización.

        Parameters:
        - X (numpy.ndarray): Matriz de datos de entrenamiento de forma (n_samples, n_features).

        Returns:
        - None
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.adjusted = True

    def normaliza(self, X: np.ndarray) -> np.ndarray:
        """
        Normaliza las características de X utilizando la normalización estándar.

        Parameters:
        - X (numpy.ndarray): Matriz de datos a normalizar de forma (n_samples, n_features).

        Returns:
        - numpy.ndarray: Matriz de datos normalizados de forma (n_samples, n_features).

        Raises:
        - NormalizadorNoAjustado: Si se llama a normaliza() antes de ajusta().
        """
        if not self.adjusted:
            raise NormalizadorNoAjustado("Debe ajustar el normalizador antes de normalizar los datos.")

        X_norm = (X - self.mean_) / self.std_
        return X_norm


normst_cancer = NormalizadorStandard()
normst_cancer.ajusta(X_cancer)
Xe_cancer_n = normst_cancer.normaliza(X_cancer)
