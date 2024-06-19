from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
import time

from ciencia_de_datos.p_final.scratch import ClasificadorNoEntrenado, rendimiento, particion_entr_prueba
from ciencia_de_datos.p_final.scratch_logistic import RegresionLogisticaMiniBatch
from carga_datos import *
from ciencia_de_datos.p_final.scratch_one import codifica_one_hot


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time for {func.__name__}: {elapsed_time} seconds")
        return result

    return wrapper


class RL_OvR:
    def __init__(self, rate=0.1, rate_decay=False, batch_tam=64):
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.classifiers = []
        self.classes = None

    @timing_decorator
    def entrena(self, X: np.ndarray, y: np.ndarray, n_epochs: int = 100, salida_epoch=False) -> None:
        """
        :param X: Dataset de entrenamiento como numpy array
        :param y:  Clase objetivo
        :param n_epochs: Numero de epchos
        :param salida_epoch: Entrenamiento verbose o no
        :return: Nada
        """
        self.classes = np.unique(y)
        self.classifiers = []

        for cls in self.classes:
            # Esto es facilmente paralalizable usando concurrent futures.
            classifier = RegresionLogisticaMiniBatch(rate=self.rate, rate_decay=self.rate_decay, n_epochs=n_epochs,
                                                     batch_tam=self.batch_tam)
            y_binary = np.where(y == cls, 1, 0)
            classifier.entrena(X, y_binary, n_epochs=n_epochs, salida_epoch=salida_epoch)
            self.classifiers.append(classifier)

    def _entrena_paralelo_auxiliar(self, cls, X: np.ndarray, y: np.ndarray, n_epochs: int = 100,
                                  salida_epoch=False) -> RegresionLogisticaMiniBatch:
        classifier = RegresionLogisticaMiniBatch(rate=self.rate, rate_decay=self.rate_decay, n_epochs=n_epochs,
                                                 batch_tam=self.batch_tam)
        y_binary = np.where(y == cls, 1, 0)
        classifier.entrena(X, y_binary, n_epochs=n_epochs, salida_epoch=salida_epoch)
        return classifier

    @timing_decorator
    def _entrena_paralelo(self, X: np.ndarray, y: np.ndarray, n_epochs: int = 100, salida_epoch=False):
        self.classes = np.unique(y)
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._entrena_paralelo_auxiliar, cls, X, y, n_epochs, salida_epoch) for cls in
                       self.classes]
            self.classifiers = [future.result() for future in futures]

    def clasifica_prob(self, ejemplos: np.ndarray) -> np.ndarray:
        """
        :param ejemplos: array de valores para realizar la predicción
        :return: array de arrays de probabilidad de pertenencer a cada clase para cada ejemplo
        """
        if not self.classifiers:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado.")

        # Ensure the input is two-dimensional
        if ejemplos.ndim == 1:
            ejemplos = ejemplos.reshape(1, -1)

        probas = np.zeros((ejemplos.shape[0], len(self.classes)))

        for i, classifier in enumerate(self.classifiers):
            probas[:, i] = classifier.clasifica_prob(ejemplos)
        return probas

    def clasifica(self, ejemplos: np.ndarray) -> np.ndarray:
        """
        :param ejemplos: array de valores para realizar la predicción
        :return: array de clases de cada uno de los ejemplos
        """
        if not self.classifiers:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado.")

        probas = self.clasifica_prob(ejemplos)
        return self.classes[np.argmax(probas, axis=1)]


#
# Xe_iris, Xp_iris, ye_iris, yp_iris = particion_entr_prueba(X_iris, y_iris)
#
# rl_iris_ovr = RL_OvR(rate=0.001, batch_tam=8)
# # print(Xe_iris, Xp_iris, ye_iris, yp_iris)
# rl_iris_ovr.entrena(Xe_iris, ye_iris, n_epochs=500)
# print(rendimiento(rl_iris_ovr, Xe_iris, ye_iris))
# rl_iris_ovr._entrena_paralelo(Xe_iris, ye_iris, n_epochs=500)
# print(rendimiento(rl_iris_ovr, Xe_iris, ye_iris))

def rendimiento2(clasificador, X: np.ndarray, y: np.ndarray) -> float:
    """
    Performance del clasificador dado un conjunto de datos de test

    :param clasificador: Naive bayes entrenado
    :param X:  Datos de test
    :param y: Clase objetivo de test
    :return: accuracy
    """

    # Make predictions for each example in X
    y_pred_accum = 0
    print(X)
    for idx, x in enumerate(X):
        print(clasificador.clasifica(x), y[idx])
        print(clasificador.clasifica(x) == y[idx])
        pred = clasificador.clasifica(x) == y[idx]
        y_pred_accum += np.sum(pred)

    # Calculate the total number of predictions
    total = len(y)

    # Calculate the accuracy
    accuracy = y_pred_accum / total
    print(y_pred_accum)

    return accuracy


print("==== MEJOR RENDIMIENTO RL_OvR SOBRE CREDITO:")
X_credito_oh = codifica_one_hot(X_credito)
Xe_credito_oh, Xp_credito_oh, ye_credito, yp_credito = particion_entr_prueba(X_credito_oh, y_credito, test=0.3)

RL_CLASIF_CREDITO = RL_OvR(rate=0.1, rate_decay=True,
                           batch_tam=16)  # ATENCIÓN: sustituir aquí por los mejores parámetros tras el ajuste
RL_CLASIF_CREDITO.entrena(Xe_credito_oh, ye_credito, n_epochs=50)  # Aumentar o disminuir los epochs si fuera necesario
print(rendimiento2(RL_CLASIF_CREDITO, Xe_credito_oh, ye_credito))
