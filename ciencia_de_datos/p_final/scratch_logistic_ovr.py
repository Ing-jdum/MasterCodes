import numpy as np

from ciencia_de_datos.p_final.scratch import particion_entr_prueba, rendimiento, ClasificadorNoEntrenado
from ciencia_de_datos.p_final.scratch_logistic import RegresionLogisticaMiniBatch
from carga_datos import *


class RL_OvR:
    def __init__(self, rate=0.1, rate_decay=False, batch_tam=64):
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.classifiers = []
        self.classes = None

    def entrena(self, X, y, n_epochs=100, salida_epoch=False):
        self.classes = np.unique(y)
        self.classifiers = []

        for cls in self.classes:
            classifier = RegresionLogisticaMiniBatch(rate=self.rate, rate_decay=self.rate_decay, n_epochs=n_epochs,
                                                     batch_tam=self.batch_tam)
            y_binary = np.where(y == cls, 1, 0)
            classifier.entrena(X, y_binary, n_epochs=n_epochs, salida_epoch=salida_epoch)
            self.classifiers.append(classifier)

    def clasifica_prob(self, ejemplos: np.ndarray) -> np.ndarray:
        if not self.classifiers:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado.")

        # Ensure the input is two-dimensional
        if ejemplos.ndim == 1:
            ejemplos = ejemplos.reshape(1, -1)

        probas = np.zeros((ejemplos.shape[0], len(self.classes)))

        for i, classifier in enumerate(self.classifiers):
            print(classifier.w)
            probas[:, i] = classifier.clasifica_prob(ejemplos)
        return probas

    def clasifica(self, ejemplos: np.ndarray) -> np.ndarray:
        if not self.classifiers:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado.")

        probas = self.clasifica_prob(ejemplos)
        return np.argmax(probas, axis=1)


Xe_iris, Xp_iris, ye_iris, yp_iris = particion_entr_prueba(X_iris, y_iris)

rl_iris_ovr = RL_OvR(rate=0.001, batch_tam=8)
# print(Xe_iris, Xp_iris, ye_iris, yp_iris)
rl_iris_ovr.entrena(Xe_iris, ye_iris, n_epochs=500)
print("hola", rl_iris_ovr.clasifica(Xe_iris))

# print(rendimiento(rl_iris_ovr, Xe_iris, ye_iris))

