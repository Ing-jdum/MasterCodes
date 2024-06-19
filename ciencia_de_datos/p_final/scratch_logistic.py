from typing import Tuple

import numpy as np
from scipy.special import expit
from carga_datos import *
from ciencia_de_datos.p_final.scratch import particion_entr_prueba, ClasificadorNoEntrenado, rendimiento
from ciencia_de_datos.p_final.scratch_3_on import NormalizadorStandard


class RegresionLogisticaMiniBatch:

    def __init__(self, rate=0.1, rate_decay=False, n_epochs=100, batch_tam=64):
        self.rate = rate
        self.rate_decay = rate_decay
        self.n_epochs = n_epochs
        self.batch_tam = batch_tam
        self.w = None  # Se inicializará en el método entrena
        self.classes = None
        self.cross_entropy_train = []
        self.accuracy_train = []
        self.cross_entropy_val = []
        self.accuracy_val = []
        self.best_epoch = None

    @staticmethod
    def _sigmoid(x):
        """
        Función sigmoide usando expit de scipy.special.
        :param x: Array de entrada.
        :return: Array con valores sigmoide.
        """

        return expit(x)

    @staticmethod
    def _cross_entropy_loss(y_true, y_pred):
        """
        Calcula la pérdida de entropía cruzada.
        :param y_true: Etiquetas verdaderas de los datos.
        :param y_pred: Probabilidades predichas.
        :return: Pérdida de entropía cruzada.
        """

        epsilon = 1e-10  # Para evitar log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip para evitar log(0) y log(1)
        loss = -np.mean(np.where(y_true == 1, np.log(y_pred), np.log(1 - y_pred)))
        return loss

    def entrena(self, X, y, Xv=None, yv=None, n_epochs=None, salida_epoch=False, early_stopping=False, paciencia=3):
        """
        Entrena el clasificador utilizando regresión logística
        con descenso de gradiente mini-batch.

        :param X: Matriz de datos de entrenamiento de forma (n_samples, n_features).
        :param y: Vector de etiquetas de entrenamiento de forma (n_samples,).
        :param Xv: Matriz de datos de validación de forma (n_samples_val, n_features).
        :param yv: Vector de etiquetas de validación de forma (n_samples_val,).
        :param n_epochs: Número máximo de epochs para el entrenamiento. Si es None, usa el valor del constructor.
        :param salida_epoch: Si es True, imprime la entropía cruzada y el rendimiento en cada epoch.
        :param early_stopping: Si es True, activa early stopping basado en la entropía cruzada de validación.
        :param paciencia: Número de epochs sin mejora para detener el entrenamiento si early_stopping es True.
        :return: None
        """

        if n_epochs is None:
            n_epochs = self.n_epochs

        # Inicializar pesos aleatorios
        np.random.seed(0)
        self.w = np.random.randn(X.shape[1])

        # Determinar las clases y su orden
        self.classes = np.unique(y).tolist()

        # Si no hay clases, no se puede entrenar
        if len(self.classes) != 2:
            raise ValueError("Debe haber exactamente dos clases para la regresión logística binaria.")

        # Si las clases no están ordenadas correctamente (negativo, positivo)
        if self.classes[0] != 0 or self.classes[1] != 1:
            raise ValueError("Las clases deben estar ordenadas como [negativo, positivo].")

        # Early stopping incializacion
        if early_stopping:
            mejor_ec_val = float('inf')
            epochs_sin_mejora = 0

        # Entrenamiento
        rate_actual = self.rate
        for epoch in range(n_epochs):
            if self.rate_decay:
                rate_actual = self.rate / (1 + epoch)

            # Shuffle de los datos
            idx_shuffle = np.random.permutation(len(X))
            X_shuffled = X[idx_shuffle]
            y_shuffled = y[idx_shuffle]

            # Mini-batch training
            for i in range(0, len(X), self.batch_tam):
                end = min(i + self.batch_tam, len(X))
                X_batch = X_shuffled[i:end]
                y_batch = y_shuffled[i:end]

                # Calcular gradiente y actualizar pesos
                y_pred = self._sigmoid(np.dot(X_batch, self.w))
                gradient = np.dot(X_batch.T, (y_pred - y_batch)) / len(X_batch)
                self.w -= rate_actual * gradient

            # Calcular métricas de entrenamiento
            accuracy_train, cross_entropy_train = self._calculate_metrics(X, y, salida_epoch, early_stopping)
            self.accuracy_train.append(accuracy_train)

            if cross_entropy_train is not None:
                self.cross_entropy_train.append(cross_entropy_train)

            # Si hay conjunto de validación, calcular métricas de validación
            if Xv is not None and yv is not None:
                accuracy_val, cross_entropy_val = self._calculate_metrics(Xv, yv, salida_epoch, early_stopping)

                # Almacenar métricas de validación
                if cross_entropy_val is not None:
                    self.cross_entropy_val.append(cross_entropy_val)
                self.accuracy_val.append(accuracy_val)

                # Early stopping
                if early_stopping:
                    if cross_entropy_val < mejor_ec_val:
                        mejor_ec_val = cross_entropy_val
                        self.best_epoch = epoch
                        epochs_sin_mejora = 0
                    else:
                        epochs_sin_mejora += 1
                        if epochs_sin_mejora >= paciencia:
                            print("PARADA TEMPRANA")
                            return

            # Imprimir métricas por epoch si salida_epoch es True
            if salida_epoch:
                print(
                    f"Epoch {epoch}, En entrenamiento EC: {cross_entropy_train}, Rendimiento: {accuracy_train} ")
                if Xv is not None and yv is not None:
                    print(
                        f"en validación    EC: {cross_entropy_val}, Rendimiento: {accuracy_val}")

    def _calculate_metrics(self, X: np.ndarray, y:np.ndarray, salida_epoch:bool, early_stopping:bool) -> Tuple[float, float]:
        """
        Función auxiliar para evitar repetir codigo en el calculo de metricas para validacion y entrenamiento
        :param X:
        :param y:
        :param salida_epoch:
        :param early_stopping:
        :return:
        """
        y_pred_val = self.clasifica_prob(X)

        # + Téngase en cuenta que el cálculo de la entropía cruzada no es necesario
        #   para el entrenamiento, aunque si salida_epoch o early_stopping es True,
        #   entonces si es necesario su cálculo. Tenerlo en cuenta para no calcularla
        #   cuando no sea necesario.
        if salida_epoch or early_stopping:
            cross_entropy_val = self._cross_entropy_loss(y, y_pred_val)
        else:
            cross_entropy_val = None

        accuracy_val = rendimiento(self, X, y)
        return accuracy_val, cross_entropy_val

    def clasifica_prob(self, ejemplos: np.ndarray) -> np.ndarray:
        """
        Calcula las probabilidades de pertenecer a la clase positiva para los ejemplos dados.
        :param ejemplos: Array de ejemplos de forma (n_samples, n_features).
        :return: Array de probabilidades de pertenecer a la clase positiva para cada ejemplo.
        """
        if self.w is None:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado.")

        return self._sigmoid(np.dot(ejemplos, self.w))

    def clasifica(self, ejemplos: np.ndarray) -> np.ndarray:
        """
        Clasifica los ejemplos dados en las clases negativas (0) o positivas (1).
        :param ejemplos: Array de ejemplos de forma (n_samples, n_features).
        :return: Array de predicciones de clases para cada ejemplo.
        """

        if self.w is None:
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado.")
        probabilidades = self.clasifica_prob(ejemplos)
        return np.where(probabilidades >= 0.5, self.classes[1], self.classes[0])


# Xev_cancer, Xp_cancer, yev_cancer, yp_cancer = particion_entr_prueba(X_cancer, y_cancer, test=0.2)
# Xe_cancer, Xv_cancer, ye_cancer, yv_cancer = particion_entr_prueba(Xev_cancer, yev_cancer, test=0.2)
# normst_cancer = NormalizadorStandard()
# normst_cancer.ajusta(Xe_cancer)
# Xe_cancer_n = normst_cancer.normaliza(Xe_cancer)
# Xv_cancer_n = normst_cancer.normaliza(Xv_cancer)
# Xp_cancer_n = normst_cancer.normaliza(Xp_cancer)
# lr_cancer = RegresionLogisticaMiniBatch(rate=0.1, rate_decay=True)
# lr_cancer.entrena(Xe_cancer_n, ye_cancer, Xv_cancer, yv_cancer)
# # print(lr_cancer.w.shape)
# # print(Xp_cancer_n[26:27].shape)
# print(lr_cancer.clasifica_prob(Xp_cancer_n[26:27]))
# print(lr_cancer.clasifica_prob(Xp_cancer_n[24:27]))
# print(lr_cancer.clasifica(Xp_cancer_n[24:27]))
# print(rendimiento(lr_cancer,Xe_cancer_n,ye_cancer))
# lr_cancer = RegresionLogisticaMiniBatch(rate=0.1, rate_decay=True)
# lr_cancer.entrena(Xe_cancer_n, ye_cancer, Xv_cancer_n, yv_cancer, salida_epoch=True, early_stopping=True)

# ye_iris = np.array(
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#         , 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
#         , 1, 1, 1, 1, 1, 1])
#
# Xe_iris = np.array([[5.5, 4.2, 1.4, 0.2]
#                        , [5., 3.2, 1.2, 0.2]
#                        , [5., 3.4, 1.6, 0.4]
#                        , [4.9, 3.1, 1.5, 0.2]
#                        , [5.7, 3.8, 1.7, 0.3]
#                        , [5., 3.4, 1.5, 0.2]
#                        , [5.8, 4., 1.2, 0.2]
#                        , [4.8, 3., 1.4, 0.3]
#                        , [5.3, 3.7, 1.5, 0.2]
#                        , [4.7, 3.2, 1.6, 0.2]
#                        , [5.7, 4.4, 1.5, 0.4]
#                        , [4.8, 3.1, 1.6, 0.2]
#                        , [5.2, 4.1, 1.5, 0.1]
#                        , [5.4, 3.9, 1.3, 0.4]
#                        , [4.4, 3.2, 1.3, 0.2]
#                        , [5.4, 3.4, 1.7, 0.2]
#                        , [5., 3.5, 1.6, 0.6]
#                        , [4.4, 2.9, 1.4, 0.2]
#                        , [4.3, 3., 1.1, 0.1]
#                        , [5., 3., 1.6, 0.2]
#                        , [5.4, 3.9, 1.7, 0.4]
#                        , [5.1, 3.5, 1.4, 0.3]
#                        , [5., 3.5, 1.3, 0.3]
#                        , [5., 3.3, 1.4, 0.2]
#                        , [4.9, 3., 1.4, 0.2]
#                        , [4.8, 3., 1.4, 0.1]
#                        , [4.9, 3.6, 1.4, 0.1]
#                        , [4.8, 3.4, 1.9, 0.2]
#                        , [4.6, 3.4, 1.4, 0.3]
#                        , [5.1, 3.3, 1.7, 0.5]
#                        , [5.5, 3.5, 1.3, 0.2]
#                        , [5.1, 3.7, 1.5, 0.4]
#                        , [5.1, 3.8, 1.5, 0.3]
#                        , [4.9, 3.1, 1.5, 0.1]
#                        , [5.1, 3.4, 1.5, 0.2]
#                        , [5.1, 3.8, 1.6, 0.2]
#                        , [4.6, 3.1, 1.5, 0.2]
#                        , [5.1, 3.5, 1.4, 0.2]
#                        , [4.6, 3.2, 1.4, 0.2]
#                        , [5.1, 3.8, 1.9, 0.4]
#                        , [5.7, 2.9, 4.2, 1.3]
#                        , [4.9, 2.4, 3.3, 1., ]
#                        , [5.4, 3., 4.5, 1.5]
#                        , [5.6, 2.5, 3.9, 1.1]
#                        , [5., 2.3, 3.3, 1., ]
#                        , [5.8, 2.7, 3.9, 1.2]
#                        , [6.6, 2.9, 4.6, 1.3]
#                        , [6.3, 2.5, 4.9, 1.5]
#                        , [5.7, 2.6, 3.5, 1., ]
#                        , [5.6, 2.7, 4.2, 1.3]
#                        , [6.8, 2.8, 4.8, 1.4]
#                        , [5.8, 2.6, 4., 1.2]
#                        , [6.4, 2.9, 4.3, 1.3]
#                        , [6.3, 3.3, 4.7, 1.6]
#                        , [6.1, 3., 4.6, 1.4]
#                        , [6.2, 2.9, 4.3, 1.3]
#                        , [5.9, 3.2, 4.8, 1.8]
#                        , [5., 2., 3.5, 1., ]
#                        , [6.6, 3., 4.4, 1.4]
#                        , [6., 2.2, 4., 1., ]
#                        , [6.1, 2.8, 4., 1.3]
#                        , [5.5, 2.3, 4., 1.3]
#                        , [5.6, 2.9, 3.6, 1.3]
#                        , [5.1, 2.5, 3., 1.1]
#                        , [6.7, 3., 5., 1.7]
#                        , [6.2, 2.2, 4.5, 1.5]
#                        , [5.9, 3., 4.2, 1.5]
#                        , [6., 3.4, 4.5, 1.6]
#                        , [5.5, 2.5, 4., 1.3]
#                        , [6.4, 3.2, 4.5, 1.5]
#                        , [5.5, 2.4, 3.7, 1., ]
#                        , [6.5, 2.8, 4.6, 1.5]
#                        , [6.7, 3.1, 4.4, 1.4]
#                        , [5.8, 2.7, 4.1, 1., ]
#                        , [5.5, 2.6, 4.4, 1.2]
#                        , [5.6, 3., 4.1, 1.3]
#                        , [5.7, 2.8, 4.5, 1.3]
#                        , [6.7, 3.1, 4.7, 1.5]
#                        , [5.7, 2.8, 4.1, 1.3]
#                        , [7., 3.2, 4.7, 1.4]])
#
# lr_cancer = RegresionLogisticaMiniBatch(rate=0.1, rate_decay=True)
#
# lr_cancer.entrena(Xe_iris, ye_iris, Xe_iris, ye_iris)
#
# print(lr_cancer.w)
