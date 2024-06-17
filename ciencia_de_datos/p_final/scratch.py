from typing import Tuple
import numpy as np
from carga_datos import *


def particion_entr_prueba(x: np.ndarray, y: np.ndarray, test: float = 0.20) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estratifica en train y test

    :param x: Training data
    :param y:  Target Data
    :param test: Proporcion de test del total
    :return: X_train, y_train, X_test, y_test
    """
    np.random.seed(0)
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


class ClasificadorNoEntrenado(Exception): pass


class NaiveBayes:

    def __init__(self, k: float = 1):
        self.k = k
        # probabilidades a priori
        self.priors: np.ndarray = None
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        # Conteo de cada clase
        self.class_count: np.ndarray = None
        # Las clases
        self.classes: np.ndarray = None
        # Para laplace, cuantos valores unicos puede tomar cada feature
        self.unique_feature_count: np.ndarray = None

    def entrena(self, X: np.ndarray, y: np.ndarray):
        """
        Este metodo calcula las probabilidades a priori y algunas cosas utiles para la clasificación
        :param X: Train set
        :param y: Target values
        :return: None
        """
        self.X = X
        self.y = y

        self.classes, self.class_count = np.unique(y, return_counts=True)
        self.priors = self.class_count / sum(self.class_count)
        # Aplana un array 2d y el valor en cada posicion corresponde a la cantidad de valores unicos de cada columna
        self.unique_feature_count = np.apply_along_axis(lambda col: len(np.unique(col)), axis=0, arr=X)

    def clasifica_prob(self, ejemplo: np.ndarray) -> dict:
        """
        Recibe un ejemplo y devuelve las probabilidades de cada clase
        :param ejemplo: array numpy con los valores de cada atributo
        :return: diccionario con la probabilidad de cada clase.
        """

        if self.priors is None:
            raise ClasificadorNoEntrenado

        result = {}
        proportional_probs = self.calculate_proportional_probabilities(ejemplo)

        # Exponentiate to obtain probabilities
        prob_props = np.exp(proportional_probs)

        # Normalize probabilities to sum to 1 (optional step)
        prob_props /= np.sum(prob_props)

        for idx, current_class in enumerate(self.classes):
            result[current_class] = prob_props[idx]

        return result

    def calculate_proportional_probabilities(self, ejemplo: np.ndarray) -> np.ndarray:
        """
        Metodo que aplica naive bayes al ejemplo con suavizado de la place y log prob.
        :param ejemplo: array a clasificar
        :return: un array de valores proporcionales a las probabilidades
        """

        # Inicializar valores
        num_features = self.X.shape[1]

        proportional_probs = np.zeros(len(self.classes))
        for idx, current_class in enumerate(self.classes):
            # Los valores del conjunto de entrenamiento que corresponden a la clase en cuestion
            X_given_class = self.X[self.y == current_class]

            # cuenta para cada features cuantos hay del valor del ejemplo
            counts = np.array([np.sum(X_given_class[:, i] == ejemplo[i]) for i in range(num_features)])

            # divisor del suavizado laplace
            numerator = counts + self.k
            denom = self.class_count[idx] + (self.k * self.unique_feature_count)

            # calculo del valor de las probabilidades condicionales
            conditional_prob = numerator / denom

            # Calcular la probabilidad proporcional
            # proportional_probs[idx] = self.__priors[idx] * np.prod(conditional_prob)
            proportional_probs[idx] = np.log(self.priors[idx]) + np.sum(np.log(conditional_prob))
        return proportional_probs

    def clasifica(self, ejemplo: np.ndarray):
        if self.priors is None:
            raise ClasificadorNoEntrenado
        prob_dict = self.clasifica_prob(ejemplo)
        return max(prob_dict, key=prob_dict.get)


def rendimiento(clasificador, X: np.ndarray, y: np.ndarray) -> float:
    """
    Performance del clasificador dado un conjunto de datos de test

    :param clasificador: Naive bayes entrenado
    :param X:  Datos de test
    :param y: Clase objetivo de test
    :return: accuracy
    """

    # Make predictions for each example in X
    y_pred_accum = 0
    for idx, x in enumerate(X):
        pred = clasificador.clasifica(x) == y[idx]
        y_pred_accum += pred

    # Calculate the total number of predictions
    total = len(y)

    # Calculate the accuracy
    accuracy = y_pred_accum / total

    return accuracy


# nb_tenis = NaiveBayes(k=0.5)
# nb_tenis.entrena(X_tenis, y_tenis)
# ej_tenis = np.array(['Soleado', 'Baja', 'Alta', 'Fuerte'])
# print(nb_tenis.clasifica_prob(ej_tenis))
# print(nb_tenis.clasifica(ej_tenis))
#
# print("rendimiento", rendimiento(nb_tenis, X_tenis, y_tenis))
