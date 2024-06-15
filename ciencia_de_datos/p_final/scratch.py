from typing import Tuple
import numpy as np
from carga_datos import *


def particion_entr_prueba(x: np.ndarray, y: np.ndarray, test: float = 0.20) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


class NaiveBayes:

    def __init__(self, k: float = 1):
        self.k = k
        # probabilidades a priori
        self.__priors: np.ndarray
        self.__X: np.ndarray
        self.__y: np.ndarray
        # Conteo de cada clase
        self.__class_count: np.ndarray
        # Las clases
        self.__classes: np.ndarray
        # Para laplace, cuantos valores unicos puede tomar cada feature
        self.__unique_feature_count: np.ndarray

    def entrena(self, X: np.ndarray, y: np.ndarray):
        """
        Este metodo calcula las probabilidades a priori y algunas cosas utiles para la clasificación
        :param X: Train set
        :param y: Target values
        :return: None
        """
        self.__X = X
        self.__y = y

        self.__classes, self.__class_count = np.unique(y, return_counts=True)
        self.__priors = self.__class_count / sum(self.__class_count)
        # Aplana un array 2d y el valor en cada posicion corresponde a la cantidad de valores unicos de cada columna
        self.__unique_feature_count = np.apply_along_axis(lambda col: len(np.unique(col)), axis=0, arr=X)

    def clasifica_prob(self, ejemplo: np.ndarray) -> dict:
        """
        Recibe un ejemplo y devuelve las probabilidades de cada clase
        :param ejemplo: array numpy con los valores de cada atributo
        :return: diccionario con la probabilidad de cada clase.
        """
        result = {}
        proportional_probs = self.calculate_proportional_probabilities(ejemplo)

        # Exponentiate to obtain probabilities
        prob_props = np.exp(proportional_probs)

        # Normalize probabilities to sum to 1 (optional step)
        prob_props /= np.sum(prob_props)

        for idx, current_class in enumerate(self.__classes):
            result[current_class] = prob_props[idx]

        return result

    def calculate_proportional_probabilities(self, ejemplo:np.ndarray) -> np.ndarray:
        """
        Metodo que aplica naive bayes al ejemplo con suavizado de la place y log prob.
        :param ejemplo: array a clasificar
        :return: un array de valores proporcionales a las probabilidades
        """

        # Inicializar valores
        num_features = self.__X.shape[1]

        proportional_probs = np.zeros(len(self.__classes))
        for idx, current_class in enumerate(self.__classes):
            # Los valores del conjunto de entrenamiento que corresponden a la clase en cuestion
            X_given_class = self.__X[self.__y == current_class]

            # cuenta para cada features cuantos hay del valor del ejemplo
            counts = np.array([np.sum(X_given_class[:, i] == ejemplo[i]) for i in range(num_features)])

            # divisor del suavizado laplace
            numerator = counts + self.k
            denom = self.__class_count[idx] + (self.k * self.__unique_feature_count)

            # calculo del valor de las probabilidades condicionales
            conditional_prob = numerator / denom

            # Calcular la probabilidad proporcional
            # proportional_probs[idx] = self.__priors[idx] * np.prod(conditional_prob)
            proportional_probs[idx] = np.log(self.__priors[idx]) + np.sum(np.log(conditional_prob))
        return proportional_probs

    def clasifica(self, ejemplo: np.ndarray):
        prob_dict = self.clasifica_prob(ejemplo)
        return max(prob_dict, key=prob_dict.get)


nb_tenis = NaiveBayes(k=0.5)
nb_tenis.entrena(X_tenis, y_tenis)
ej_tenis = np.array(['Soleado', 'Baja', 'Alta', 'Fuerte'])
print(nb_tenis.clasifica_prob(ej_tenis))
print(nb_tenis.clasifica(ej_tenis))
