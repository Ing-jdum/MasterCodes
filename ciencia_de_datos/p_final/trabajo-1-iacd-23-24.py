# Inteligencia Artificial para la Ciencia de los Datos
# Implementación de clasificadores 
# Dpto. de C. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================


# --------------------------------------------------------------------------
# Autor(a) del trabajo:
#
# APELLIDOS:
# NOMBRE: 
#
# Segundo componente (si se trata de un grupo):
#
# APELLIDOS:
# NOMBRE:
# ----------------------------------------------------------------------------


# *****************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen, por lo
# que debe realizarse de manera individual o con la pareja del grupo. 
# La discusión y el intercambio de información de carácter general con los 
# compañeros se permite, pero NO AL NIVEL DE CÓDIGO. Igualmente el remitir 
# código de terceros, OBTENIDO A TRAVÉS DE LA RED, 
# DE HERRAMIENTAS DE GENERACIÓN DE CÓDIGO o cualquier otro medio, 
# se considerará plagio. Si tienen dificultades para realizar el ejercicio, 
# consulten con el profesor. En caso de detectarse plagio, supondrá 
# una calificación de cero en la asignatura, para todos los alumnos involucrados. 
# Sin perjuicio de las medidas disciplinarias que se pudieran tomar. 
# *****************************************************************************


# MUY IMPORTANTE: 
# ===============    

# * NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES, MÉTODOS
#   Y ATRIBUTOS QUE SE PIDEN. EN PARTICULAR: NO HACERLO EN UN NOTEBOOK.

# * En este trabajo NO SE PERMITE USAR Scikit Learn. 

# * Se recomienda (y se valora especialmente) el uso eficiente de numpy. Todos 
#   los datasets se suponen dados como arrays de numpy. 

# * Este archivo (con las implementaciones realizadas), ES LO ÚNICO QUE HAY QUE ENTREGAR.
#   AL FINAL DE ESTE ARCHIVO hay una serie de ejemplos a ejecutar que están comentados, y que
#   será lo que se ejecute durante la presentación del trabajo al profesor.
#   En la versión final a entregar, descomentar esos ejemplos del final y no dejar 
#   ninguna otra ejecución de ejemplos. 


import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

import numpy as np
from scipy.special import expit

# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar (casi) todos los conjuntos de datos,
# basta con tener descomprimido el archivo datos-trabajo-1-iacd.tgz (en el mismo sitio
# que este archivo) Y CARGARLOS CON LA SIGUIENTE ORDEN. 

from carga_datos import *


# Como consecuencia de la línea anterior, se habrán cargado los siguientes 
# conjuntos de datos, que pasamos a describir, junto con los nombres de las 
# variables donde se cargan. Todos son arrays de numpy: 


# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.  

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresita (0:republicano o 1:demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos. 

# * Datos sobre concesión de prestamos en una entidad bancaria. En el propio
#   archivo datos/credito.py se describe con más detalle. Se carga en las
#   variables X_credito, y_credito.   


# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer.


# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos del dataset original. 
#   Los textos se han vectorizado usando CountVectorizer de Scikit Learn, con la opción
#   binary=True. Como vocabulario, se han usado las 609 palabras que ocurren
#   más frecuentemente en las distintas críticas. La vectorización binaria
#   convierte cada texto en un vector de 0s y 1s en la que cada componente indica
#   si el correspondiente término del vocabulario ocurre (1) o no ocurre (0)
#   en el texto (ver detalles en el archivo carga_datos.py). Los datos se
#   cargan finalmente en las variables X_train_imdb, X_test_imdb, y_train_imdb,
#   y_test_imdb.    

# Además, en la carpeta datos/digitdata se tiene el siguiente dataset, que
# habrá de ser procesado y cargado:  

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En la carpeta digitdata están todos los datos.
#   Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles). 


# ==================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA 
# ==================================================

# Definir una función 

#           particion_entr_prueba(X,y,test=0.20)

# que recibiendo un conjunto de datos X, y sus correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test. La división ha de ser ALEATORIA y
# ESTRATIFICADA respecto del valor de clasificación. Por supuesto, en el orden 
# en el que los datos y los valores de clasificación respectivos aparecen en
# cada partición debe ser consistente con el orden original en X e y.   
# 

def particion_entr_prueba(X: np.ndarray, y: np.ndarray, test: float = 0.20) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estratifica en train y test

    :param X: Training data
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

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:

# Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:
# print(y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0])


# Las proporciones entre las clases son (aprox) las mismas en los dos conjuntos de
# datos, y la misma que en el total: 267/168=178/112=89/56

# print(np.unique(y_votos,return_counts=True))
#  (array([0, 1]), array([168, 267]))
# >>> np.unique(ye_votos,return_counts=True)
#  (array([0, 1]), array([112, 178]))
# >>> np.unique(yp_votos,return_counts=True)
#  (array([0, 1]), array([56, 89]))

# La división en trozos es aleatoria y, por supuesto, en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con los datos del cáncer, en el que se observa que las proporciones
# entre clases se conservan en la partición. 

# >>> Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)

# >>> np.unique(y_cancer,return_counts=True)
# (array([0, 1]), array([212, 357]))

# >>> np.unique(yev_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yp_cancer,return_counts=True)
# (array([0, 1]), array([42, 71]))    


# Podemos ahora separar Xev_cancer, yev_cancer, en datos para entrenamiento y en 
# datos para validación.

# >>> Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)

# >>> np.unique(ye_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yv_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))


# Otro ejemplo con más de dos clases:

# >>> Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)

# >>> np.unique(y_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([202, 228, 220]))

# >>> np.unique(ye_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([121, 137, 132]))

# >>> np.unique(yp_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([81, 91, 88]))
# ------------------------------------------------------------------


# ========================================================
# EJERCICIO 2: IMPLEMENTACIÓN DEL CLASIFICADOR NAIVE BAYES
# ========================================================

# Se pide implementar el clasificador Naive Bayes, en su versión categórica
# con suavizado y log probabilidades (descrito en el tema 2, diapositivas 22 a
# 34). En concreto:


# ----------------------------------
# 2.1) Implementación de Naive Bayes
# ----------------------------------

# Definir una clase NaiveBayes con la siguiente estructura:

# class NaiveBayes():

#     def __init__(self,k=1):
#                 
#          .....

#     def entrena(self,X,y):

#         ......

#     def clasifica_prob(self,ejemplo):

#         ......

#     def clasifica(self,ejemplo):

#         ......


# * El constructor recibe como argumento la constante k de suavizado (por
#   defecto 1) 
# * Método entrena, recibe como argumentos dos arrays de numpy, X e y, con los
#   datos y los valores de clasificación respectivamente. Tiene como efecto el
#   entrenamiento del modelo sobre los datos que se proporcionan. NOTA: Se valorará
#   que el entrenamiento se haga con un único recorrido del dataset. 
# * Método clasifica_prob: recibe un ejemplo (en forma de array de numpy) y
#   devuelve una distribución de probabilidades (en forma de diccionario) que
#   a cada clase le asigna la probabilidad que el modelo predice de que el
#   ejemplo pertenezca a esa clase. 
# * Método clasifica: recibe un ejemplo (en forma de array de numpy) y
#   devuelve la clase que el modelo predice para ese ejemplo.   

# Si se llama a los métodos de clasificación antes de entrenar el modelo, se
# debe devolver (con raise) una excepción:

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

        # Estos pasos son solo necesarios para obtener una probabilidad y no un valor proporcional a la probabilidad.
        # Convertir a probabilidad eliminando las operaciones de logaritmo
        prob_props = np.exp(proportional_probs)

        # Normalizar
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
            proportional_probs[idx] = np.log(self.priors[idx]) + np.sum(np.log(conditional_prob))
        return proportional_probs

    def clasifica(self, ejemplo: np.ndarray):
        if self.priors is None:
            raise ClasificadorNoEntrenado
        if not isinstance(ejemplo, np.ndarray):
            print("Error: No es un array de numpy")
            return
        prob_dict = self.clasifica_prob(ejemplo)
        return max(prob_dict, key=prob_dict.get)


# Ejemplo "jugar al tenis":


# nb_tenis=NaiveBayes(k=0.5)
# nb_tenis.entrena(X_tenis,y_tenis)
# ej_tenis=np.array(['Soleado','Baja','Alta','Fuerte'])
# print(nb_tenis.clasifica_prob(ej_tenis))
# {'no': 0.7564841498559081, 'si': 0.24351585014409202}
# >>> nb_tenis.clasifica(ej_tenis)
# 'no'


# ----------------------------------------------
# 2.2) Implementación del cálculo de rendimiento
# ----------------------------------------------

# Definir una función "rendimiento(clasificador,X,y)" que devuelve la
# proporción de ejemplos bien clasificados (accuracy) que obtiene el
# clasificador sobre un conjunto de ejemplos X con clasificación esperada y. 

# Ejemplo:

# >>> rendimiento(nb_tenis,X_tenis,y_tenis)
# 0.9285714285714286

def rendimiento(clasificador, X: np.ndarray, y: np.ndarray) -> float:
    """
    Performance del clasificador dado un conjunto de datos de test

    :param clasificador: Clasificador entrenado con metodo clasifica
    :param X:  Datos de test
    :param y: Clase objetivo de test
    :return: accuracy
    """

    # Acumula los correctos
    y_pred_accum = 0
    for idx, x in enumerate(X):
        pred = clasificador.clasifica(x) == y[idx]
        y_pred_accum += pred

    total = len(y)
    accuracy = y_pred_accum / total
    return accuracy


# --------------------------
# 2.3) Aplicando Naive Bayes
# --------------------------

# Usando el clasificador Naive Bayes implementado, obtener clasificadores 
# con el mejor rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Concesión de prestamos
# - Críticas de películas en IMDB 

# En todos los casos, será necesario separar un conjunto de test para dar la
# valoración final de los clasificadores obtenidos (ya realizado en el ejerciio 
# anterior). Ajustar también el valor del parámetro de suavizado k, usando un 
# conjunto de validación. 

# Describir (dejándolo comentado) el proceso realizado en cada caso, 
# y los rendimientos obtenidos. 


# ==================================
# EJERCICIO 3: NORMALIZADOR ESTÁNDAR
# ==================================


# Definir la siguiente clase que implemente la normalización "standard", es 
# decir aquella que traslada y escala cada característica para que tenga
# media 0 y desviación típica 1. 

# En particular, definir la clase: 


# class NormalizadorStandard():

#    def __init__(self):

#         .....

#     def ajusta(self,X):

#         .....        

#     def normaliza(self,X):

#         ......

#

# donde el método ajusta calcula las corresondientes medias y desviaciones típicas
# de las características de X necesarias para la normalización, y el método 
# normaliza devuelve el correspondiente conjunto de datos normalizados. 

# Si se llama al método de normalización antes de ajustar el normalizador, se
# debe devolver (con raise) una excepción:

class NormalizadorNoAjustado(Exception): pass


class NormalizadorStandard:

    def __init__(self):
        self.mean = None
        self.std = None
        self.adjusted = False

    def ajusta(self, X: np.ndarray) -> None:
        """
         Calcula las medias y desviaciones estándar de las características de X necesarias para la normalización.
        :param X: Matriz de datos de entrenamiento de forma (n_samples, n_features).

        :return: None
        """

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.adjusted = True

    def normaliza(self, X: np.ndarray) -> np.ndarray:
        """
        Normaliza las características de X utilizando la normalizacin estándar.

        :param X: Matriz de datos a normalizar de forma (n_samples, n_features).
        :return: Matriz de datos normalizados de forma (n_samples, n_features).
        """

        if not self.adjusted:
            raise NormalizadorNoAjustado("Debe ajustar el normalizador antes de normalizar los datos.")

        X_norm = (X - self.mean) / self.std
        return X_norm


# Por ejemplo:

#

# Una vez realizado esto, la media y desviación típica de Xe_cancer_n deben ser 
# 0 y 1, respectivamente. No necesariamente ocurre lo mismo con Xv_cancer_n, 
# ni con Xp_cancer_n. 


# ------


# ===========================================
# EJERCICIO 4: REGRESIÓN LOGÍSTICA MINI-BATCH
# ===========================================


# En este ejercicio se propone la implementación de un clasificador lineal 
# binario basado regresión logística (mini-batch), con algoritmo de entrenamiento 
# de descenso por el gradiente mini-batch (para minimizar la entropía cruzada).
# Diapositiva 50 del tema 3. 


# En concreto se pide implementar una clase: 

# class RegresionLogisticaMiniBatch():

#    def __init__(self,rate=0.1,rate_decay=False,n_epochs=100,
#                 batch_tam=64):

#         .....

#     def entrena(self,X,y,Xv=None,yv=None,n_epochs=100,salida_epoch=False,
#                     early_stopping=False,paciencia=3):

#         .....        

#     def clasifica_prob(self,ejemplos):

#         ......

#     def clasifica(self,ejemplos):

#          ......


# * El constructor tiene los siguientes argumentos de entrada:


#   + rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#     durante todo el aprendizaje. Si rate_decay es True, rate es la
#     tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#   + rate_decay, indica si la tasa de aprendizaje debe disminuir en
#     cada epoch. En concreto, si rate_decay es True, la tasa de
#     aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#     con la siguiente fórmula: 
#        rate_n= (rate_0)*(1/(1+n)) 
#     donde n es el número de epoch, y rate_0 es la cantidad introducida
#     en el parámetro rate anterior. Su valor por defecto es False. 
#  
#   + batch_tam: tamaño de minibatch


# * El método entrena tiene como argumentos de entrada:
#   
#     +  Dos arrays numpy X e y, con los datos del conjunto de entrenamiento 
#        y su clasificación esperada, respectivamente. Las dos clases del problema 
#        son las que aparecen en el array y, y se deben almacenar en un atributo 
#        self.clases en una lista. La clase que se considera positiva es la que 
#        aparece en segundo lugar en esa lista.
#     
#     + Otros dos arrays Xv,yv, con los datos del conjunto de  validación, que se 
#       usarán en el caso de activar el parámetro early_stopping. Ambos con 
#       valor None por defecto. 

#     + n_epochs es el número máximo de epochs en el entrenamiento. 

#     + salida_epoch (False por defecto). Si es True, al inicio y durante el 
#       entrenamiento, cada epoch se imprime  el valor de la entropía cruzada 
#       del modelo respecto del conjunto de entrenamiento, y su rendimiento 
#       (proporción de aciertos). Igualmente para el conjunto de validación, si lo
#       hubiera. Esta opción puede ser útil para comprobar 
#       si el entrenamiento  efectivamente está haciendo descender la entropía
#       cruzada del modelo (recordemos que el objetivo del entrenamiento es 
#       encontrar los pesos que minimizan la entropía cruzada), y está haciendo 
#       subir el rendimiento.
# 
#     + early_stopping (booleano, False por defecto) y paciencia (entero, 3 por defecto).
#       Si early_stopping es True, dejará de entrenar cuando lleve un número de
#       epochs igual a paciencia sin disminuir la menor entropía conseguida hasta el momento
#       en el conjunto de validación 
#       NOTA: esto se suele hacer con un conjunto de validación, y mecanismo de 
#       "callback" para recuperar el mejor modelo, pero por simplificar implementaremos
#       esta versión más sencilla.  
#        


# * Método clasifica: recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY de clases que el modelo predice para esos ejemplos. 

# * Un método clasifica_prob, que recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY con las probabilidades que el modelo 
#   asigna a cada ejemplo de pertenecer a la clase positiva.       


# Si se llama a los métodos de clasificación antes de entrenar el modelo, se
# debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception): pass


# RECOMENDACIONES:


# + IMPORTANTE: Siempre que se pueda, tratar de evitar bucles for para recorrer 
#   los datos, usando en su lugar funciones de numpy. La diferencia en eficiencia
#   es muy grande. 

# + Téngase en cuenta que el cálculo de la entropía cruzada no es necesario
#   para el entrenamiento, aunque si salida_epoch o early_stopping es True,
#   entonces si es necesario su cálculo. Tenerlo en cuenta para no calcularla
#   cuando no sea necesario.     

# * Definir la función sigmoide usando la función expit de scipy.special, 
#   para evitar "warnings" por "overflow":

#   from scipy.special import expit    
#
#   def sigmoide(x):
#      return expit(x)

# * Usar np.where para definir la entropía cruzada. 

# -------------------------------------------------------------

# Ejemplo, usando los datos del cáncer de mama (los resultados pueden variar):


# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)

# >>> lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer,yv_cancer)

# >>> lr_cancer.clasifica(Xp_cancer_n[24:27])
# array([0, 1, 0])   # Predicción para los ejemplos 24,25 y 26 

# >>> yp_cancer[24:27]
# array([0, 1, 0])   # La predicción anterior coincide con los valores esperado para esos ejemplos

# >>> lr_cancer.clasifica_prob(Xp_cancer_n[24:27])
# array([7.44297196e-17, 9.99999477e-01, 1.98547117e-18])


# Por ejemplo, los rendimientos sobre los datos (normalizados) del cáncer:

# >>> rendimiento(lr_cancer,Xe_cancer_n,ye_cancer)
# 0.9824561403508771

# >>> rendimiento(lr_cancer,Xp_cancer_n,yp_cancer)
# 0.9734513274336283


# Ejemplo con salida_epoch y early_stopping:

# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)

# >>> lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer_n,yv_cancer,salida_epoch=True,early_stopping=True)

# Inicialmente, en entrenamiento EC: 155.686323940485, rendimiento: 0.873972602739726.
# Inicialmente, en validación    EC: 43.38533009881579, rendimiento: 0.8461538461538461.
# Epoch 1, en entrenamiento EC: 32.7750241863029, rendimiento: 0.9753424657534246.
#          en validación    EC: 8.4952918658522,  rendimiento: 0.978021978021978.
# Epoch 2, en entrenamiento EC: 28.0583715052223, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.665719133490596, rendimiento: 0.967032967032967.
# Epoch 3, en entrenamiento EC: 26.857182744289368, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.09511082759361, rendimiento: 0.978021978021978.
# Epoch 4, en entrenamiento EC: 26.120803184993328, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.327991940213478, rendimiento: 0.967032967032967.
# Epoch 5, en entrenamiento EC: 25.66005010760342, rendimiento: 0.9808219178082191.
#          en validación    EC: 8.376171724729662, rendimiento: 0.967032967032967.
# Epoch 6, en entrenamiento EC: 25.329200890122557, rendimiento: 0.9808219178082191.
#          en validación    EC: 8.408704771704937, rendimiento: 0.967032967032967.
# PARADA TEMPRANA

# Nótese que para en el epoch 6 ya que desde la entropía cruzada obtenida en el epoch 3 
# sobre el conjunto de validación, ésta no se ha mejorado. 


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

    def _calculate_metrics(self, X: np.ndarray, y: np.ndarray, salida_epoch: bool, early_stopping: bool) -> Tuple[
        float, float]:
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


# -----------------------------------------------------------------

# ===================================================
# EJERCICIO 5: APLICANDO LOS CLASIFICADORES BINARIOS
# ===================================================


# Usando la regeresión logística implementada en el ejercicio 2, obtener clasificadores
# con el mejor rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama 
# - Críticas de películas en IMDB

# Ajustar los parámetros (tasa, rate_decay, batch_tam) para mejorar el rendimiento 
# (no es necesario ser muy exhaustivo, tan solo probar algunas combinaciones). 
# Usar para ello un conjunto de validación. 

# Dsctbir el proceso realizado en cada caso, y los rendimientos finales obtenidos
# sobre un conjunto de prueba (dejarlo todo como comentario)     


# =====================================================
# EJERCICIO 6: CLASIFICACIÓN MULTICLASE CON ONE vs REST
# =====================================================

# Se pide implementar un algoritmo de regresión logística para problemas de
# clasificación en los que hay más de dos clases, usando  la técnica One vs Rest. 


#  Para ello, implementar una clase  RL_OvR con la siguiente estructura, y que 
#  implemente un clasificador OvR (one versus rest) usando como base el
#  clasificador binario RegresionLogisticaMiniBatch


# class RL_OvR():

#     def __init__(self,rate=0.1,rate_decay=False,
#                   batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs=100,salida_epoch=False):

#        .......

#     def clasifica(self,ejemplos):

#        ......


#  Los parámetros de los métodos significan lo mismo que en el apartado
#  anterior, aunque ahora referido a cada uno de los k entrenamientos a 
#  realizar (donde k es el número de clases) (
#  Por simplificar, supondremos que no hay conjunto de validación ni parada
#  temprana.  


#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------
# >>> Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)

# >>> rl_iris_ovr=RL_OvR(rate=0.001,batch_tam=8)

# >>> rl_iris_ovr.entrena(Xe_iris,ye_iris)

# >>> rendimiento(rl_iris_ovr,Xe_iris,ye_iris)
# 0.8333333333333334

# >>> rendimiento(rl_iris_ovr,Xp_iris,yp_iris)
# >>> 0.9
# --------------------------------------------------------------------

# Por diversión para comparar la versión paralela con la lineal
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
            # Esto es facilmente paralalizable usando concurrent futures, pero, para mantenerlo enfocado en numpy.
            classifier = RegresionLogisticaMiniBatch(rate=self.rate, rate_decay=self.rate_decay, n_epochs=n_epochs,
                                                     batch_tam=self.batch_tam)
            y_binary = np.where(y == cls, 1, 0)
            classifier.entrena(X, y_binary, n_epochs=n_epochs, salida_epoch=salida_epoch)
            self.classifiers.append(classifier)

    # Por diversión para comparar la velocidad respecto a la versión lineal.
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


# =====================================================
# EJERCICIO 7: APLICANDO LOS CLASIFICADORES MULTICLASE
# =====================================================


# -------------------------
# 8.1) Codificación one-hot
# -------------------------


# Los conjuntos de datos en los que algunos atributos son categóricos (es decir,
# sus posibles valores no son numéricos, o aunque sean numéricos no hay una 
# relación natural de orden entre los valores) no se pueden usar directamente
# con los modelos de regresión logística, o con redes neuronales, por ejemplo.

# En ese caso es usual transformar previamente los datos usando la llamada
# "codificación one-hot". Básicamente, cada columna se reemplaza por k columnas
# en los que los valores psoibles son 0 o 1, y donde k es el número de posibles 
# valores del atributo. El valor i-ésimo del atributo se convierte en k atributos
# (0 ...0 1 0 ...0 ) donde todas las posiciones son cero excepto la i-ésima.  

# Por ejemplo, sin un atributo tiene tres posibles valores "a", "b" y "c", ese atributo 
# se reemplazaría por tres atributos binarios, con la siguiente codificación:
# "a" --> (1 0 0)
# "b" --> (0 1 0)
# "c" --> (0 0 1)    

# Definir una función:    

#     codifica_one_hot(X) 

# que recibe un conjunto de datos X (array de numpy) y devuelve un array de numpy
# resultante de aplicar la codificación one-hot a X.Por simplificar supondremos 
# que el array de entrada tiene todos sus atributos categóricos, y que por tanto 
# hay que codificarlos todos.

# NOTA: NO USAR PANDAS NI SKLEARN PARA ESTA FUNCIÓN

def codifica_one_hot(X: np.ndarray) -> np.ndarray:
    """

    :param X: dataset a codificar
    :return: array codificado
    """
    result = []
    for i in range(X.shape[1]):
        clases = np.unique(X[:, i])
        for cls in clases:
            arr = np.where(X[:, i] == cls, 1, 0)
            result.append(arr)
    return np.array(result).T


# Aplicar la función para obtener una codificación one-hot de los datos sobre
# concesión de prestamo bancario.     

# >>> Xc=np.array([["a",1,"c","x"],
#                  ["b",2,"c","y"],
#                  ["c",1,"d","x"],
#                  ["a",2,"d","z"],
#                  ["c",1,"e","y"],
#                  ["c",2,"f","y"]])

# >>> codifica_one_hot(Xc)
# 
# array([[1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
#        [0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0.],
#        [0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
#        [1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.],
#        [0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.],
#        [0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0.]])

# En este ejemplo, cada columna del conjuto de datos original se transforma en:
#   * Columna 0 ---> Columnas 0,1,2
#   * Columna 1 ---> Columnas 3,4
#   * Columna 2 ---> Columnas 5,6,7,8
#   * Columna 3 ---> Columnas 9, 10,11     


# ---------------------------------------------------------
# 8.2) Conjunto de datos de la concesión de crédito
# ---------------------------------------------------------

# Aplicar la implementación OvR del ejercicio anterior y la de one-hot del
# apartado anterior, para obtener un clasificador que aconseje la concesión, 
# estudio o no concesión de un préstamo, basado en los datos X_credito, y_credito. 

# Ajustar adecuadamente los parámetros (nuevamente, no es necesario ser demasiado 
# exhaustivo). Describirlo en los comentarios. 


# ---------------------------------------------------------
# 8.3) Clasificación de imágenes de dígitos escritos a mano
# ---------------------------------------------------------


#  Aplicar la implementación OvR del ejercicio anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en la 
#  carpeta datos/digitdata que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador. 

#  Los datos están ya separados en entrenamiento, validación y prueba. 

# Se pide:

# * Definir las funciones auxiliares necesarias para cargar el dataset desde los 
#   archivos de texto, y crear variables:
#       X_entr_dg, y_entr_dg
#       X_val_dg, y_val_dg
#       X_test_dg, y_test_dg
#   que contengan arrays de numpy con el dataset proporcionado (USAR ESOS NOMBRES).  

# * Obtener un modelo de clasificación RL_OvR    

# * Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
#   rate_decay para tratar de obtener un rendimiento aceptable (por encima del
#   75% de aciertos sobre test). 


# --------------------------------------------------------------------------
# ********** TOMADO DE LAS PRACTICAS DE CLASE
def cargaImágenes(fichero, ancho, alto):
    def convierte_0_1(c):
        if c == " ":
            return 0
        else:
            return 1

    with open(fichero) as f:
        lista_imagenes = []
        ejemplo = []
        cont_lin = 0
        for lin in f:
            ejemplo.extend(list(map(convierte_0_1, lin[:ancho])))
            cont_lin += 1
            if cont_lin == alto:
                lista_imagenes.append(ejemplo)
                ejemplo = []
                cont_lin = 0
    return np.array(lista_imagenes)


def cargaClases(fichero):
    with open(fichero) as f:
        return np.array([int(c) for c in f])


trainingdigits = "datos/digitdata/trainingimages"

validationdigits = "datos/digitdata/validationimages"

testdigits = "datos/digitdata/testimages"

trainingdigitslabels = "datos/digitdata/traininglabels"

validationdigitslabels = "datos/digitdata/validationlabels"

testdigitslabels = "datos/digitdata/testlabels"

X_entr_dg = cargaImágenes(trainingdigits, 28, 28)

y_entr_dg = cargaClases(trainingdigitslabels)

X_val_dg = cargaImágenes(validationdigits, 28, 28)

y_val_dg = cargaClases(validationdigitslabels)

X_test_dg = cargaImágenes(testdigits, 28, 28)

y_test_dg = cargaClases(testdigitslabels)

# ********************************************************************************

# EJEMPLOS DE PRUEBA

# LAS SIGUIENTES LLAMADAS SERÁN EJECUTADAS POR EL PROFESOR EL DÍA DE LA PRESENTACIÓN.
# UNA VEZ IMPLEMENTADAS LAS DEFINICIONES Y FUNCIONES (INCLUIDAS LAS AUXILIARES QUE SE
# HUBIERAN NECESITADO) Y REALIZADOS LOS AJUSTES DE HIPERPARÁMETROS, 
# DEJAR COMENTADA CUALQUIER LLAMADA A LAS FUNCIONES QUE SE TENGA EN ESTE ARCHIVO 
# Y DESCOMENTAR LAS QUE VIENE A CONTINUACIÓN.

# EN EL APARTADO FINAL DE RENDINIENTOS FINALES, USAR LA MEJOR COMBINACIÓN DE 
# HIPERPARÁMETROS QUE SE HAYA OBTENIDO EN CADA CASO, EN LA FASE DE AJUSTE. 

# ESTE ARCHIVO trabajo-1-iacd-23-24.py SERA CARGADO POR EL PROFESOR, 
# TENIENDO EN LA MISMA CARPETA LOS ARCHIVOS OBTENIDOS
# DESCOMPRIMIENDO datos-trabajo-1-iacd.zip.
# ES IMPORTANTE QUE LO QUE SE ENTREGA SE PUEDA CARGAR SIN ERRORES Y QUE SE EJECUTEN LOS 
# EJEMPLOS QUE VIENEN A CONTINUACIÓN. SI ALGUNO DE LOS EJERCICIOS NO SE HA REALIZADO 
# O DEVUELVE ALGÚN ERROR, DEJAR COMENTADOS LOS CORRESPONDIENTES EJEMPLOS. 


# *********** DESCOMENTAR A PARTIR DE AQUÍ
#
# print("************ PRUEBAS EJERCICIO 1:")
# print("**********************************\n")
Xe_votos, Xp_votos, ye_votos, yp_votos = particion_entr_prueba(X_votos, y_votos, test=1 / 3)
# print("Partición votos: ", y_votos.shape[0], ye_votos.shape[0], yp_votos.shape[0])
# print("Proporción original en votos: ", np.unique(y_votos, return_counts=True))
# print("Estratificación entrenamiento en votos: ", np.unique(ye_votos, return_counts=True))
# print("Estratificación prueba en votos: ", np.unique(yp_votos, return_counts=True))
# print("\n")
#
Xev_cancer, Xp_cancer, yev_cancer, yp_cancer = particion_entr_prueba(X_cancer, y_cancer, test=0.2)
# print("Proporción original en cáncer: ", np.unique(y_cancer, return_counts=True))
# print("Estratificación entr-val en cáncer: ", np.unique(yev_cancer, return_counts=True))
# print("Estratificación prueba en cáncer: ", np.unique(yp_cancer, return_counts=True))
Xe_cancer, Xv_cancer, ye_cancer, yv_cancer = particion_entr_prueba(Xev_cancer, yev_cancer, test=0.2)
# print("Estratificación entrenamiento cáncer: ", np.unique(ye_cancer, return_counts=True))
# print("Estratificación validación cáncer: ", np.unique(yv_cancer, return_counts=True))
# print("\n")
#
Xe_credito, Xp_credito, ye_credito, yp_credito = particion_entr_prueba(X_credito, y_credito, test=0.4)
# print("Estratificación entrenamiento crédito: ", np.unique(ye_credito, return_counts=True))
# print("Estratificación prueba crédito: ", np.unique(yp_credito, return_counts=True))
# print("\n\n\n")
#
# # print("************ PRUEBAS EJERCICIO 2:")
# # print("**********************************\n")
#
# nb_tenis = NaiveBayes(k=0.5)
# nb_tenis.entrena(X_tenis, y_tenis)
# ej_tenis = np.array(['Soleado', 'Baja', 'Alta', 'Fuerte'])
# print("NB Clasifica_prob un ejemplo tenis: ", nb_tenis.clasifica_prob(ej_tenis))
# print("NB Clasifica un ejemplo tenis: ", nb_tenis.clasifica([ej_tenis]))
# print("\n")
#
# nb_votos = NaiveBayes(k=1)
# nb_votos.entrena(Xe_votos, ye_votos)
# print("NB Rendimiento votos sobre entrenamiento: ", rendimiento(nb_votos, Xe_votos, ye_votos))
# print("NB Rendimiento votos sobre test: ", rendimiento(nb_votos, Xp_votos, yp_votos))
# print("\n")
#
# nb_credito = NaiveBayes(k=1)
# nb_credito.entrena(Xe_credito, ye_credito)
# print("NB Rendimiento crédito sobre entrenamiento: ", rendimiento(nb_credito, Xe_credito, ye_credito))
# print("NB Rendimiento crédito sobre test: ", rendimiento(nb_credito, Xp_credito, yp_credito))
# print("\n")
#
# nb_imdb = NaiveBayes(k=1)
# nb_imdb.entrena(X_train_imdb, y_train_imdb)
# print("NB Rendimiento imdb sobre entrenamiento: ", rendimiento(nb_imdb, X_train_imdb, y_train_imdb))
# print("NB Rendimiento imdb sobre test: ", rendimiento(nb_imdb, X_test_imdb, y_test_imdb))
# print("\n")
#
# print("************ PRUEBAS EJERCICIO 3:")
# print("**********************************\n")
#
# #
# normst_cancer=NormalizadorStandard()
# normst_cancer.ajusta(Xe_cancer)
# Xe_cancer_n=normst_cancer.normaliza(Xe_cancer)
# Xv_cancer_n=normst_cancer.normaliza(Xv_cancer)
# Xp_cancer_n=normst_cancer.normaliza(Xp_cancer)
#
# # Esta prueba es erronea porque calcula la media del datase original y no del normalizado:
# print("Normalización cancer entrenamiento: ",np.mean(Xe_cancer,axis=0))
# print("Normalización cancer validación: ",np.mean(Xv_cancer,axis=0))
# print("Normalización cancer test: ",np.mean(Xp_cancer,axis=0))
#
# # Esta las creamos nostros:
#
# print("Normalización cancer entrenamiento: ",np.mean(Xe_cancer_n,axis=0))
# print("Normalización cancer validación: ",np.mean(Xv_cancer_n,axis=0))
# print("Normalización cancer test: ",np.mean(Xp_cancer_n,axis=0))

# print("\n\n\n")
#
#
# print("************ PRUEBAS EJERCICIO 4:")
# print("**********************************\n")
#
#
# lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)
# lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer,yv_cancer)
# print("LR clasifica cuatro ejemplos cáncer (y valor esperado): ",lr_cancer.clasifica(Xp_cancer_n[17:21]),yp_cancer[17:21])
# print("LR clasifica_prob cuatro ejemplos cáncer: ", lr_cancer.clasifica_prob(Xp_cancer_n[17:21]))
# print("LR rendimiento cáncer entrenamiento: ", rendimiento(lr_cancer,Xe_cancer_n,ye_cancer))
# print("LR rendimiento cáncer prueba: ", rendimiento(lr_cancer,Xp_cancer_n,yp_cancer))
#
# print("\n\n CON SALIDA Y EARLY STOPPING**********************************\n")
#
# lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)
# lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer_n,yv_cancer,salida_epoch=True,early_stopping=True)
#
# print("\n\n\n")
#
# print("************ PRUEBAS EJERCICIO 6:")
# print("**********************************\n")
#
# Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)
#
# rl_iris_ovr=RL_OvR(rate=0.001,batch_tam=16)
#
# rl_iris_ovr.entrena(Xe_iris, ye_iris)
#
# print("OvR Rendimiento entrenamiento iris: ",rendimiento(rl_iris_ovr,Xe_iris,ye_iris))
# print("OvR Rendimiento prueba iris: ",rendimiento(rl_iris_ovr,Xp_iris,yp_iris))
# print("\n\n\n")


print("************ RENDIMIENTOS FINALES REGRESIÓN LOGÍSTICA EN CRÉDITO, IMDB y DÍGITOS")
print("*******************************************************************************\n")

# ATENCIÓN: EN CADA CASO, USAR LA MEJOR COMBINACIÓN DE HIPERPARÁMETROS QUE SE HA
# DEBIDO OBTENER EN EL PROCESO DE AJUSTE
# #
# print("==== MEJOR RENDIMIENTO RL SOBRE VOTOS:")
# RL_VOTOS = RegresionLogisticaMiniBatch(rate=0.5, rate_decay=False,
#                                        batch_tam=16)  # ATENCIÓN: sustituir aquí por los mejores parámetros tras el ajuste
# RL_VOTOS.entrena(Xe_votos, ye_votos, n_epochs=50)  # Aumentar o disminuir los epochs si fuera necesario
# print("Rendimiento RL entrenamiento sobre votos: ", rendimiento(RL_VOTOS, Xe_votos, ye_votos))
# print("Rendimiento RL test sobre votos: ", rendimiento(RL_VOTOS, Xp_votos, yp_votos))
# print("\n")
# #
# #
# print("==== MEJOR RENDIMIENTO RL SOBRE CÁNCER:")
# RL_CANCER = RegresionLogisticaMiniBatch(rate=0.05, rate_decay=True,
#                                         batch_tam=32)  # ATENCIÓN: sustituir aquí por los mejores parámetros tras el ajuste
# RL_CANCER.entrena(Xe_cancer, ye_cancer, n_epochs=200)  # Aumentar o disminuir los epochs si fuera necesario
# print("Rendimiento RL entrenamiento sobre cáncer: ", rendimiento(RL_CANCER, Xe_cancer, ye_cancer))
# print("Rendimiento RL test sobre cancer: ", rendimiento(RL_CANCER, Xp_cancer, yp_cancer))
# print("\n")
#

#
# print("==== MEJOR RENDIMIENTO RL_OvR SOBRE CREDITO:")
# X_credito_oh = codifica_one_hot(X_credito)
# Xe_credito_oh, Xp_credito_oh, ye_credito, yp_credito = particion_entr_prueba(X_credito_oh, y_credito, test=0.3)
#
# RL_CLASIF_CREDITO = RL_OvR(rate=0.1, rate_decay=False,
#                            batch_tam=16)  # ATENCIÓN: sustituir aquí por los mejores parámetros tras el ajuste
# RL_CLASIF_CREDITO.entrena(Xe_credito_oh, ye_credito, n_epochs=80)  # Aumentar o disminuir los epochs si fuera necesario
# print("Rendimiento RLOVR  entrenamiento sobre crédito: ", rendimiento(RL_CLASIF_CREDITO, Xe_credito_oh, ye_credito))
# print("Rendimiento RLOVR  test sobre crédito: ", rendimiento(RL_CLASIF_CREDITO, Xp_credito_oh, yp_credito))
# print("\n")

#
#
# print("==== MEJOR RENDIMIENTO RL SOBRE IMDB:")
# RL_IMDB=RegresionLogisticaMiniBatch(rate=0.5,rate_decay=False,batch_tam=16) # ATENCIÓN: sustituir aquí por los mejores parámetros tras el ajuste
# RL_IMDB.entrena(X_train_imdb,y_train_imdb, n_epochs=40) # Aumentar o disminuir los epochs si fuera necesario
# print("Rendimiento RL entrenamiento sobre imdb: ",rendimiento(RL_IMDB,X_train_imdb,y_train_imdb))
# print("Rendimiento RL test sobre imdb: ",rendimiento(RL_IMDB,X_test_imdb,y_test_imdb))
# print("\n")
#
#
print("==== MEJOR RENDIMIENTO RL SOBRE DIGITOS:")
RL_DG=RL_OvR(rate=0.05, rate_decay=False, batch_tam=2) # ATENCIÓN: sustituir aquí por los mejores parámetros tras el ajuste
RL_DG._entrena_paralelo(X_entr_dg,y_entr_dg, n_epochs=5, salida_epoch=False) # Aumentar o disminuir los epochs si fuera necesario
print("Rendimiento RL entrenamiento sobre dígitos: ",rendimiento(RL_DG,X_entr_dg,y_entr_dg))
print("Rendimiento RL validación sobre dígitos: ",rendimiento(RL_DG,X_val_dg,y_val_dg))
print("Rendimiento RL test sobre dígitos: ",rendimiento(RL_DG,X_test_dg,y_test_dg))
