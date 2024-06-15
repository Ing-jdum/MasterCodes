#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: jruiz
"""
# CONJUNTOS DE DATOS A USAR EN EL TRABAJO 1 DE LA ASIGNATURA
# "INTELIGENCIA ARTIFICIAL PARA LA CIANCIA DEL DATO"

import numpy as np


# ----------------------------------------------------

# JUGAR AL TENIS

from datos import jugar_tenis

X_tenis=jugar_tenis.X_tenis
y_tenis=jugar_tenis.y_tenis


# ----------------------------------------------------



# ----------------------------------------------------

# CONCESIÓN DE UN PRÉSTAMO

from datos import credito

X_credito=np.array([d[:-1] for d in credito.datos_con_la_clase])
y_credito=np.array([d[-1] for d in credito.datos_con_la_clase])

# ----------------------------------------------------

# CLASIFICACIÓN DE LA PLANTA DE IRIS

from sklearn.datasets import load_iris

iris=load_iris()
X_iris=iris.data
y_iris=iris.target


# --------------------------------------------------

# VOTOS EN EL CONGRESO USA

from datos import votos
X_votos=votos.datos
y_votos=votos.clasif


#--------------------------------------------------

# CÁNCER DE MAMA

from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()

X_cancer=cancer.data
y_cancer=cancer.target

#-------------------------------------------

# CRÍTICAS DE PELÍCULAS EN IMDB

# Los datos están obtebidos de esta manera

#import random as rd
#from sklearn.datasets import load_files
#
#reviews_train = load_files("datos/aclImdb/train/")
#muestra_entr=rd.sample(list(zip(reviews_train.data,reviews_train.target)),k=2000)
#text_train=[d[0] for d in muestra_entr]
#text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
#y_train=np.array([d[1] for d in muestra_entr])
#print("Ejemplos por cada clase: {}".format(np.bincount(y_train)))
#
#reviews_test = load_files("datos/aclImdb/test/")
#muestra_test=rd.sample(list(zip(reviews_test.data,reviews_test.target)),k=400)
#text_test=[d[0] for d in muestra_test]
#text_test = [doc.replace(b"<br />", b" ") for doc in text_test]
#y_test=np.array([d[1] for d in muestra_test])
#print("Ejemplos por cada clase: {}".format(np.bincount(y_test)))
#
#
#from sklearn.feature_extraction.text import CountVectorizer
#
#vect = CountVectorizer(min_df=50, stop_words="english",binary=True).fit(text_train)
#print("Tamaño del vocabulario: {}".format(len(vect.vocabulary_)))
#X_train = vect.transform(text_train).toarray()
#X_test = vect.transform(text_test).toarray()
#
#np.save("datos/imdb_sentiment/vect_train_text.npy",X_train)
#np.save("datos/imdb_sentiment/vect_test_text.npy",X_test)
#np.save("datos/imdb_sentiment/y_train_text.npy",y_train)
#np.save("datos/imdb_sentiment/y_test_text.npy",y_test)

X_train_imdb=np.load("datos/imdb_sentiment/vect_train_text.npy")
X_test_imdb=np.load("datos/imdb_sentiment/vect_test_text.npy")
y_train_imdb=np.load("datos/imdb_sentiment/y_train_text.npy")
y_test_imdb=np.load("datos/imdb_sentiment/y_test_text.npy")

# ----------------------------------------------------------------

# DÍGITOS ESCRITOS A MANO

# ver la carpeta digitdata
