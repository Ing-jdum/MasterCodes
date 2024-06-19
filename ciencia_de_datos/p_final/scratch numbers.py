import numpy as np

from ciencia_de_datos.p_final.scratch import rendimiento
from ciencia_de_datos.p_final.scratch_logistic_ovr import RL_OvR


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

X_train_dg = cargaImágenes(trainingdigits, 28, 28)

y_train_dg = cargaClases(trainingdigitslabels)

X_valid_dg = cargaImágenes(validationdigits, 28, 28)

y_valid_dg = cargaClases(validationdigitslabels)

X_test_dg = cargaImágenes(testdigits, 28, 28)

y_test_dg = cargaClases(testdigitslabels)


rl_iris_ovr = RL_OvR(rate=0.001, batch_tam=2)

rl_iris_ovr.entrena(X_train_dg, y_train_dg, n_epochs=5)

print(rendimiento(rl_iris_ovr, X_test_dg, y_test_dg))
