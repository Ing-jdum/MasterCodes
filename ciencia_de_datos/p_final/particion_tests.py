from scratch import particion_entr_prueba
from carga_datos import *
import numpy as np




def test_votos_data():
    Xe_votos, Xp_votos, ye_votos, yp_votos = particion_entr_prueba(X_votos, y_votos, test=1 / 3)

    assert y_votos.shape[0] == 435
    assert ye_votos.shape[0] == 290
    assert yp_votos.shape[0] == 145

    assert np.unique(y_votos, return_counts=True)[1].tolist() == [168, 267]
    assert np.unique(ye_votos, return_counts=True)[1].tolist() == [112, 178]
    assert np.unique(yp_votos, return_counts=True)[1].tolist() == [56, 89]


def test_cancer_data():
    Xev_cancer, Xp_cancer, yev_cancer, yp_cancer = particion_entr_prueba(X_cancer, y_cancer, test=0.2)

    assert np.unique(y_cancer, return_counts=True)[1].tolist() == [212, 357]
    assert np.unique(yev_cancer, return_counts=True)[1].tolist() == [170, 286]
    assert np.unique(yp_cancer, return_counts=True)[1].tolist() == [42, 71]

    Xe_cancer, Xv_cancer, ye_cancer, yv_cancer = particion_entr_prueba(Xev_cancer, yev_cancer, test=0.2)

    assert np.unique(ye_cancer, return_counts=True)[1].tolist() == [136, 229]
    assert np.unique(yv_cancer, return_counts=True)[1].tolist() == [34, 57]


def test_credito_data():
    Xe_credito, Xp_credito, ye_credito, yp_credito = particion_entr_prueba(X_credito, y_credito, test=0.4)

    assert np.unique(y_credito, return_counts=True)[1].tolist() == [202, 228, 220]
    assert np.unique(ye_credito, return_counts=True)[1].tolist() == [122, 137, 132]
    assert np.unique(yp_credito, return_counts=True)[1].tolist() == [80, 91, 88]
