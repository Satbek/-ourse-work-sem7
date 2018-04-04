"""
Вариационный метод восстановления волнового фронта.
2 параметра регуляризации \alfa, \gamma.
"""

import numpy as np
import lib.variational_method.eigenvalues as eigenvalues
import lib.variational_method.matrixes as matrixes


def _f1(g1, h1):
    """
    F1 составлющая матрицы F
    :param g1: 2d array. Градиент по x
    :param h1: шаг сетки по x
    :return: 2d array
    """
    dim = g1.shape
    tmp = np.dot(matrixes.create_g1(dim[0], h1), g1)
    return np.dot(tmp, matrixes.create_b(dim[1]), h1)


def _f2(g2, h2):
    """
    F2 составляющая матрицы F
    :param g2: 2d array. Градиент по y
    :param h2: шаг сетки по y
    :return: 2d array
    """
    dim = g2.shape
    tmp = np.dot(matrixes.create_b(dim[0], h2), g2)
    return np.dot(tmp, matrixes.create_g2(dim[1], h2))


def _get_f_matrix(g1, g2, h1, h2):
    """
    Матрица F. Правая часть в разностной схеме метода.
    :param g1: 2d array. Градиент по x
    :param g2: 2d array. Градиент по y
    :param h1: Шаг сетки по x
    :param h2: Шаг сетки по y
    :return: 2d array
    """
    return _f1(g1, h1) + _f2(g2, h2)


def method(g1, g2, h1, h2, alfa, gamma):
    """
    Метод. Принимает градиенты по x, y. Параметры регуляризации alga, gamma.
    Возвращает восстановленный волновой фронт.
    :param g1: 2d array. Градиент по x
    :param g2: 2d array. Градиент по y
    :param h1: Шаг сетки по x
    :param h2: Шаг сетки по y
    :param alfa: параметр регуляризации
    :param gamma: параметр регуляризации
    :return:
    """
    f = np.fft.fft2(_get_f_matrix(g1, g2, h1, h2))
    lambda_ = eigenvalues.get_lambda(f.shape[0], h1)
    mu_ = eigenvalues.get_mu(f.shape[1], h2)
    res = np.zeros(f.shape, dtype=complex)
    for k in range(res.shape[0]):
        for l in range(res.shape[1]):
            res[k, l] = (lambda_[k] * mu_[l] + mu_[k] * lambda_[l] +
                         alfa * mu_[k] * mu_[l] + gamma * lambda_[k] * lambda_[l])
    res = np.fft.ifft2(f / res)
    return np.real(res)
