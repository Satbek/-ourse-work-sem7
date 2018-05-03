"""
Вариационный метод восстановления волнового фронта.
2 параметра регуляризации alpha, gamma.
"""

import numpy as np
import lib.variational_method.eigenvalues as eigenvalues
import lib.variational_method.matrixes as matrixes


def _f1(g1, h1, h2):
    """
    F1 составлющая матрицы F
    :param g1: 2d array. Градиент по x
    :param h1: шаг сетки по x
    :param h2: шаг сетки по y
    :return: 2d array
    """
    dim = g1.shape
    tmp = np.dot(matrixes.create_g1(dim[0], h1), g1)
    return np.dot(tmp, matrixes.create_b(dim[1], h2))


def _f2(g2, h1, h2):
    """
    F2 составляющая матрицы F
    :param g2: 2d array. Градиент по y
    :param h1: шаг сетки по x
    :param h2: шаг сетки по y
    :return: 2d array
    """
    dim = g2.shape
    tmp = np.dot(matrixes.create_b(dim[0], h1), g2)
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
    return _f1(g1, h1, h2) + _f2(g2, h1, h2)


def method(g1, g2, h1, h2, alpha, gamma):
    """
    Метод. Принимает градиенты по x, y. Параметры регуляризации alga, gamma.
    Возвращает восстановленный волновой фронт.
    :param g1: 2d array. Градиент по x
    :param g2: 2d array. Градиент по y
    :param h1: Шаг сетки по x
    :param h2: Шаг сетки по y
    :param alpha: параметр регуляризации
    :param gamma: параметр регуляризации
    :return:
    """
    f = np.fft.fft2(_get_f_matrix(g1, g2, h1, h2))
    lambda1 = eigenvalues.get_lambda(f.shape[0], h1)
    lambda2 = eigenvalues.get_lambda(f.shape[1], h2)

    mu1 = eigenvalues.get_mu(f.shape[0], h1)
    mu2 = eigenvalues.get_mu(f.shape[1], h2)

    res = f
    for k in range(res.shape[0]):
        for l in range(res.shape[1]):
            t = (lambda1[k] * mu2[l] + mu1[k] * lambda2[l] +
                    alpha * mu1[k] * mu2[l] + gamma * lambda1[k] * lambda2[l])
            res[k,l] = f[k,l] / t
    res = np.fft.ifft2(res)
    return np.real(res)
