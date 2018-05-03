"""
Генерация собственных значений lambda, mu для матриц в вариационном методе
"""
import numpy as np


def get_lambda(n, h):
    """
    :param n: количество собственных значений. Будет от 0 до n-1
    :param h: шаг сектки(дискретизации)
    :return: np.array([])
    """
    return np.array([4 / h ** 2 * np.sin(k * h / 2) ** 2 for k in range(n)])


def get_mu(n, h):
    """
    :param n: количество собственных значений. Будет от 0 до n-1
    :param h: шаг сектки(дискретизации)
    :return: np.array([])
    """
    return 1 - h ** 2 * get_lambda(n, h) / 6
