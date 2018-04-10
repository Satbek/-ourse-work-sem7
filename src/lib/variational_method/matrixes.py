"""
Генерация трехдиагональных матриц для вариационного метода
"""

import numpy as np


def create_l(dim, h):
    """
    Матрица L, 2-я производная
    :param dim: размерность
    :param h: шаг сетки
    :return:[[ 2, -1 0],
             [-1, 2, -1],
             [0,-1, 2]] / h^2
    """
    diag1 = np.array([2 if i == j else 0 for i in range (dim) for j in range(dim)]).reshape(dim,dim)
    diag2 = np.array([-1 if np.abs(i - j) == 1 else 0 for i in range (dim) for j in range(dim)]).reshape(dim,dim)
    return (diag1 + diag2) / h**2


def create_b(dim, h):
    """
    Матрица B
    :param dim: размерность
    :param h: шаг сетки
    :return: I - 1/6 * h^2 * L
            [[ 2/3, 1/6, 0],
             [ 1/6, 2/3, 1/6],
             [ 0, 1/6, 2/3]]
    """
    return np.eye(dim) - 1 / 6 * create_l(dim, h) * h**2


def _create_g(dim):
    """
    Матрица G1, без деления на h
    :param dim: размерность
    :return: [[0, -0.5, 0],
             [[0.5, 0, -0.5],
             [[0, 0.5, 0]]
    """
    diag1 = np.array([-1 if i - j == -1 else 0 for i in range(dim) for j in range(dim)]).reshape(dim, dim)
    diag2 = np.array([1 if i - j == 1 else 0 for i in range(dim) for j in range(dim)]).reshape(dim, dim)
    return 0.5 * (diag1 + diag2)


def create_g1(dim, h):
    """
    Матрица G1
    :param dim: размерность
    :param h: шаг сетки
    :return: [[0, -0.5, 0],
             [[0.5, 0, -0.5],
             [[0, 0.5, 0]]
    """
    diag1 = np.array([-1 if i - j == -1 else 0 for i in range (dim) for j in range(dim)]).reshape(dim,dim)
    diag2 = np.array([1 if i - j == 1 else 0 for i in range (dim) for j in range(dim)]).reshape(dim,dim)
    return _create_g(dim) / h


def create_g2(dim, h):
    """
    Матрица G2
    :param dim: размерность
    :param h: шаг сетки
    :return: [[ 0. ,  0.5,  0. ],
             [-0.5,  0. ,  0.5],
             [ 0. , -0.5,  0. ]]
    """
    return _create_g(dim).T / h
