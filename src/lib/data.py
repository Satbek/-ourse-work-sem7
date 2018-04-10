"""
В этом модуле находятся функции, используемые для задания данных для
вычислений. Иначе говоря каждая функция здесь создает двумерный массив
особого вида, который будет необходим для дальнейштх вычислений.
"""
import numpy as np


def get_plane(x_s, x_e, y_s, y_e, M):
    """
        Возвращает 2 матрицы размера 2^M * 2^M, которые используются как
        сетка для вычисления дальнейших функцями, которые задает пользователь.
        Keyword arguments:
            x_s - x start , начальная точка x
            x_e - x end, конечная точка x
            y_s - y start, начальная точка y
            y_e - конечная точка y
            M - определяет размерность матрицы. Она будет равна 2^M * 2^M
            ---------->x -расположение осей.
            |
            |
            |
            |
            \/
            y
    """
    dim = 2 ** M * 1j
    Y, X = np.mgrid[x_s:x_e:dim, y_s:y_e:dim]
    return X, Y  # нужно исправить


def get_Poisson_noise(image, photons):
    """
        Функция возвращает зашумленную матрицу.
        image - матрица. (необходимо , чтобы в ней были только
                            положительные аргументы)
        photons - нормировочный коэффициент
    """
    original_type = image.dtype
    step = image.min()
    noised_image = image.astype(np.float64)
    if (step < 0):
        noised_image -= step
    scale_factor = photons / noised_image.max()
    noised_image = noised_image * scale_factor
    noised_image = np.random.poisson(noised_image)
    noised_image = noised_image / scale_factor
    if (step < 0):
        noised_image += step
    noised_image = noised_image.astype(original_type)
    return noised_image


def super_gauss(x, y, a, N):
    """
    Супергаусс на сетке x,y с параметрами a, N
    :param x: сетка по x
    :param y: сетка по y
    :param a: параметр, отвечает за размер области, которую срезает супергаусс
    :param N: отвечает за "крутизну" супергаусс, чем больше, тем быстрее функция стремится к 0
    :return: 2d array
    """
    return np.exp(-((x**2 + y**2) / a**2) ** N)


def normalize(front):
    """
    Нормирует исходный волновой фронт на квадрат [-1, 1] x [-1, 1]
    :param front: волновой фронт
    :return: отнормированный волновой фронт
    """
    front -= front.min()
    front /= front.max()
    return front
