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
