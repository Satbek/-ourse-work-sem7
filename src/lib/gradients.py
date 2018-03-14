"""
В данном модуле находятся функции реализующие нахождение градиента от
двумерного массива.
Функция приминимает одну матрицу, возвращает две. Градиент по X и
Градиент по Y.
"""
import numpy as np
from .haar_wawelet_method import util as ut


def fried_model_gradient(image):
    """
        Возврщает градиент по геометрии Фрайда. Hаходится с помощью
        сверток с фильтрами пробразования Хаара.
        Keyword arguments:
            image - исходный массив из которого будут получать градиенты.
    """
    tmp = ut.convolve_2d(image, ut.GetH_h(1), mode='horizontal')
    X = ut.convolve_2d(tmp, ut.GetH_l(1), mode='vertical')
    tmp = ut.convolve_2d(image, ut.GetH_l(1), mode='horizontal')
    Y = ut.convolve_2d(tmp, ut.GetH_h(1), mode='vertical')
    return X, Y


def Hudgin_gradien_model(image):
    """
    Возврщает градиент по геометрии Хаджина. Находится с помощью
    сверток с фильтрами пробразования Хаара.
    Keyword arguments:
        image - исходный массив из которого будут получать градиенты.
    """
    X = np.sqrt(2) * ut.convolve_2d(image, ut.GetH_h(1), mode='horizontal')
    Y = np.sqrt(2) * ut.convolve_2d(image, ut.GetH_h(1), mode='vertical')
    return X, Y
