"""
Модуль реализует функции используемые при тестировании. Необходим для курсовой работы.
Модули непосредственно реализующие метод никак от него не зависят.
"""
import decompozition as decomp
import lib.util as ut
import lib.data as data
import lib.mesurements as msrm
import lib.gradients as gr
import numpy as np
def get_all(func, grad_x, grad_y, x_s, x_e, y_s, y_e, M, LL_0 = 1.0):
    """
        Функция возвращает основную статистику о работе метода.
        func - функция, которую необходимо восстановить
        grad_x - функция, градиент по x, в явном виде.
        grad_y - функция, градиент по y, в явном виде.
        x_s, x_e, y_s, y_e, M - задают сетку на которой будут вычислены значения функций.
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
    x,y = data.get_plane(x_s,x_e,y_s,y_e,M)
    res = dict()
    res['orig'] = func(x,y)
    res['grX'] = grad_x(x,y)
    res['grY'] = grad_y(x,y)
    res['X_H'] = res['grX'][::,:-1:]
    res['Y_H'] = res['grY'][:-1:,::]
    res['X_F'] = res['grX'][:-1:,:-1:]
    res['Y_F'] = res['grY'][:-1:,:-1:]
    LL = LH = HL = HH = dict()
    LL[0] = np.array([[LL_0]])
    res['LL_0'] = np.array([[np.mean(res['orig']) * (2 ** M)]])
    LH, HL, HH = decomp.analyze(res['grX'],res['grY'], res['grX'], res['grY'])
    LL = decomp.syntesis(LL,LH, HL, HH, M)
    res['LL'] = LL
    res['LH'] = LH
    res['HL'] = HL
    res['HH'] = HH
    res['mse'] = msrm.mse(res['orig'], LL[M])
    res['M'] = M
    return res