"""
Это основной модуль. Здесь реализуется разложение двумерного сигнала по вейвлетам Хаара.
Основной функцией является analyze. - именно она реализует преобразование.
"""
import numpy as np
import lib.util as ut

def _GetLH(m, Y):
    """
        Принимает Y - градиент по геометрии Фрайда.
        Возращает LH - квадрант составляющую на m-ом уровне. m >=1
    """
    coef = np.power(np.sqrt(2), (m - 1))
    tmp = Y
    for i  in range (1, m):
        tmp = ut.convolve_2d(tmp, ut.GetH_l(2 ** i), 'horizontal')
    for i in range (m - 1):
        tmp = ut.convolve_2d(tmp, ut.GetH_l(2 ** i), 'vertical')
        tmp = ut.convolve_2d(tmp, ut.GetH_l(2 ** i), 'vertical')
    return coef * ut.down_sample(tmp, 2 ** m)


def _GetHL(m, X):
    """
        Принимает X - градиент по геометрии Фрайда.
        Возращает LH - квадрант составляющую на m-ом уровне. m >=1
    """
    coef = np.power(np.sqrt(2), (m - 1))
    tmp = X
    for i in range (m - 1):
        tmp = ut.convolve_2d(tmp, ut.GetH_l(2 ** i), 'horizontal')
        tmp = ut.convolve_2d(tmp, ut.GetH_l(2 ** i), 'horizontal')
    for i  in range (1, m):
        tmp = ut.convolve_2d(tmp, ut.GetH_l(2 ** i), 'vertical') 
    return coef * ut.down_sample(tmp, 2 ** m)

def _GetHH(m, X) :
    """
        Принимает X - градиент по геометрии Фрайда.
        Возращает HH - квадрант составляющую на m-ом уровне. m>=2
    """
    coef = np.sqrt(2 ** (m-1))
    buf = ut.convolve_2d(X, ut.GetH_l(2**0), 'horizontal')
    buf = ut.convolve_2d(buf, ut.GetH_l(2**0), 'horizontal')
    buf = ut.convolve_2d(buf, ut.GetH_h(2**(m - 1)), 'vertical')
    if (m == 2) :
        return ut.down_sample(buf, 4).T * coef
    for k in range(1, m - 1):
        buf = ut.convolve_2d(buf, ut.GetH_l(2**k), 'horizontal')
        buf = ut.convolve_2d(buf, ut.GetH_l(2**k), 'horizontal')
        buf = ut.convolve_2d(buf, ut.GetH_l(2**k), 'vertical')
    return ut.down_sample(buf, 2**m).T * coef

def _get_HH_right(X_H, Y_H):
    """
        Принимает X_H, Y_H - градиенты по геометрии Хаджина.
        Возращает HH - квадрант на 1-ом уровне.
    """
    coef = np.sqrt(2) * 0.25
    tmp1 = ut.convolve_2d(X_H, ut.GetH_h(1), 'vertical')
    tmp2 = ut.convolve_2d(Y_H, ut.GetH_h(1), 'horizontal')
    return ut.down_sample(tmp1 + tmp2,2) * coef

def analyze(gradX, gradY, X_H, Y_H):
    """
        Самая главная функция. Принимает градиенты по геометриям
        Хаджина и Фрайда. Возращает словари LH, HL, HH с разложением по уровням.
        минусы и транспонирования добавлены для совместимости с модулем Pywawelets.
        analyze вернет при точных параметрах такое же значение как и dwt2(array,'haar')
        из модуля pywt(Pywawelets). 
    """
    M = int(np.log2(len(gradX)))
    LH = dict()
    HL = dict()
    HH = dict()
    HH[M - 1] = _get_HH_right(X_H, Y_H).T
    HL[M - 1] = -ut.down_sample(gradX, 2).T
    LH[M - 1] = -ut.down_sample(gradY, 2).T
    for k in range(M - 1):
        LH[k] = -_GetLH(M - k, gradY).T
        HL[k] = -_GetHL(M - k, gradX).T
        if (k != M - 1):
            HH[k] = _GetHH(M - k, gradX)
    return LH, HL, HH

import pywt
def syntesis(LL,LH, HL, HH, M):
    """
        Воосстанавливает исходный двумерный сигнал по известным составляющим. 
        Возвращает словарь LL, который представляет собой LL составляющую восстанавливаемого
        сигнала.
        LL - словарь. Изначально с нем содерживатся среднее значение сигнала. Оно равно np.mean(orig)* 2^M.
                именно этот параметр определяет насоклько точно будет восстановлен сигнал.
        LH - словарь. LH - квадрант исходного сигнала. Предполагается, что он полностью известнен и получен из analyze.
        HL - словарь. HL - квадрант исходного сигнала. Предполагается, что он полностью известнен и получен из analyze.
        HH - словарь. HH - квадрант исходного сигнала. Предполагается, что он полностью известнен и получен из analyze.
        M - определяет разрешение декомпозиции. LL[M] будет размера 2^M * 2^M
        Функция по своей сути является оберткой над библиотекой pywt.
    """
    for k in range(M):
        LL[k + 1] = pywt.idwt2([LL[k],(LH[k], HL[k], HH[k])], 'haar')
    return LL