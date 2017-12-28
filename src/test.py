"""
Модуль реализует функции используемые при тестировании. Необходим для курсовой работы.
Модули непосредственно реализующие метод никак от него не зависят.
"""
import decomposition as decomp
import lib.util as ut
def get_all(func, grad_x, grad_y, x_s, x_e, y_s, y_e, M):
    """
        Функция возвращает основную статистику о работе метода.
        
    """
    x,y = get_plane(x_s,x_e,y_s,y_e,M)
    res = dict()
    res['orig'] = func(x,y)
    res['grX'] = grad_x(x,y)
    res['grY'] = grad_y(x,y)
    res['X_H'] = res['grX'][::,:-1:]
    res['Y_H'] = res['grY'][:-1:,::]
    res['X_F'] = res['grX'][:-1:,:-1:]
    res['Y_F'] = res['grY'][:-1:,:-1:]
    LL = LH = HL = HH = dict()
    LL[0] = np.array([[np.mean(res['orig']) * (2 ** M)]])
    LH, HL, HH = analyze(res['grX'],res['grY'], res['grX'], res['grY'])
    LL = syntesis(LL,LH, HL, HH, M)
    res['LL'] = LL
    res['LH'] = LH
    res['HL'] = HL
    res['HH'] = HH
    res['mse'] = mse(res['orig'], LL[M])
    res['M'] = M
    return res