"""
В модуле реализованы функции для визуализации результатов работы.
"""
import test
import scipy.misc
import matplotlib.pyplot as plt

def show_all(func, grad_x, grad_y, x_s, x_e, y_s, y_e, M):
	"""
	Визулизация функции test.get_all.
	Принимает те же параметры, что и get_all
	Отрисовывает исходную и восстновленную функции.
	Выводит статистику.
	Возвращает результат функции get_all
	"""
	res = test.get_all(func, grad_x, grad_y, x_s, x_e, y_s, y_e, M)
	f, axarr = plt.subplots(1,2,figsize=(30,30))
	axarr[0].imshow(res['LL'][res['M']], cmap="gray")
	axarr[1].imshow(res['orig'], cmap = 'gray')
	