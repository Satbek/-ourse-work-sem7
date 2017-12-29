"""
В модуле реализованы функции для визуализации результатов работы.
"""
import research
import matplotlib.pyplot as plt
import lib.gradients as gr
import lib.mesurements as msrm
import numpy as np

def show_all(func, grad_x, grad_y, x_s, x_e, y_s, y_e, M,LL_0=1):
	"""
	Визулизация функции test.get_all.
	Принимает те же параметры, что и get_all
	Отрисовывает исходную и восстновленную функции.
	Выводит статистику.
	Возвращает результат функции get_all 
	"""
	res = research.get_all(func, grad_x, grad_y, x_s, x_e, y_s, y_e, M, LL_0)
	f, axarr = plt.subplots(1,2,figsize=(15,15))
	axarr[0].set_title('восстановленное')
	axarr[0].imshow(res['LL'][res['M']], cmap="gray")
	axarr[1].set_title('исходное')
	axarr[1].imshow(res['orig'], cmap = 'gray')
	print("MSE = ", res['mse'])
	print("M = ", res['M'])
	X_H, Y_H = gr.Hudgin_gradien_model(res['orig'])
	X_F, Y_F = gr.fried_model_gradient(res['orig'])
	print( "Погрешность X_H=", msrm.mse( X_H[::,:-1:], res['X_H']) )
	print( "Погрешность Y_H=", msrm.mse( Y_H[:-1:,::], res['Y_H']) )
	print( "Погрешность X_F=", msrm.mse( X_F[:-1:,:-1:], res['X_F']) )
	print( "Погрешность Y_F=", msrm.mse( Y_F[:-1:,:-1:], res['Y_F']) )
	print( "LL[0] исходного изображения ", res['LL_0'] )
	print( "LL[0] восстновленного изображения ", np.mean( res['LL'][res['M']] ) * (2**res['M']) )
	print( "Максимум исходного изображения", np.max(res['orig']))
	print( "Максимум восстновленного изображения", np.max(res['LL'][res['M']]))
	print( "Минимум исходного изображения", np.min(res['orig']))
	print( "Минимум восстновленного изображения", np.min(res['LL'][res['M']]))
	return res