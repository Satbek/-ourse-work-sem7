"""
В этом модуле находятся служебные функции необходимые для реализации основных функций
This module provides utilyty functions, to write another functions
Эти функции используются для работы с z-преобразованием, свертки сигналов
Theese functions are used to work with z-thatsform, singals convolution
"""
import numpy as np
def down_sample(array ,sample_factor):
	"""
		Производит операцию dounsample(стрелочка вниз)удаляет каждый второй элемент в массиве. 
		Работает как с одномерными так и с двумерными массивами. В случае двумерного удаляет каждый второй элемент
		и по строкам и по столбцам.
		Keyword arguments:
			array - одномерный или двумерный массив
			sample_factor - int > 0. определяет какой элемент будет удаляться. Обычно это 2^k, но необязательно. Можно удалить и каждый 3-ий.
	"""
	array = np.array(array, dtype=float)
	array = array[::sample_factor]
	if (len(array.shape) == 2):
		array = array.transpose()
		array = array[::sample_factor]
	return array

from scipy.signal import convolve2d as conv2d
def convolve_2d(array_2d, kernel, mode):
	"""
		Осуществляет свертку двумерного массива array_2d с ядром kernel либо по строкам,
		либо по столбцам.
		Keyword arguments:
			array_2d - двумерный массив.
			kernel - одномерный массив. Преполагается, что это z-преобразование фильтра.
			mode - параметр, отвечающий за то, с чем будет свертка: со строками или со стобцами.
					'verical' - по стобцам, 'horizontal' - по строкам.
		Свертка производится с дополнениеми нулями справа.
	"""
	res = np.array(array_2d, dtype=float)
	kernel = np.array(kernel)
	if (mode == 'horizontal'):
		res = conv2d(res, kernel[None, :])
		res = res[:,len(kernel)-1:]
	if (mode == 'vertical'):
		res = conv2d(res, kernel[:, None])
		res = res[len(kernel)-1:,:]
	return res

def GetH_h(pow_):
	"""
		Возварщает вектор , соответствующий z-преобразованию H_H(z^pow_)
		(высокочастотного фильтра преобразования Хаара).
		Keyword arguments:
			pow_ - степень при H_H(z)
	"""
	res = np.zeros(pow_ + 1)
	res[0] = 1
	res[-1] = -1
	return res / np.sqrt(2)

def GetH_l(pow_):
	"""
		Возварщает вектор , соответствующий z-преобразованию H_L(z^pow_)
		(низкочастотного фильтра преобразования Хаара).
		Keyword arguments:
			pow_ - степень при H_L(z)
	"""
	res = np.zeros(pow_ + 1)
	res[0] = res[-1] = 1
	return res / np.sqrt(2)