"""
Модуль отвечает за функции реализующие нормы схожести матриц.
"""
import numpy as np
def mse(A, B):
	"""
	Средне квадратичное отклонение
	"""
	coef = A.shape[0]*A.shape[1]
	return ((A - B) ** 2).mean()/coef