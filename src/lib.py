import numpy as np
import pywt
from scipy import signal
from sklearn.metrics import mean_squared_error

H_h = np.array([1, -1])/np.sqrt(2)
H_l = np.array([1, 1])/np.sqrt(2)

from scipy.signal import convolve2d as conv2d

def down_sample(array ,sample_factor):
    array = np.array(array, dtype=float)
    array = array[::sample_factor]
    if (len(array.shape) == 2):
        array = array.transpose()
        array = array[::sample_factor]
    return array

def convolve_2d(array_2d, kernel, mode):
    res = np.array(array_2d, dtype=float)
    kernel = np.array(kernel)
    if (mode == 'horizontal'):
        res = conv2d(res, kernel[None, :])
        res = res[:,len(kernel)-1:]
    if (mode == 'vertical'):
        res = conv2d(res, kernel[:, None])
        res = res[len(kernel)-1:,:]
    return res

def fried_model_gradient(image):
    tmp = convolve_2d(image, H_h, mode='horizontal')
    X = convolve_2d(tmp, H_l, mode='vertical')
    tmp = convolve_2d(image, H_l, mode='horizontal')
    Y = convolve_2d(tmp, H_h, mode='vertical')
    return X, Y

def process_next_X (prevX):
    tmp1 = convolve_2d(prevX, H_l, mode='horizontal')
    tmp2 = convolve_2d(tmp1, H_l, mode='horizontal')
    return down_sample(np.sqrt(2) * convolve_2d(tmp2, np.array([1,0,1]) / np.sqrt(2), mode='vertical'), 2)

def process_next_Y (prevY):
    tmp1 = convolve_2d(prevY, np.array([1,0,1]) / np.sqrt(2), mode = 'horizontal')
    tmp2 = convolve_2d(tmp1, H_l, mode = 'vertical')
    return np.sqrt(2) * down_sample(convolve_2d(tmp2, H_l, mode = 'vertical'), 2)

