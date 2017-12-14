
# coding: utf-8

# In[1]:

import numpy as np
import pywt
from scipy import signal
from sklearn.metrics import mean_squared_error


# In[2]:

H_h = np.array([1, -1])/np.sqrt(2)
H_l = np.array([1, 1])/np.sqrt(2)


# In[165]:

def down_sample(array ,sample_factor):#works
    array = np.array(array, dtype=float)
    array = array[::sample_factor]
    if (len(array.shape) == 2):
        array = array.transpose()
        array = array[::sample_factor]
    return array


# In[164]:

from scipy.signal import convolve2d as conv2d
def convolve_2d(array_2d, kernel, mode):#works
    res = np.array(array_2d, dtype=float)
    kernel = np.array(kernel)
    if (mode == 'horizontal'):
        res = conv2d(res, kernel[None, :])
        res = res[:,len(kernel)-1:]
    if (mode == 'vertical'):
        res = conv2d(res, kernel[:, None])
        res = res[len(kernel)-1:,:]
    return res


# In[5]:

def fried_model_gradient(image):
    tmp = convolve_2d(image, H_h, mode='horizontal')
    X = convolve_2d(tmp, H_l, mode='vertical')
    tmp = convolve_2d(image, H_l, mode='horizontal')
    Y = convolve_2d(tmp, H_h, mode='vertical')
    return X, Y


# In[6]:

def process_next_X (prevX):#Не работает, неправильный алгоритм
    tmp1 = convolve_2d(prevX, H_l, mode='horizontal')
    tmp2 = convolve_2d(tmp1, H_l, mode='horizontal')
    return down_sample(np.sqrt(2) * convolve_2d(tmp2, np.array([1,0,1]) / np.sqrt(2), mode='vertical'), 2)


# In[7]:

def process_next_Y (prevY):
    tmp1 = convolve_2d(prevY, np.array([1,0,1]) / np.sqrt(2), mode = 'horizontal')
    tmp2 = convolve_2d(tmp1, H_l, mode = 'vertical')
    return np.sqrt(2) * down_sample(convolve_2d(tmp2, H_l, mode = 'vertical'), 2)


# In[8]:

def process_next_HH (X):
    tmp1 = convolve_2d(X, H_l, mode = 'horizontal')
    tmp2 = convolve_2d(tmp1, H_l, mode = 'horizontal')
    tmp3 = convolve_2d(tmp2, np.array([1,0,-1]) / np.sqrt(2), mode = 'vertical')
    return np.sqrt(2) * down_sample(tmp3, 4)


# In[9]:

def process_left_quadrant (grad_X, grad_Y):
    M = int(np.log2(len(grad_X)))
    X = dict({M : grad_X})
    Y = dict({M : grad_Y})
    HL = dict({M - 1 : down_sample(X[M], 2)})
    LH = dict({M - 1 : down_sample(Y[M], 2)})
    HH = dict()
    for k in range(2, M + 1)[::-1]:
        print (k)
        X[k - 1] = process_next_X(X[k])
        Y[k - 1] = process_next_Y(Y[k])
        LH[k - 2] = down_sample(Y[k - 1], 2)
        HL[k - 2] = down_sample(X[k - 1], 2)
        HH[k - 2] = process_next_HH(X[k])
    return LH, HL, HH


# In[12]:

G_l = H_l
G_h = -H_h


# In[13]:

def up_2(array):
    return np.hstack([array[:, None], np.zeros((len(array), 1))]).ravel()


# In[163]:

def UP_2(array):#работает
    shape = array.shape
    array = array.reshape(shape[0] * shape[1])
    array = up_2(array)
    array = array.reshape(shape[0], shape[1] * 2)
    shape = array.T.shape
    array = array.T.reshape(shape[0] * shape[1])
    array = up_2(array)
    array = array.reshape(shape[0], shape[1] * 2)
    return array.T


# In[162]:

def convolve_2d_syn(array_2d, kernel, mode):#работает
    if (mode == 'horizontal'):
        array_2d = np.column_stack((np.zeros(array_2d.shape[1]), array_2d))
        return convolve_2d(array_2d, kernel, 'horizontal')[:,:-1]
    if (mode == 'vertical'):
        array_2d = np.row_stack((np.zeros(array_2d.shape[1]), array_2d))
        return convolve_2d(array_2d, kernel, 'vertical')[:-1,:]


# In[166]:

def Hudgin_gradien_model(image):#работает
    X = np.sqrt(2) * convolve_2d(image, H_h, 'horizontal')
    Y = np.sqrt(2) * convolve_2d(image, H_h, 'vertical')
    return X, Y


# In[167]:

def get_HH(X_H, Y_H):#works
    coef = np.sqrt(2) * 0.25
    tmp1 = convolve_2d(X_H, H_h, 'vertical')
    tmp2 = convolve_2d(Y_H, H_h, 'horizontal')
    return down_sample(tmp1 + tmp2,2) * coef


# In[160]:

def reconstract_one_step(LL, LH, HL, HH):#Работает
    tmp = convolve_2d_syn(UP_2(LL), G_l, 'vertical')
    tmp2 = convolve_2d_syn(UP_2(LH), G_h, 'vertical')
    res1 = (tmp + tmp2)
    tmp_ = convolve_2d_syn(UP_2(HL), G_l, 'vertical')
    tmp2_ = convolve_2d_syn(UP_2(HH), G_h, 'vertical')
    res2 = tmp2_+ tmp_
    res1 = convolve_2d_syn(res1, G_l, 'horizontal')
    res2 = convolve_2d_syn(res2, G_h, 'horizontal')
    return res1 + res2


# In[10]:

exmpl = np.array(([[5, 7, 5, 4, 6, 0, 0, 6],
       [5, 4, 9, 3, 6, 9, 7, 0],
       [4, 1, 1, 5, 8, 0, 5, 7],
       [1, 8, 9, 2, 4, 4, 9, 9],
       [0, 4, 3, 2, 7, 1, 0, 4],
       [3, 7, 7, 7, 1, 9, 0, 3],
       [9, 5, 3, 9, 8, 7, 5, 9],
       [2, 0, 7, 6, 2, 8, 8, 2]]))


# In[ ]:



