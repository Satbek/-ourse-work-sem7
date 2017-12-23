
# coding: utf-8

# In[1]:

import numpy as np


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


"""def up_2(array):
    return np.hstack([array[:, None], np.zeros((len(array), 1))]).ravel()

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
        return convolve_2d(array_2d, kernel, 'vertical')[:-1,:]"""


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

# def reconstract_one_step(LL, LH, HL, HH):#Работает
#     tmp = convolve_2d_syn(UP_2(LL), G_l, 'vertical')
#     tmp2 = convolve_2d_syn(UP_2(LH), G_h, 'vertical')
#     res1 = (tmp + tmp2)
#     tmp_ = convolve_2d_syn(UP_2(HL), G_l, 'vertical')
#     tmp2_ = convolve_2d_syn(UP_2(HH), G_h, 'vertical')
#     res2 = tmp2_+ tmp_
#     res1 = convolve_2d_syn(res1, G_l, 'horizontal')
#     res2 = convolve_2d_syn(res2, G_h, 'horizontal')
#     return res1 + res2


# In[10]:

# In[ ]:



# def convolve_2d_syn(array_2d, kernel, mode):#работает
#     if (mode == 'horizontal'):
#         array_2d = np.column_stack((np.zeros(array_2d.shape[1]), array_2d))
#         return convolve_2d(array_2d, kernel, 'horizontal')[:,:-1]
#     if (mode == 'vertical'):
#         array_2d = np.row_stack((np.zeros(array_2d.shape[1]), array_2d))
#        return convolve_2d(array_2d, kernel, 'vertical')[:-1,:]

def GetH_h(pow_):
    res = np.zeros(pow_ + 1)
    res[0] = 1
    res[-1] = -1
    return res / np.sqrt(2)

def GetH_l(pow_):
    res = np.zeros(pow_ + 1)
    res[0] = res[-1] = 1
    return res / np.sqrt(2)



def GetLH(m, Y):
    coef = np.power(np.sqrt(2), (m - 1))
    tmp = Y
    for i  in range (1, m):
        tmp = convolve_2d(tmp, GetH_l(2 ** i), 'horizontal')
    for i in range (m - 1):
        tmp = convolve_2d(tmp, GetH_l(2 ** i), 'vertical')
        tmp = convolve_2d(tmp, GetH_l(2 ** i), 'vertical')
    return coef * down_sample(tmp, 2 ** m)


def GetHL(m, X):
    coef = np.power(np.sqrt(2), (m - 1))
    tmp = X
    for i in range (m - 1):
        tmp = convolve_2d(tmp, GetH_l(2 ** i), 'horizontal')
        tmp = convolve_2d(tmp, GetH_l(2 ** i), 'horizontal')
    for i  in range (1, m):
        tmp = convolve_2d(tmp, GetH_l(2 ** i), 'vertical') 
    return coef * down_sample(tmp, 2 ** m)


def GetHHleft(m, X):
    coef = np.power(np.sqrt(2), (m - 1))
    tmp = X
    for i in range (m - 1):
        tmp = convolve_2d(tmp, GetH_l(2 ** i), 'horizontal')
        tmp = convolve_2d(tmp, GetH_l(2 ** i), 'horizontal')
    tmp = convolve_2d(tmp, GetH_h(2 ** m - 1), 'vertical')
    for i  in range (1, m - 1):
        tmp = convolve_2d(tmp, GetH_l(2 ** i), 'vertical')
    #print (tmp)
    return coef * down_sample(tmp, 2 ** m)