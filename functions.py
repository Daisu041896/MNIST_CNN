import numpy as np
import pandas as pd
from keras.datasets import mnist

#----sigmoid----------
def sigmoid(x):
    x = np.where(x<-20, 0, 1/(1+np.exp(-x)))
    return x

def sigmoid_diff(x):
    y = 1/(1+np.exp(-x))
    return y*(1-y)
#-------------------

#----softmax--------
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def softmax_batch(x):
    batch_size = x.shape[0]
    for j in range(batch_size):
        softmaxed = softmax(x[j])
        if j == 0:
            y_save = softmaxed.reshape(1, softmaxed.shape[0])
            # print("y_save when j == 0", y_save)
            # print("x:", x)
        else:
            # print("type_y_save", type(y_save))
            # print("softmax(x[j])", softmax(x[j]))
            y_save = np.concatenate([y_save, softmax(x[j]).reshape(1, softmaxed.shape[0])])
            # print("y_save when j!= 0", y_save)
    return  y_save


def softmax_diff(x):
    return np.dot(np.diag(x)-np.dot(x, x.T), np.ones(x.shape[0]))
#-------------------

#---------relu----------
def relu(x):
    return np.maximum(0, x)
def relu_diff(x):
    return np.where(x>0, 1, 0)
#-------------------

def batch_normalization(x):
    # print("====batch_normalization====")
    # print("x")
    ave = np.mean(x, axis = 0).astype("float32")
    std_normal = np.std(x, axis = 0).astype("float32")
    # print("ave:", ave)
    # print("std_normal:", std_normal+0.01)
    # print("std_normal == 0", std_normal == 0)
    y = (x-ave)/(std_normal+0.001).astype("float32")
    # print("y", y)
    # print("====end _batch normal =====")
    return y

def one_hot(t, class_num = 10):
    num_zero = np.zeros([class_num])
    num_zero[t] = 1
    return num_zero

def one_hot_batch(t, class_num = 10):
    y = []
    for i in range(t.shape[0]):
        num_zero = np.zeros([class_num])
        num_zero[t[i]] = 1
        y.append(num_zero)
    y = np.array(y)
    return y

def checking(t, x):
    if np.argmax(x) == np.argmax(t):
        return 1
    else:
        return 0

def checking_batch(x, t):
    batch_size = x.shape[0]
    count = 0
    for j in range(batch_size):
        count += checking(x[j], t[j])
    return count


def cross_entropy_errror(t, y):
    delta = 1e-7;
    return -np.sum(t*np.log(y+delta))

def cross_entropy_errror_batch(t, y):
    batch_size = t.shape[0]
    loss = 0
    for j in range(batch_size):
        loss += cross_entropy_errror(t[j],y[j])
    return loss/batch_size

def update_weight(w, dw, lr):
    for i in range(len(w)):
        w[i] = w[i]-lr*dw[i]
    return w
#----まえに結合していく--------
def affine_forward(x, w, b):
    y = np.dot(w, x) + b
    return y

def affine_backward(w, y):
    x = np.dot(w.T, y)
    return x

def affine_weight(y, x):
    dw = np.dot(y.reshape(y.shape[0], 1), x.reshape(1, x.shape[0]))
    return dw
#------------------------


def Conv_forward(x, w, pad = 0, stride = 1):
    filter_dim = w.shape[0]
    if filter_dim %2 ==0:
        d = int(filter_dim/2) #filterを前に入れている
    else:
        d = int((filter_dim-1)/2)
    x_pad = np.pad(x,[(pad),(pad)],"constant")#paddingしている画像を生成
    # print("x.shape", x.shape)
    # print("x_pad.shape", x_pad.shape)
    height, width = x_pad.shape[0], x_pad.shape[1] #height, widthを保存
    count_i = [] #x方向のlist
    if filter_dim %2 ==0:
        for i in range(d, height-d+1, stride):
            count_j = []#y方向のlist
            for j in range(d, width-d+1, stride):
                # print("i:", i, "j:", j)
                count_j.append(np.sum(x_pad[i-d:i+d, j-d:j+d]*w)) #filterをかけていって、畳み込んでいる
            count_i.append(count_j)
    else:
        for i in range(d, height-d, stride):
            count_j = []#y方向のlist
            for j in range(d, width-d, stride):
                count_j.append(np.sum(x_pad[i-d:i+d+1, j-d:j+d+1]*w)) #filterをかけていって、畳み込んでいる
            count_i.append(count_j)
    y = np.array(count_i)
    # print("y.shape in Conv:", y.shape)
    return y

def Conv_backward(y,w, pad = 0, stride = 1):
    # print("Conv_backward")
    # print("w:", w)
    w_shape = np.array(w.shape)
    # w_shape  = w_shape -1 #0paddingするようにしている
    w = np.flip(w, axis = 0)#np.flipして、180度回転
    w = np.flip(w, axis = 1)#np.flipして、180度回転
    # print("w:", w)
    # print("y.shape", y.shape)
    # y = np.pad(y, w_shape-1, "constant")#paddingして0を増やしていく
    # print("y_pad:", y.shape)
    # print("pad: ", pad, "\t stride", stride)
    x = Conv_forward(y, w, pad, stride)
    # print("x.shape", x.shape)
    return x

def flatten_forward(x):
    y = x.flatten()
    return y
#--------------------

def flatten_backward(x, dims):
    y = x.reshape(dims)
    return y

def pooling_forward(x, filter_dim, stride = 0):
    # print("x.shape in pooling_forward:", x.shape)
    if stride == 0:
        stride = filter_dim
    d = int((filter_dim-1)/2) #filterを前に入れている
    height, width = x.shape[0], x.shape[1] #height, widthを保存
    count_i = [] #x方向のlist
    count_max_index_i = []
    for i in range(d, height-d, stride):
        count_j = []#y方向のlist
        count_max_index_j = []
        for j in range(d, width-d, stride):
            small_portion = x[i-d:i+d+1, j-d:j+d+1]
            pixel = np.max(small_portion)
            index = np.unravel_index(np.argmax(small_portion), small_portion.shape)
            count_j.append(pixel) #最大値を取っている
            count_max_index_j.append(index)#最大値のindexを記録している
        count_max_index_i.append(count_max_index_j)
        count_i.append(count_j)
    y = np.array(count_i)
    y_index = np.array(count_max_index_i)
    # print("y_index.shape in pooling_forward: ", y_index.shape)
    # print("y.shape in pooling_forward: ", y.shape)
    return y, y_index, x.shape

def pooling_backward(y, y_index, x_shape, filter_dim, stride):
    d = filter_dim -1

    # print("y.shape", y.shape, "stride", stride)
    # print("y_index.shape in pooling_backward: ", y_index.shape)
    # print("y.shape in pooling_backward: ", y.shape)
    # print("y.shap&stride:", np.array(y.shape)*stride)

    base = np.zeros((np.array(x_shape)))
    height, width = y.shape[0], y.shape[1] #height, widthを保存
    count_i = [] #x方向のlistx
    count_max_index_i = []
    for i in range(0, height):
        count_j = []#y方向のlist
        count_max_index_j = []
        index_img_i  = i*stride
        # print("index_img_i:", index_img_i)
        for j in range(0, width):
            # print("i:", i, "j:", j)
            index_img_j = j * stride
            pix_i, pix_j = y_index[i, j][0], y_index[i, j][1]
            # print("index_img_i+pix_i:", index_img_i+pix_i, "index_img_j+pix_j:", index_img_j+pix_j)
            # print("i, j: ", i, j)
            # print("index_img_i", index_img_i, "index_img_j", index_img_j)
            # print("pix_i", pix_i, "pix_j", pix_j)
            base[index_img_i+pix_i][index_img_j+pix_j] = y[i, j]
    return base





'''
#-------affineして前に渡す----------
def affine_forward(self, x, w, b):
    y = np.dot(w, x) + b
    return y
#-----------------

#-----------------
def Conv_forward(self, x, w, pad, filter_dim, stride = 1):
    d = int((filter_dim-1)/2) #filterを前に入れている
    x_pad = np.pad(x,[(pad,pad),(pad, pad)],"constant")#paddingしている画像を生成
    height, width = x_pad.shape[0], x_pad.shape[1] #height, widthを保存
    count_i = [] #x方向のlist
    for i in range(d, height-d, stride):
        count_j = []#y方向のlist
        for j in range(d, width-d, stride):
            count_j.append(np.sum(img_np[i-d:i+d+1, j-d:j+d+1]*w)) #filterをかけていって、新しいものを作っている
        count_i.append(count_j)
    y = np.array(count_i)
    return y
#-----------------

#-------flattenを行って、前に進める---------
def flatten_forward(self, x):
    y = x.flatten()
    return y
#------------------

#----sigmoid----------
def sigmoid(self, x):
     return  1/(1+exp(-x))

def sigmoid_diff(self, x):
    return sigmoid(x)*(1-sigmoid(x))
#-------------------

#----softmax--------
def softmax(self, x):
    return np.exp(x)/np.sum(np.exp(x))

def softmax_diff(self):
    x = self.last_y
    return np.dot(np.diag(x)-np.dot(x, x.T), np.ones(x.shape[0]))
#-------------------

#---------relu----------
def relu(self, x):
    return np.maximum(0, x)
def relu_diff(self, x):
    return np.where(x>0, 1, 0)
#-------------------

#-----------------
def cross_entropy_errror(self):
    delta = 1e-7;
    return -np.sum(self.t*np.log(self.last_y+delta))
#-----------------
'''
