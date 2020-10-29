'''
2019/8/3
CNN_MNISTを一から書いてみる
'''
import numpy as np
import pandas as pd
from keras.datasets import mnist
from functions import *
# ''
#
class CNN_MNIST:
    def __init__(self):
        self.layers_param = {} #layerの種類とそのパラメータを順番に保存
        self.layer_num = 0 #layerの総数を保存
        self.layer_name = []#layerの名前のみを保存
        self.x = {}
        self.dims = []
    #最初のベクトルを決定
    def start(self, x):
        self.dims.append(np.array(x.shape))

    def layers_definition(self, name):
        layer_name =name+str(self.layer_num)
        self.layer_name.append(layer_name)
        self.layers_param[layer_name] = {}
        params = self.layers_param[layer_name]
        self.layer_num += 1
        print("self.dims[-1]", self.dims[-1])
        # print("type(self.dims[-1])", type(self.dims[-1]))
        # print("self.dims[-1].shape[0]", self.dims[-1].shape)
        return params

    #Affineを定義（モデルの定義）
    def Affine(self, next_dim):
        params = self.layers_definition("affine")
        i = self.dims[-1][0]
        params["w"]  = np.random.normal(0, 0.1, (next_dim,  int(self.dims[-1][0]))).astype(np.float32)
        # params["w"]  = 0.01*np.random.rand(next_dim,  int(self.dims[-1][0])).astype(np.float32)
        params["b"]  = np.random.normal(0, 0.1, (next_dim)).astype(np.float32)
        # params["b"]  = 0.01*np.random.rand(next_dim).astype(np.float32)


        self.dims.append(np.array([next_dim]))

    def Conv_gray(self, filter_dim, pad, stride=1):
        params = self.layers_definition("conv_gray")
        params["w"]  = np.random.normal(0, 0.01, (filter_dim, filter_dim)).astype(np.float32)
        # params["w"]  = 0.01*np.random.rand(filter_dim, filter_dim).astype(np.float32)
        params["pad"] = pad
        params["stride"] = stride
        d =pad*2-(filter_dim-1)
        self.dims.append((np.array(self.dims[-1])+d)/stride)


    def Flatten(self):
        params = self.layers_definition("flatten")
        y = np.prod(np.array(self.dims[-1])).astype("float32")
        y = np.array([y])
        # y = y.reshape((y.shape[0]))
        self.dims.append(y)

    def max_pooling(self, filter_dim, stride):
        params = self.layers_definition("max_pool")
        params['filter_dim'] = filter_dim
        params['stride'] = stride
        d = filter_dim-1
        self.dims.append((np.array(self.dims[-1])-d)/stride)

    def types_forward(self, x, i, activation = True):
        layer_name = self.layer_name[i]
        params = self.layers_param[layer_name]
        if "affine" in layer_name:
            y = affine_forward(x, params["w"], params["b"])
            if activation:
                # y = relu(y)
                y = sigmoid(y)
        elif "conv_gray" in layer_name:
            y = Conv_forward(x, params["w"], params["pad"], params["stride"])
            if activation:
                # y = relu(y)
                y = sigmoid(y)
        elif "flatten" in layer_name:
            y = flatten_forward(x)
            params["dim_shape"] = x.shape
        elif "max_pool" in layer_name:
            y, params['y_index'], params['x_shape']  = pooling_forward(x, params['filter_dim'], params['stride'])
        return y

    def forward_propagation_batch(self, x, t, class_num):
        batch_size = x.shape[0]
        self.t = one_hot_batch(t, class_num)
        self.x = {}
        for i in range(len(self.layer_name)):
            if i == 0:
                x_vec = x
            else:
                x_vec = y_save
            self.x[self.layer_name[i]] = x_vec
            for j in range(batch_size):
                if i != len(self.layer_name) -1:
                    next_y = self.types_forward(x_vec[j], i)
                else:
                    next_y = self.types_forward(x_vec[j], i, False)
                if j == 0:
                    y_save = next_y[np.newaxis, ]
                else:
                    y_save = np.concatenate([y_save, next_y[np.newaxis,     ]], axis = 0)
            if i != len(self.layer_name) -1 and ("affine" or "conv_gray" in self.layer_name[i]):
                y_save = batch_normalization(y_save)
        # print("y_save before ", y_save)
        # print("y: ", y)
        self.predicted_t = softmax_batch(y_save)
        # print("softmax(y):", self.predicted_t)
        self.count = checking_batch(self.t, self.predicted_t)
        self.loss = cross_entropy_errror_batch(self.t, self.predicted_t)

    def back_propagation_batch(self):
        batch_size = self.predicted_t.shape[0]
        for j in range(batch_size):
            y = self.predicted_t[j] - self.t[j]
            # print("error kyori"+str(j)+":", y)
            for i in range(len(self.layer_name)-1, -1, -1):
                y = self.types_backward_batch(y,i, j)

    def types_backward_batch(self, y, i, j):
        layer_name  =  self.layer_name[i]
        params = self.layers_param[layer_name]
        x = self.x[layer_name][j]
        if "affine" in layer_name:
            if j == 0:
                params["dw"] = affine_weight(y, x)
                params["db"] = y
            else:
                params["dw"] += affine_weight(y, x)
                params["db"] += y
            y = affine_backward(params["w"], y)
            # y_activate = relu_diff(y)
            y_activate = sigmoid_diff(y)
            y = y*y_activate
        elif "conv_gray" in layer_name:
            if j == 0:
                params["dw"] = Conv_forward(x, y, params["pad"], params['stride'])
            else:
                params["dw"] += Conv_forward(x, y, params["pad"], params['stride'])
            y = Conv_backward(y, params["w"], params["pad"], params["stride"])
            # y_activate = relu_diff(y)
            y_activate = sigmoid_diff(y)
            y = y * y_activate
        elif "flatten" in layer_name:
            y = flatten_backward(y, params['dim_shape'])
        elif "max_pool" in layer_name:
            y = pooling_backward(y,  params['y_index'], params['x_shape'], params['filter_dim'], params['stride'])
        return y

    def updating_weight(self, lr):
        for layer_name in self.layer_name:
            params = self.layers_param[layer_name]
            if "affine" in layer_name:
                params["w"] = params['w'] - lr * params['dw']
                params["b"] = params['b'] - lr* params['db']
            elif "conv_gray" in layer_name:
                params["w"] = params['w'] - lr*params['dw']
        return

    def train(self, epochs, batch_size, lr):
        print("self.dims", self.dims)
        count = 0
        right_count = 0
        class_num = 10
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.astype('float32')   # int型をfloat32型に変換
        X_train /= 255;
        for i in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            batch_imgs = X_train[idx]
            batch_ans  = y_train[idx]
            self.forward_propagation_batch(batch_imgs, batch_ans, class_num)
            self.back_propagation_batch();
            self.updating_weight(lr);
            right_count+=self.count
            count = (i+1)*batch_size
            print("i:",i, "\tloss:", self.loss, "\ttotal_acc:", right_count/count, "\tbatch_acc: ", self.count/batch_size)
            lr = lr*0.99

def main():
    epochs = 30000
    batch_size = 100
    lr = 0.01
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("X_train.shape:", X_train.shape)
    model =CNN_MNIST()
    print("")
    model.start(X_train[0])
    print("X_train[0]", X_train[0].shape)
    # '''
    model.Conv_gray(filter_dim= 3, pad = 1, stride = 1)
    model.max_pooling(filter_dim=3, stride = 1)
    # model.Conv_gray(filter_dim= 5, pad = 2, stride = 1)
    # model.max_pooling(filter_dim=3, stride = 1)
    # model.Conv_gray(filter_dim= 3, pad = 1, stride = 1)
    # model.max_pooling(filter_dim=3, stride = 1)
    '''
    model.Conv_gray(filter_dim= 3, pad = 1, stride = 1)
    model.max_pooling(filter_dim=3, stride = 1)
    '''
    model.Flatten()
    # '''
    model.Affine(400)
    # model.Affine(30)
    model.Affine(300)
    # '''
    model.Affine(10)
    model.train(epochs, batch_size, lr)

if __name__=="__main__":
    main()
