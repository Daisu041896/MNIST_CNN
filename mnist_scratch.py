'''
2019/8/3
MNISTを一から書いてみる
'''
import numpy as np
import pandas as pd
from keras.datasets import mnist
from functions import *
# ''
#
class MNIST:
    def __init__(self):
        self.w = []
        self.b  = []

    #最初のベクトルを決定
    def start(self, x):
        self.x = x;

    #Affineを定義（モデルの定義）
    def Affine(self, next_dim):
        #-------最初の層の定義-------
        if len(self.w) == 0:
            w = np.random.normal(0, 0.1, (next_dim, self.x.shape[0])).astype(np.float32)
            b = np.random.normal(0, 0.1, (next_dim)).astype(np.float32)
            # w = 0.01*np.random.rand(next_dim,  int(self.x.shape[0])).astype(np.float32)
            # b = 0.01*np.random.rand(next_dim).astype(np.float32)
            self.w.append(w)
            self.b.append(b)
        #-------------------

        #---------2層目以降の定義----------
        else:
            w = np.random.normal(0, 0.1, (next_dim,  self.w[-1].shape[0])).astype(np.float32)
            b = np.random.normal(0, 0.1, (next_dim)).astype(np.float32)
            # w = 0.01*np.random.rand(next_dim,  int(self.w[-1].shape[0])).astype(np.float32)
            # b = 0.01*np.random.rand(next_dim).astype(np.float32)
            self.w.append(w)
            self.b.append(b)
        #-------------------

    def show(self):
        for k in range(len(self.w)):
            print("self.w["+str(k)+"]", self.w[k])
            print("self.b["+str(k)+"]", self.b[k])
            print("self.w["+str(k)+"]", self.w[k].shape)
            print("self.b["+str(k)+"]", self.b[k].shape)
    #-------------------

    #------------------
    #AFFINEでyを作っている
    def affine_forward(self, x, t):
        # print("num_zero[t]", num_zero)
        self.t = one_hot(t)
        self.y = []
        self.x = x
        self.y.append(x)
        for i in range(len(self.w)):
            y = np.dot(self.w[i], self.y[i])
            self.b[i] = self.b[i].reshape(self.b[i].shape[0])
            y = y+self.b[i]
            # print("y.shape", y.shape)
            if i != len(self.w)-1:
                y = relu(y)
                self.y.append(y)
        self.last_y = softmax(y)
        self.count = checking(self.t, self.last_y)
        # print("self.t.shape[0]", self.t.shape[0])
        # print("self.last_y", self.last_y)
        self.E = cross_entropy_errror(self.t, self.last_y)
        # print("self.E: ", self.E)

    #AFFINEを戻ってく
    def affine_backward(self):
        self.w_new = []
        self.b_new = []
        y = self.last_y - self.t
        # print("y: ", y)
        for j in range(len(self.w)-1, -1, -1):
            y = y.reshape(y.shape[0], 1)
            if j != len(self.w) - 1:
                y =y* relu_diff(self.y[j+1])

            self.y[j] = self.y[j].reshape(self.y[j].shape[0], 1)
            self.w_new.append(np.dot(y, self.y[j].T))
            self.b_new.append(y.reshape(y.shape[0]))
            y = np.dot(self.w[j].T, y)

        self.w_new.reverse()
        self.b_new.reverse()
    #----------------------

    #-----------------
    def train(self, epochs, batch_size, lr):
        count = 0
        right_count = 0
        loss_ave = 0
        # count_epoch = 0
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.astype('float32')   # int型をfloat32型に変換
        X_train /= 255;
        # y_train = y_train.astype('float32')
        #-------X_trainの2次元画像をベクトル化----------
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
        #------------------------

        for i in range(epochs):
            num_img = i*batch_size
            batch_count = 0
            loss_ave = 0
            for j in range(batch_size):
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                lis_idx = list(idx)

                self.affine_forward(X_train[num_img+j], y_train[num_img+j])
                self.affine_backward()
                self.w = update_weight(self.w, self.w_new, lr)
                self.b = update_weight(self.b, self.b_new, lr)

                right_count += self.count
                batch_count += self.count
                count += 1
                loss_ave += self.E

            print("i:",i, "\tloss:", loss_ave/batch_size, "\ttotal_acc:",right_count/count, "\tbatch_acc: ", batch_count/batch_size)

# '''

def main():
    epochs = 60
    batch_size = 1000
    lr = 0.05
    # '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
    # X_train = X_train.reshape([X_train.shape[0],m])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])

    # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]+X_test.shape[2])
    print("X_train.shape:", X_train.shape)

    model =MNIST()
    model.start(X_train[0])
    # model.Affine(200)
    # model.Affine(100)

    model.Affine(40)
    model.Affine(50)
    model.Affine(10)
    model.train(epochs, batch_size, lr)
    '''
    a1 = np.array([1, 2, 3, 4, 5])
    a2 = np.array([2, 3, 1, 3, 5])
    print(np.dot(a1, a2))
    # '''
    '''
    x = np.array([1, 2, 3])
    w = np.array([[1, -2, -3], [3, 4,5]])
    b = np.ones(w.shape[0])
    y = np.dot(w, x)+b
    z = np.array([3, 4, 7])
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("x.shape: ", x.shape)
    print("w.shape: ", w.shape)
    print("b.shape: ", b.shape)
    print("w:\n ", w)
    print("b:\n ", b)
    print("x: \n", x)
    print("y: \n", y)
    print("x: \n", x/np.sum(x))
    print("x*z: \n", x*z)
    for j in range(4, -1, -1):
        print("j:", j)
    print("X_train.shape",X_train.shape, "y_train.shape:", y_train.shape)

    a = np.arange(4).reshape(2, 2)
    b = np.arange(8).reshape(2, 4)
    lis = [a, b]
    c = np.arange(4, 8).reshape(2, 2)
    d = np.arange(3, 11).reshape(2, 4)
    lis2 = [c, d]
    print(np.maximum(0, w))
    print("np.where(w>0, 1, 0)", np.where(w>0, 1, 0))
    '''

if __name__=="__main__":
    main()
