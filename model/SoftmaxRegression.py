#coding=utf-8
from time import time

import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import math
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split


class SR(object):
    def __init__(self, input_size, output_size):
        self.w = np.random.randn(input_size + 1, output_size) / math.sqrt(input_size + 1)
        self.pre_w = np.zeros(shape=(input_size + 1, output_size))
        self.w_cache =  np.zeros(shape=(input_size + 1, output_size))
        self.eps = 1e-9
        # self.w = np.zeros(shape=(input_size + 1, output_size))
        self.input_size = input_size
        self.output_size = output_size

    def mod(self, a, b):
        return a / b

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference

    def update(self, data, label, gama=0.2):
        # wx+b: batch x input * input x output = batch x output
        output = np.dot(data, self.w)
        # ps : batch x output
        ps = self.softmax(output)
        for idx in range(0, len(ps)):
            ps[idx][int(label[idx])] -= 1.0

        delta = (1.0 / len(data)) * np.dot(data.T, ps)
        # print 'delta:', delta
        # 比较不同的GD优化方法
        # sgd： 0.292571428571
        # Delta = self.alpha * delta + 0.3 * self.w

        # adagrad
        self.w_cache = self.w_cache + delta**2
        Delta = self.alpha * delta/(np.sqrt(self.w_cache)+self.eps)
        # Momentum:
        # Delta = gama * self.pre_w + self.alpha * delta
        # self.pre_w = Delta


        self.w -= Delta

        # print 'w:', self.w
        pass

    def Nesterov_update(self, data, label, gama=0.9):
        w = self.w - gama*self.pre_w
        output = np.dot(data, w)
        ps = self.softmax(output)
        for idx in range(0, len(ps)):
            ps[idx][int(label[idx])] -= 1.0
        delta = (1.0 / len(data)) * np.dot(data.T, ps)

        # Nesterov
        Delta = gama * self.pre_w + self.alpha * delta
        self.pre_w = Delta
        self.w -= Delta


    def train(self, datas, labels, X_test, y_test, batch_num=100, alpha=0.1, iterats=25, stop_condition=1e-5):
        self.batch = batch_num
        self.alpha = alpha

        datas = np.c_[datas, np.ones(len(datas))]
        iter_num = (len(datas) + batch_num - 1) / batch_num

        xs = []
        acc = []
        idx = 0
        for i in range(0, iterats):
            print 'iter:', i
            for j in range(0, iter_num):
                start = j * batch_num
                end = min((j + 1) * batch_num, len(datas))
                data = datas[start: end]
                label = labels[start: end]
                self.update(data, label)
                # self.Nesterov_update(data, label)
                xs.append(idx)
                acc.append(self.test(X_test, y_test, o='no_output'))
                idx += 1
        pl.plot(range(0, len(acc)), acc)  # use pylab to plot x and y
        pl.show()
        pass

    def predict(self, data):
        data = np.c_[data, np.ones(len(data))]
        output = np.dot(data, self.w)
        ps = self.softmax(output)
        # exp_output = np.exp(output)
        # ps = exp_output / (np.tile(np.sum(exp_output, axis=1), self.output_size).reshape(len(data), self.output_size))

        return np.argmax(ps, axis=1)

    def test(self, datas, label, o='output'):
        predict_label = self.predict(datas)
        # print predict_label
        error = 0
        for idx in range(0, len(label)):
            if (int(predict_label[idx]) != int(label[idx])):
                error += 1
        if o=='output':
            print 'error:', error / float(len(datas))
        return error / float(len(datas))

if __name__ == '__main__':
    sr = SR(784, 10)
    mnist = fetch_mldata('MNIST original')
    n_samples = len(mnist.data)
    datas = mnist.data.reshape((n_samples, -1))
    print datas.shape
    width = datas.shape[1]
    print width
    for idx in range(0, len(datas)):
        for j in range(0, width):
            if datas[idx][j] > 0:
                datas[idx][j] = 1
    labels = mnist.target

    X_train, X_test, y_train, y_test = train_test_split(datas, labels, test_size=0.05, random_state=42)
    print 'starting train.....'
    now = time()
    sr.train(X_train, y_train, X_test, y_test)
    train_time = time() - now
    print 'finish train :', train_time, 's'
    sr.test(X_test, y_test)
    sr.test(X_train, y_train)
