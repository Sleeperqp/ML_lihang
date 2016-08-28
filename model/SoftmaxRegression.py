from time import time

import numpy as np
import math
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split


class SR(object):
    def __init__(self, input_size, output_size):
        self.w = np.random.randn(input_size + 1, output_size) / math.sqrt(input_size + 1)
        # self.w = np.zeros(shape=(input_size + 1, output_size))
        self.input_size = input_size
        self.output_size = output_size

    def mod(self, a, b):
        return a / b

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference

    def update(self, data, label):
        # wx+b: batch x input * input x output = batch x output
        # print 'data:', data
        output = np.dot(data, self.w)
        # print output
        # print 'output', output
        ps = self.softmax(output)
        # exp_output = np.exp(output- np.max(output)/2)
        # print 'exe_output:', exp_output
        # print exp_output.shape
        # print (np.tile(np.sum(exp_output, axis=1), self.output_size).reshape(len(data), self.output_size)).shape
        # ps = exp_output / (np.tile(np.sum(exp_output, axis=1), self.output_size).reshape(len(data), self.output_size))
        for idx in range(0, len(ps)):
            ps[idx][int(label[idx])] -= 1.0
        # ps : batch x output
        #  intput x batch * batch x output = input x output
        # print 'ps:', ps
        delta = (1.0 / len(data)) * np.dot(data.T, ps)
        # print 'delta:', delta
        self.w -= self.alpha * delta + 0.05 * self.w

        # print 'w:', self.w
        pass

    def train(self, datas, labels, batch_num=200, alpha=0.03, iterats=500, stop_condition=1e-5):
        self.batch = batch_num
        self.alpha = alpha

        datas = np.c_[datas, np.ones(len(datas))]
        iter_num = (len(datas) + batch_num - 1) / batch_num

        for i in range(0, iterats):
            print 'iter:', i
            for j in range(0, iter_num):
                start = j * batch_num
                end = min((j + 1) * batch_num, len(datas))
                data = datas[start: end]
                label = labels[start: end]
                self.update(data, label)
        pass

    def predict(self, data):
        data = np.c_[data, np.ones(len(data))]
        output = np.dot(data, self.w)
        ps = self.softmax(output)
        # exp_output = np.exp(output)
        # ps = exp_output / (np.tile(np.sum(exp_output, axis=1), self.output_size).reshape(len(data), self.output_size))

        return np.argmax(ps, axis=1)

    def test(self, datas, label):
        predict_label = self.predict(datas)
        print predict_label
        error = 0
        print type(predict_label)
        print type(label)
        for idx in range(0, len(label)):
            if (int(predict_label[idx]) != int(label[idx])):
                error += 1
        print 'error:', error / float(len(datas))


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

    X_train, X_test, y_train, y_test = train_test_split(datas, labels, test_size=0.33, random_state=42)
    print 'starting train.....'
    now = time()
    sr.train(X_train, y_train)
    train_time = time() - now
    print 'finish train :', train_time, 's'
    sr.test(X_test, y_test)
