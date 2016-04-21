# encoding: utf-8

"""
@version: 1.0
@author: sleeper_qp
@contact: qipeng1992@gmail.com
@software: PyCharm Community Edition
@file: SVC.py
@time: 2016/4/19 20:31
"""
import numpy as np


class SVC(object):
    def __init__(self, c=10):
        self.N = 0
        self.X = None
        self.y = None
        self.w = None
        self.wn = 0
        self.K = np.zeros((self.N, self.N))
        self.a = np.zeros((self.N, 1))
        self.b = 0
        self.C = c

    # 核函数
    @staticmethod
    def kernel_function(x1, x2):
        return np.dot(x1, x2)

    # 核矩阵
    def kernel_matrix(self, x):

        for i in range(0, len(x)):
            for j in range(i, len(x)):
                self.K[j][i] = self.K[i][j] = self.kernel_function(x[i], x[j])

    # 根据对偶问题得到w
    def get_w(self):
        ay = self.a * self.y
        w = np.zeros((1, self.wn))
        # print w
        for i in range(0, self.N):
            w += self.X[i] * ay[i]

        return w

    # 计算更新B
    def get_b(self, a1, a2, a1_old, a2_old):
        y1 = self.y[a1]
        y2 = self.y[a2]
        a1_new = self.a[a1]
        a2_new = self.a[a2]

        b1_new = -self.E[a1] - y1 * self.K[a1][a1] * (a1_new - a1_old) - y2 * self.K[a2][a1] * (
            a2_new - a2_old) + self.b
        b2_new = -self.E[a2] - y1 * self.K[a1][a2] * (a1_new - a1_old) - y2 * self.K[a2][a2] * (
            a2_new - a2_old) + self.b
        if (0 < a1_new) and (a1_new < self.C) and (0 < a2_new) and (a2_new < self.C):
            return b1_new[0]
        else:
            return (b1_new[0] + b2_new[0]) / 2.0

    # 判别函数g(x)
    def g_x(self, x):
        return np.dot(self.w, x) + self.b

    def fabs(self, x):
        if x < 0:
            return -x
        return x

    # 计算更新拉格朗日乘子的E
    def get_E(self):
        E = np.zeros((self.N, 1))
        for i in range(0, self.N):
            E[i] = self.g_x(self.X[i]) - self.y[i]
        return E

    # 判断样本点是否满足kkt条件
    def satisfy_kkt(self, a):
        idx = a[1]
        if a[0] == 0:
            if self.g_x(self.X[idx]) > 1:
                return 1
            return 0
        elif a[0] < self.C:
            if self.g_x(self.X[idx]) == 1:
                return 1
            return 0
        elif a[0] == self.C:
            if self.g_x(self.X[idx]) < 1:
                return 1
            return 0
        return 0

    # 选择第一个拉格朗日乘子
    # 返回拉格朗日乘子的下标
    def chose_first_a(self):
        # 首先在边界上的乘子对应的样本是否满足kkt条件
        data_on_bound = filter(lambda a: ((a[0] > 0) and (a[0] < self.C)),
                               map(lambda a, b: (a, b), self.a, range(0, len(self.a))))
        # print data_on_bound
        if len(data_on_bound) == 0:
            for data in map(lambda a, b: (a, b), self.a, range(0, len(self.a))):
                if self.satisfy_kkt(data) != 1:
                    return data[1]
        else:
            # 此处为第一个不满足kkt条件的点即返回
            for data in data_on_bound:
                if self.satisfy_kkt(data) != 1:
                    return data[1]
        return -1

    # 选择第二个拉格朗日乘子对应的样本下标
    def chose_second_a(self, first_a_idx):
        max = 0
        id = -1
        for i in range(0, self.N):
            if i != first_a_idx and (max < self.fabs(self.E[i] - self.E[first_a_idx])):
                max = self.fabs(self.E[i] - self.E[first_a_idx])
                id = i
        return id

    # 拉格朗日乘子的裁剪函数
    def clip_func(self, a_new, a1_old, a2_old, y1, y2):

        if (y1 == y2):
            L = max(0, a1_old + a2_old - self.C)
            H = min(self.C, a1_old + a2_old)
        else:
            L = max(0, a2_old - a1_old)
            H = min(self.C, self.C + a2_old - a1_old)
        if a_new < L:
            a_new = L
        if a_new > H:
            a_new = H
        return a_new

    # 更新a1,a2
    def update_a(self, a1, a2):
        tau = self.K[a1][a1] + self.K[a1][a1] - 2 * self.K[a1][a2]
        # 更新公式
        # a2_new = a2_old + (y2*(E2-E1))/tau)
        a2_new = self.a[a2] + (self.y[a2] * (self.E[a2] - self.E[a1]) / tau)
        a2_new_cliped = self.clip_func(a2_new, self.a[a1], self.a[a2], self.y[a1], self.y[a2])
        a1_new = self.a[a1] + self.y[a1] * self.y[a2] * (self.a[a2] - a2_new_cliped)
        self.a[a1] = a1_new
        self.a[a2] = a2_new_cliped

    # 模型训练函数
    # 输入: 数据集样本点(x,y)
    def train(self, x, y, max_iternum=1):
        # 初始化
        x_len = len(x)
        self.X = x
        self.N = x_len
        self.wn = len(x[0])
        self.y = np.array(y).reshape((self.N, 1))
        # 核矩阵
        self.K = np.zeros((self.N, self.N))
        self.kernel_matrix(x)
        # b
        self.b = 0
        # 拉格朗日乘子
        self.a = np.zeros((self.N, 1))
        self.w = self.get_w()
        self.E = self.get_E()
        for i in range(0, max_iternum):
            # 选择不满足kkt条件的拉格朗日乘子
            # 因为a,w,b均满足kkt条件时原始问题与对偶问题的解相等且a是对偶问题的解,w,b为原始问题的解
            first_a = self.chose_first_a()
            print first_a
            if first_a == -1:
                break
            second_a = self.chose_second_a(first_a)
            print second_a
            a1_old = self.a[first_a]
            a2_old = self.a[second_a]
            # 更新拉格朗日乘子 
            self.update_a(first_a, second_a)
            print self.a[first_a], self.a[second_a]
            # 更新W与B
            self.w = self.get_w()
            self.b = self.get_b(first_a, second_a, a1_old, a2_old)
            # 更新E
            self.E = self.get_E()
        print self.a


if __name__ == "__main__":
    svc = SVC()
    X = np.array([(1.2, 4), (3, 5), (6, 13.2), (2.1, 1), (17, 5), (4, 2)])
    Y = np.array([-1, -1, -1, 1, 1, 1])
    svc.train(X, Y)
