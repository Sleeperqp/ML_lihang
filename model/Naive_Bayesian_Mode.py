#coding=utf-8
import numpy as np
from collections import defaultdict

#朴素贝叶斯模型
class NB(object):
    def __init__(self):
        print "朴素贝叶斯模型"
        self.prior  = {}
        self.conditional = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))

    #模型训练
    #输入: 训练数据集,对应类别集,以及Laplace参数,默认为1,当设置为0时,不是使用拉普拉斯平滑
    def train(self, X=np.array([[1,1],[1,2],[1,2],[1,1],[1,1],[2,1],[2,2],[2,2],[2,3],[2,3],[3,3],[3,2],[3,2],[3,3],[3,3]]),
              Y = [-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1], Laplace=1):
        if len(X) != len(Y):
            print "数据集X, Y无法对应"
            return
        if Laplace < 0:
            print "拉普拉斯平滑参数输入有误,应大于零！"
            return

        num = 1.0*len(X)
        self.X_len = len(X[0])
        n = len(X[0])
        # print groupby(sorted(Y))
        sorted_Y = sorted(Y)
        pre = None
        for i in sorted_Y:
            if i != pre:
                self.prior[i] = 1
                pre = i
            else:
                self.prior[i] += 1

        for i in range(0, int(num)):
            for j in range(0, n):
                # print Y[i], j, X[i][j]
                self.conditional[Y[i]][j][X[i][j]] += 1
        # print "===============条件概率==============="
        for i in self.conditional:
            for j in self.conditional[i]:
                for k in self.conditional[i][j]:
                    self.conditional[i][j][k] = (self.conditional[i][j][k] + Laplace)/(1.*self.prior[i] + Laplace*len(self.conditional[i][j]))
                    # print "Y=",i,j,k,":",self.conditional[i][j][k]
        # print "=============先验===================="
        for i in self.prior:
            self.prior[i] = (self.prior[i]+Laplace)/(num+Laplace*len(self.prior))
            # print self.prior[i]
    #预测输入的数据的类别
    #输入:单条预测数据
    #输出:所预测的类别
    def predict(self, data):
        if len(data) != self.X_len:
            print "输入数据错误"

        max_p = -1.0
        label = None
        for i in self.prior:
            p = self.prior[i]
            # print "***************************"
            # print p
            for j in range(0,self.X_len):
                # print p, i, j, data[j], self.conditional[i][j][data[j]]
                p *= self.conditional[i][j][data[j]]
                # print self.conditional[i][j][data[j]],p
            # print i, p
            if p > max_p:
                max_p = p
                label = i
        return label

    #测试
    def test(self):
        self.train()
        print "[2,1]的类别为",self.predict([2, 1])

if __name__=="__main__":
    nb = NB()
    nb.test()