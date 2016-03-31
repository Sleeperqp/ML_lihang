#coding=utf-8
import numpy as np
from math import exp, fabs


class LR:
    def __init__(self):
        pass
    # 根据类型得到函数类型
    # logic为逻辑函数
    # linear为线性函数
    def f(self, x, w, LR_type):
        # print w, x
        if LR_type == "logic":
            return  1.0/(exp(np.dot(x, w))+1)
        elif LR_type == "linear":
            return np.dot(x, w)
        else:
            -1

    # 迭代更新
    # 采用梯度下降法
    # 逻辑回归为采用最大似然估计
    # 迭代公式为 \theta += \sum{(f(x)-y)*\theta}
    # 线性回归采用平方损失函数
    # 迭代公式为 \theta -= \summ{(f(x)-y)*\theta}
    def iteration_update(self, w, dataset, labels, LR_type, iterats, alpha, stop_condition):
        itr = 0
        while(itr < iterats):
            delta = 0.0
            for idx in range(0, len(dataset)):
                # print idx, dataset[idx], labels[idx]
                f_value = self.f(dataset[idx], w, LR_type)
                # print f_value
                delta += (f_value - labels[idx])*dataset[idx]
            if sum(map(lambda  x:fabs(x), delta)) < stop_condition:
                break
            delta /= float(len(dataset))
            if self.LR_type == "logic":
                w +=  alpha*delta
            elif self.LR_type == "linear":
                w -= alpha*delta
            itr += 1
        self.w =  w
    # 预测
    def predict(self, x):
        pre = x
        x = [1] + x
        p_1 = self.f(x, self.w, LR_type=self.LR_type)
        if self.LR_type == "logic":
            print str(pre)+"为类别1的概率:"+str(p_1)
            if p_1 >= 0.5:
                return 1
            else:
                return 0
        elif self.LR_type == "linear":
            print str(pre)+"预测值为:"+str(p_1)
            return p_1

    # 训练
    def train(self, dataset, labels, LR_type="logic", iterats=50000, alpha=0.1, stop_condition=1e-15):
        self.LR_type = LR_type
        b = np.ones(len(dataset))
        dataset = np.c_[b, dataset]
        w = np.zeros(dataset.shape[1])
        self.iteration_update(w, dataset, labels, LR_type, iterats, alpha, stop_condition)
        print "finish train."

    # 测试用例
    def test(self):
        dataset  = np.array([[1, 1], [0, 0]])
        b = np.ones(len(dataset))
        dataset = np.c_[b, dataset]
        # print dataset
        labels = [1,0]
        w = [0, 0, 0]
        self.iteration_update(w, dataset, labels)
        print self.w
        print self.predict([1, 0])

if __name__=="__main__":
    lr = LR()
    dataset  = np.array([[1, 1], [0, 0]])
    labels = [1,0]
    lr.train(dataset, labels,LR_type="logic")
    label = lr.predict([0, 0])
    print "[0, 0]的类别为",label