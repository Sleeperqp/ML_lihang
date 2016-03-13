#coding=utf-8
import matplotlib.pyplot as plt
import  math
import numpy as np
from datastruct.kdtree import kdTree

def KNN_K(X=[(1,1),(1,3),(4,1),(6,4),(0,0),(2,4),(3.5,5),(7,6),(5,2)], Y=[0,0,0,0,1,1,1,1,1]):

    #得到每个点对应不同k值所得到的类别,加入散点图数组中
    k = range(1, 6)
    test_x = []
    test_y = []
    for i in k:
        test_x.append([[],[]])
        test_y.append([[],[]])
    for i in range(0, 100):
        i = i/10.0
        for j in range(0, 100):
            j = j/10.0
            #得到点(i, j)到所有训练集点的(距离,类别)。然后根据距离排序,得到前K近点所对应的类别
            dis = map(lambda x, idx:(L_p(x,(i,j), p=2), Y[idx]),X, range(0, 9))
            dis = sorted(dis)
            for h in k:
                topK = map(lambda x: x[1], dis[:h])
                # print topK
                n0 = 0
                n1 = 0
                for m in topK:
                    if m == 0:
                        n0 += 1
                    elif m == 1:
                        n1 += 1
                if n0 < n1:
                    test_x[h-1][1].append(i)
                    test_y[h-1][1].append(j)
                else :
                    test_x[h-1][0].append(i)
                    test_y[h-1][0].append(j)
    for i in k:
        plt.xlabel(u'X0')
        plt.ylabel(u'X1')

        #根据不同K值得到需要绘制点的类别
        #根据训练集为训练图的散点图配色
        test0_x = test_x[i-1][0]
        test0_y = test_y[i-1][0]
        test1_x = test_x[i-1][1]
        test1_y = test_y[i-1][1]
        plt.scatter(test0_x,test0_y, s=10, c='red')
        plt.scatter(test1_x,test1_y, s=10, c='green')

        #根据训练集为训练图的散点图配色
        label0_x = []
        label0_y = []
        label1_x = []
        label1_y = []
        for idx in range(0, len(X)):
            if Y[idx] == 0:
                label0_x.append(X[idx][0])
                label0_y.append(X[idx][1])
            if Y[idx] == 1:
                label1_x.append(X[idx][0])
                label1_y.append(X[idx][1])

        plt.scatter(label0_x, label0_y, s=100,c='red')
        plt.scatter(label1_x, label1_y, s=100,c='green')

        plt.show()
def L_p(data1, data2, p=2):
    if len(data1) != len(data2):
            print '长度不一致'
            return -1
    dis = 0
    for i in range(0, len(data1)):
        dis += pow(math.fabs(data1[i]-data2[i]), p)
    #print  dis
    return  math.pow(dis, 1.0/p)

class KNN(object):
    def __init__(self):
        self.kdtree = kdTree()
    #读取(X,y)，建立kdtree
    def readdataset(self, X ,Y):
        self.kdtree.buidTree(X)
        self.Y = Y
    #输入X,找到与它最近的数据点，预测其类别
    def predict(self,data):
        mindis, idx = self.kdtree.findMindis(data)
        if idx != -1:
            print data,"的类别为",self.Y[idx]

if __name__=="__main__":
    KNN_K()
    X = np.array([[2,3,1],[5,4,1],[9,6,1],[4,7,1],[8,1,1],[7,2,1]])
    Y = [0, 0, 1, 0, 1, 2]
    knn = KNN()
    knn.readdataset(X, Y)
    knn.predict([7, 0.9, 1])
    