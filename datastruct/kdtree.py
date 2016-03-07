#coding=utf-8

from tree import Node
import  numpy as np
import math

class kdTree(object):
    def __init__(self):
        self.root = None

    def addNode(self):
        print "add"

    def buildNode(self, nownode, dataset, datalist):
        if len(dataset) == 0 or len(datalist) == 0:
            return
        if len(datalist) == 1:
            nownode.data = dataset[datalist[0]]
            nownode.idx = datalist[0]
            return
        #得到有datalist所指向的集合
        datas = []
        for idx in datalist:
            datas.append(dataset[idx])
        datas = np.array(datas)
        #得到当前深度所需的坐标轴depth%k数据以及对于的下标 (data, idx) 并排序
        datas = sorted(map(lambda x,y:(x,y), datas[:,nownode.depth%len(dataset[0])],datalist))

        #将集合根据中值划分开
        left = []
        right = []
        mid = int(math.ceil((len(datas))/2))

        for i in range(0, len(datas)):
            if i < mid:
                left.append(datas[i][1])
            elif i > mid:
                right.append(datas[i][1])

        nownode.data = dataset[datas[mid][1]]
        nownode.idx = datas[mid][1]
        #递归地建树
        if left and nownode.left == None:
            nownode.left = kdtreeNode(data=-1, depth=nownode.depth+1)
            self.buildNode(nownode.left, dataset, left)
        if right and nownode.right == None:
            nownode.right = kdtreeNode(data=-1, depth=nownode.depth+1)
            self.buildNode(nownode.right, dataset, right)

    #建立kd树
    def buidTree(self, dataset):
        num = len(dataset)
        datalist = range(0, num)
        self.root = kdtreeNode(data=-1, depth=0)
        self.buildNode(self.root, dataset, datalist)

    #遍历kd树
    def dfs(self, node):
        if node == None:
            return
        self.dfs(node.left)
        str = ""
        for i in range(0, node.depth):
            str += '\t'
        print str,node.data,node.idx
        self.dfs(node.right)

    #输入预测点,找到最近的训练集点(下称为数据点):
    #首先，定义全局最小距离nowdis,
    #递归找到数据点所在的超区域
    #   对于每个数据点:
    #       计算当前数据点到预测点的距离predis
    #       predis与nowdis比较，更新nowdis以及对应下标minidx
    #       根据当前维切分,找到与预测点更近的超区域,返回该超区域的最小距离
    #       更新nowdis并更新对应下标minidx
    #       判断相反的超区域是否可能有更近的数据点:通过nowdis与(该数据点与预测点的当前维的距离)判断
    #       若小于nowdis,则说明有可能有数据点离预测点更近
    #       因此进入相反区域,返回该超区域的最小距离
    #       更新nowdis与下标mindix
    def find_nearest(self, nownode, data, nowdis, minidx):
        if nownode == None:
            return  (float('inf'),-1)

        print nownode.data

        a = data[nownode.depth % len(data)]
        b = nownode.data[nownode.depth % len(data)]
        predis = self.distance(data, nownode.data)
        if predis < nowdis:
            nowdis = predis
            minidx = nownode.idx

        if a < b:
            (leftdis, leftidx) = self.find_nearest(nownode.left, data, nowdis, minidx)
            #nowdis = min(predis, leftdis, nowdis)
            if leftdis < nowdis:
                nowdis = leftdis
                minidx = leftidx

            if nowdis >= math.fabs(a-b):
                (rightdis, rightidx) = self.find_nearest(nownode.right, data, nowdis, minidx)
                #nowdis = min(nowdis, rightdis)
                if rightdis < nowdis:
                    nowdis = rightdis
                    minidx = rightidx
        else:
            (rightdis, rightidx) = self.find_nearest(nownode.right, data, nowdis, minidx)
            #nowdis = min(predis, rightdis, nowdis)
            if rightdis < nowdis:
                nowdis = rightdis
                minidx = rightidx
            if nowdis >= math.fabs(a-b):
                (leftdis, leftidx) = self.find_nearest(nownode.left, data, nowdis, minidx)
                #nowdis = min(nowdis, leftdis)
                if leftdis < nowdis:
                    nowdis = leftdis
                    minidx = leftidx

        return  (nowdis, minidx)

    def findMindis(self, data):
        return self.find_nearest(self.root, data, float('inf'), -1)

    def distance(self, data1, data2):
        if len(data1) != len(data2):
            print '长度不一致'
            return -1
        dis = 0
        for i in range(0, len(data1)):
            dis += (data1[i]-data2[i])*(data1[i]-data2[i])
        #print  dis
        return  math.sqrt(dis)

    def test(self):
        dataset = np.array([[2,3,1],[5,4,1],[9,6,1],[4,7,1],[8,1,1],[7,2,1]])
        self.buidTree(dataset)
        print "++++++++++++++++++++++++++++++++++++++++++"
        self.dfs(self.root)

        print self.find_nearest(self.root, [2, 5, 1], float('inf'), -1)

class kdtreeNode(Node):
    def __init__(self, data=-1, depth=-1, left=None, right=None, idx=None):
        Node.__init__(self, data, depth, left, right)
        self.idx = idx

if __name__=="__main__":
    kdtree = kdTree()
    kdtree.test()