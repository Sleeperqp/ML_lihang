#coding=utf-8

import numpy as np
import math

class DecisionTree(object):
    def __init__(self):
        print "决策树"

    def ID3(self, datasets, labels):
        self.root = decisionNode()
        self.decisiontree = self.build_decisiontree(node=self.root, dataset=datasets, labels=labels, type="C4.5")
        pass

    def build_decisiontree(self, node, dataset, labels, gain_threshold=0.05, delete_feature_list=[], type="ID3"):
        if len(dataset) == 0:
            return None
        all = True
        tmp_dict = {}
        for i in labels:
            if i not in tmp_dict:
                tmp_dict[i] = 1
            else:
                tmp_dict[i] += 1
            if len(tmp_dict) >= 2:
                all = False
                break
        if True == all:
            node.label = labels[0]
            return node

        max_gain = -1.0
        idx = -1
        H_D = self.get_entropy_from_labels(labels)
        for i in range(0, len(dataset[0])):
            partition_set = {}
            for j in range(0, len(dataset)):
                # print j
                if dataset[j][i] not in partition_set:
                    partition_set[dataset[j][i]] = [j]
                else:
                    partition_set[dataset[j][i]].append(j)
            H_D_A = 0.0
            for (k, v) in partition_set.items():
                a = (float(len(v))/len(dataset))
                b = (self.get_entropy_from_labels(map(lambda i: labels[i], v)))
                H_D_A += a*b
            H_A_D = self.get_entropy_from_labels(dataset[:, i])+1e-10
            # print "-----------=============--------------"
            print H_D-H_D_A
            # print (H_D-H_D_A)/H_A_D
            gain = 0
            if type == "ID3":
                gain = H_D-H_D_A
            else:
                gain = (H_D-H_D_A)/H_A_D

            if (max_gain <gain):
                max_gain = gain
                idx = i
        print idx,":",max_gain
        # 删去该特征
        delete_feature_list.append(idx)

        # 如信息增益小于阈值,则该节点为叶节点,停止建树,选择类别中最多的类作为本节点的类别
        if max_gain < gain_threshold:
            (num, label)=max(map(lambda x: (labels.count(x), x), labels))
            node.label = label
            return node

        # 递归建立子树
        partition_set = {}
        for j in range(0, len(dataset)):
            if dataset[j][idx] not in partition_set:
                partition_set[dataset[j][idx]] = [j]
            else:
                partition_set[dataset[j][idx]].append(j)
        for (key, lists) in partition_set.items():
            print "第",idx,"特征：",key
            child = decisionNode()
            child = self.build_decisiontree(child, np.array(map(lambda i: dataset[i], lists)),np.array(map(lambda i: labels[i], lists)), gain_threshold, delete_feature_list, type)
            if child != None:
                print child.label
                node.child[key] = child
        pass

    def get_entropy_from_labels(self, labels=[]):
        Ps = self.get_p(labels)
        entropy = 0.0
        for (label, p) in Ps.items():
            entropy += self.p_logp(p)
        return  -entropy

    def get_gini_from_labels(self, labels=[]):
        gini = 0.0

        Ps = self.get_p(labels)
        for (label, p) in Ps.items():
            # print label, p
            gini += p*p

        return  1-gini

    # 输入:要求是list 或者 一位数组
    # 输出:得到(label, 概率)的dict
    def get_p(self, labels):
        Ps = {}
        for label in labels:
            if label not in Ps:
                Ps[label] = 1
            else:
                Ps[label] += 1

        length = float(len(labels))

        for (label,num) in Ps.items():
            Ps[label] /= length
            # print label, ":", Ps[label]
        return  Ps

    def p_logp(self, p):
        if p <= 0:
            return 0
        return p*math.log(p, 2)

    def test(self):
        dataset = np.array([
            [1, 0, 0, 1],
            [1, 0, 0 ,2],
            [1, 1, 0, 2],
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [2, 0, 0, 1],
            [2, 0, 0, 2],
            [2, 1, 1, 2],
            [2, 0, 1, 3],
            [2, 0, 1, 3],
            [3, 0, 1, 3],
            [3, 0, 1, 2],
            [3, 1, 0, 2],
            [3, 1, 0, 3],
            [3, 0, 0, 1],
        ])
        labels = [0, 0, 1, 1, 0,
                  0, 0, 1, 1, 1,
                  1, 1, 1, 1, 0]
        print self.get_gini_from_labels(labels)
        # for i in range(0, len(dataset)):
        #     print (i+1), dataset[i], labels[i]
        # print self.get_entropy_from_labels(labels)
        # self.ID3(dataset, labels)

class decisionNode():
    def __init__(self, data=-1, depth=-1, idx=-1):
        # 属性值
        self.data = data
        # 树的深度
        self.depth = depth
        # 所使用的特征的在数据中的下标
        self.idx = idx
        # 子节点:每个key为特征的一个取值
        self.child = {}
        # 叶节点的类别,内部节点为None
        self.label = None


if __name__=="__main__":
    decisiontree = DecisionTree()
    decisiontree.test()