#coding=utf-8

import numpy as np
import math

class DecisionTree(object):
    def __init__(self):
        print "决策树"

    def ID3(self, datasets, labels):
        self.root = decisionNode()
        datasets = self.trans_matrix_2_dict(datasets)
        self.decisiontree = self.build_decisiontree(node=self.root, dataset=datasets, labels=labels, type="ID3")
        pass

    def build_decisiontree(self, node, dataset, labels, gain_threshold=0.05, type="ID3"):
        if len(dataset) == 0:
            return None
        # 若labels中所有的类别都一样,则该节点停止建树,成为叶节点
        unique = set(labels)
        if 1 == len(unique):
            node.label = labels[0]
            return node

        #计算最优的特征
        max_gain = -1.0
        max_idx = -1
        H_D = self.get_entropy_from_labels(labels)
        for (idx, feature_lists) in dataset.items():
            unique_feature_list = list(set(feature_lists))
            H_D_A = 0.0
            for i in unique_feature_list:
                filter_list = self.split_dataset(dataset, idx, i)
                # print i, filter_list
                H_D_A += (float(len(filter_list))/len(labels))*(self.get_entropy_from_labels(map(lambda i: labels[i], filter_list)))
            H_A_D = self.get_entropy_from_labels(dataset[idx]) + 1e-10

            gain = 0
            if type == "ID3":
                gain = H_D-H_D_A
            else:
                gain = (H_D-H_D_A)/H_A_D
            # print idx, gain
            if (max_gain <gain):
                max_gain = gain
                max_idx = idx
        # print max_idx,":",max_gain

        # 如信息增益小于阈值,则该节点为叶节点,停止建树,选择类别中最多的类作为本节点的类别
        if max_gain < gain_threshold:
            (num, label)=max(map(lambda x: (labels.count(x), x), labels))
            node.label = label
            return node

        # 切分出当且维的特征数据
        unique_feature_list = set(dataset[max_idx])
        now_featur = {}
        now_featur[max_idx] = dataset[max_idx]
        dataset.pop(max_idx)

        # 根据不同的特征值建立子树
        for i in unique_feature_list:
            filter_idx_list = self.split_dataset(now_featur, max_idx, i)
            child = decisionNode()
            child_dataset = {}
            for (key, lists) in dataset.items():
                child_dataset[key] = list(map(lambda idx:dataset[key][idx], filter_idx_list))

            child = self.build_decisiontree(child, child_dataset,list(map(lambda i: labels[i], filter_idx_list)), gain_threshold, type)
            if None != child:
                # print "第",max_idx,"特征：", i,"\t",child.label
                node.child[i] = child
        return node

    # 将numpy矩阵组成的数据集转换成以维度作为key的dict形式
    # 原因是 1 方便数据切分,数据集中的下标没有存在意义,而维度下标一直使用
    #       2 矩阵形式的数据在决策树模型中没有计算联系,不同维数据间没有交集
    #
    def trans_matrix_2_dict(self, datasets):
        feature_dict = {}

        for i in range(0,datasets.T.shape[0]):
            feature_dict[i] = list(datasets.T[i])
        # for (idx, lists) in feature_dict.items():
        #     print idx, lists
        return feature_dict

    # 根据特征的维度,属性值以及切分方式提取符合切分条件的数据
    # 返回符合条件的数据下标所组成的list
    def split_dataset(self, dataset, idx, value, filter_type="eq"):
        type_map_fuc = {"eq":(lambda x, y: x==y),
                    "greater":(lambda x, y: x>y),
                    "low": (lambda x, y: x<y),
                    "geq": (lambda x, y: x>=y),
                    "leq": (lambda x, y: x<=y)}
        if filter_type not in type_map_fuc:
            print filter_type,"不是符合输入的条件"
            return None
        filter_fuc =  type_map_fuc[filter_type]

        # 根据过滤函数得到符合条件的数据下标
        idx_list = filter(lambda j: (filter_fuc(value,dataset[idx][j])), range(0, len(dataset[idx])))
        return idx_list

    # 输入partiton的类别list 得到熵
    def get_entropy_from_labels(self, labels=[]):
        Ps = self.get_p(labels)
        entropy = 0.0
        for (label, p) in Ps.items():
            entropy += self.p_logp(p)
        return  -entropy

    # 输入partiton的类别list 得到gini指数
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
        #print self.get_gini_from_labels(labels)
        # print self.trans_matrix_2_dict(dataset)
        # print self.split_dataset(self.trans_matrix_2_dict(dataset), 3, 3)
        # for i in range(0, len(dataset)):
        #     print (i+1), dataset[i], labels[i]
        # print self.get_entropy_from_labels(labels)
        self.ID3(dataset, labels)

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