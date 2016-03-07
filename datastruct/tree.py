#coding=utf-8

class Node(object):
    def __init__(self, data=-1, depth=-1, left=None, right=None):
        self.data = data
        self.depth = depth
        self.left = left
        self.right = right