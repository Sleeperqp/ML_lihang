#coding=utf-8
import numpy as np

#感知器原始形式
#输入:X, Y, delat, max_itr=5000
#输出:w,b
def perceptron_model():
    X = [np.array([[3],[3]]),np.array([[4],[3]]), np.array([[1],[1]])]
    Y = [1, 1, -1]
    w = np.array([0, 0])
    b = 0
    delat = 1
    while 1:
        XY_list = map(lambda  x, y: (x, y), X, Y)
        #print XY_list
        errorlist = filter(lambda  value: value if (value[1]*(w.dot(value[0])+b) <= 0) else None, XY_list)
        #print errorlist
        if not errorlist:
            break

        updatex, updatey = errorlist[0]
        w = w + delat*updatey*(updatex.T)
        b = b + delat*updatey

    print "===========perceptron_model====================="
    print "w:",w
    print "b:",b

#感知器的对偶模式
#输入: X, Y, delat, max_itr=5000
#输出: w, b
def perceptron_dual_model():
    print "==========================================="
    X = [np.array([3, 3]),np.array([4, 3]), np.array([1, 1])]
    Y = np.array([1, 1, -1])
    a = np.array([0, 0, 0])
    b = 0
    delat = 1


    X = np.array(X).T
    gramX = np.dot(np.array(X).T, np.array(X))

    while 1:
        error_list = filter(lambda idx: idx if (Y[idx-1]*(np.dot(np.array(gramX[idx-1, :]), (a*Y).T)+b) <= 0) else None, range(1, len(X[0])+1))
        error_list = map(lambda i : i-1, error_list)
        # error_list =[]
        # for idx in range(0, len(X[0])):
        #     if Y[idx]*(np.dot(np.array(gramX[idx, :]), (a*Y).T)+b) <= 0:
        #         error_list.append(idx)
        #print error_list
        if not error_list:
            break

        idx = error_list[0]
        a[idx] += delat
        b += delat*Y[idx]


    pre_w = map(lambda  x, y: (x*y), a*Y, X.T)
    pre_w = np.array(pre_w)
    w = reduce(lambda x, y: x +y,pre_w)
    print "w:",w
    print "b:",b

if __name__=="__main__":
    perceptron_model()
    perceptron_dual_model()