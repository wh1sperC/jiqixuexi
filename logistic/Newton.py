from numpy import *
import matplotlib.pyplot as plt
import random
from matplotlib.pyplot import MultipleLocator
import warnings

def normalization(x):
    mu = mean(x, axis=0)
    sigma = std(x, axis=0)
    return (x - mu) / sigma

def loadDataSet():
    f = open('Exam/train/x.txt')
    data = []
    for line in f:
        lineArr = line.strip().split()
        data.append([float(lineArr[0]), float(lineArr[1])])
    f.close()
    f = open('Exam/train/y.txt')
    label = []
    for line in f:
        lineArr = line.strip().split()
        label.append(int(float(lineArr[0])))
    f.close()
    # data归一化 添加1
    data = normalization(data)
    data1 = []
    for i in data:
        data1.append([1, i[0], i[1]])
    return data1, label

def sigmoid(x):
    return 1.0 / (1 + exp(-x))

def mat2list(x):
    xlist = []
    for i in range(shape(x)[0]):
        xlist.append(x[i, 0])
    return xlist

def newtonMethod(data, label):
    dataMat = mat(data)
    labelMat = mat(label).transpose()
    n, m = shape(dataMat)  # n samples, m features
    theta = mat([[1], [-1], [1]])
    alpha = 0.001
    maxCycle = 10000
    episilon = 0.01
    h = sigmoid(dataMat * theta)
    error = h - labelMat
    precost = (-labelMat.transpose() * log(h) - (ones(
            (n, 1)) - labelMat).transpose() * log(ones((n, 1)) - h))
    deltaJTheta = (dataMat.transpose() * error) / n
    para = multiply(h, ones((n, 1)) - h)
    paralist = mat2list(para)
    H = dataMat.transpose() * mat(diag(paralist)) * dataMat
    plt.ion()
    xs = [0, 0]
    ys = [0, precost[0, 0]]
    for k in range(maxCycle):
        theta = theta - linalg.inv(H / n) * deltaJTheta
        h = sigmoid(dataMat * theta)
        cost = (-labelMat.transpose() * log(h) - (ones(
            (n, 1)) - labelMat).transpose() * log(ones((n, 1)) - h))
        error = h - labelMat
        deltaJTheta = (dataMat.transpose() * error) / n
        para = multiply(h, ones((n, 1)) - h)
        paralist = mat2list(para)
        H = dataMat.transpose() * mat(diag(paralist)) * dataMat
        xs[0] = xs[1]
        ys[0] = ys[1]
        xs[1] = k + 1
        ys[1] = cost[0, 0]
        fig = plt.figure(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax = plt.subplot(121)
        ax.set_title("cost", fontsize = 8)
        ax.plot(xs, ys)
        plotResult(data, label, theta, fig)
        plt.pause(0.2)
        if abs(precost - cost) < episilon:
            break
        precost = cost

    print(k)
    return theta

def plotResult(data, label, theta, fig):
    dataMat = mat(data)
    labelMat = mat(label).transpose()
    xcord1 = []
    ycord1 = []
    xcord0 = []
    ycord0 = []
    for i in range(shape(data)[0]):
        if (label[i]) == 0:
            xcord0.append(dataMat[i, 1])
            ycord0.append(dataMat[i, 2])
        else:
            xcord1.append(dataMat[i, 1])
            ycord1.append(dataMat[i, 2])
    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax = plt.subplot(122)
    plt.cla()
    ax.scatter(xcord0, ycord0, s=30, c='red', marker='s')
    ax.scatter(xcord1, ycord1, s=30, c='green')
    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(2)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    x = arange(-3, 3, 0.1)
    y = (-theta[1, 0] * x - theta[0, 0]) / theta[2, 0]
    ax.plot(x, y)

def main():
    data1, label = loadDataSet()
    theta = newtonMethod(data1, label)
    print(theta)
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
