from numpy import *
import matplotlib.pyplot as plt
import random

def normalization(x):
    mu = mean(x, axis=0)
    sigma = std(x, axis=0)
    return (x - mu) / sigma

def softmax(mat):
    expsum = sum(exp(mat), axis = 1)
    return exp(mat) / expsum

def calError(h, label, type):
    ans = zeros((shape(h)[0], 1))
    for i in range(shape(h)[0]):
        if label[i, 0] == type:
            ans[i, 0] = 1 - h[i, type]
        else:
            ans[i, 0] = -h[i, type]
    return ans

def loadTrainSet():
    f = open('Iris/train/x.txt')
    data = []
    for line in f:
        lineArr = line.strip().split()
        data.append([float(lineArr[0]), float(lineArr[1])])
    f.close()
    f = open('Iris/train/y.txt')
    label = []
    for line in f:
        lineArr = line.strip().split()
        label.append(int(float(lineArr[0])))
    f.close()
    data = normalization(data)
    data1 = []
    for i in data:
        data1.append([1, i[0], i[1]])
    return data1, label

def loadTestSet():
    f = open('Iris/test/x.txt')
    data = []
    for line in f:
        lineArr = line.strip().split()
        data.append([float(lineArr[0]), float(lineArr[1])])
    f.close()
    f = open('Iris/test/y.txt')
    label = []
    for line in f:
        lineArr = line.strip().split()
        label.append(int(float(lineArr[0])))
    f.close()
    data = normalization(data)
    data1 = []
    for i in data:
        data1.append([1, i[0], i[1]])
    return data1, label

def getTestPredict(theta, testdata):
    testdataMat = mat(testdata)
    htest = testdataMat * theta
    label = argmax(htest, axis=1)
    return label

def getCorrectRate(predictlabel, testlabel):
    correctnum = 0
    for i in range(shape(testlabel)[0]):
        if predictlabel[i, 0] == testlabel[i, 0]:
            correctnum = correctnum + 1
    return correctnum / shape(testlabel)[0]

def calLikelihood(h, labelMat):
    ans = 0
    for i in range(shape(h)[0]):
        ans = ans + log(h[i, labelMat[i, 0]])
    return ans

def softmaxRegression(data, label):
    dataMat = mat(data)
    labelMat = mat(label).transpose()
    n, m = shape(dataMat)   # n samples, m features
    theta = zeros((m, 4))   # m features, 3 tpyes + 1
    alpha = 0.001
    maxCycle = 10000
    episilon = 0.0005
    preLikelihood = 0.0
    for k in range(maxCycle):
        h = softmax(dataMat * theta)
        likelihood = calLikelihood(h, labelMat)
        if abs(likelihood - preLikelihood) < episilon:
            break
        preLikelihood = likelihood
        for i in range(shape(h)[1]):
            delta = alpha * dataMat.transpose() * calError(h, labelMat, i)

            theta[:, i] = theta[:, i] + delta.transpose()
    print(k)
    return theta         

def stocSoftmaxRegression(data, label):
    dataMat = mat(data)
    labelMat = mat(label).transpose()
    n, m = shape(dataMat)   # n samples, m features
    theta = zeros((m, 4))   # m features, 3 tpyes + 1
    alpha = 0.001
    maxCycle = 50000
    episilon = 1e-7
    preLikelihood = 0.0
    for k in range(maxCycle):
        h = softmax(dataMat * theta)
        likelihood = calLikelihood(h, labelMat)
        if abs(likelihood - preLikelihood) < episilon:
            break
        preLikelihood = likelihood
        # choose one sample only
        rand = random.randint(0, n - 1)
        for i in range(shape(h)[1]):
            if labelMat[rand, 0] == i:
                delta = alpha * (1 - h[rand, i]) * dataMat[rand]
            else:
                delta = alpha * (-h[rand, i]) * dataMat[rand]
            theta[:, i] = theta[:, i] + delta
    print(k)
    return theta  

def plotBestFit(fig, data, label, theta, name, subplot):
    dataMat = mat(data)
    labelMat = mat(label).transpose()
    xcord0 = []; ycord0 = []
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(shape(data)[0]):
        if label[i] == 0:
            xcord0.append(dataMat[i, 1])
            ycord0.append(dataMat[i, 2])
        elif label[i] == 1:
            xcord1.append(dataMat[i, 1])
            ycord1.append(dataMat[i, 2])
        elif label[i] == 2:
            xcord2.append(dataMat[i, 1])
            ycord2.append(dataMat[i, 2])

    ax = fig.add_subplot(subplot)
    ax.set_title(name, fontsize=8)
    
    ax.scatter(xcord0, ycord0, s = 30, c = 'red')
    ax.scatter(xcord1, ycord1, s = 30, c = 'green')
    ax.scatter(xcord2, ycord2, s = 30, c = 'blue')

    plotBoundary(theta[0, 0] - theta[0, 1], theta[1, 0] - theta[1, 1], theta[2,0] - theta[2, 1], "red-green")
    plotBoundary(theta[0, 0] - theta[0, 2], theta[1, 0] - theta[1, 2], theta[2,0] - theta[2, 2], "red-blue")
    plotBoundary(theta[0, 1] - theta[0, 2], theta[1, 1] - theta[1, 2], theta[2,1] - theta[2, 2], "green-blue")

    

def plotBoundary(para0, para1, para2, name):
    x = arange(-3, 3, 0.1)
    y = (-para1 * x - para0) / para2
    plt.plot(x, y, label = name)
    plt.legend()
    

def main():
    fig = plt.figure()
    data1, label = loadTrainSet()
    testdata, testlabel = loadTestSet()
    print("SoftmaxRegression:")
    print("theta:")
    theta1 = softmaxRegression(data1, label)
    print(theta1)

    print("to TestDataSet:")    
    predictlabel1 = getTestPredict(theta1, testdata)
    print("accuracy:")
    print(getCorrectRate(predictlabel1, mat(testlabel).transpose()))
    plotBestFit(fig, testdata, testlabel, theta1, "Softmax, ToTestDataSet", 221)

    print("to TrainDataSet:")    
    predictlabel2 = getTestPredict(theta1, data1)
    print("accuracy:")
    print(getCorrectRate(predictlabel2, mat(label).transpose()))
    plotBestFit(fig, data1, label, theta1, "Softmax, ToTrainDataSet", 222)

    print("stocSoftmaxRegression:")
    print("theta")
    theta2 = stocSoftmaxRegression(data1, label)
    print(theta2)

    print("to TestDataSet:")    
    predictlabel1 = getTestPredict(theta2, testdata)
    print("accuracy:")
    print(getCorrectRate(predictlabel1, mat(testlabel).transpose()))
    plotBestFit(fig, testdata, testlabel, theta2, "stocSoftmax, ToTestDataSet", 223)

    print("to TrainDataSet:")    
    predictlabel2 = getTestPredict(theta2, data1)
    print("accuracy:")
    print(getCorrectRate(predictlabel2, mat(label).transpose()))
    plotBestFit(fig, data1, label, theta2, "stocSoftmax, toTrainDataSet", 224)

    plt.show()

if __name__=='__main__':
    main()

