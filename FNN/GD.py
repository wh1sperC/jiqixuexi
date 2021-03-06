import numpy as np
from numpy import *
import matplotlib.pyplot as plt


def normalization(x): #数据归一化处理
    for i in range(x.shape[0]):
        max_num=max(x[i])
        min_num=min(x[i])
        for j in range(x.shape[1]):
            x[i][j]=(x[i][j]-min_num)/(max_num-min_num)
    return x

def sigmoid(x): # sigmoid函数
    return 1.0/(1.0+np.exp(-x))

def tanH(x):#tanh函数
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def softmax(x):#softmax函数，这里用作最后输出的激活函数处理
    return np.exp(x)/np.sum(np.exp(x),axis=1)

actfun=['tanh','sigmoid','softmax']
lossfun='least_square' #'cross_entropy'
nodes=[2,7,6,3]
layer=len(nodes)
alpha=0.01
type=nodes[-1]
epoches=10000 #训练次数

#导入数据
dataset='Iris'
train_x=np.loadtxt("./FNN/{}/train/x.txt".format(dataset),dtype=float,ndmin=2)
train_y=np.loadtxt("./FNN/{}/train/y.txt".format(dataset),dtype=float,ndmin=2)
test_x=np.loadtxt("./FNN/{}/test/x.txt".format(dataset),dtype=float,ndmin=2)
test_y=np.loadtxt("./FNN/{}/test/y.txt".format(dataset),dtype=float,ndmin=2)
train_x=normalization(train_x.T).T
test_x=normalization(test_x.T).T
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False
fig = plt.figure(facecolor = 'lightgrey',figsize=[3*5,1*5])

M=train_x.shape[0] #训练集规模

#存放坐标分类
x_0=[];y_0=[]
x_1=[];y_1=[]
x_2=[];y_2=[]
for j in range(M):
    if train_y[j]==1:
        x_1.append(train_x[j][0])
        y_1.append(train_x[j][1])
    elif train_y[j]==2:
        x_2.append(train_x[j][0])
        y_2.append(train_x[j][1])
    else:
        x_0.append(train_x[j][0])
        y_0.append(train_x[j][1])

def plotboundry(p0,p1,p2,name):
    x=arange(0,1,0.1)
    y=(-p1*x-p0)/p2
    plt.plot(x,y,label=name)
    plt.legend()

def plot1(theta):
    plt.scatter(x_0,y_0,s=30,c='',marker='o',edgecolors='b')
    plt.scatter(x_1,y_1,s=30,c='b',marker='+')
    plotboundry(theta[0,1]-theta[0,0], theta[1, 1] - theta[1, 0], theta[2,1] - theta[2, 0], "o-+")

def plot2(theta):
    plt.scatter(x_0,y_0,s=30,marker='o',edgecolors='k')
    plt.scatter(x_1,y_1,s=30,c='k',marker='+')
    plt.scatter(x_2,y_2,s=30,c='k',marker='*')
    plotboundry(theta[0, 1] - theta[0, 0], theta[1, 1] - theta[1, 0], theta[2,1] - theta[2, 0], "o-+")
    plotboundry(theta[0, 2] - theta[0, 0], theta[1, 2] - theta[1, 0], theta[2,2] - theta[2, 0], "o-*")
    plotboundry(theta[0, 2] - theta[0, 1], theta[1, 2] - theta[1, 1], theta[2,2] - theta[2, 1], "+-*")

#对标签进行处理
labels=np.zeros((M,type))
for i in range(M):
    labels[i][int(train_y[i])]=1

def least_square(hatx,label): # 最小二乘估计
    loss=0
    for i in range(label.shape[0]):
        loss+=((hatx[0][i]-label[i])**2)/2
    return loss

def cross_entropy(hatx,label): # 交叉熵损失
    loss=0
    for i in range(label.shape[0]):
        loss+=label[i]*np.log(hatx[0][i])
    return loss

def onehot_encode(x): # 神经网络输出层的独热码翻译
    n=x.shape[0]
    hatx=np.zeros(n)
    for i in range(n):
        hatx[i]=argmax(x[i][0])
    return hatx

def accuracy(hatx,label): # 模型预测准确率
    num=0
    N=hatx.shape[0]
    for i in range(N):
        if hatx[i]==label[i]:
            num+=1
    return num/N

class Network(): # 神经网络类
    def __init__(self,layers,nodes,TYPE):
        if len(nodes)!=layers or layers==0: #先检查神经网络是否可构成
            raise Exception("Error layers or nodes")
        self.layers=layers #网络层数
        self.nodes=nodes #节点分布
        self.TYPE=TYPE #分类类数
        self.hidden_layers=[] # 隐藏层结果储存
        self.weights=[np.random.rand(x,y) for x,y in zip(nodes[:-1],nodes[1:])]
        self.bayes=[np.zeros((1,y)) for y in nodes[1:]] #权重和贝叶斯偏置

    def feedforward(self,idx): #前向传播
        x=idx
        self.hidden_layers.append(x)
        for i in range(self.layers-1):
            x=np.dot(x,self.weights[i])+self.bayes[i]
            if actfun[i]=='tanh':
                x=tanH(x)
            if actfun[i]=='sigmoid':
                x=sigmoid(x)
            if actfun[i]=='softmax':
                x=softmax(x)
            self.hidden_layers.append(x)
        return x #得到最后的输出结果

    def backward(self,label): # 反向更新参数
        errors=[]
        nodes=self.nodes
        n=1
        idx=self.hidden_layers[self.layers-2]
        idy=self.hidden_layers[-1]
        if lossfun=='least_square':
            error_last=idy[0]-label
        if lossfun=='cross_entropy':
            error_last=-label/idy[0]
        weight=np.ones((nodes[-1],1))
        error=np.zeros(nodes[-1])
        for i in range(nodes[-1]): #输出层误差
            for j in range(n):
                error[i]+=error_last[i]*weight[i][j]*idy[0][i]*(1-idy[0][i])
        errors.append(error)

        for i in range(self.layers-2): # 隐藏层误差
            idx=self.hidden_layers[self.layers-3-i]
            idy=self.hidden_layers[self.layers-2-i]
            n=nodes[self.layers-1-i]
            error_last=error
            error=np.zeros(nodes[self.layers-2-i])
            weight=self.weights[self.layers-2-i]
            for j in range(nodes[self.layers-2-i]):
                for k in range(n):
                    error[j]+=error_last[k]*weight[j][k]*idy[0][j]*(1-idy[0][j])
            errors.append(error)
       
        for i in range(self.layers-1): #参数更新
            idx=self.hidden_layers[self.layers-2-i]
            idy=self.hidden_layers[self.layers-1-i]
            self.weights[self.layers-2-i]-=np.dot(idx.T.reshape(nodes[self.layers-2-i],1),errors[i].reshape((1,nodes[self.layers-1-i])))*alpha
            self.bayes[self.layers-2-i]-=errors[i]*alpha
    
    def train(self): #训练
        lost=[]
        xlabel=[]
        acc=[]
        for epoch in range(epoches):
            loss=0
            px=[]
            for i in range(train_x.shape[0]):
                hatx=self.feedforward(train_x[i])
                self.backward(labels[i])
                self.hidden_layers.clear()
                if lossfun=='least_square':
                    loss+=least_square(hatx,labels[i])
                if lossfun=='cross_entropy':
                    loss-=cross_entropy(hatx,labels[i])
                px.append(hatx)
            px=onehot_encode(np.array(px))
            #print(px)
            a=accuracy(px,train_y)
            acc.append(a)
            lost.append(loss)
            xlabel.append(epoch)
            if (epoch+1)%100 == 0:
                print("epoch {}:loss:{:.6f}".format(epoch+1,loss))
                plt.clf()
                plt.figure(1)
                plt.subplot(131)
                plt.title('loss')
                plt.plot(xlabel,lost,c='r')
                plt.subplot(132)
                plt.title('accuracy')
                plt.plot(xlabel,acc,c='b')
                plt.subplot(133)
                if self.nodes[-1]==2:
                    temp1=np.dot(self.bayes[0],self.weights[1])+self.bayes[1]
                    temp2=np.dot(self.weights[0],self.weights[1])
                    theta=np.append(temp1,temp2,axis=0)
                    plot1(theta)
                if self.nodes[-1]==3:
                    temp1=self.bayes[0]@self.weights[1]@self.weights[2]+self.bayes[1]@self.weights[2]+self.bayes[2]
                    temp2=self.weights[0]@self.weights[1]@self.weights[2]
                    theta=np.append(temp1,temp2,axis=0)
                    plot2(theta)
                plt.pause(0.001)
                plt.ioff()
        plt.show()

    def test(self): #测试
        hatx=[]
        for i in range(test_x.shape[0]):
            x=test_x[i]
            for j in range(self.layers-1):
                x=np.dot(x,self.weights[j])+self.bayes[j]
                if actfun[j]=='tanh':
                    x=tanH(x)
                if actfun[j]=='sigmoid':
                    x=sigmoid(x)
                if actfun[j]=='softmax':
                    x=softmax(x)
            hatx.append(x)
        #print(hatx)
        hatx=onehot_encode(np.array(hatx))
        print("predict:")
        print(hatx)
        print(test_y)
        print("accuracy:{:.6f}".format(accuracy(hatx,test_y)))
        return hatx

#建立一个神经网络对象并实现训练和测试
FNN=Network(layer,nodes,type)
FNN.train()
FNN.test()
print('final weight:')
print(FNN.weights)
print('final bayes:')
print(FNN.bayes)



