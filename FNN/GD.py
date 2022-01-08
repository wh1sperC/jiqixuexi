import numpy as np
from numpy import *
import matplotlib.pyplot as plt


def normalization(x):
    for i in range(x.shape[0]):
        max_num=max(x[i])
        min_num=min(x[i])
        for j in range(x.shape[1]):
            x[i][j]=(x[i][j]-min_num)/(max_num-min_num)
    return x

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def tanH(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

actfun=['tanh','tanh','sigmoid']
lossfun='least_square'
nodes=[2,7,6,3]
layers=len(nodes)
alpha=0.01
type=3
epoches=2000 #训练次数

train_x=np.loadtxt("./FNN/Iris/train/x.txt",dtype=float,ndmin=2)
train_y=np.loadtxt("./FNN/Iris/train/y.txt",dtype=float,ndmin=2)
test_x=np.loadtxt("./FNN/Iris/test/x.txt",dtype=float,ndmin=2)
test_y=np.loadtxt("./FNN/Iris/test/y.txt",dtype=float,ndmin=2)
train_x=normalization(train_x.T).T
test_x=normalization(test_x.T).T
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False
fig = plt.figure(facecolor = 'lightgrey')

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

#对标签进行处理
labels=np.zeros((train_x.shape[0],type))
for i in range(train_x.shape[0]):
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

def accuracy(hat_x,label): # 模型预测准确率
    num=0
    N=hat_x.shape[0]
    for i in range(N):
        if hat_x[i]==label[i]:
            num+=1
    return num/N

class Network(): # 神经网络类
    def __init__(self,layers,nodes,TYPE):
        if len(nodes)!=layers or layers==0: #先检查神经网络是否可构成
            raise Exception("Error layers or nodes")
        self.layers=layers
        self.nodes=nodes
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
                #output=tanH(x)
                x=tanH(x)
            if actfun[i]=='sigmoid':
                #output=sigmoid(x)
                x=sigmoid(x)
            #self.hidden_layers.append(output)
            self.hidden_layers.append(x)
        #return output #得到最后的输出结果
        return x

    def backward(self,label):
        errors=[]
        nodes=self.nodes
        n=1
        idx=self.hidden_layers[layers-2]
        idy=self.hidden_layers[-1]
        if lossfun=='least_square':
            error_last=idy[0]-label
        if lossfun=='cross_entropy':
            error_last=-label/idy[0]
        weight=np.ones((nodes[-1],1))
        error=np.zeros(nodes[-1])
        for j in range(nodes[-1]):
            for l in range(n):
                error[j]+=error_last[j]*weight[j][l]*idy[0][j]*(1-idy[0][j])
        errors.append(error)

        for i in range(layers-2):
            idx=self.hidden_layers[layers-3-i]
            idy=self.hidden_layers[layers-2-i]
            n=nodes[layers-1-i]
            error_last=error
            error=np.zeros(nodes[layers-2-i])
            weight=self.weights[layers-2-i]
            for j in range(nodes[layers-2-i]):
                for l in range(n):
                    error[j]+=error_last[l]*weight[j][l]*idy[0][j]*(1-idy[0][j])

            errors.append(error)
       
        for i in range(layers-1):
            idx=self.hidden_layers[layers-2-i]
            idy=self.hidden_layers[layers-1-i]
            self.weights[layers-2-i]-=np.dot(idx.T.reshape(nodes[layers-2-i],1),errors[i].reshape((1,nodes[layers-1-i])))*alpha
            self.bayes[layers-2-i]-=errors[i]*alpha
    
    def train(self):
        lost=[]
        xlabel=[]
        for epoch in range(epoches):
            loss=0
            for k in range(train_x.shape[0]):
                hat_x=self.feedforward(train_x[k])
                self.backward(labels[k])
                self.hidden_layers.clear()
                if lossfun=='least_square':
                    loss+=least_square(hat_x,labels[k])
                if lossfun=='cross_entropy':
                    loss-=cross_entropy(hat_x,labels[k])
            
            lost.append(loss)
            xlabel.append(epoch)
            if (epoch+1)%50 == 0:
                print("epoch {}:loss:{:.6f}".format(epoch+1,loss))
                plt.clf()
                plt.figure(1)
                plt.plot(xlabel,lost)
                plt.pause(0.01)
                plt.ioff()

    def test(self):
        hat_x=[]
        for k in range(test_x.shape[0]):
            x=test_x[k]
            for i in range(layers-1):
                x=np.dot(x,self.weights[i])+self.bayes[i]
                if actfun[i]=='tanh':
                    x=tanH(x)
                if actfun[i]=='sigmoid':
                    x=sigmoid(x)
            hat_x.append(x)
        print(hat_x)
        hat_x=onehot_encode(np.array(hat_x))
        print("predict:")
        print(hat_x)
        print(test_y)
        print("accuracy:{:.4f}".format(accuracy(hat_x,test_y)))

    def predict(self,X,Y):
        z=[]
        ninput=[]
        for i in range(300):
            for j in range(300):
                ninput.append([X[i][j],Y[i][j]])
        ninput=np.array(ninput).reshape((90000,2))
        for k in range(layers-1):
            ninput=np.dot(ninput,self.weights[k])+self.bayes[k]
        hat_x=onehot_encode(ninput).reshape(300,300)
        return hat_x
        



FNN=Network(layers,nodes,type)
FNN.train()
FNN.test()
