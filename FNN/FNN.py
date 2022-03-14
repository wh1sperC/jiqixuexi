from matplotlib.colors import ListedColormap
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
nodes=[2,3,6,3]
layer=len(nodes)
alpha=0.01
type=nodes[-1]
epoches=2000 #训练次数

#导入数据
if type==3:
    dataset='Iris'
elif type==2:
    dataset='Exam'
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
        self.bayes=[np.random.rand(1,y) for y in nodes[1:]] #权重和贝叶斯偏置

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
                print("epoch {}:loss:{:.6f},accuracy:{:.6f}".format(epoch+1,loss,a))
                self.plotBestFit(train_x,train_y,lost,xlabel,acc)
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
    
    def plotBestFit(self,datamat,labelmat,lost,xlabel,acc):
        N,M=200,200
        x1_min,x1_max=datamat[:,0].min()-.5,datamat[:,0].max()+.5
        x2_min,x2_max=datamat[:,1].min()-.5,datamat[:,1].max()+.5
        t1=np.linspace(x1_min,x1_max,N)
        t2=np.linspace(x2_min,x2_max,M)
        x1,x2=np.meshgrid(t1,t2)
        x_show=np.stack((x1.flat,x2.flat),axis=1)
        y_predict=[]
        for i in range(x_show.shape[0]):
            x=self.feedforward(x_show[i])
            y_predict.append(x)
        y_predict=onehot_encode(np.array(y_predict))
        plt.figure(1)
        plt.clf()
        plt.subplot(131)
        plt.title('loss')
        plt.plot(xlabel,lost,c='r')
        plt.subplot(132)
        plt.title('accuracy')
        plt.plot(xlabel,acc,c='b')
        plt.subplot(133)
        cl_bg=ListedColormap(['g','r','b'])#['g','r','b']
        cl_pt=ListedColormap(['g','r','b'])#['k','purple','y']
        plt.xlim(x1_min,x1_max)
        plt.ylim(x2_min,x2_max)
        plt.pcolormesh(x1,x2,y_predict.reshape(x1.shape),cmap=cl_bg,shading='auto')
        plt.scatter(datamat[:,0],datamat[:,1],c=labelmat,cmap=cl_pt,marker='o',edgecolors='k')
        plt.title('分类结果')
        plt.pause(0.1)
        plt.ioff()

#建立一个神经网络对象并实现训练和测试
FNN=Network(layer,nodes,type)
FNN.train()
FNN.test()
print('final weight:')
print(FNN.weights)
print('final bayes:')
print(FNN.bayes)



