# 使用梯度下降实现FNN神经网络
import numpy as np
import matplotlib.pyplot as plt
import chineseize_matplotlib
import random


plt.rcParams['axes.unicode_minus'] = False
np.set_printoptions(precision=8)

def normalization(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    return (x - mu) / sigma

# 加载数据
def load_data(path,m):
    data=np.loadtxt(path,dtype=float,ndmin=m)
    #data=normalization(data)
    return data

x_path1='./FNN/Exam/train/x.txt'
x_path2='./FNN/Iris/train/x.txt'
y_path1='./FNN/Exam/train/y.txt'
y_path2='./FNN/Iris/train/y.txt'
x1=load_data(x_path1,2)
x1=normalization(x1)
x2=load_data(x_path2,2)
x2=normalization(x2)
y1=load_data(y_path1,2)
y2=load_data(y_path2,2)

labels=np.zeros((x1.shape[0],2))
for k in range(x1.shape[0]):
    labels[k][int(y1[k])]=1
print(labels)
# 定义神经网络的组成
class FNN:
    #初始化网络结构
    # 其中sizes是[输入层，任意多隐藏层，输出层]的节点数的列表，f是我们需要训练的数据的特征数
    #f是数据维度,f=2代表输入数据是二维的
    def __init__(self,sizes,data,label,alpha): 
        self.layer_nums=len(sizes) #获取网络层数
        self.sizes=sizes
        self.data=data
        self.label=label
        self.hidden=[]
        self.alpha=alpha
        self.bayes=[np.zeros((i,1)) for i in sizes[1:]]
        #self.bayes=[np.random.rand(i,1) for i in sizes[1:]] #获取每层贝叶斯偏置
        self.weights=[np.zeros((i,j)) for i,j in zip(sizes[:-1],sizes[1:])]
        #self.weights=[np.random.rand(i,j) for i,j in zip(sizes[:-1],sizes[1:])] #获取每一层的权重
        
    # 定义激活函数，这里使用sigmoid
    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))
    
    # 定义前馈函数
    def feedforward(self,ix):
        x=ix
        self.hidden.append(x)
        for i in range(len(self.sizes)-1):
            x=np.array([np.dot(x[j],self.weights[i])+self.bayes[i][0] for j in range(x.shape[0])])
            netout=self.sigmoid(x)
            self.hidden.append(netout)
        return netout

    # 定义反馈更新函数
    def backforward(self,label):
        idx=self.hidden[1]
        idy=self.hidden[2]
        print(idy)
        print(idy[0][0]-label)
        print(idy.shape)
        print(label)
        error_out=np.zeros(self.sizes[2])
        #print(error_out)
        for j in range(self.sizes[2]):
            error_out[j]=(idy[0][j]-label[j])*idy[0][j]*(1-idy[0][j])
        x=self.hidden[0]
        y=self.hidden[1]
        error_hidden=np.zeros(self.sizes[1])
        for j in range(self.sizes[1]):
            for i in range(self.sizes[2]):
                error_hidden[j]+=error_out[i]*self.weights[1][j][i]*y[0][j]*(1-y[0][j])
        self.weights[1]-=np.dot(idx.T,error_out.reshape((1,2)))*self.alpha
        self.bayes[1]-=error_out*self.alpha
        self.weights[0]-=np.dot(x.T.reshape(2,1),error_hidden.reshape(1,3))*self.alpha
        self.bayes[0]-=error_hidden*self.alpha
    
    # 定义训练函数
    def train(self,iter):
        lost=[]
        for i in range(iter):
            loss=0
            for j in range(x1.shape[0]):
                hx=self.feedforward(x1[j])
                print(labels[j])
                self.backforward(labels[j])
                self.hidden.clear()
                loss+=((hx[0][0]-labels[j][0])**2+(hx[0][1]-labels[j][1])**2)/2
            lost.append(loss)
            plt.clf()
            plt.figure(1)
            plt.plot(range(i+1),lost)
            plt.pause(0.01)
            plt.ioff()
        
def loss_function(hx,label):
    loss=0
    for i in range(label.shape[0]):
        loss+=((hx[0][i]-label[i])**2)/2
    return loss
        



train_data1=np.hstack([x1,y1])
train_data2=np.hstack([x2,y2])
#print(x1)
#print(y1)
#print(x1.shape)
#print(y1.shape)
#print(train_data1)
net=FNN([x1.shape[1],5,x1.shape[1]],x1,y1,0.01)
#print(net.layer_nums)
#print(net.sizes)
#print(net.bayes)
#print(net.weights)
#print(len(net.weights))
#print(net.data)
#print(zip(train_data1[:,-2],train_data1[:,-1]))
'''print(type(tmp))
tmp1=np.sum(tmp,axis=2)
print(tmp1)
print(type(tmp1))
print(np.shape(tmp1))'''
#print(net.gd(iters=1,data=train_data1,alpha=0.001))
net.train(1)