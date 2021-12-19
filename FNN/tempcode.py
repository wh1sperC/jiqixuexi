# 使用梯度下降实现FNN神经网络
import numpy as np
import matplotlib.pyplot as plt
import chineseize_matplotlib
import random


plt.rcParams['axes.unicode_minus'] = False
np.set_printoptions(precision=8)

# 加载数据
def load_data(path,m):
    data=np.loadtxt(path,dtype=float,ndmin=m)
    return data

# 定义神经网络的组成
class FNN:
    #初始化网络结构
    # 其中sizes是[输入层，任意多隐藏层，输出层]的节点数的列表，f是我们需要训练的数据的特征数
    #f是数据维度,f=2代表输入数据是二维的
    def __init__(self,sizes,data,label): 
        self.layer_nums=len(sizes) #获取网络层数
        self.sizes=sizes
        self.data=data
        self.label=label
        self.hidden=[]
        self.bayes=[np.random.rand(i,1) for i in sizes[1:]] #获取每层贝叶斯偏置
        self.weights=[np.random.rand(i,j) for i,j in zip(sizes[:-1],sizes[1:])] #获取每一层的权重
    
    # 定义激活函数，这里使用sigmoid
    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))
    
    # 定义前馈函数
    def feedforward(self):
        x=self.data
        for i in range(len(self.sizes)-1):
            x=np.array([np.dot(x[j],self.weights[i])+self.bayes[i][0] for j in range(x.shape[0])])
            netout=self.sigmoid(x)
        return netout

    # 定义反馈更新函数
    def backforward(self):
        pass
    
    # 定义训练函数
    def train(self):
        print(np.around(self.feedforward(),8))
        

if __name__ == '__main__':
    x_path1='./FNN/Exam/train/x.txt'
    x_path2='./FNN/Iris/train/x.txt'
    y_path1='./FNN/Exam/train/y.txt'
    y_path2='./FNN/Iris/train/y.txt'
    x1=load_data(x_path1,2)
    x2=load_data(x_path2,2)
    y1=load_data(y_path1,2)
    y2=load_data(y_path2,2)
    train_data1=np.hstack([x1,y1])
    train_data2=np.hstack([x2,y2])
    print(x1)
    print(y1)
    print(x1.shape)
    print(y1.shape)
    print(train_data1)
    net=FNN([x1.shape[1],5,y1.shape[1]],x1,y1)
    print(net.layer_nums)
    print(net.sizes)
    print(net.bayes)
    print(net.weights)
    print(len(net.weights))
    #print(zip(train_data1[:,-2],train_data1[:,-1]))
    '''print(type(tmp))
    tmp1=np.sum(tmp,axis=2)
    print(tmp1)
    print(type(tmp1))
    print(np.shape(tmp1))'''
    #print(net.gd(iters=1,data=train_data1,alpha=0.001))
    net.train()