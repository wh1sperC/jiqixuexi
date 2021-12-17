# 使用梯度下降实现FNN神经网络
import numpy as np
import matplotlib.pyplot as plt
import chineseize_matplotlib
import random


# 加载数据
def load_data(path,m):
    data=np.loadtxt(path,dtype=float,ndmin=m)
    return data

# 定义神经网络的组成
class FNN:
    #初始化网络结构
    # 其中sizes是[输入层，任意多隐藏层，输出层]的节点数的列表，f是我们需要训练的数据的特征数
    #f是数据维度,f=2代表输入数据是二维的
    def __init__(self,sizes,f): 
        self.layer_nums=len(sizes) #获取网络层数
        self.sizes=sizes
        self.bayes=[np.random.rand(i,1) for i in sizes[1:]] #获取每层贝叶斯偏置
        self.weights=[np.random.rand(i,j) for i,j in zip(sizes[1:],sizes[:-1])] #获取每一层的权重
    
    # 定义激活函数，这里使用sigmoid
    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))
    
    # 定义梯度下降函数
    def gd(self,iters):
        for j in range(iters):
            pass

if __name__ == '__main__':
    x_path1='./FNN/Exam/train/x.txt'
    x_path2='./FNN/Iris/train/x.txt'
    y_path1='./FNN/Exam/train/y.txt'
    y_path2='./FNN/Iris/train/y.txt'
    x1=load_data(x_path1,2)
    x2=load_data(x_path2,2)
    y1=load_data(y_path1,2)
    y2=load_data(y_path2,2)
    print(x1.T)
    #print(y1.T)
    print(x1.shape)
    net=FNN([x1.shape[0],2,x1.shape[0]],x1.shape[1])
    print(net.layer_nums)
    print(net.sizes)
    print(net.bayes)
    print(net.weights)
    #print(net.weights[0])
    #print(np.random.randn(1,2,2))
    print(np.matmul(net.weights[0],x1))
    print(np.shape(np.matmul(net.weights[0],x1)))