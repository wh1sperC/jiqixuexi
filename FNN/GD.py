# 使用梯度下降实现FNN神经网络
import numpy as np
import matplotlib.pyplot as plt
import chineseize_matplotlib
import random


# 加载数据
def load_data(path,m):
    data=np.loadtxt(path,dtype=float,ndmin=m)
    return data

x_path1='./FNN/Exam/train/x.txt'
x_path2='./FNN/Iris/train/x.txt'
y_path1='./FNN/Exam/train/y.txt'
y_path2='./FNN/Iris/train/y.txt'
data1=load_data(x_path1,2)
data2=load_data(x_path2,2)
label1=load_data(y_path1,2)
label2=load_data(y_path2,2)
print(data1)
print(label1)

# 定义神经网络的组成
class FNN:
    #初始化网络结构，其中sizes是[输入层，任意多隐藏层，输出层]的节点数的列表，f是我们需要训练的数据的特征数
    def __init__(self,sizes): 
        self.layer_nums=len(sizes) #获取网络层数
        self.sizes=sizes
        self.bayes=[np.random.randn(i,1) for i in sizes[1:]] #获取每层贝叶斯偏置
        self.weights=[np.random.randn(i,j) for i,j in zip(sizes[1:],sizes[:-1])] #获取每一层的权重
    
    # 定义激活函数，这里使用sigmoid
    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))
    
    # 定义梯度下降函数
    def gd(self,iters):
        for j in range(iters):
            pass

if __name__ == '__main__':
    net=FNN([3,2,1])
    print(net.layer_nums)
    print(net.sizes)
    print(net.bayes)
    print(net.weights)