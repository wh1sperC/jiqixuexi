# 纯numpy实现可变神经网络
import numpy as np
import matplotlib.pyplot as plt
import chineseize_matplotlib

plt.rcParams['axes.unicode_minus']=False


# 编辑神经网络架构类
class FNN:
    def __init__(self,sizes):
        self.sizes=sizes #这里的sizes是一个列表，用来记录网络结构
        self.layer_nums=len(sizes) #网络层数
        self.bayes=[np.random.rand(i,1) for i in sizes[1:]]
        self.weights=[np.random.rand(j,i) for i,j in zip(sizes[:-1],sizes[1:])]
    
    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))
    
    def forward():
        pass