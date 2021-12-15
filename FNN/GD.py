# 使用梯度下降实现FNN神经网络
import numpy as np
import matplotlib.pyplot as plt
import chineseize_matplotlib
from numpy.lib.npyio import load


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