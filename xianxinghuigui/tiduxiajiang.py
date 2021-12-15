# -*- coding:gbk -*-
# 使用梯度下降方法构建线性回归模型
import numpy as np
import matplotlib.pyplot as plt
import chineseize_matplotlib


# 导入数据
x = []
y = []
with open("./Price/x.txt") as f1:
    for line in f1:
        x.append(int(line))
with open("./Price/y.txt") as f2:
    for line in f2:
        y.append(float(line))
print('x=', x)
print('y=', y)
m=len(x) # 样本个数
x = np.mat(x).T
x-=2000
y = np.mat(y).T # 这里换成列向量
x0=np.ones((m,1))
X=np.append(x0,x,axis=1)
print(X)


#初始化参量
theta=np.mat([2,2]).T
alpha=0.01
iter=100

#h=np.dot(X,theta)
#print(h)
#h-=y
#print(h)

# 损失函数
def costf(X,y,theta,m):
    tmp=np.dot(X,theta)-y
    return 0.5*(np.dot(np.transpose(tmp),tmp))/m

# 梯度计算
def gradiant(X,y,theta,m):
    tmp = np.dot(X, theta) - y
    temp=np.dot(np.transpose(tmp),X).T
    return temp/m

# 迭代计算
def iterate(X,y,theta,m,alpha,iter):
    grad_lst=[]
    for i in range(iter):
        grad=gradiant(X,y,theta, m)
        grad_lst.append(grad)
        theta=theta-alpha*grad
    return theta,grad_lst


theta,grad_lst=iterate(X,y,theta,m,alpha,iter)
print(theta)
#print(grad_lst)
new_grad=[np.array(grad_lst[i][1,:]).tolist() for i in range(len(grad_lst))]
#print(new_grad)

#画图
#print(type(x))
x=np.array(x)
#print(type(x))
y=np.array(y)
plt.scatter(x+2000,y,c='b')
plt.plot(x+2000,theta[0,0]+theta[1,0]*x,'k-')
plt.scatter(2014,theta[0,0]+theta[1,0]*(2014-2000),c='r')
print("预测2014年的房价为：{0}".format(theta[0,0]+theta[1,0]*(2014-2000)))
plt.show()
plt.plot(range(iter),[new_grad[i][:][0] for i in range(len(new_grad))])
plt.show()