# --coding:gbk --
# 使用解析解方法构建线性回归模型
import numpy as np
import matplotlib.pyplot as plt
import chineseize_matplotlib


# 读入数据
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
x = np.array(x)
x-=2000
y = np.array(y)

# 建立模型
plt.scatter(x, y, c='b')  # 显示数据的散点图

xx = np.mean(x, axis=None)  # 平均值
x2 = np.sum([i ** 2 for i in x])  # 平方和
#print(xx)
#print(x2)
m = len(x)  # 样本个数
w = (np.sum([x[i] * y[i] for i in range(m)]) - xx * np.sum(y)) / (x2 - np.sum(x) ** 2 / m)  # 参数w
b = np.sum([y[i] - w * x[i] for i in range(m)]) / m  # 参数b
print('w=', w)
print('b=', b)

#print(w * x + b)  # 画线
plt.plot(x, w * x + b, 'k-')

# 预测
x_predict = 2014-2000
y_predict = w * x_predict + b
plt.scatter(x_predict, y_predict, c='r')
print('预测2014年的房价为：', y_predict)
plt.title('南京房价预测')
plt.show()
