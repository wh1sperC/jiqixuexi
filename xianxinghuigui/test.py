import numpy as np


class Liner():

    def __init__(self,alpha,iter):
        self.alpha=alpha
        self.iter=iter

    def fit(self,X,y):
        X=np.array(X)
        y=np.array(y)
        self.theta=np.zeros(1+X.shape[1])
        self.loss=[]
        for i in range(self.iter):
            h=self.theta[0]+np.dot(X,self.theta[1:])
            error=y-h
            self.loss.append(np.sum(error**2)/2)
            self.theta[0]+=self.alpha*np.sum(error)
            self.theta[1:]+=self.alpha*np.dot(X.T,error)

    def predict(self,X):
        X=np.array(X)
        result = np.dot(X, self.theta[1:]) + self.theta[0]
        return result

if __name__ == '__main__':
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
    m = len(x)  # 样本个数
    x = np.mat(x).T
    x -= 2000
    y = np.mat(y).T  # 这里换成列向量
    x0 = np.ones((m, 1))
    X = np.append(x0, x, axis=1)
    l=Liner(alpha=0.0001,iter=100)
    l.fit(X,y)
    print(l.theta)

