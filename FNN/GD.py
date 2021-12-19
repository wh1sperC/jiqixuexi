# -*- coding: utf-8 -*-
import numpy as np
from numpy import *
import time
import math
import matplotlib.pyplot as plt
import argparse

path="./FNN/Exam"
train_x=np.loadtxt("{}/train/x.txt".format(path))
train_y=np.loadtxt("{}/train/y.txt".format(path))
test_x=np.loadtxt("{}/test/x.txt".format(path))
test_y=np.loadtxt("{}/test/y.txt".format(path))
plt.rcParams['font.family'] = 'SimHei'
fig = plt.figure(facecolor = 'lightgrey')
plt.rcParams['axes.unicode_minus']=False



#转换标签
labels=zeros((train_x.shape[0],2))
for k in range(train_x.shape[0]):
    labels[k][int(train_y[k])]=1

def sigmoid(x):
    return 1/(1+exp(-x))
            

class Network():
    def __init__(self,sizes,type,alpha):
        self.sizes=sizes
        self.type=type
        self.alpha=alpha
        self.hidden=[]
        self.weights=[np.zeros((x,y)) for x,y in zip(sizes[:-1],sizes[1:])]
        self.biases=[np.zeros((1,y)) for y in sizes[1:]]
        
    def feedforward(self,idx):
        x=idx
        self.hidden.append(x)
        for i in range(len(self.sizes)-1):
            x=np.dot(x,self.weights[i])+self.biases[i]
            noutput=sigmoid(x)
            self.hidden.append(noutput)
        return noutput

    def backforward(self,label):
        '''for i in range(len(self.sizes)-1):
            index=len(self.sizes)-1-i
            if i==0:
                dx=self.hidden[index-1]
                dy=self.hidden[index]
                error=np.zeros(self.sizes[index])
                for j in range(self.sizes[index]):
                    error[j]=(dy)'''
        
        idx=self.hidden[1]
        idy=self.hidden[2]
        
        error_out=np.zeros(self.sizes[2])
        print(error_out)
        for j in range(self.sizes[2]):
            error_out[j]=(idy[0][j]-label[j])*idy[0][j]*(1-idy[0][j])

        ix=self.hidden[0]
        iy=self.hidden[1]
        error_hidden=np.zeros(self.sizes[1])
        for j in range(self.sizes[1]):
            for l in range(self.sizes[2]):
                error_hidden[j]+=error_out[l]*self.weights[1][j][l]*iy[0][j]*(1-iy[0][j])
        print(error_hidden)
        self.weights[1]-=np.dot(idx.T,error_out.reshape((1,2)))*self.alpha
        self.biases[1]-=error_out*self.alpha
        self.weights[0]-=np.dot(ix.T.reshape(self.sizes[0],1),error_hidden.reshape(1,self.sizes[1]))*self.alpha
        self.biases[0]-=error_hidden*self.alpha
        

    def train(self,iter):
        lost=[]
        for epoch in range(iter):
            loss=0
            for k in range(train_x.shape[0]):
                hat_x=self.feedforward(train_x[k])
                self.backforward(labels[k])
                self.hidden.clear()
                loss+=((hat_x[0][0]-labels[k][0])**2+(hat_x[0][1]-labels[k][1])**2)/2
                if epoch==iter-1:
                    print("[{},{}],[{},{}]".format(hat_x[0][0],hat_x[0][1],labels[k][0],labels[k][1]))
            
            lost.append(loss)
            plt.clf()
            plt.figure(1)
            plt.plot(range(epoch+1),lost)
            plt.pause(0.01)
            plt.ioff()

        
def loss_function(hat_x,label):
    loss=0
    for i in range(label.shape[0]):
        loss+=((hat_x[0][i]-label[i])**2)/2
    return loss


def onehot_encode(x):
    n=x.shape[0]
    hat_x=np.zeros(n)
    for i in range(n):
        x_one=list(x[i])
        hat_x[i]=x_one.index(max(x_one))
    return hat_x
    
FNN=Network([train_x.shape[1],4,2],2,0.01)
FNN.train(1000)

