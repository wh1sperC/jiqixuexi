# -*- coding: utf-8 -*-
import numpy as np
from numpy import *
import time
import math
import matplotlib.pyplot as plt
import argparse

parser=argparse.ArgumentParser("Feedforward neural network ")
parser.add_argument('--dataset',      type=str,   default='Exam',            help='data')
parser.add_argument('--acfun',        type=list,  default=['tanh','sigmoid','sigmoid'],help='activity function')
parser.add_argument('--lossfun',      type=str,   default='least_square',    help='loss function')
parser.add_argument('--layers',       type=int,   default=4,                 help='nums of all layers')
parser.add_argument('--nodes',        type=list,  default=[2,7,6,2],           help='nodes of each layer')
parser.add_argument('--drop_percent', type=float, default=0.01,              help='drop_percent')
parser.add_argument('--TYPE',         type=int,   default=2,                 help='type of answer')
args=parser.parse_args()


path="./FNN/{}".format(args.dataset)
train_x=np.genfromtxt("{}/train/x.txt".format(path))
train_y=np.genfromtxt("{}/train/y.txt".format(path))
test_x=np.genfromtxt("{}/test/x.txt".format(path))
test_y=np.genfromtxt("{}/test/y.txt".format(path))
plt.rcParams['font.family'] = 'SimHei'
fig = plt.figure(facecolor = 'lightgrey')
plt.rcParams['axes.unicode_minus']=False

epoches=2000
layers=args.layers
nodes=args.nodes
drop_percent=args.drop_percent
TYPE=args.TYPE
M=train_x.shape[0]

x_0=[];y_0=[]
x_1=[];y_1=[]
x_2=[];y_2=[]
for j in range(M):
    if train_y[j]==1:
        x_1.append(train_x[j][0])
        y_1.append(train_x[j][1])
    elif train_y[j]==2 :
        x_2.append(train_x[j][0])
        y_2.append(train_x[j][1])
    else:
        x_0.append(train_x[j][0])
        y_0.append(train_x[j][1])

labels=zeros((train_x.shape[0],TYPE))
for k in range(train_x.shape[0]):
    labels[k][int(train_y[k])]=1

def normalization(x):
    for i in range(x.shape[0]):
        max_num=max(x[i])
        min_num=min(x[i])
        for j in range(x.shape[1]):
            x[i][j]=(x[i][j]-min_num)/(max_num-min_num)
    return x

def sigmoid(x):
    return 1/(1+exp(-x))

def tanh(x):
    return (exp(x)-exp(-x))/(exp(x)+exp(-x))  

class Network():
    def __init__(self,layers,nodes,TYPE):
        if len(nodes)!=layers or layers==0:
            raise Exception("error layers or nodes")
        self.layers=layers
        self.nodes=nodes
        self.TYPE=TYPE
        self.hidden_layers=[]
        self.weights=[np.random.rand(x,y) for x,y,i in zip(nodes[:-1],nodes[1:],range(0,layers,1))]
        self.biases=[np.zeros((1,y)) for y in nodes[1:]]

        
    def feedforward(self,idx):
        x=idx
        self.hidden_layers.append(x)
        for i in range(self.layers-1):
            x=np.dot(x,self.weights[i])+self.biases[i]
            if args.acfun[i]=='tanh':
                noutput=tanh(x)
            if args.acfun[i]=='sigmoid':
                noutput=sigmoid(x)
            self.hidden_layers.append(noutput)
        return noutput

    def backward(self,label):
        errors=[]
        nodes=self.nodes
        n=1
        idx=self.hidden_layers[layers-2]
        idy=self.hidden_layers[-1]
        if args.lossfun=='least_square':
            error_last=idy[0]-label
        if args.lossfun=='cross_entropy':
            error_last=-label/idy[0]
        weight=np.ones((nodes[-1],1))
        error=np.zeros(nodes[-1])
        for j in range(nodes[-1]):
            for l in range(n):
                error[j]+=error_last[j]*weight[j][l]*idy[0][j]*(1-idy[0][j])
        errors.append(error)

        for i in range(layers-2):
            idx=self.hidden_layers[layers-3-i]
            idy=self.hidden_layers[layers-2-i]
            n=nodes[layers-1-i]
            error_last=error
            error=np.zeros(nodes[layers-2-i])
            weight=self.weights[layers-2-i]
            for j in range(nodes[layers-2-i]):
                for l in range(n):
                    error[j]+=error_last[l]*weight[j][l]*idy[0][j]*(1-idy[0][j])

            errors.append(error)
       
        for i in range(layers-1):
            idx=self.hidden_layers[layers-2-i]
            idy=self.hidden_layers[layers-1-i]
            self.weights[layers-2-i]-=np.dot(idx.T.reshape(nodes[layers-2-i],1),errors[i].reshape((1,nodes[layers-1-i])))*args.drop_percent
            self.biases[layers-2-i]-=errors[i]*args.drop_percent
    
    def train(self):
        lost=[]
        xlabel=[]
        for epoch in range(epoches):
            loss=0
            for k in range(train_x.shape[0]):
                hat_x=self.feedforward(train_x[k])
                self.backward(labels[k])
                self.hidden_layers.clear()
                if args.lossfun=='least_square':
                    loss+=least_square(hat_x,labels[k])
                if args.lossfun=='cross_entropy':
                    loss-=cross_entropy(hat_x,labels[k])
            
            lost.append(loss)
            xlabel.append(epoch)
            if (epoch+1)%50 == 0:
                print("epoch {}:loss:{:.6f}".format(epoch+1,loss))
                plt.clf()

                plt.figure(1)
                plt.plot(xlabel,lost)
        
        
                plt.pause(0.01)
                plt.ioff()

    def test(self):
        hat_x=[]
        for k in range(test_x.shape[0]):
            x=test_x[k]
            for i in range(layers-1):
                x=np.dot(x,self.weights[i])+self.biases[i]
                if args.acfun[i]=='tanh':
                    x=tanh(x)
                if args.acfun[i]=='sigmoid':
                    x=sigmoid(x)
            hat_x.append(x)
        print(hat_x)
        hat_x=onehot_encode(np.array(hat_x))
        print("predict:")
        print(hat_x)
        print(test_y)
        print("accuracy:{:.4f}".format(accuracy(hat_x,test_y)))

    def predict(self,X,Y):
        z=[]
        ninput=[]
        for i in range(300):
            for j in range(300):
                ninput.append([X[i][j],Y[i][j]])
        ninput=np.array(ninput).reshape((90000,2))
        for k in range(layers-1):
            ninput=np.dot(ninput,self.weights[k])+self.biases[k]
        hat_x=onehot_encode(ninput).reshape(300,300)
        return hat_x
        
def least_square(hat_x,label):
    loss=0
    for i in range(label.shape[0]):
        loss+=((hat_x[0][i]-label[i])**2)/2
    return loss

def cross_entropy(hat_x,label):
    loss=0
    for i in range(label.shape[0]):
        loss+=label[i]*log(hat_x[0][i])
    return loss


def onehot_encode(x):
    n=x.shape[0]
    hat_x=np.zeros(n)
    for i in range(n):
        hat_x[i]=argmax(x[i][0])
    return hat_x

def accuracy(hat_x,label):
    num=0
    N=hat_x.shape[0]
    for i in range(N):
        if hat_x[i]==label[i]:
            num+=1
    return num/N

#if __name__=='__main__':
train_x=normalization(train_x.T).T
test_x=normalization(test_x.T).T
FNN=Network(layers,nodes,TYPE)
FNN.train()
FNN.test()
