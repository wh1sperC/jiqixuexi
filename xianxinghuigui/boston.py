from tkinter import N
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as stm
from matplotlib import cm

boston=load_boston()
#print(boston)
#print(boston['data'])
#print(boston['target'])
print(boston.data.shape)
print(boston.target.shape)
X,y=boston.data,boston.target.reshape((boston.data.shape[0],1))
df=pd.DataFrame(X,columns=boston.feature_names)

print(df.head(10))
print(df.describe())
print(df.info())
print(y)
df['TARGET']=y
print(df)
df.to_csv('./xianxinghuigui/housing.csv',index=False)