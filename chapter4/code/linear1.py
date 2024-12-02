#利用线性回归模型做房价预测（scikit-learn版）
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler
import pandas as pd
#载入数据并观察数据特征
table=np.genfromtxt("../data/USA_Housing.csv",delimiter=',',dtype=str)
header=table[0]#数据列的名称
data=table[1:].astype(float)
#划分训练集和测试集
ratio=0.8
split=int(len(data)*ratio)#训练集和测试集的idx分界点
shuffled_data=np.random.permutation(data)#打乱数据集
train,test=shuffled_data[:split],shuffled_data[split:]
#利用scikit-learn库的接口做数据标准化工作
scalar=StandardScaler()
scalar.fit(train)#只使用训练集的数据计算均值和方差
train=scalar.transform(train)
test=scalar.transform(test)
#划分输入数据和导出的结果
x_train,y_train=train[:,:-1],train[:,-1].flatten()#这里的flatten用于将数组扁平化，方便遍历
x_test,y_test=test[:,:-1],test[:,-1].flatten()
#初始化线性回归模型
linreg=LinearRegression()
#LinearRegression()已经考虑了线性回归的常数项，无需再拼接1
linreg.fit(x_train,y_train)
#coef_是训练所得回归系数,intercept_是常数项
print(f'回归系数是{linreg.coef_},常数项是{linreg.intercept_}')
y_pred=linreg.predict(x_test)

#计算预测值和真实值之间的rmse
rmse_loss=np.sqrt(np.square(y_test-y_pred).mean())
print(f'rmse is {rmse_loss}')
