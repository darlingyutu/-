#利用线性回归模型做房价预测（主体数据使用numpy处理）
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
#在数据矩阵x的最后一列添加1，方便后边运算的时候吸收w系数向量的常数项w0(具体关注线性回归方程的公式推导)
x=np.concatenate([x_train,np.ones((len(x_train),1))],axis=-1)#axis=-1代表沿着最后一个维度拼接
#@代表矩阵相乘，x.T代表矩阵转置,np.linalg.inv计算矩阵的逆
w=np.linalg.inv(x.T @ x) @ x.T @ y_train #训练集训练出的w权重向量
#在测试集上使用训练出的w权重向量进行预测
x_test=np.concatenate([x_test,np.ones((len(x_test),1))],axis=-1)
y_pred=x_test @ w
#计算真实值和预测值之间的均方根误差

rmse_loss=np.sqrt(np.square(y_test-y_pred).mean())
print(f'rmse is {rmse_loss}')



