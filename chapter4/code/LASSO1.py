#SGD接口实现LASSO回归
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor

# 读取数据
table = np.genfromtxt("../data/USA_Housing.csv", delimiter=',', dtype=float, skip_header=1)
x_origin = table[:, :-1]
y_origin = table[:, -1]

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x_origin, y_origin, test_size=0.2)

# 数据标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)  # 仅使用训练集的数据计算均值和方差
x_test = scaler.transform(x_test)  # 使用相同的均值和标准差对测试集进行标准化

# 使用SGD训练模型参数
model = SGDRegressor( max_iter=50000,eta0=0.001)  # 增加最大迭代次数
model.fit(x_train, y_train)  # 训练模型

# 使用模型预测
y_pred = model.predict(x_test)

# 计算均方根误差（RMSE）
rmse=np.sqrt(np.mean(np.square(y_pred-y_test)))
fb=rmse/y_pred.mean()#预测偏差
print(f'rmse is {rmse} dollars,forecast bias is {fb*100:.4f}%')