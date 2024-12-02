#基于scikit-learn实现的logitstic模型
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 数据加载与划分
data = np.genfromtxt("../data/lr_dataset.csv", dtype=float, delimiter=',')
x_origin = data[:, 0:2]
y_origin = data[:, -1:] 
# 1. 数据标准化
scaler = StandardScaler()
x = scaler.fit_transform(x_origin)  # 标准化特征数据
# 2. 数据集划分：70% 训练集, 30% 测试集
X_train, X_test, y_train, y_test = train_test_split(x, y_origin, test_size=0.3)
# 3. 构建 Logistic 回归模型
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
# 4. 预测
y_pred = log_reg.predict(X_test)
# 5. 模型评估
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
