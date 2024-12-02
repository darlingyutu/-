import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_auc_score

# 数据加载与划分
data = np.genfromtxt("../data/lr_dataset.csv", dtype=float, delimiter=',')
x_origin = data[:, 0:2]
y_origin = data[:, -1:]
rng = np.random.default_rng()
ratio = 0.8
idx = rng.permutation(len(x_origin))
x_total, y_total = x_origin[idx], y_origin[idx]
x_train, y_train = x_total[:int(ratio * len(idx))], y_total[:int(ratio * len(idx))]
x_test, y_test = x_total[int(ratio * len(idx)):], y_total[int(ratio * len(idx)):]

# 模型评价指标
def acc(y_true, y_pred):
    return np.mean(y_pred == y_true)

def auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

# logistic函数
def logistic(z):
    return 1 / (1 + np.exp(-z))

# 梯度下降函数
def GD(num_steps, learning_rate, l2_coef):
    # 初始化模型参数，确保theta是列向量
    theta = np.random.normal(size=(X.shape[1], 1))  # 这里将theta初始化为(3, 1)的列向量
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    train_auc = []
    test_auc = []
    
    for i in range(num_steps):
        # 预测值
        pred = logistic(X @ theta)  # 预测值维度是 (N, 1)
        
        # 梯度计算，确保操作维度一致
        grad = -X.T @ (y_train - pred) + l2_coef * theta
        
        # 更新theta
        theta -= learning_rate * grad
        
        # 计算训练集的损失
        train_loss = - y_train.T @ np.log(pred) - (1 - y_train).T @ np.log(1 - pred) + l2_coef * np.linalg.norm(theta) ** 2 / 2
        train_losses.append(train_loss.item())  # 确保train_loss是标量
        
        # 计算测试集的损失
        test_pred = logistic(X_test @ theta)
        test_loss = - y_test.T @ np.log(test_pred) - (1 - y_test).T @ np.log(1 - test_pred)
        test_losses.append(test_loss.item())  # 确保test_loss是标量
        
        # 记录训练和测试的准确率
        train_acc.append(acc(y_train, pred >= 0.5))
        test_acc.append(acc(y_test, test_pred >= 0.5))
        
        # 记录训练和测试的AUC
        train_auc.append(auc(y_train, pred))
        test_auc.append(auc(y_test, test_pred))
    
    return theta, train_losses, test_losses, train_acc, test_acc, train_auc, test_auc


# 定义参数
num_steps = 250
learning_rate = 0.002
l2_coef = 1.0
np.random.seed(0)

# 拼接1列用于偏置
X = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
X_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)

# 训练
theta, train_losses, test_losses, train_acc, test_acc, train_auc, test_auc = GD(num_steps, learning_rate, l2_coef)

# 预测准确率
y_pred = np.where(logistic(X_test @ theta) >= 0.5, 1, 0)
final_acc = acc(y_test, y_pred)
print('预测准确率：', final_acc)
print('回归系数：', theta)

# 可视化
plt.figure(figsize=(13, 9))
xticks = np.arange(num_steps) + 1

# 准确率
plt.subplot(222)
plt.plot(xticks, train_acc, color='blue', label='train accuracy')
plt.plot(xticks, test_acc, color='red', ls='--', label='test accuracy')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# AUC
plt.subplot(223)
plt.plot(xticks, train_auc, color='blue', label='train AUC')
plt.plot(xticks, test_auc, color='red', ls='--', label='test AUC')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()

# 决策边界
plt.subplot(224)
plot_x = np.linspace(-1.1, 1.1, 100)
plot_y = -(theta[0] * plot_x + theta[2]) / theta[1]
pos_index = np.where(y_total == 1)
neg_index = np.where(y_total == 0)
plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', color='coral', s=10)
plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', color='blue', s=10)
plt.plot(plot_x, plot_y, ls='-.', color='green')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.xlabel('X1 axis')
plt.ylabel('X2 axis')
plt.savefig('../result/output_16_1.png')
plt.show()
