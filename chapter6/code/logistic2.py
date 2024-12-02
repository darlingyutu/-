import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc


# 数据加载与划分
data = np.genfromtxt("../data/lr_dataset.csv", dtype=float, delimiter=',')
x_origin = data[:, 0:2]
y_origin = data[:, -1:]
rng = np.random.default_rng()
ratio = 0.7
idx = rng.permutation(len(x_origin))
x_total, y_total = x_origin[idx], y_origin[idx]
x_train, y_train = x_total[:int(ratio * len(idx))], y_total[:int(ratio * len(idx))]
x_test, y_test = x_total[int(ratio * len(idx)):], y_total[int(ratio * len(idx)):]

# 评估指标
def acc(y_true, y_pred):
    return np.mean(y_pred == y_true)


# Logistic 函数
def logistic(z):
    return 1 / (1 + np.exp(-z))

# 梯度下降
def GD0(x, y, epoch, learning_rate, l2):
    x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)  # 添加偏置项
    theta = np.random.normal(size=(x.shape[1], 1))
    for i in range(epoch):
        theta=(1-l2*learning_rate)*theta+learning_rate*x.T@(y-logistic(x@theta))
    return theta

# 梯度下降
def GD(x, y, epoch, learning_rate, l2):
    x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)  # 添加偏置项
    theta = np.random.normal(size=(x.shape[1], 1))
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    train_auc = []
    test_auc = []
    
    for i in range(epoch):
        pred = logistic(x @ theta)  # 预测值
        grad = -x.T @ (y - pred) + l2 * theta  # 梯度
        theta -= learning_rate * grad  # 更新 theta
        
        # 计算训练集的损失
        train_loss = - y.T @ np.log(pred) - (1 - y).T @ np.log(1 - pred) + l2 * np.linalg.norm(theta) ** 2 / 2
        train_losses.append(train_loss.item())#这里的item()是numpy中的接口，用于把train_loss转化成标量，不然它会是1一个1维数组
         
        # 计算测试集的损失
        test_pred = logistic(np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1) @ theta)
        test_loss = - y_test.T @ np.log(test_pred) - (1 - y_test).T @ np.log(1 - test_pred)
        test_losses.append(test_loss.item())
        
        # 记录训练和测试的准确率
        train_acc.append(acc(y, pred >= 0.5))
        test_acc.append(acc(y_test, test_pred >= 0.5))
        
        # 计算训练集的 AUC
        fpr_train, tpr_train, _ = roc_curve(y, pred)
        train_auc.append(auc(fpr_train, tpr_train))

        # 计算测试集的 AUC
        fpr_test, tpr_test, _ = roc_curve(y_test, test_pred)
        test_auc.append(auc(fpr_test, tpr_test))

    
    return theta, train_losses, test_losses, train_acc, test_acc, train_auc, test_auc

# 定义参数
num_steps = 300
learning_rate = 0.01
l2 = 1.0

# 训练模型
theta, train_losses, test_losses, train_acc, test_acc, train_auc, test_auc = GD(x_train, y_train, num_steps, learning_rate, l2)

# 预测准确率
y_pred = np.where(logistic(np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1) @ theta) >= 0.5, 1, 0)
final_acc = acc(y_test, y_pred)
print(f'预测精度为: {final_acc}')

#绘图
fig,((ax0,ax1),(ax2,ax3))=plt.subplots(2,2,figsize=(12,9))
#损失函数曲线
train_loss = np.array(train_losses).reshape(-1, 1)
test_loss = np.array(test_losses).reshape(-1, 1)
train_loss=preprocessing.MinMaxScaler().fit_transform(train_loss)
test_loss=preprocessing.MinMaxScaler().fit_transform(test_loss)
ax0.plot(range(num_steps),train_loss,label="train_losses")
ax0.plot(range(num_steps),test_loss,label="test_losses")
ax0.set_xlabel("Epoch")
ax0.set_ylabel("Loss")
ax0.set_xlim([0,100])
ax0.legend()
#acc
ax1.plot(range(num_steps),train_acc,label="train_acc")
ax1.plot(range(num_steps),test_acc,label="test_acc")
ax1.legend()
#auc
ax2.plot(range(num_steps),train_auc,label="train_auc")
ax2.plot(range(num_steps),test_auc,label="test_auc")
ax2.legend()
#绘制数据分布散点图，并绘制出决策边界
position_idx=np.where(y_total==1)
negative_idx=np.where(y_total==0)
ax3.scatter(x_total[position_idx,0],x_total[position_idx,1],label="positive")
ax3.scatter(x_total[negative_idx,0],x_total[negative_idx,1],label="negative")
#以下是绘制决策边界的绘图
plot_x = np.linspace(-1.1, 1.1, 100)
plot_y = -(theta[0]*plot_x+theta[2])/theta[1]
ax3.plot(plot_x,plot_y,label="Decision Boundary",color="black")
ax3.legend()
plt.show()

 