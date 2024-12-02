#采用随机梯度下降法训练线性回归模型(在现代机器学习中，SGD与MBGD叫法差别不大，MBGD是小批量梯度下降)
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

def batch_generator(x,y,batch_size,shuffle=True):
    """
       生成小批量数据，用于训练模型。

    该函数将输入的数据集 `x` 和对应的标签 `y` 划分为多个批次， 
    每个批次的大小为 `batch_size`。可以选择是否对数据进行随机打乱。
    在每一次迭代中，返回一批次的数据和标签，供训练使用。
    Args:
        x (ndarray): 训练集
        y (ndarray): 训练集结果标签
        batch_size (int): 将训练集划分为batch_size批进行训练
        shuffle (bool, optional): 选择是否随机打乱数据集. Defaults to True.
    """
    batch_count=0 #批量计数器
    if shuffle:
        #随机生成0到len(x)-1的下标
        idx=np.random.permutation(len(x))
        x=x[idx]
        y=y[idx]
        
    while True:
        #选定批量数据集的范围为[start:end)
        start = batch_count*batch_size 
        end = min(start+batch_size,len(x))
        if start >= end:
            #若已经遍历完一遍,结束生成
            break
        batch_count+=1
        yield x[start:end],y[start:end]
               
def SGD(num_epoch, learning_rate, batch_size):
    # 拼接原始矩阵
    X = np.concatenate([x_train, np.ones((len(x_train), 1))], axis=-1)
    X_test = np.concatenate([x_test, np.ones((len(x_test), 1))], axis=-1)
    # 随机初始化参数
    theta = np.random.normal(size=X.shape[1])

    # 小批量随机梯度下降
    # 为了观察迭代过程，我们记录每一次迭代后在训练集和测试集上的均方根误差
    train_losses = []
    test_losses = []
    for i in range(num_epoch):
        # 初始化批量生成器
        batch_g = batch_generator(X, y_train, batch_size, shuffle=True)
        train_loss = 0
        for x_batch, y_batch in batch_g:
            # 计算梯度
            grad = x_batch.T @ (x_batch @ theta - y_batch)
            # 更新参数
            theta = theta - learning_rate * grad / len(x_batch)
            # 累加平方误差
            train_loss += np.square(x_batch @ theta - y_batch).sum()
        # 计算训练和测试误差
        train_loss = np.sqrt(train_loss / len(X))#均方根误差
        train_losses.append(train_loss)
        test_loss = np.sqrt(np.square(X_test @ theta - y_test).mean())
        test_losses.append(test_loss)

    # 输出结果，绘制训练曲线
    print('回归系数：', theta)
    return theta, train_losses, test_losses

# 设置迭代次数，学习率与批量大小
num_epoch = 20
learning_rate = 0.01
batch_size = 32
# 设置随机种子
np.random.seed(0)

_, train_losses, test_losses = SGD(num_epoch, learning_rate, batch_size)
    
# 将损失函数关于运行次数的关系制图，可以看到损失函数先一直保持下降，之后趋于平稳
plt.plot(np.arange(num_epoch), train_losses, color='blue', 
    label='train loss')
plt.plot(np.arange(num_epoch), test_losses, color='red', 
    ls='--', label='test loss')
# 由于epoch是整数，这里把图中的横坐标也设置为整数
# 该步骤也可以省略
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.show()