#MF矩阵分解模型实现用户对电影评分的预测

import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm #进度条工具
from sklearn.model_selection import train_test_split

#数据读取和预处理
data=np.genfromtxt("../data/movielens_100k.csv",dtype=int,delimiter=',')
data[:,:2]=data[:,:2]-1 #规整化编号，使电影id和用户id从0开始
train,test=train_test_split(data,test_size=0.2)
#统计用户总数和电影总数;统计训练集中每个用户和电影出现的数量，作为正则化约束强度
user_num=len(np.unique_values(data[:,0]))
item_num=len(np.unique_values(data[:,1]))
user_cnt=np.bincount(train[:,0],minlength=user_num)
item_cnt=np.bincount(train[:,1],minlength=item_num)
#用户和电影的编号作为下标，需保存为整数
user_train,user_test=train[:,0],test[:,0]
item_train,item_test=train[:,1],test[:,1]
y_train,y_test=train[:,2],test[:,2]#这里是电影的评分，也就是结果集

#实现MF分解的类
class MF:
    def __init__(self,N,M,d):
        """
        N是用户数量，M是电影数量，d是特征的维度
        Args:
            N (int): 用户数量
            M (int): 电影数量
            d (int): 特征的维度
        """
        #初始化参数
        self.user_params=np.ones((N,d))
        self.item_params=np.ones((M,d))
    def pred(self,user_id,item_id):
        """
        预测用户user_id对电影item_id的分数
        Args:
            user_id (int): 用户编号
            item_id (int): 电影编号
        """
        user_param=self.user_params[user_id]
        item_param=self.item_params[item_id]
        #返回预测的分数
        return np.sum(user_param*item_param,axis=1)
    def update(self,user_grad,item_grad,lr):
        """
        基于梯度下降的参数更新(公式可以参考《动手学习机器学习》第83页)
        Args:
            user_grad (ndarray):损失函数对用户特征向量的梯度
            item_grad (ndarray): 损失函数对电影特征向量的梯度
            lr (float):梯度下降算法的学习率 
        """
        self.user_params-=lr*user_grad
        self.item_params-=lr*item_grad
        
#训练函数
def train(model,learning_rate,lbd,max_training_step,batch_size):
    train_losses = []
    test_losses = []
    batch_num = int(np.ceil(len(user_train) / batch_size))#ceil是向上取整操作
    with tqdm(range(max_training_step * batch_num)) as pbar:
        for epoch in range(max_training_step):
            # 随机梯度下降
            train_rmse = 0#训练的均方根误差
            for i in range(batch_num):
                # 获取当前批量
                st = i * batch_size
                ed = min(len(user_train), st + batch_size)
                user_batch = user_train[st: ed]
                item_batch = item_train[st: ed]
                y_batch = y_train[st: ed]
                # 计算模型预测
                y_pred = model.pred(user_batch, item_batch)
                # 计算梯度
                P = model.user_params
                Q = model.item_params
                errs = y_batch - y_pred
                P_grad = np.zeros_like(P)
                Q_grad = np.zeros_like(Q)
                for user, item, err in zip(user_batch, item_batch, errs):
                    P_grad[user] = P_grad[user] - err * Q[item] + lbd * P[user]
                    Q_grad[item] = Q_grad[item] - err * P[user] + lbd * Q[item]
                #注意，这里除了一个len()去做了一个梯度归一化的操作，梯度归一化有许多方法和优点(实际上使用的只是小批量梯度下降的一般形式)
                model.update(P_grad /len(user_batch), Q_grad /len(user_batch), learning_rate)
                
                train_rmse += np.mean(errs ** 2)
                # 更新进度条
                pbar.set_postfix({
                    'Epoch': epoch,
                    'Train RMSE': f'{np.sqrt(train_rmse / (i + 1)):.4f}',
                    'Test RMSE': f'{test_losses[-1]:.4f}' if test_losses else None
                })
                pbar.update(1)

            # 计算 RMSE 损失
            train_rmse = np.sqrt(train_rmse / len(user_train))
            train_losses.append(train_rmse)
            y_test_pred = model.pred(user_test, item_test)
            test_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
            test_losses.append(test_rmse)
    
    return train_losses, test_losses

#训练模型
#可视化训练损失函数曲线
# 超参数
feature_num = 16 # 特征数
learning_rate = 0.1 # 学习率
lbd = 1e-4 # 正则化强度
max_training_step = 30
batch_size = 64 # 批量大小

# 建立模型
model = MF(user_num, item_num, feature_num)
# 训练部分
train_losses, test_losses = train(model, learning_rate, lbd, 
    max_training_step, batch_size)

plt.figure()
x = np.arange(max_training_step) + 1
plt.plot(x, train_losses, color='blue', label='train loss')
plt.plot(x, test_losses, color='red', ls='--', label='test loss')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.show()