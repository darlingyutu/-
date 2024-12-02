#基于python标准库实现的knn算法(手写数字图像分类MNIST数据集,灰度图像素为28*28)
import numpy as np
import matplotlib.pyplot as plt
import os
#读取手写数字的数据集
m_x=np.genfromtxt('../data/mnist_x',delimiter=' ')
m_y=np.genfromtxt('../data/mnist_y')
#划分训练集和验证集比例(8:2)
ratio=0.8
split=int(len(m_x)*ratio)#用于选择训练的数据集
#打乱数据集
rng=np.random.default_rng()
idx=rng.permutation(len(m_x))#生成乱序array，其中的值是range(0,len(m_x))的乱序版
m_x=m_x[idx]#用乱序标签idx打乱数据集
m_y=m_y[idx]#这里的结果也需要相同的idx打乱，保证验证的可靠性
x_train,x_test=m_x[0:split],m_x[split:]
y_train,y_test=m_y[0:split],m_y[split:]

#数据集可视化，显示手写数字(28*28像素)
data=np.reshape(np.array(m_x[0],dtype=int),[28,28])
#这里mnist_x每一行代表一个数字，但是要先reshape成28*28的像素图
fig,ax=plt.subplots()
ax.imshow(data,cmap='gray')
#plt.show()#这里同时可以用来验证是否打乱了数据集

#定义对样本点相似度的度量，这里使用的是欧式距离
def distance(a:np.array,b:np.array)->int:
    return np.sqrt(np.sum(np.square(a-b)))

class knn1:
    def __init__(self,k,label_num):
        """
        knn1初始化器，定义临近样本点的数量k和类别的数量label_num
        Args:
            k (int): 使用k个样本点来构建邻居集合
            label_num (int): 类别的数量
        """
        self.k=k
        self.label_num=label_num
        
    def fit(self,x_train,y_train):
        """
        在类中保存训练数据
        Args:
            x_train (np.array): _description_
            y_train (np.array): _description_
        """
        self.x_train=x_train
        self.y_train=y_train
    
    def get_knn_indices(self,x):
        """
        获取距离目标样本点最近的k个样本点的idx
        Args:
            x (np.array): 输入的目标样本点
        """
        dis=[distance(a,x) for a in self.x_train]#计算已知样本点(x)到目标样本点的距离
        knn_indices=np.argsort(dis)#argsort返回元素从小到大的索引
        return knn_indices[0:self.k]#取最近的k个点的idx
    
    def get_label(self,x):
        """
        knn具体实现,观察k邻近并使用np.argmax获取数量最多的类别
        Args:
            x (np.array): 输入样本点
        """
        knn_indices=self.get_knn_indices(x)
        #类别计数
        label_type=np.zeros(shape=[self.label_num])
        for idx in knn_indices:
            label=int(self.y_train[idx])
            label_type[label]+=1
        #返回数量最多的类别
        return np.argmax(label_type)#argmax返回的是最大值的索引，这里索引的值刚好就是手写数字
    
    def predict(self,x_test):
        """
        训练集预测
        Args:
            x_test (np.array): 输入训练集
        """
        predict_labels=np.zeros(shape=len(x_test),dtype=int)#预测样本x_test的类别
        for i,x in enumerate(x_test):
            #enumerate(iterable,start=0)
            #enumerate() 返回一个枚举对象，这个对象是一个迭代器，产生的每个元素都是一个包含索引和对应值的元组。
            #遍历测试集，将预测结果加入predict_labels
            predict_labels[i]=self.get_label(x)
        return predict_labels
    
    
#接下来使用knn验证
for k in range(1,10):
    knn=knn1(k,label_num=10)#因为数字只有0-9十类
    knn.fit(x_train,y_train)#载入数据
    predicted_labels=knn.predict(x_test)
    accuracy=np.mean(predicted_labels==y_test)
    print(f'when k is {k},the accuracy of the prediction is {accuracy*100:.2f}%')
    

    
    