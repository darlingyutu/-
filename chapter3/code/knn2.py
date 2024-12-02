#基于scikit-learn实现的knn算法对gauss.csv的分类
from sklearn.neighbors import KNeighborsClassifier #knn分类器
from matplotlib.colors import ListedColormap
import matplotlib.pyplot  as plt
import numpy as np

data=np.genfromtxt('../data/gauss.csv',delimiter=',')
x_train=data[:,:2]#点的二维坐标
y_train=data[:,2]#点的类别
#这里我们使用在一定范围内随机生成的网格数据来做knn分类的测试集
x_min,x_max=np.min(x_train[:,0])-1,np.max(x_train[:,0])+1
y_min,y_max=np.min(x_train[:,1])-1,np.max(x_train[:,1])+1#以上用于定义网格边界
#构造网格
step=0.2#采样步长为0.2（也可以通过x或者y坐标的极差和点的数量做更合理的量化）
x,y=np.meshgrid(np.arange(x_min,x_max,step),np.arange(y_min,y_max,step))
grid_data=np.concatenate([x.reshape(-1,1),y.reshape(-1,1)],axis=1)#这里的一行就是一个随机点的坐标
#定义knn分类器训练,绘图
fig,ax=plt.subplots(2,2,figsize=(12,16),tight_layout=True)
cmap_light = ListedColormap(['royalblue', 'lightcoral'])  # 使用 ListedColormap 生成颜色图
ax=ax.flatten()#将子图展平，方便访问
for k in range(6,10):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    z=knn.predict(grid_data)
    #绘图
    ax[k-6].pcolormesh(x,y,z.reshape(x.shape),cmap=cmap_light,alpha=0.7)
    ax[k-6].scatter(x_train[y_train==0,0],x_train[y_train==0,1],c='blue',marker='o')
    ax[k-6].scatter(x_train[y_train==1,0],x_train[y_train==1,1],c='red',marker='x')
    ax[k-6].set_xlabel('X axis')
    ax[k-6].set_ylabel('Y axis')
    ax[k-6].set_title(f'K={k}')
    ax[k-6].legend()
    

plt.show()

