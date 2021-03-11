# Description：MLP搭建非线性二分类模型
# Author：朱勇
# Email: yong_zzhu@163.com
# Time：2021/3/11 22:06

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Activation
from sklearn.metrics import accuracy_score

#数据导入与可视化
data = pd.read_csv('task1_data.csv')
x = data.drop(['y'],axis=1)
y = data.loc[:,'y']
fig1 = plt.figure(figsize=(5,5))
plt.scatter(x.loc[:,'x1'][y==1],x.loc[:,'x2'][y==1],label='label1')
plt.scatter(x.loc[:,'x1'][y==0],x.loc[:,'x2'][y==0],label='label0')
plt.title('row data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
#数据预处理（数据分离）
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
#建立模型
mlp = Sequential()
mlp.add(Dense(units=25,input_dim=2,activation='sigmoid'))
mlp.add(Dense(units=1,activation='sigmoid'))
mlp.summary()
#模型配置
mlp.compile(optimizer='adam',loss='binary_crossentropy')
#模型训练
mlp.fit(x_train,y_train,epochs=8000)
#模型预测
y_train_predict = mlp.predict_classes(x_train)
y_test_predict = mlp.predict_classes(x_test)
#模型评估
accuracy_train = accuracy_score(y_train,y_train_predict)
print(accuracy_train)
accuracy_test = accuracy_score(y_test,y_test_predict)
print(accuracy_test)
#结果可视化，生成新的数据点，用于画出决策边界
xx,yy = np.meshgrid(np.arange(0,100,1),np.arange(0,100,1))
x_range = np.c_[xx.ravel(),yy.ravel()]
y_range_predict = mlp.predict_classes(x_range)
#预测结果数据类型转化，用于可视化时检索
y_range_predict_format = pd.Series(i[0] for i in y_range_predict)
#结果可视化
fig2 = plt.figure(figsize=(5,5))
plt.scatter(x_range[:,0][y_range_predict_format==1],x_range[:,1][y_range_predict_format==1],label='label1_predict')
plt.scatter(x_range[:,0][y_range_predict_format==0],x_range[:,1][y_range_predict_format==0],label='label0_predict')
plt.scatter(x.loc[:,'x1'][y==1],x.loc[:,'x2'][y==1],label='label1')
plt.scatter(x.loc[:,'x1'][y==0],x.loc[:,'x2'][y==0],label='label0')
plt.title('row data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

