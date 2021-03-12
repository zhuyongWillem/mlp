# Description：
# Author：朱勇
# Email: yong_zzhu@163.com
# Time：2021/3/11 22:06

from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

#数据的导入
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
print(type(x_train),x_train.shape)
fig1 = plt.figure(figsize=(3,3))
img1 = x_train[0]
plt.imshow(img1)
plt.title('raw img 1')
plt.show()
#输入数据预处理
feature_size = img1.shape[0]*img1.shape[1]
x_train_format = x_train.reshape(x_train.shape[0],feature_size)
x_test_format = x_test.reshape(x_test.shape[0],feature_size)
print(x_train_format.shape,x_train.shape)
#归一化
x_train_normal = x_train_format / 255
x_test_normal = x_test_format / 255
#输出结果预处理one_hot
y_train_format = to_categorical(y_train)
y_test_format = to_categorical(y_test)
#建立MLP模型
mlp = Sequential()
mlp.add(Dense(units=392,input_dim=784,activation='relu'))
mlp.add(Dense(units=196,activation='relu'))
mlp.add(Dense(units=10,activation='softmax'))
mlp.summary()
#配置参数与模型训练
mlp.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
mlp.fit(x_train_normal,y_train_format,epochs=10)
#结果预测
y_train_predict = mlp.predict_classes(x_train_normal)
accuracy_train = accuracy_score(y_train,y_train_predict)
print(accuracy_train)
y_test_predict = mlp.predict_classes(x_test_normal)
accuracy_test = accuracy_score(y_test,y_test_predict)
print(accuracy_test)
#创建结果标签字典
label_dict = {0:'T shirt',1:'裤子',2:'套头衫',3:'裙子',4:'外套',5:'凉鞋',6:'衬衫',7:'运动鞋',8:'包',9:'裸靴'}
img2 = x_train[80]
fig2 = plt.figure(figsize=(3,3))
plt.imshow(img2)
plt.title(label_dict[y_train_predict[80]],{'family':'SimHei'})
plt.show()