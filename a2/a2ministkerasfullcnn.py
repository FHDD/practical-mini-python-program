from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import a2svmministsample_gendatapropro

#调用mnist数据集
train_images, train_labels, test_images, test_labels, train_nums, test_nums = a2svmministsample_gendatapropro.run()
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#将数据格式变为（图片数量，28*28），取值为0-1，因为每个点的像素值为8位，即范围为0-255，所以归一化的时候除以255
train_images = train_images.reshape((60000, 28,28,1))
train_images = train_images.astype('float32') / 255
# train_images.dtype='float32'
# train_images=train_images/255
#
test_images = test_images.reshape((10000, 28,28,1))
test_images = test_images.astype('float32') / 255
# test_images.dtype='float32'
# test_images=train_images/255

#将训练标签向量化，将每个标签标示为全零向量，只有标签索引对应的元素为1，即标签为1对应的标签向量为[1,0,0,0,0,0,0,0,0,0]
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# #搭建网络
# network = models.Sequential()
# #设置第一层全连接层
# # network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
# network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
# #设置第一层全连接层，其返回一个由10个概率值组成的数组
# network.add(layers.Dense(10, activation='softmax'))
# network.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics=['accuracy'])
#
# #训练网络
# network.fit(train_images, train_labels, epochs=5, batch_size=128)
#
# #测试模型在测试集上的性能
# test_loss, test_acc = network.evaluate(test_images, test_labels)
# print('test_acc:', test_acc)

#搭建网络
model = models.Sequential()
#第一层卷积层
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
#第二层最大池化层
model.add(layers.MaxPooling2D((2,2)))
#第三层卷积层
model.add(layers.Conv2D(64, (3,3), activation='relu'))
#第四层最大池化层
model.add(layers.MaxPooling2D((2,2)))
#第五层卷积层
model.add(layers.Conv2D(64, (3,3), activation='relu'))
#第六层Flatten层，将3D张量平铺为向量
model.add(layers.Flatten())
#第七层全连接层
model.add(layers.Dense(64, activation='relu'))
#第八层softmax层，进行分类
model.add(layers.Dense(10, activation='softmax'))

print(model.summary()) #输出网络结构

model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs = 5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc = ', test_acc)