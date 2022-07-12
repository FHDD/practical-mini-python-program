import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import gzip
import os
import torchvision
import cv2
import matplotlib.pyplot as plt




# 设置网络结构
class Net(nn.Module):
    def __init__(self, in_features, out_features):
        super(Net, self).__init__()
        self.dnn1 = nn.Linear(in_features, 512)  # 第一层全连接层
        self.dnn2 = nn.Linear(512, out_features)  # 第二层全连接层

    def forward(self, x):
        x = F.relu(self.dnn1(x))  # relu激活
        x = self.dnn2(x)
        return x


net = Net(28 * 28, 10)

# 加载MNIST数据集

# #************************************a2torchloadlocalminist*********************************************************
class DealDataset(Dataset):
    """
        读取数据、初始化数据
    """

    def __init__(self, folder, data_name, label_name, transform=None):
        (train_set, train_labels) = load_data(folder, data_name,
                                              label_name)  # 其实也可以直接使用torch.load(),读取之后的结果为torch.Tensor形式
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)


def load_data(data_folder, data_name, label_name):
    with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:  # rb表示的是读取二进制数据
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    return (x_train, y_train)


train_dataset = DealDataset(r'E:\研究生\HW\大数据机器学习\第一次实验\data', "train-images-idx3-ubyte.gz",
                           "train-labels-idx1-ubyte.gz", transform=transforms.ToTensor())
test_dataset = DealDataset(r'E:\研究生\HW\大数据机器学习\第一次实验\data', "t10k-images-idx3-ubyte.gz",
                           "t10k-labels-idx1-ubyte.gz", transform=transforms.ToTensor())

# #************************************a2torchloadlocalminist*********************************************************

# train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
# test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
print(net)

# 开始训练
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.02)

for epoch in range(5):
    running_loss, running_acc = 0.0, 0.0
    for i, data in enumerate(train_loader):
        img, label = data
        img = img.reshape(-1, 28 * 28)
        out = net(img)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * label.size(0)
        _, predicted = torch.max(out, 1)
        running_acc += (predicted == label).sum().item()
        print('Epoch [{}/5], Step [{}/{}], Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, i + 1, len(train_loader), loss.item(), (predicted == label).sum().item() / 128))
    # 测试
    test_loss, test_acc = 0.0, 0.0
    for i, data in enumerate(test_loader):
        img, label = data
        img = img.reshape(-1, 28 * 28)
        out = net(img)
        loss = criterion(out, label)

        test_loss += loss.item() * label.size(0)
        _, predicted = torch.max(out, 1)
        test_acc += (predicted == label).sum().item()

    print("Train {} epoch, Loss: {:.6f}, Acc: {:.6f}, Test_Loss: {:.6f}, Test_Acc: {:.6f}".format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset)),
        test_loss / (len(test_dataset)), test_acc / (len(test_dataset))))
