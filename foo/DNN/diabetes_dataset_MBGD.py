import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import time

"""utils是工具包，data是数据集，Dataset是数据集抽象类，DataLoader是数据加载器"""
"""Dataset 用于加载数据集，DataLoader 用于加载数据集的迭代器"""


class MyDataset(Dataset):
    def __init__(self, filepath):
        df = pd.read_csv(filepath, index_col=0)  # 读取数据
        arr = df.values  # 转换为 numpy 数组
        arr = arr.astype(np.float32)  # 转换数据类型
        ts = torch.tensor(arr)  # 转换为张量
        ts = ts.to('cuda')  # 把训练集移动到 GPU
        self.x = ts[:, :-1]  # 前八列输入特征
        self.y = ts[:, -1].reshape(-1, 1)  # 最后一列输出特征
        self.len = ts.size(0)  # 数据集大小

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


class DNN(nn.Module):
    def __init__(self):
        """搭建神经网络各层"""
        super(DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32), nn.Sigmoid(),  # 第一层 输入层 全连接层
            nn.Linear(32, 8), nn.Sigmoid(),  # 第二层 隐藏层 全连接层
            nn.Linear(8, 4), nn.Sigmoid(),  # 第三层 隐藏层 全连接层
            nn.Linear(4, 1), nn.Sigmoid()  # 第四层 输出层 全连接层
        )

    def forward(self, x):
        """前向传播"""
        y = self.net(x)
        return y


if __name__ == '__main__':
    data = MyDataset('.../assets/input/data.csv')
    train_size = int(len(data) * 0.7)  # 训练集大小
    test_size = len(data) - train_size  # 测试集大小
    train_data, test_data = random_split(data, [train_size, test_size])

    train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    model = DNN().to('cuda')  # 实例化模型

    loss_fn = nn.BCELoss(reduction='mean')  # 定义损失函数

    # 优化算法的选择
    learning_rate = 0.005  # 学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("deep learning model training begin")
    start_time = time.time()

    epochs = 500  # 训练次数
    losses = []  # 记录损失函数变化

    for epoch in range(epochs):
        for (x, y) in train_loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    end_time = time.time()
    print(f"DNN model training end, cost {end_time - start_time} seconds")

    # 保存模型
    torch.save(model, '../../assets/output/diabetes_dataset_MiniBGD_model.pth')

    # 绘制损失函数变化曲线
    Fig = plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Iteration')
    plt.ylabel('loss')
    plt.show()

    correct = 0
    total = 0

    with torch.no_grad():  # 关闭梯度计算
        for (x, y) in test_loader:
            pred = model(x)  # 前向传播(小批量)
            pred = (pred >= 0.5).float()  # 将预测值转换为0或1
            correct += torch.sum((pred == y).all(dim=1))  # 预测正确的样本
            total += y.size(0)  # 全部的样本数量

    print(f'测试集精准度: {100 * correct / total} %')
