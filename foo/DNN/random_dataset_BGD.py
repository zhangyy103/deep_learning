import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np


# 搭建神经网络
class DNN(nn.Module):
    def __init__(self):
        """ 搭建神经网络各层"""
        super(DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 5), nn.ReLU(),  # 第一层 输入层 全连接层
            nn.Linear(5, 5), nn.ReLU(),  # 第二层 隐藏层 全连接层
            nn.Linear(5, 5), nn.ReLU(),  # 第三层 隐藏层 全连接层
            nn.Linear(5, 3)  # 第四层 输出层 全连接层
        )

    def forward(self, x):
        """ 前向传播"""
        y = self.net(x)
        return y


if __name__ == '__main__':

    # 输入特征
    x1 = torch.rand(10000, 1)
    x2 = torch.rand(10000, 1)
    x3 = torch.rand(10000, 1)
    # x1 x2 x3 为 10000 行 1 列的随机数 范围是 [0, 1)

    # 输出特征
    y1 = ((x1 + x2 + x3) > 1).float()
    y2 = ((1 < (x1 + x2 + x3)) & ((x1 + x2 + x3) < 2)).float()
    y3 = ((x1 + x2 + x3) > 2).float()
    # 经过运算y1 y2 y3 为 10000 行 1 列的 0 1 矩阵 矩阵中 0 1 出现的概率各为 0.5

    # 拼接数据
    data = torch.cat([x1, x2, x3, y1, y2, y3], dim=1)
    # data 为 10000 行 6 列的数据集 前三列为输入特征 后三列为输出特征

    # 打乱数据，划分训练集和测试集
    train_size = int(0.7 * len(data))
    test_size = len(data) - train_size
    data = data[torch.randperm(len(data))]  # 打乱数据
    train_data = data[:train_size]
    test_data = data[train_size:]

    model = DNN().to('cuda')

    # 定义损失函数
    loss_fn = nn.MSELoss()

    # 优化算法的选择
    learning_rate = 0.01  # 学习率
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print("deep learning model training begin")
    start_time = time.time()

    epochs = 1000  # 训练次数
    losses = []

    # 给训练集划分输入和输出
    x_train = train_data[:, :3].to('cuda')  # 输入
    y_train = train_data[:, 3:].to('cuda')  # 输出

    for _ in range(epochs):
        pred = model(x_train)  # 前向传播
        loss = loss_fn(pred, y_train)  # 计算损失
        losses.append(loss.item())  # 记录损失
        optimizer.zero_grad()  # 清理上一轮滞留的梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 优化内部参数

    # 将模型保存到assets文件夹
    torch.save(model, '../../assets\\output\\random_dataset_model.pth')

    end_time = time.time()
    Fig = plt.figure()
    plt.plot(range(epochs), losses)
    plt.ylabel('loss')
    plt.xlabel('Iteration')
    plt.show()

    print(f"DNN model training end, cost {end_time - start_time} seconds")

    new_model = torch.load('../../assets/output/random_dataset_model.pth')

    # 给测试集划分输入与输出
    X = test_data[:, :3].to('cuda')  # 前 3 列为输入特征，移动到 GPU
    Y = test_data[:, -3:].to('cuda')  # 后 3 列为输出特征，移动到 GPU

    with torch.no_grad():  # 该局部关闭梯度计算功能
        pred = new_model(X)  # 一次前向传播（批量）
        pred = (pred > 0.5).float()  # 将预测值转换为0或1
        correct = torch.sum((pred == Y).all(1))  # 预测正确的样本
        total = Y.size(0)  # 全部的样本数量
        print(f'测试集精准度: {100 * correct / total} %')


