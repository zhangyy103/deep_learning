import torch
import torch.nn as nn
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    # 读取数据
    df = pd.read_csv('../../assets/input/data.csv', index_col=0)
    arr = df.values
    arr = arr.astype(np.float32)  # 转换数据类型
    ts = torch.tensor(arr)  # 转换为张量
    ts = ts.to('cuda')  # 把训练集移动到 GPU

    # 划分训练集和测试集
    train_size = int(len(ts) * 0.7)  # 训练集大小
    test_size = len(ts) - train_size  # 测试集大小
    ts = ts[torch.randperm(ts.size(0)), :]  # 打乱数据集

    train_data = ts[:train_size, :]  # 训练集
    test_data = ts[train_size:, :]  # 测试集

    model = DNN().to('cuda')  # 实例化模型

    loss_fn = nn.BCELoss(reduction='mean')  # 定义损失函数

    # 优化算法的选择
    learning_rate = 0.005  # 学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("deep learning model training begin")
    start_time = time.time()

    epochs = 5000  # 训练次数
    losses = []

    # 给训练集划分输入和输出
    x_train = train_data[:, :-1]  # 前八列输入特征
    y_train = train_data[:, -1].reshape(-1, 1)  # 最后一列输出特征
    # 此处的reshape(-1, 1)是为了将y_train转换为二维张量，以便后续计算损失

    for _ in range(epochs):
        pred = model(x_train)  # 前向传播
        loss = loss_fn(pred, y_train)  # 计算损失
        losses.append(loss.item())  # 记录损失函数的变化
        optimizer.zero_grad()  # 清理上一轮滞留的梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 优化内部参数

    end_time = time.time()
    print(f"DNN model training end, cost {end_time - start_time} seconds")

    # 保存模型
    torch.save(model, '../../assets/output/diabetes_dataset_model.pth')

    Fig = plt.figure()
    plt.plot(range(epochs), losses)
    plt.xlabel('Iteration')
    plt.ylabel('loss')
    plt.show()

    new_model = torch.load('../../assets/output/diabetes_dataset_model.pth')  # 加载模型

    test_x = test_data[:, :-1]  # 测试集输入
    test_y = test_data[:, -1].reshape(-1, 1)  # 测试集输出

    with torch.no_grad():
        pred = new_model(test_x)  # 一次前向传播
        pred = (pred > 0.5).float()  # 将预测值转换为0或1
        correct = torch.sum((pred == test_y).all(1))  # 预测样本

        total = test_y.size(0)  # 测试集样本总数
        print(f'测试集精准度: {100 * correct / total} %')
