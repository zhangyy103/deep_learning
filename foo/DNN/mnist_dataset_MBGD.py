import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import time

# 制作数据集

# 数据集转换参数
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])

# 下载训练集和测试集
train_data = datasets.MNIST(
    root='../../assets/input/mnist',  # 数据集存放路径
    train=True,  # 是训练集?
    download=True,  # 如果数据集不存在，是否进行下载
    transform=transform  # 数据集转换参数
)

test_data = datasets.MNIST(
    root='../../assets/input/mnist',  # 数据集存放路径
    train=False,  # 是测试集?
    download=True,  # 如果数据集不存在，是否进行下载
    transform=transform  # 数据集转换参数
)

# 搭建神经网络
class DNN(nn.Module):
    def __init__(self):
        """搭建神经网络各层"""
        super(DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 512), nn.ReLU(),  # 第一层 输入层 全连接层
            nn.Linear(512, 256), nn.ReLU(),  # 第二层 隐藏层 全连接层
            nn.Linear(256, 128), nn.ReLU(),  # 第三层 隐藏层 全连接层
            nn.Linear(128, 64), nn.ReLU(),  # 第四层 隐藏层 全连接层
            nn.Linear(64, 10)  # 第五层 输出层 全连接层

        )

    def forward(self, x):
        """前向传播"""
        x = x.view(x.size(0), -1)  # 展平输入
        y = self.net(x)
        return y


if __name__ == '__main__':

    # 批次加载器
    train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    model = DNN().to('cuda')  # 实例化模型

    loss_fn = nn.CrossEntropyLoss()  # 定义损失函数(自带softmax激活函数)

    # 优化算法的选择
    learning_rate = 0.01  # 学习率
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)

    print("deep learning model training begin")
    start_time = time.time()

    epochs = 10  # 训练次数
    losses = []  # 记录损失函数变化

    for epoch in range(epochs):
        for (x, y) in train_loader:
            x, y = x.to('cuda'), y.to('cuda')
            pred = model(x)
            loss = loss_fn(pred, y)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    end_time = time.time()
    print(f"DNN model training end, cost {end_time - start_time} seconds")

    # 保存模型
    torch.save(model, '../../assets/output/DNN_mnist_model.pth')

    Fig = plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Iteration')
    plt.ylabel('loss')
    plt.show()

    correct = 0
    total = 0

    with torch.no_grad():  # 关闭梯度计算
        for (x, y) in test_loader:
            x, y = x.to('cuda'), y.to('cuda')
            pred = model(x)
            _, predicted = torch.max(pred, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print(f'测试集精准度: {100 * correct / total} %')
