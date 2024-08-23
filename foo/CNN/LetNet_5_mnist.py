import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
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


class CNN(nn.Module):
    def __init__(self):
        """搭建神经网络各层"""
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Tanh(),  # 第一层 卷积层
            nn.AvgPool2d(kernel_size=2, stride=2),  # 第二层 池化层
            nn.Conv2d(6, 16, kernel_size=5), nn.Tanh(),  # 第三层 卷积层
            nn.AvgPool2d(kernel_size=2, stride=2),  # 第四层 池化层
            nn.Conv2d(16, 120, kernel_size=5), nn.Tanh(),  # 第五层 卷积层
            nn.Flatten(),  # 展平
            nn.Linear(120, 84), nn.Tanh(),  # 第六层 全连接层
            nn.Linear(84, 10)  # 第七层 输出层
        )

    def forward(self, x):
        """前向传播"""
        y = self.net(x)
        return y


# 批次加载器
train_loader = DataLoader(train_data, shuffle=True, batch_size=256)
test_loader = DataLoader(test_data, shuffle=False, batch_size=256)

model = CNN().to('cuda')  # 实例化模型

loss_fn = nn.CrossEntropyLoss()  # 定义损失函数

# 优化算法的选择
learning_rate = 0.1  # 学习率
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

if __name__ == '__main__':

    print("CNN model training begin")
    start_time = time.time()

    # 训练网络
    epochs = 10
    losses = []  # 记录损失函数变化的列表

    for epoch in range(epochs):
        for (x, y) in train_loader:  # 获取小批次的 x 与 y
            x, y = x.to('cuda:0'), y.to('cuda:0')
            Pred = model(x)  # 一次前向传播（小批量）
            loss = loss_fn(Pred, y)  # 计算损失函数
            losses.append(loss.item())  # 记录损失函数的变化
            optimizer.zero_grad()  # 清理上一轮滞留的梯度
            loss.backward()  # 一次反向传播
            optimizer.step()  # 优化内部参数

    end_time = time.time()
    print(f"CNN model training end, cost {end_time - start_time} seconds")

    # 保存模型
    torch.save(model, '../../assets/output/LetNet-5_mnist_model.pth')

    Fig = plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Iteration')
    plt.ylabel('loss')
    plt.show()

    correct = 0
    total = 0
    with torch.no_grad():  # 该局部关闭梯度计算功能
        for (x, y) in test_loader:  # 获取小批次的 x 与 y
            x, y = x.to('cuda:0'), y.to('cuda:0')
            Pred = model(x)  # 一次前向传播（小批量）
            _, predicted = torch.max(Pred.data, dim=1)
            # correct += torch.sum((predicted == y))
            correct += (predicted == y).sum().item()
            total += y.size(0)
    print(f'测试集精准度: {100 * correct / total} %')
