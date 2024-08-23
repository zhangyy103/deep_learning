"""
2014 年，获得 ImageNet 图像分类竞赛的冠军是 GoogLeNet，其解决了一个
重要问题：滤波器超参数选择困难，如何能够自动找到最佳的情况。
其在网络中引入了一个小网络——Inception 块，由 4 条并行路径组成，4 条
路径互不干扰。这样一来，超参数最好的分支的那条分支，其权重会在训练过程
中不断增加，这就类似于帮我们挑选最佳的超参数
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

# 制作数据集

# 数据集转换参数
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)
])

# 下载训练集和测试集
train_data = datasets.FashionMNIST(
    root='../../assets/input/fashion_mnist',
    train=True,  # 是测试集?
    download=True,  # 如果数据集不存在，是否进行下载
    transform=transform  # 数据集转换参数
)
test_data = datasets.FashionMNIST(
    root='../../assets/input/fashion_mnist',
    train=False,  # 是测试集?
    download=True,  # 如果数据集不存在，是否进行下载
    transform=transform  # 数据集转换参数
)

# 制作数据集加载器
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)


# Inception 块
class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.Conv2d(24, 24, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, dim=1)


# GoogLeNet
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Inception(in_channels=10),
            nn.Conv2d(88, 20, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Inception(in_channels=20),
            nn.Flatten(),
            nn.Linear(1408, 10)
        )

    def forward(self, x):
        y = self.net(x)
        return y


model = CNN().to('cuda')

# 损失函数的选择
loss_fn = nn.CrossEntropyLoss()

# 优化器的选择
learning_rate = 0.1
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

if __name__ == '__main__':
    epochs = 10
    losses = []

    print("GoogLeNet model training begin")
    start_time = time.time()

    for epoch in range(epochs):
        for (x, y) in train_loader:
            x, y = x.to('cuda'), y.to('cuda')
            model.train()
            pred = model(x)
            loss = loss_fn(pred, y)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    end_time = time.time()
    print(f"Google model training end, cost {end_time - start_time} seconds")

    # 保存模型
    torch.save(model, '../../assets/output/GoogleNet_fashion_mnist.pth')

    Fig = plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Iteration')
    plt.ylabel('loss')
    plt.show()

    # 测试模型
    correct = 0
    total = 0

    with torch.no_grad():
        for (x, y) in test_loader:
            x, y = x.to('cuda'), y.to('cuda')
            model.eval()
            pred = model(x)
            _, predicted = torch.max(pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print(f'测试集精准度: {100 * correct / total} %')
