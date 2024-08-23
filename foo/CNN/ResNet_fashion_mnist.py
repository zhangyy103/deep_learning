import torch
import torch.nn as nn
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


# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        y = self.nn(x)
        return nn.functional.relu(x + y)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.ReLU(),
            ResidualBlock(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            ResidualBlock(32),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 10)
        )

    def forward(self, x):
        y = self.net(x)
        return y


model = CNN().to('cuda')

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化算法的选择
learning_rate = 0.1  # 学习率
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

if __name__ == '__main__':
    epochs = 10
    losses = []  # 记录训练损失

    print("ResNet model training begin")
    start_time = time.time()

    for epoch in range(epochs):
        for (x, y) in train_loader:
            x, y = x.to('cuda:0'), y.to('cuda:0')
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    end_time = time.time()
    print(f"ResNet model training end, cost {end_time - start_time:.2f}s")

    # 保存模型
    torch.save(model, '../../assets/output/ResNet_fashion_mnist.pth')

    # 绘制损失函数曲线
    Fig = plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    correct = 0
    total = 0

    with torch.no_grad():
        for (x, y) in test_loader:
            x, y = x.to('cuda'), y.to('cuda')
            pred = model(x)
            _, predicted = torch.max(pred, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print(f'测试集精准度: {100 * correct / total} %')
