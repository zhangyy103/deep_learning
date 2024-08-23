"""
考虑到如果使用 ImageNet 训练集会导致训练时间过长，这里使用稍低一档
的 1×28×28 的 Fashion MNIST 数据集，并手动将其分辨率从 1×28×28 提到 1×224×224，
同时输出从 1000 个类别降到 10 个，修改后的网络结构见表 3-1。
Fashion MNIST：包含时尚商品（如衣服、鞋子、包等）的图像。
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import time

# 数据集转换参数
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
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

# 批次加载器
train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=128, shuffle=False)


class CNN(nn.Module):
    """定义网络结构"""

    #  输入和输出的形状分别是 1×224×224 和 10因此这并不是一个完整的 AlexNet 网络，只是一个简化版
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        y = self.net(x)
        return y


model = CNN().to('cuda')  # 实例化模型

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化算法的选择
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

if __name__ == '__main__':
    print("CNN model training begin")
    start_time = time.time()

    # 训练网络
    epochs = 10
    losses = []

    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to('cuda'), y.to('cuda')
            pred = model(x)
            loss = loss_fn(pred, y)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    end_time = time.time()
    print(f"Total cost time: {end_time - start_time}")

    # 保存模型
    torch.save(model, '../../assets/output/AlexNet_fashion_mnist_model.pth')

    # 绘制损失函数变化曲线
    Fig = plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Iteration')
    plt.ylabel('loss')
    plt.show()

    # 测试网络
    correct = 0
    total = 0
    with torch.no_grad():  # 该局部关闭梯度计算功能
        for (x, y) in test_loader:  # 获取小批次的 x 与 y
            x, y = x.to('cuda:0'), y.to('cuda:0')
            pred = model(x)  # 一次前向传播（小批量）
            _, predicted = torch.max(pred.data, dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    print(f'测试集精准度: {100 * correct / total} %')
