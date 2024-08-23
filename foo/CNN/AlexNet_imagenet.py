"""
130G的ImageNet数据集，包含1000个类别，每个类别有1000张图片，每张图片的大小为224*224*3。
epoch=5 NVIDIA 4060Ti 8GB显存，要跑5-10个小时

跑不了，这代码没运行过
"""
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# 数据集转换参数
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 下载训练集和测试集
train_data = datasets.ImageNet(
    root='../../assets/input/imagenet',  # 数据集存放路径
    split='train',  # 是训练集
    download=True,  # 如果数据集不存在，是否进行下载
    transform=transform  # 数据集转换参数
)
test_data = datasets.ImageNet(
    root='../../assets/input/imagenet',  # 数据集存放路径
    split='val',  # 是测试集
    download=True,  # 如果数据集不存在，是否进行下载
    transform=transform  # 数据集转换参数
)


class CNN(nn.Module):
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


# 加载预训练的 AlexNet 模型
model = CNN().to('cuda')


# 定义损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化算法的选择
learning_rate = 0.001  # 学习率
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

if __name__ == '__main__':
    # 批次加载器
    train_loader = DataLoader(train_data, shuffle=True, batch_size=256)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=256)

    print("AlexNet model training begin")
    start_time = time.time()

    # 训练网络
    epochs = 10
    losses = []  # 记录损失函数变化的列表

    for epoch in range(epochs):
        model.train()
        for (x, y) in train_loader:  # 获取小批次的 x 与 y
            x, y = x.to('cuda:0'), y.to('cuda:0')
            Pred = model(x)  # 一次前向传播（小批量）
            loss = loss_fn(Pred, y)  # 计算损失函数
            losses.append(loss.item())  # 记录损失函数的变化
            optimizer.zero_grad()  # 清理上一轮滞留的梯度
            loss.backward()  # 一次反向传播
            optimizer.step()  # 优化内部参数

    end_time = time.time()
    print(f"AlexNet model training end, cost {end_time - start_time} seconds")

    # 保存模型
    torch.save(model, '../../assets/output/AlexNet_imagenet_model.pth')

    Fig = plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Iteration')
    plt.ylabel('loss')
    plt.show()

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():  # 该局部关闭梯度计算功能
        for (x, y) in test_loader:  # 获取小批次的 x 与 y
            x, y = x.to('cuda:0'), y.to('cuda:0')
            Pred = model(x)  # 一次前向传播（小批量）
            _, predicted = torch.max(Pred.data, dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    print(f'测试集精准度: {100 * correct / total} %')
