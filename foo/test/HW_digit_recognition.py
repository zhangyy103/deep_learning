import torch
import pygame
from foo.DNN.mnist_dataset_MBGD import DNN
from foo.CNN.LetNet_5_mnist import CNN
from PIL import Image
import numpy as np
from torchvision import transforms


# 加载模型
model = torch.load('../../assets/output/LetNet-5_mnist_model.pth')
model.eval()  # 设置模型为评估模式

# 初始化 pygame
pygame.init()
screen = pygame.display.set_mode((400, 200))
pygame.display.set_caption("Handwritten Digit Recognition")
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
canvas = pygame.Surface((200, 200))
canvas.fill(WHITE)
font = pygame.font.Font(None, 60)


def predict_digit(image):
    pil_image = Image.frombytes("RGB", image.get_size(), pygame.image.tostring(image, "RGB"))
    pil_image = pil_image.convert("L").resize((28, 28))
    pil_image = Image.fromarray(255 - np.array(pil_image))  # 检查图像是否反转
    img_tensor = transforms.ToTensor()(pil_image).unsqueeze(0).to('cuda')
    img_tensor = transforms.Normalize((0.1307,), (0.3081,))(img_tensor)  # 正确归一化
    print(f'Image tensor shape: {img_tensor.shape}')  # 打印张量形状
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    print(f'Predicted digit: {predicted.item()}')  # 打印预测结果
    return predicted.item()


running = True
is_drawing = False
digit = None  # 初始化预测结果
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            is_drawing = True
        if event.type == pygame.MOUSEBUTTONUP:
            is_drawing = False
            digit = predict_digit(canvas)
            canvas.fill(WHITE)  # 清空画布
        if event.type == pygame.MOUSEMOTION and is_drawing:
            mouseX, mouseY = event.pos
            pygame.draw.circle(canvas, BLACK, (mouseX, mouseY), 8)

    screen.fill(WHITE)
    screen.blit(canvas, (0, 0))
    if digit is not None:
        text1 = font.render("Predict:", True, BLACK)
        text2 = font.render(f"{digit}", True, BLACK)
        screen.blit(text1, (220, 50))
        screen.blit(text2, (270, 100))
    pygame.display.flip()

pygame.quit()
