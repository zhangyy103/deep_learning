import torch
from PIL import Image
from torchvision import transforms
from foo.CNN.AlexNet_fashion_mnist import CNN
from foo.CNN.GoogleNet_fashion_mnist import CNN
from foo.CNN.GoogleNet_fashion_mnist import Inception
from foo.CNN.ResNet_fashion_mnist import CNN
from foo.CNN.ResNet_fashion_mnist import ResidualBlock

#  model_path = '../../assets/output/AlexNet_fashion_mnist_model.pth'
#  model_path = '../../assets/output/GoogleNet_fashion_mnist.pth'
model_path = '../../assets/output/ResNet_fashion_mnist.pth'
# 加载模型
# model = torch.load(model_path)
model = torch.load(model_path)
model.to('cuda')
model.eval()

# 定义与训练时相同的预处理操作
preprocess = transforms.Compose([
    #  AlexNet 是 224 * 224 的输入,  GoogleNet 是 28 * 28 的输入,  ResNet 是 28 * 28 的输入
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.Grayscale(num_output_channels=1),  # 转换为灰度图像
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# 预测函数
def predict_image(picture_paths):
    image = Image.open(picture_paths)
    img_tensor = preprocess(image).unsqueeze(0).to('cuda')  # 扩展维度并移动到GPU
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()


# 服装类别名称
"""
'T恤/上衣'， '裤子'， '套头衫'， '连衣裙'， '外套'，
'凉鞋'， '衬衫'， '运动鞋'， '包'， '踝靴'
"""
categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

dress_path = '../../assets/input/fashion/fmnist_0.jpg'
trouser_path = '../../assets/input/fashion/fmnist_1.jpg'
pullover_path = '../../assets/input/fashion/fmnist_2.jpg'
dress2_path = '../../assets/input/fashion/fmnist_3.jpg'
coat_path = '../../assets/input/fashion/fmnist_4.jpg'
sneaker_path = '../../assets/input/fashion/fmnist_5.jpg'
shirt_path = '../../assets/input/fashion/fmnist_6.jpg'
sneaker2_path = '../../assets/input/fashion/fmnist_7.jpg'
bag_path = '../../assets/input/fashion/fmnist_8.jpg'
ankle_boot_path = '../../assets/input/fashion/fmnist_9.jpg'

picture_path = [dress_path, trouser_path, pullover_path, dress2_path, coat_path,
                sneaker_path, shirt_path, sneaker2_path, bag_path, ankle_boot_path]

print(f'model: {model_path[20:]}')
for image_path in picture_path:
    category_idx = predict_image(image_path)
    print(f'{image_path[27:35]}: {categories[category_idx]}')

"""
model: AlexNet_fashion_mnist_model.pth
fmnist_0: T-shirt/top
fmnist_1: Trouser
fmnist_2: Pullover
fmnist_3: Dress
fmnist_4: Coat
fmnist_5: Sneaker
fmnist_6: Shirt
fmnist_7: Sneaker
fmnist_8: Bag
fmnist_9: Ankle boot

model: GoogleNet_fashion_mnist.pth
fmnist_0: Dress
fmnist_1: Trouser
fmnist_2: Sandal
fmnist_3: Dress
fmnist_4: Coat
fmnist_5: Sandal
fmnist_6: Pullover
fmnist_7: Sneaker
fmnist_8: Bag
fmnist_9: Ankle boot
"""
"""
model: ResNet_fashion_mnist.pth
fmnist_0: Shirt
fmnist_1: Trouser
fmnist_2: Sandal
fmnist_3: Dress
fmnist_4: Shirt
fmnist_5: Sandal
fmnist_6: Pullover
fmnist_7: Sneaker
fmnist_8: Bag
fmnist_9: Ankle boot
"""
