# 你需要安装 torchvision 和 torch 以使用 transforms 和 DataLoader

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

text_to_image = {
    "dog.jpg": "dog",
    "dog2.jpg": "dog",
    "dog3.jpg": "dog",
    "dog4.jpg": "dog",
    "dog5.jpg": "dog",
    "woman.jpg": "woman",
    "woman2.jpg": "woman",
    "woman3.jpg": "woman",
    "woman4.jpg": "woman",
}

class MultimodalGuidedDiffusion(nn.Module):    #创建一个新类
    def __init__(self, num_channels):   #创建类的实例
        super(MultimodalGuidedDiffusion, self).__init__()

        # 定义图像特征提取网络（使用预训练的 ResNet-50）
        self.image_encoder = nn.Sequential(*list(resnet50(pretrained=True).children())[:-1])

        # 定义更复杂的解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=1, padding=1), # 转置卷积层1
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=1, padding=1), # 转置卷积层2
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1), # 转置卷积层3
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1), # 转置卷积层4
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1), # 转置卷积层5
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1), # 转置卷积层6
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_channels, kernel_size=3, stride=1, padding=1), # 转置卷积层7
            nn.Tanh(),  # 使用Tanh激活函数输出范围[-1, 1]
        )

    def forward(self, image): #前向传播
        # 提取图像特征
        features = self.image_encoder(image)
        features = features.view(features.size(0), -1)

        # 解码得到输出
        output = self.decoder(features.unsqueeze(2).unsqueeze(3))
        return output


# 设置超参数
num_channels = 3 #RGB
learning_rate = 0.001
batch_size = 10 # 设置为10，每次输入10个样本进行训练
num_epochs = 10


# 创建数据集和数据加载器，并添加数据增强操作
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(15),  # 随机旋转角度[-15, 15]
    transforms.ToTensor(),#转换PIL为torch.Tensor
])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, text_to_image, root_dir, transform=None):
        self.text_to_image = text_to_image
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.text_to_image)

    def __getitem__(self, index):
        image_file = list(self.text_to_image.keys())[index]
        image_file = image_file.replace('/', os.sep)
        text = self.text_to_image[image_file]
        image_path = os.path.join(self.root_dir, text, image_file)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

dataset = CustomDataset(text_to_image, root_dir='C:/rubbish/ai3/绘画模型2/绘画模型', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = MultimodalGuidedDiffusion(num_channels)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 将模型设为训练模式
model.train()

# 开始训练
for epoch in range(num_epochs):
    for images in dataloader:
        # 前向传播
        outputs = model(images)

        # 调整目标张量的尺寸与输出张量一致
        target = images

        # 计算均方误差损失
        loss = nn.functional.mse_loss(outputs, target)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 打印每个epoch的损失
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), 'model.pth')
print('模型已保存。')
