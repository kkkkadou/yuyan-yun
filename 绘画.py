import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import time
import matplotlib.pyplot as plt

loss_list = []

# 定义函数获取特征
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# 定义函数将风格图像应用到目标图像
def neural_style_transfer(content_image, style_image, output_path, num_steps=500, style_weight=100000, content_weight=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练的VGG19模型
    vgg = models.vgg19(pretrained=True).features
    vgg.to(device).eval()

    total_time = 0  # 初始化总时间为0
    for step in range(num_steps):
        start_time = time.time()   # 开始计时

    # 冻结所有参数，使我们不对其进行梯度下降
    for param in vgg.parameters():
        param.requires_grad_(False)

    content_image = content_image.to(device)
    style_image = style_image.to(device)

    # 获取内容和风格特征
    content_features = get_features(content_image, vgg)
    style_features = get_features(style_image, vgg)

    # 将目标图像初始化为内容图像的副本
    target_image = content_image.clone().requires_grad_(True).to(device)

    # 定义优化器
    optimizer = optim.Adam([target_image], lr=0.01)

    for step in range(num_steps):
        target_features = get_features(target_image, vgg)

        # 计算内容损失
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

        # 计算风格损失
        style_loss = 0
        for layer in style_features:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = gram_matrix(style_features[layer])
            _, d, h, w = target_feature.shape
            layer_style_loss = torch.mean((target_gram - style_gram)**2) #风格损失是目标图像和风格图像的Gram矩阵之间的均方误差
            layer_style_loss /= d * h * w #除以特征的维度
            style_loss += layer_style_loss

        # 总损失
        total_loss = content_weight * content_loss + style_weight * style_loss
        loss_list.append(total_loss.item())

        # 更新目标图像
        optimizer.zero_grad() #梯度重置
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}, Total loss: {total_loss.item()}")


        elapsed_time = time.time() - start_time  # 计算所需的时间
        total_time += elapsed_time  # 更新总时间

        if step % 50 == 0:
            print(f"Step {step}, Total loss: {total_loss.item()}, Time per step: {elapsed_time:.2f} seconds")

    print(f"Total elapsed time for {num_steps} steps: {total_time:.2f} seconds")

    # 将张量转换回图像，还原归一化，并保存最终的风格化图像
    target_image = target_image.squeeze(0)
    target_image = target_image.cpu().detach().numpy()
    target_image = target_image.transpose(1, 2, 0)                 #[channels, height, width]格式转换为[height, width, channels]格式
    target_image = (target_image * 255).astype('uint8')
    target_image = Image.fromarray(target_image)
    target_image.save(output_path)

# 定义函数计算Gram矩阵
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# 加载内容图像和风格图像
content_image = Image.open("C:/rubbish/ai3/绘画模型2/绘画模型/实体树.jpg").convert('RGB')
style_image = Image.open("C:/rubbish/ai3/绘画模型2/绘画模型/莫奈.jpg").convert('RGB')

# 对图像进行预处理和转换为张量
transform = transforms.Compose([
    transforms.Resize((512, 512)),                             # 使用较大的图像尺寸以获得更好的效果
    transforms.ToTensor(),
])
content_image = transform(content_image).unsqueeze(0)
style_image = transform(style_image).unsqueeze(0)

# 应用神经风格迁移
neural_style_transfer(content_image, style_image, "output.jpg", num_steps=1000, style_weight=1000000, content_weight=1)

plt.plot(loss_list)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over Iteration')
plt.show()


