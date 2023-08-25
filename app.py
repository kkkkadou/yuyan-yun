import tkinter as tk
from PIL import ImageTk, Image
import torch
from torchvision.transforms import transforms
from model import MultimodalGuidedDiffusion

# 加载模型
model = MultimodalGuidedDiffusion(num_channels=3)
model.load_state_dict(torch.load('model.pth'))
model.eval()  # 设置模型为评估模式


# 定义文本到图像的映射
text_to_image = {
    "dog": "dog/dog.jpg",
    "dog": "dog/dog2.jpg",
    "dog": "dog/dog3.jpg",
    "dog": "dog/dog4.jpg",
    "dog": "dog/dog4.jpg",
    "woman": "woman/woman.jpg",
    "woman": "woman/woman.jpg",
    "woman": "woman/woman.jpg",
    "woman": "woman/woman.jpg",
}

# 定义变换
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),  # 将 PIL 图像转换为张量
])

# 创建主窗口
root = tk.Tk()

# 创建 Canvas 组件用于显示图像
canvas = tk.Canvas(root, width=256, height=256)
canvas.pack()

def generate_image():
    # 获取输入文本
    text = text_entry.get()
    if text in text_to_image:
        # 加载图像并应用变换
        image_path = text_to_image[text]
        image = Image.open(image_path)

        image.show()  # 这会打开图片

        image_tensor = transform(image).unsqueeze(0)

        # 生成图像
        output_tensor = model(image_tensor)
        rgb_tensor = output_tensor[0, [2, 1, 0], :, :]  # 转换为 RGB

        # 转换为 PIL 图像
        output_image = transforms.ToPILImage()(rgb_tensor)

        # 显示图像
        tk_image = ImageTk.PhotoImage(output_image)
        canvas.create_image(0, 0, anchor='nw', image=tk_image)
        canvas.image = tk_image  # 保持对图像的引用以防止垃圾收集
    else:
        print('无法识别的输入')



# 创建文本输入框
text_entry = tk.Entry(root)
text_entry.pack()

# 创建按钮
button = tk.Button(root, text='generate', command=generate_image)
button.pack()

# 开始主循环
root.mainloop()
