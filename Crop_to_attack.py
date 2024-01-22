import os
import torch
import cv2
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from PIL import Image
from torchvision import models
import json
import numpy as np
import matplotlib.pyplot as plt
from torchray.attribution.grad_cam import grad_cam
from PIL import ImageDraw

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.resnet50(pretrained=True)
model.to(device)
model.eval()

with open('imagenet_class_index.json') as f:
    class_idx = json.load(f)
classes = [class_idx[str(i)][0] for i in range(1000)]

transform = transforms.Compose([
    transforms.ToTensor()
])

img_path = './save-data-food/res50/adv-cut/clean_img'
img_paths = [os.path.join(img_path, f) for f in os.listdir(img_path)]
imgs = []
for img_path in img_paths:
    img = Image.open(img_path)
    img = transform(img)
    imgs.append(img)

imgs = torch.stack(imgs).to(device)
outputs = model(imgs)
probs, labels = torch.max(torch.softmax(outputs, dim=1), dim=1)
fig, axs = plt.subplots(10, 5, figsize=(24, 36))
axs = axs.ravel()

# 遍历图像
for i, img_path in enumerate(img_paths):
    label = labels[i].item()
    prob = probs[i].item()
    img = Image.open(img_path)
    img = transform(img)
    label_text = classes[label]
    prob_text = '{:.2f}%'.format(prob * 100)
    folder_path = os.path.join('./save-data-food/res50/adv-cut')
    img = img.to(device)
    # 计算 Grad-CAM 并可视化
    saliency_map = grad_cam(model, img.unsqueeze(0), label, saliency_layer='layer4')
    saliency_map = saliency_map.detach().cpu().numpy()
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    saliency_map = saliency_map.squeeze()
    saliency_map = plt.get_cmap('jet')(saliency_map)[:, :, :3]
    saliency_map = (saliency_map * 255).astype(np.uint8)
    saliency_map = transforms.ToPILImage()(saliency_map.squeeze())
    img = transforms.ToPILImage()(img.squeeze())
    img = img.convert('RGB')
    w, h = img.size
    saliency_map = saliency_map.resize((w, h), resample=Image.Resampling.BILINEAR)
    saliency_map = saliency_map.convert('RGB')

    # 获取热力图中从红色到黄色的连续区域的边界框
    mask = np.array(saliency_map)[:, :, 0]
    mask[(mask < 50) | (mask > 200)] = 0
    mask = mask.astype(np.uint8)
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    bbox = [x0, y0, x1, y1]
    # Crop to attack
    img_crop =  img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    save_path = os.path.join(folder_path, f'gradcam_clean_cut_{i}.png')
    img_crop.save(save_path)
    print(f'已保存图像到文件夹：{folder_path}')

    axs[i].imshow(saliency_map)

    axs[i].axis('off')
    axs[i].plot([bbox[0], bbox[2]], [bbox[1], bbox[1]], color='r', linewidth=2)
    axs[i].plot([bbox[0], bbox[2]], [bbox[3], bbox[3]], color='r', linewidth=2)
    axs[i].plot([bbox[0], bbox[0]], [bbox[1], bbox[3]], color='r', linewidth=2)
    axs[i].plot([bbox[2], bbox[2]], [bbox[1], bbox[3]], color='r', linewidth=2)


fig.tight_layout()
plt.show()
plt.savefig('./save-data/res50/adv-cut', dpi=400, bbox_inches='tight')





