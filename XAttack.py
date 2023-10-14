import numpy as np
import json
import os
import sys
import time
import math
import io
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchattacks.attack import Attack
from utils import *
from compression import *
from decompression import *
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# 信息丢弃类
class XAttack(Attack):
    def __init__(self, model, height=224, width=224, steps=40, batch_size=20, block_size=8, q_size=10, targeted=False):
        super(XAttack, self).__init__("XAttack",model)
        self.steps = steps
        self.targeted = targeted
        self.batch_size = batch_size
        self.height = height
        self.width = width
        # Value for quantization range
        self.factor_range = [5, q_size]  # 量化因子
        self.alpha_range = [0.1, 1e-20]
        self.alpha = torch.tensor(self.alpha_range[0])  # self.alpha=0.1
        self.alpha_interval = torch.tensor(
            (self.alpha_range[1] - self.alpha_range[0]) / self.steps)
        block_n = np.ceil(height / block_size) * np.ceil(height / block_size)  # np.ceil向上取整
        q_ini_table = np.empty((batch_size, int(block_n), block_size, block_size), dtype=np.float32)
        q_ini_table.fill(q_size)
        '''
        色彩空间Y,cb,cr
        '''
        self.q_tables = {"y": torch.from_numpy(q_ini_table),
                         "cb": torch.from_numpy(q_ini_table),
                         "cr": torch.from_numpy(q_ini_table)}  # ndarry->tensor

    def dlr_loss(self,x,y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(x.shape[0])
        eps = 1e-12  # 加上一个很小的常量
        loss = -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (
                    x_sorted[:, -1] - x_sorted[:, -3] + eps)
        return loss.mean()

    def cw_loss(self, x, y):
        # 计算目标类别的得分
        target_scores = x.gather(1, y.view(-1, 1)).squeeze()
        # 计算其他类别的最大得分
        max_nontarget_scores = x.scatter(1, y.view(-1, 1), float('-inf')).max(dim=1)[0]
        # 计算 CW 损失
        loss = max_nontarget_scores - target_scores

        return loss.mean()

    # 前项推断
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        q_table = None
        self.alpha = self.alpha.to(self.device)
        self.alpha_interval = self.alpha_interval.to(self.device)
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        #cw损失
        adv_loss=self.cw_loss
        # Adam优化器
        optimizer = torch.optim.Adam([self.q_tables["y"], self.q_tables["cb"], self.q_tables["cr"]], lr=0.01)
        # images维度变换
        images = images.permute(0, 2, 3, 1)
        components = {'y': images[:, :, :, 0], 'cb': images[:, :, :, 1], 'cr': images[:, :, :, 2]}
        for i in range(self.steps):
            # 允许计算梯度
            self.q_tables["y"].requires_grad = True
            self.q_tables["cb"].requires_grad = True
            self.q_tables["cr"].requires_grad = True
            upresults = {}
            for k in components.keys():
                comp = block_splitting(components[k])
                comp = dct_8x8(comp)
                comp = quantize(comp, self.q_tables[k], self.alpha)
                comp = dequantize(comp, self.q_tables[k])
                comp = idct_8x8(comp)
                merge_comp = block_merging(comp, self.height, self.width)
                upresults[k] = merge_comp
            rgb_images = torch.cat(
                [upresults['y'].unsqueeze(3), upresults['cb'].unsqueeze(3), upresults['cr'].unsqueeze(3)], dim=3)
            rgb_images = rgb_images.permute(0, 3, 1, 2)
            outputs = self.model(rgb_images)
            _, pre = torch.max(outputs.data, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            #有无目标攻击准确率计算
            if self.targeted:
                suc_rate = ((pre == labels).sum()/self.batch_size).cpu().detach().numpy()
            else:
                suc_rate = ((pre != labels).sum()/self.batch_size).cpu().detach().numpy()

            # 计算损失
            adv_cost = adv_loss(outputs, labels.long())

            if not self.targeted:
                adv_cost = -1 * adv_cost

            total_cost = adv_cost
            optimizer.zero_grad()
            total_cost.backward()
            self.alpha += self.alpha_interval
            # XAI-AT
            for k in self.q_tables.keys():
                self.q_tables[k] = self.q_tables[k].detach() - torch.sign(self.q_tables[k].grad)
                self.q_tables[k] = torch.clamp(self.q_tables[k], self.factor_range[0], self.factor_range[1]).detach()
            if i % 10 == 0:
                print('Step: ', i, "  Loss: ", total_cost.item(), "  Current Suc rate: ", suc_rate)
            # END Attack
            if suc_rate >= 1:
                print('End at step {} with suc. rate {}'.format(i, suc_rate))
                q_images = torch.clamp(rgb_images, min=0, max=255.0).detach()
                return q_images, pre, i
        q_images = torch.clamp(rgb_images, min=0, max=255.0).detach()

        return q_images, pre, q_table


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        input = input / 255.0
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


def save_img(img, img_name, save_dir):
    create_dir(save_dir)
    img_path = os.path.join(save_dir, img_name)
    img_pil = Image.fromarray(img.astype(np.uint8))
    img_pil.save(img_path)


def pred_label_and_confidence(model, input_batch, labels_to_class):
    input_batch = input_batch.cuda()
    with torch.no_grad():
        out = model(input_batch)
    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1) * 100
    # print(percentage.shape)
    pred_list = []
    for i in range(index.shape[0]):
        pred_class = labels_to_class[index[i]]
        pred_conf = str(round(percentage[i][index[i]].item(), 2))
        pred_list.append([pred_class, pred_conf])
    return pred_list


if __name__ == "__main__":
    with open('imagenet_class_index.json', 'r') as f:
        class_index = json.load(f)
    label_to_class = {}
    for k, v in class_index.items():
        label_to_class[int(k)] = v[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_idx = json.load(open("./imagenet_class_index.json"))  # ["n01440764", "tench"]
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]  # "tench"
    class2label = [class_idx[str(k)][0] for k in range(len(class_idx))]  # "n01440764"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), ])

    resize=transforms.Resize((224,224))

    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    resnet_model = nn.Sequential(
        norm_layer,
        models.resnet50(pretrained=True)
    ).to(device)
    resnet_model = resnet_model.eval()
    batch_size = 20
    tar_cnt = 1000
    q_size = 40
    cur_cnt = 0
    suc_cnt = 0
    save_dir = "./save-data"
    data_dir = "./test-data"
    # 数据清洗
    data_clean(data_dir)
    # n01440764—>tench
    normal_data = image_folder_custom_label(root=data_dir, transform=transform, idx2label=class2label)
    normal_loader = torch.utils.data.DataLoader(normal_data, batch_size=batch_size, shuffle=False)
    normal_iter = iter(normal_loader)

    for i in range(tar_cnt//batch_size):
        print("Iter: ", i)
        images, labels = next(normal_iter)

        # For target attack: set random target.
        # Comment if you set untargeted attack.
        labels = torch.from_numpy(np.random.randint(0, 1000, size=batch_size))

        images = images * 255.0
        attack = XAttack(resnet_model, batch_size=batch_size, q_size=q_size, steps=200, targeted=True)
        at_images, at_labels, suc_step = attack(images, labels)

        # Uncomment following codes if you wang to save the adv imgs
        at_images_np = at_images.detach().cpu().numpy()
        images_np = images.detach().cpu().numpy()
        at_labels_str = [str(label.item()) for label in at_labels]
        adv_img = at_images_np[0]
        clean_img = images_np[0]
        at_label = at_labels_str[0]
        adv_img = np.moveaxis(adv_img, 0, 2)
        clean_img = np.moveaxis(clean_img, 0, 2)



        adv_dir = os.path.join(save_dir, "advdrop_adv")
        clean_dir = os.path.join(save_dir, "advdrop_clean")
        adv_img_name = "adv_{}.jpg".format(i)
        clean_img_name = "clean_{}.jpg".format(i)
        save_img(adv_img, adv_img_name, adv_dir)
        save_img(clean_img, clean_img_name, clean_dir)

        # Predict the label and confidence of the adversarial example
        at_images = at_images.to(device)
        with torch.no_grad():
            at_prob = resnet_model(at_images)
            at_pred = at_prob.argmax(dim=1).cpu().numpy()[0]
            at_confidence = at_prob[0][at_pred].item()
        labels = labels.to(device)
        #对于有目标攻击来说
        suc_cnt += (at_labels == labels).sum().item()
        #对于无目标攻击来说
        #suc_cnt += (at_labels != labels).sum().item()
        print("Current suc. rate: ", suc_cnt / ((i + 1) * batch_size))
    score_list = np.zeros(tar_cnt)
    score_list[:suc_cnt] = 1.0
    stderr_dist = np.std(np.array(score_list)) / np.sqrt(len(score_list))
    print('Avg suc rate: %.5f +/- %.5f' % (suc_cnt / tar_cnt, stderr_dist))