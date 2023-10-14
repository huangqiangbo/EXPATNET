import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as dsets
# Standard libraries
import numpy as np
import os
# PyTorch
import torch
import torch.nn as nn

y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60,
                                        55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103,
                                        77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T

y_table = nn.Parameter(torch.from_numpy(y_table))
#
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                            [24, 26, 56, 99], [47, 66, 99, 99]]).T
c_table = nn.Parameter(torch.from_numpy(c_table))


def diff_round(x):
    """ Differentiable rounding function
    Input:
        x(tensor)
    Output:
        x(tensor)
    """
    return torch.round(x) + (x - torch.round(x))**3

def phi_diff(x, alpha):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    alpha = torch.where(alpha >= 2.0, torch.tensor([2.0]).cuda(), alpha)
    s = 1/(1-alpha).to(device)
    k = torch.log(2/alpha -1).to(device) 
    phi_x = torch.tanh((x - (torch.floor(x) + 0.5)) * k) * s
    x_ = (phi_x + 1)/2 + torch.floor(x)
    return x_

def quality_to_factor(quality):
    """ Calculate factor corresponding to quality
    Input:
        quality(float): Quality for jpeg compression
    Output:
        factor(float): Compression factor
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality*2
    return quality / 100.


def imshow(img, title):
    #npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    #plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()

#将图像文件夹索引类转变为一般类别——n01440764—>tench
def image_folder_custom_label(root, transform, idx2label) :
    
    # custom_label
    # type : List
    # index -> label
    # example) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']

    #import torchvision.datasets as dsets ->torchvision.datasets.ImageFolder
    #加载测试图片
    old_data = dsets.ImageFolder(root=root, transform=transform)#return (img_data,class_id)
    old_classes = old_data.classes#根据文件夹名字确定的类别 "n01440764"
    
    label2idx = {}
    #print(label2idx)
    #enumerate枚举返回索引及对应的元素值 (0,tench)
    #idx2label:n01440764
    for i, item in enumerate(idx2label) :
        #i表示索引0—>999  item表示对应索引元素n01440764
        print(i,item)
        label2idx[item] = i
    #print('*****************************************')
    #print(label2idx)
    #root=./test/data
    new_data = dsets.ImageFolder(root=root, transform=transform, 
                                 target_transform=lambda x: idx2label.index(old_classes[x]))
    """
    print(dataset.classes)  #根据分的文件夹的名字来确定的类别
    print(dataset.class_to_idx) #按顺序为这些类别定义索引为0,1...
    """
    new_data.classes = idx2label
    #print(idx2label)
    new_data.class_to_idx = label2idx
    #print(label2idx)
    return new_data

def create_dir(dir, print_flag = False):
    if not os.path.exists(dir):
        os.mkdir(dir)
        if print_flag:
            print("Create dir {} successfully!".format(dir))
    elif print_flag:
        print("Directory {} is already existed. ".format(dir))
        
def data_clean(data_dir):
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isfile(class_path):
            os.remove(class_path)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if not img_name.endswith(".png"):
                os.remove(img_path)
