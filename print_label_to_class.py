import json

# 读取 imagenet_class_index.json 文件
with open('imagenet_class_index.json', 'r') as f:
    class_index = json.load(f)

# 将类别信息转换为字典格式
label_to_class = {}
for k, v in class_index.items():
    label_to_class[int(k)] = v[1]

print(label_to_class)
