import json
import os
# 配置路径
JSON_PATH = '/root/autodl-tmp/MOTdeer/annotations/train_half.json'  # train.json 路径
LABELS_ROOT = '/root/autodl-tmp/data/datasets/data_path/MOT17/labels_with_ids/train'  # 包含各个序列文件夹
OUTPUT_TRAIN_FILE = 'mot17.train'                # 输出的 .train 文件




# 读取 train.json
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

# 存储最终有效的图像路径
valid_image_paths = []

# 遍历 json 中的所有图像
for img_info in data['images']:
    file_name = img_info['file_name']  # 如: "MOT17_0006/img1/000001.jpg"

    # 解析路径：获取序列名和文件名
    parts = file_name.split('/')
    if len(parts) < 3:
        print(f"跳过无效路径: {file_name}")
        continue

    seq_name = parts[0]  # 如: MOT17_0006
    img_subdir = parts[1]  # 如: img1
    img_filename = parts[2]  # 如: 000001.jpg

    # 提取文件名（不含扩展名），加上 .txt
    base_name = os.path.splitext(img_filename)[0]  # → 000001
    txt_filename = base_name + '.txt'

    # 构造对应的 .txt 文件路径
    txt_path = os.path.join(LABELS_ROOT, seq_name, img_subdir, txt_filename)

    # 检查 .txt 文件是否存在
    if os.path.exists(txt_path):
        # 构造 .train 中要写入的路径：MOT17/images/train/MOT17_0006/img1/000001.jpg
        train_path = f'MOT17/images/train/{seq_name}/{img_subdir}/{img_filename}'
        valid_image_paths.append(train_path)
    else:
        # 可选：打印缺失的标注
        # print(f"⚠️ 无标注文件: {txt_path}")
        pass

# 写入 .train 文件
with open(OUTPUT_TRAIN_FILE, 'w') as f:
    for path in sorted(valid_image_paths):
        f.write(path + '\n')

print(f"✅ 成功生成 {len(valid_image_paths)} 个有效路径到 '{OUTPUT_TRAIN_FILE}'")
print("示例路径:")
for i in range(min(10, len(valid_image_paths))):
    print(valid_image_paths[i])