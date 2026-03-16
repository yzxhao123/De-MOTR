import os

# 配置路径
labels_dir = '/root/autodl-tmp/data1/datasets/data_path/MOT17/labels_with_ids/test'  # labels_with_ids/train 根目录
output_file = '/root/autodl-tmp/data1/datasets/data_path/antmot17.train'

# 白名单：只处理这几个序列
allowed_seqs = {
    'MOTdolphin_1',
    'MOTdolphin_2',
    'MOTdolphin_3',
    'MOTduck_1',
    'MOTduck_2',
    'MOTduck_3',
    'MOTgoose_1',
    'MOTgoose_2',
    'MOTgoose_3',
    'MOTpenguin_1',
    'MOTpenguin_2',
    'MOTpenguin_3'

    # 'MOT1-0003',
    # 'MOT1-0023',
    # 'MOT1-0047',
    # 'MOT1-0071',
    # 'MOT1-0076',
    # 'MOT1-0090',
    # 'MOT1-0095',
    # 'MOT1-0130',
    # 'MOTdeer_1',
    # 'MOTdeer_2',
    # 'MOTdeer_3',
    # 'MOThorse_1',
    # 'MOThorse_2',
    # 'MOThorse_3',
    # 'MOTzebra_1',
    # 'MOTzebra_2',

    # 'MOTdeer_4',
    # 'MOTdeer_5',
    # 'MOTdeer_6',
    # 'MOTdeer_7',
    # 'MOThorse_4',
    # 'MOThorse_5',
    # 'MOThorse_6',
    # 'MOThorse_7',
    # 'MOTzebra_3',
    # 'MOTzebra_4',
    # 'MOTzebra_5',
    #
    # 'MOT17_0015',
    # 'MOT17_0019',
    # 'MOT17_0017',
    # 'MOT17_0016',
    # 'MOT17_0010',
    # 'MOT17_0006',
    # 'MOT17_0007',
    # 'MOT17_0008',
    # 'MOT17_0009',
    # 'MOT17_0011',
    # 'MOT17_0012',
    # 'MOT_0040',
    # 'MOT_0041',
    # 'MOT_0042',
    # 'MOT_0043',
    # 'MOT_0044',
    # 'MOT_0045',
    # 'MOT_0046',
    # 'MOT_0021',
    # 'MOT_0024',
    # 'MOT_0027',
    # 'MOT_0028',
    # 'MOT_0031',
    # 'MOT_0030',
    # 'MOT-0051',
    # 'MOT-0052',
    # 'MOT-0053',
    # 'MOT148'
}



lines_to_write = []


seq_list = [d for d in os.listdir(labels_dir) if os.path.isdir(os.path.join(labels_dir, d))]
seq_list.sort()

for seq_name in seq_list:
    if allowed_seqs and seq_name not in allowed_seqs:
        continue

    seq_path = os.path.join(labels_dir, seq_name)

    # 进入 img1 子目录
    img1_dir = os.path.join(seq_path, 'img1')
    if not os.path.isdir(img1_dir):
        print(f"⚠️ 警告: {img1_dir} 不存在，跳过序列 {seq_name}")
        continue

    # 获取 img1 下所有 .txt 文件或图片文件（根据你真实存放的类型选择）
    # 这里假设 img1 下存在 frame_id.txt 或 frame_id.jpg；你的原脚本用的是 txt 文件名作为帧号来源
    txt_files = [f for f in os.listdir(img1_dir) if f.endswith('.txt')]

    # 如果你希望直接根据 img 文件名（jpg）来生成列表，可以改成：
    # txt_files = [f for f in os.listdir(img1_dir) if f.endswith('.jpg')]

    # 按帧号的数字值排序：提取文件名主体并转成 int，支持"000001"这种零填充形式
    def frame_key(fname):
        base = os.path.splitext(fname)[0]
        try:
            return int(base)
        except:
            # 如果不能转 int（如有前缀），退回到字符串排序
            return base

    txt_files_sorted = sorted(txt_files, key=frame_key)

    for txt_file in txt_files_sorted:
        frame_id = os.path.splitext(txt_file)[0]  # 如 '000001'
        # 构造图像路径（跟你原来保持一致）
        img_path = f"MOT17/images/test/{seq_name}/img1/{frame_id}.jpg"
        lines_to_write.append(img_path + '\n')

# 写入输出文件
with open(output_file, 'w') as f:
    f.writelines(lines_to_write)

print(f"✅ 已生成 {len(lines_to_write)} 行")
print(f"结果已保存到: {output_file}")
