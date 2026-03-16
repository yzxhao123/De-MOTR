
import os
from PIL import Image

def generate_seqinfo(seq_path):
    """
    根据 img1 文件夹生成 seqinfo.ini 文件
    seq_path: 序列文件夹路径，例如 dataset/MOT17-02-DPM
    """
    img_folder = os.path.join(seq_path, "img1")
    img_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')])

    if not img_files:
        print("img1 文件夹为空！")
        return

    # 读取第一张图片获取宽高
    first_img_path = os.path.join(img_folder, img_files[0])
    img = Image.open(first_img_path)
    imWidth, imHeight = img.size

    seqLength = len(img_files)
    seqName = os.path.basename(seq_path)
    frameRate = 30  # 默认值，可根据需要修改
    imExt = os.path.splitext(img_files[0])[1]  # 自动获取图片扩展名，例如 .jpg

    ini_text = (
        "[Sequence]\n"
        f"name={seqName}\n"
        "imDir=img1\n"
        f"frameRate={frameRate}\n"
        f"seqLength={seqLength}\n"
        f"imWidth={imWidth}\n"
        f"imHeight={imHeight}\n"
        f"imExt={imExt}\n"
    )

    # 写入 seqinfo.ini
    ini_file = os.path.join(seq_path, "seqinfo.ini")
    with open(ini_file, 'w') as f:
        f.write(ini_text)

    print(f"已生成 seqinfo.ini 文件：{ini_file}")

# 示例用法
if __name__ == "__main__":
    seq_folder = "/root/autodl-tmp/data1/datasets/data_path/MOTanimal/images/train/MOTpenguin_6"  # 替换为你的序列路径
    generate_seqinfo(seq_folder)