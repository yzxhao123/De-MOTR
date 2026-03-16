import cv2
import os

def images_to_video(img_dir, output_path, fps=30):
    # 获取并排序图片（已是000001格式，直接按文件名排序即可）
    imgs = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    if len(imgs) == 0:
        print("No images found!")
        return

    # 读取第一帧尺寸
    first_frame = cv2.imread(os.path.join(img_dir, imgs[0]))
    h, w, _ = first_frame.shape

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # 写入所有帧
    for name in imgs:
        frame = cv2.imread(os.path.join(img_dir, name))
        if frame is None:
            print(f"Skipping {name}... cannot read")
            continue
        video.write(frame)

    video.release()
    print("Video saved to:", output_path)


# 使用示例
images_to_video(
    img_dir="/root/autodl-tmp/data1/datasets/data_path/MOT17/images/train/MOT_0028/img1",   # 图片文件夹路径
    output_path="/root/autodl-tmp/data1/figs/30.mp4",    # 输出视频
    fps=30                       # 帧率
)
