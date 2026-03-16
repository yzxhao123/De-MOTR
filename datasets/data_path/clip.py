

import cv2
import os

# 视频文件夹路径
video_dir = "/root/autodl-tmp/data1/datasets/data_path/MOTanimal/video"
# 输出根文件夹
output_root = ("/root/autodl-tmp/data1/datasets/data_path/MOTanimal/test")

# 遍历 videos 文件夹下的所有 mp4 文件
for video_file in os.listdir(video_dir):
    if not video_file.endswith(".mp4"):
        continue

    video_path = os.path.join(video_dir, video_file)
    video_name = os.path.splitext(video_file)[0]  # 去掉后缀名
    # 构建输出文件夹路径：test/MOT_00 + 视频名/img1
    output_dir = os.path.join(output_root, f"MOT{video_name}", "img1")
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    frame_id = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 构建帧文件名，6位数字
        frame_name = f"{frame_id:06d}.jpg"
        frame_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(frame_path, frame)
        frame_id += 1

    cap.release()
    print(f"视频 {video_file} 切帧完成，保存到 {output_dir}")


