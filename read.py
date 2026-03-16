import pandas as pd

# 文件路径
file_path = ("/root/autodl-tmp/data1/datasets/data_path/MOT1/train/MOT17-02-SDP/gt/gt.txt"
             )
file_path = ("/root/autodl-tmp/data1/exps/data_path/MOT1/train/MOT17-02-SDP/gt/gt.txt"
             )
try:
    df = pd.read_csv(file_path, sep=",", header=None)  # 自动按列读取
    print(df.head())
    print(f"总列数: {df.shape[1]}")
except Exception as e:
    print(f"读取文件失败: {e}")
