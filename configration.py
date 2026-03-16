import os

# 指定要查看的文件夹路径
folder_path = "/root/autodl-tmp/data1/exps/e2e_motr_r50_joint/checkpoint.pth"

# 获取文件夹下的所有内容
contents = os.listdir(folder_path)

print("该文件夹下的内容有：")
for item in contents:
    print(item)


