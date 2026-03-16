import os
import shutil

root_dir = "/root/autodl-tmp/data1"  # 改成你的路径

for root, dirs, files in os.walk(root_dir):
    for name in files:
        if name.startswith(".ip"):
            path = os.path.join(root, name)
            print("Removing file:", path)
            os.remove(path)

    for name in list(dirs):
        if name.startswith(".ip"):
            path = os.path.join(root, name)
            print("Removing dir:", path)
            shutil.rmtree(path)
