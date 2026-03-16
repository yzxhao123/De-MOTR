import os

def delete_dsstore_files(root_dir):
    """
    删除指定目录及其子目录下的所有 .DS_Store 文件
    """
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file == '.DS_Store' or file.endswith('.DS'):
                file_path = os.path.join(dirpath, file)
                try:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
                    count += 1
                except Exception as e:
                    print(f"删除失败: {file_path}, 原因: {e}")
    print(f"\n共删除 {count} 个 .DS_Store 文件。")

# 示例使用：把路径换成你的文件夹路径
delete_dsstore_files("/root/autodl-tmp/data1/datasets/data_path/MOT17")
