import os
import requests

# 本地保存路径
MODEL_DIR = "/root/autodl-tmp/data/Bert"
os.makedirs(MODEL_DIR, exist_ok=True)

# BERT-base-uncased 文件列表
files = {
    "config.json": "https://huggingface.co/bert-base-uncased/resolve/main/config.json",
    "vocab.txt": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
    "pytorch_model.bin": "https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin"
}

def download_file(url, local_path):
    if os.path.exists(local_path):
        print(f"{local_path} 已存在，跳过下载")
        return
    print(f"正在下载 {url} ...")
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(local_path, 'wb') as f:
        downloaded = 0
        for data in response.iter_content(chunk_size=1024*1024):
            f.write(data)
            downloaded += len(data)
            done = int(50 * downloaded / total) if total else 0
            print(f"\r[{'█'*done}{'.'*(50-done)}] {downloaded/1024/1024:.2f}MB/{total/1024/1024 if total else 0:.2f}MB", end='')
    print(f"\n下载完成: {local_path}")

def main():
    for filename, url in files.items():
        local_path = os.path.join(MODEL_DIR, filename)
        download_file(url, local_path)
    print("\nBERT-base-uncased 模型下载完成！")
    print(f"目录结构如下：\n{os.listdir(MODEL_DIR)}")
    print(f"\n请在 MOTR 的 text_feature.py 或 args.model_dir 指向该目录，并加上 local_files_only=True")

if __name__ == "__main__":
    main()

