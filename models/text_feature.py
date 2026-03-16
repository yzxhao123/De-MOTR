import torch
from transformers import BertTokenizer, BertModel

class DefaultTextEmb:
    """
    默认文本特征生成器，用 BERT 将文本编码为固定向量。
    生成一次 embedding 后可在整个训练或推理中重复使用。
    """
    def __init__(self, model_dir="/root/autodl-tmp/data/Bert",
                 text="The persons present in each video frame, used to assist tracking.",
                 device="cuda"):
        self.device = device
        self.text = text

        # 加载 tokenizer 和 BERT 模型

        self.tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = BertModel.from_pretrained(model_dir, local_files_only=True).to(device)
        self.model.eval()

        # 生成 embedding 并放到指定 device
        self.embedding = self._build_embedding().to(self.device)

    def _build_embedding(self):
        with torch.no_grad():
            inputs = self.tokenizer(self.text, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            # 取 CLS token 作为文本特征
            text_emb = outputs.last_hidden_state[:, 0:1, :]  # (1,1,C)
        return text_emb

    def get(self):
        """返回生成好的文本特征，保证在指定 device"""
        return self.embedding.to(self.device)
