from transformers import BertTokenizer, BertModel
import torch

# 1. 加载预训练的 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# 2. 输入文本
text = "The humans present in each video frame, used to assist tracking."

# 3. 分词 + 转换为张量
inputs = tokenizer(text, return_tensors="pt")

# 4. 提取特征
with torch.no_grad():  # 不计算梯度，加快推理
    outputs = model(**inputs)

# 5. 输出结果
last_hidden_state = outputs.last_hidden_state   # 每个 token 的向量 (batch_size, seq_len, hidden_size)
cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] 向量 (batch_size, hidden_size)

print("句子特征维度:", cls_embedding.shape)
print("句子特征:", cls_embedding)
