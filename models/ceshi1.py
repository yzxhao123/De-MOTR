import torch
from torch import nn
from util.misc import NestedTensor

B, C, H, W = 1, 512, 108, 192  # 第一层特征大小
feat_tensor = torch.randn(B, C, H, W)
mask = torch.zeros(B, H, W, dtype=torch.bool)

nested_feat = NestedTensor(feat_tensor, mask)

# 随机文本向量
text_emb = torch.randn(1, 786)

# 定义 JoinerWithTextFusion
class DummyJoiner(nn.Module):
    def __init__(self, num_topk=100, embed_dim=256):
        super().__init__()
        self.num_topk = num_topk
        self.input_proj = nn.Conv2d(512, 256, 1,1)
        self.text_proj = nn.Linear(786, embed_dim)
        self.coord_proj = nn.Linear(2, embed_dim)
        self.text_emb = text_emb

    def forward(self, nested_feat):
        src, mask = nested_feat.decompose()
        src = self.input_proj(src)

        B, C, H, W = src.shape
        feat_flat = src.view(B, C, -1).permute(0, 2, 1)  # B, HW, C
        scores = feat_flat.mean(-1)
        topk_vals, topk_idx = torch.topk(scores, self.num_topk, dim=1)

        ys = torch.div(topk_idx, W, rounding_mode='floor').float() / H
        xs_coord = (topk_idx % W).float() / W
        coords = torch.stack([xs_coord, ys], dim=-1)  # B, topk, 2
        pos_embed = self.coord_proj(coords)

        text_proj = self.text_proj(self.text_emb.reshape(1, 1, -1)).expand(B, 1, -1)
        return NestedTensor(src, mask), [pos_embed], text_proj

# 测试
model = DummyJoiner()
projected_feat, pos_embed_list, text_proj = model(nested_feat)

print("Projected Feature shape:", projected_feat.tensors.shape)
print("Positional Embedding shape:", pos_embed_list[0].shape)
print("Text Projection shape:", text_proj.shape)
