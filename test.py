# test_transformer_loss.py
import torch
import torch.nn as nn
from models.deformable_transformer_mamba import build_deforamble_transformer

def test_transformer_loss():
    # ---- 配置 ----
    class Args:
        hidden_dim = 256
        nheads = 8
        enc_layers = 3
        dec_layers = 3
        dim_feedforward = 1024
        dropout = 0.1
        num_feature_levels = 4
        dec_n_points = 4
        enc_n_points = 4
        num_queries = 10
        sigmoid_attn = False
        extra_track_attn = False

    args = Args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 构建模型 ----
    model = build_deforamble_transformer(args).to(device)
    model.train()

    # ---- 伪数据 ----
    bs = 2
    C = args.hidden_dim
    Hs = [16, 8, 4, 2]
    Ws = [16, 8, 4, 2]

    srcs, masks, pos_embeds = [], [], []
    for h, w in zip(Hs, Ws):
        src = torch.rand(bs, C, h, w, device=device)
        mask = torch.zeros(bs, h, w, dtype=torch.bool, device=device)
        pos_embed = torch.rand(bs, C, h, w, device=device)
        srcs.append(src)
        masks.append(mask)
        pos_embeds.append(pos_embed)

    # ---- Encoder ----
    src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], [], []
    for lvl, (src, mask, pos) in enumerate(zip(srcs, masks, pos_embeds)):
        B, C, H, W = src.shape
        spatial_shapes.append((H, W))
        src_flatten.append(src.flatten(2).transpose(1, 2))
        mask_flatten.append(mask.flatten(1))
        lvl_pos_embed_flatten.append(pos.flatten(2).transpose(1, 2))

    src_flatten = torch.cat(src_flatten, dim=1)
    mask_flatten = torch.cat(mask_flatten, dim=1)
    lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, dim=1)
    spatial_shapes = torch.as_tensor(spatial_shapes, device=device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)),
                                   spatial_shapes.prod(1).cumsum(0)[:-1]))
    valid_ratios = torch.ones(bs, args.num_feature_levels, 2, device=device)

    # ---- Query Embeddings ----
    num_queries = args.num_queries
    query_embed = torch.rand(num_queries, 2 * C, device=device)

    # ---- Ground truth (随机) ----
    gt_classes = torch.randint(0, 2, (bs, num_queries), dtype=torch.long, device=device)
    gt_bboxes = torch.rand(bs, num_queries, 4, device=device)

    # ---- 损失函数 ----
    cls_criterion = nn.CrossEntropyLoss()
    bbox_criterion = nn.L1Loss()

    # ---- 前向 ----
    hs, init_ref, inter_refs, class_logits, bbox_deltas = model(
        srcs, masks, pos_embeds, query_embed=query_embed
    )

    # 取最后一层 decoder 输出
    pred_logits = class_logits[-1]          # (B, num_queries, num_classes)
    pred_bboxes = bbox_deltas[-1]           # (B, num_queries, 4)

    # ---- 计算 loss ----
    cls_loss = cls_criterion(pred_logits.transpose(1, 2), gt_classes)
    bbox_loss = bbox_criterion(pred_bboxes, gt_bboxes)

    total_loss = cls_loss + bbox_loss

    print("Class loss:", cls_loss.item())
    print("BBox L1 loss:", bbox_loss.item())
    print("Total loss:", total_loss.item())

if __name__ == "__main__":
    test_transformer_loss()
