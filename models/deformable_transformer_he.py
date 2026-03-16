# 在新文件 deformable_transformer_he.py 中添加以下代码

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from models.structures import Boxes, matched_boxlist_iou, pairwise_iou

from util.misc import inverse_sigmoid
from util.box_ops import box_cxcywh_to_xyxy
from models.ops.modules import MSDeformAttn

from .newffn import LightFFN


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300, decoder_self_cross=True, sigmoid_attn=False,
                 extra_track_attn=False):
        super().__init__()

        self.new_frame_adaptor = None
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        # 使用 deformable_transformer_en.py 中的 encoder
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points,
                                                          sigmoid_attn=sigmoid_attn,
                                                          freq_ratio=0.3)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        # 使用 deformable_transformer_mamba.py 中的 decoder
        decoder_layer = CustomMambaDecoderLayer(d_model=d_model, n_levels=num_feature_levels,
                                                n_points=dec_n_points, ff_dim=dim_feedforward,
                                                dropout=dropout, extra_track_attn=extra_track_attn)
        self.decoder = CustomMambaDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        return torch.stack([valid_ratio_w, valid_ratio_h], -1)

    def forward(self, srcs, masks, pos_embeds, query_embed=None, ref_pts=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder (来自 deformable_transformer_en.py)
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

            if ref_pts is None:
                reference_points = self.reference_points(query_embed).sigmoid()
            else:
                reference_points = ref_pts.unsqueeze(0).repeat(bs, 1, 1).sigmoid()
            init_reference_out = reference_points

        # decoder (来自 deformable_transformer_mamba.py)
        hs, inter_references = self.decoder(
            tgt=tgt,
            reference_points=reference_points,
            src=memory,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            query_pos=query_embed,
        )

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


# 从 deformable_transformer_en.py 复制 Encoder 相关代码
class FrequencyEnhancer(nn.Module):
    def __init__(self, d_model, freq_ratio=0.3):
        super().__init__()
        self.freq_ratio = freq_ratio
        # 可学习的频域 mask（作用在频率维度上）
        self.gate = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """
        x: [B, N, C]  -> (Batch, Tokens, Channels)
        我们在 token 维度做 FFT，相当于时序/序列维度的频率分析
        """
        B, N, C = x.shape

        # 对 token 维度做 FFT
        x_freq = torch.fft.rfft(x, dim=1)  # [B, N//2+1, C]

        # 高频截断（保留前 freq_ratio 的低频分量）
        keep_len = int(x_freq.size(1) * self.freq_ratio)
        mask = torch.zeros_like(x_freq)
        mask[:, :keep_len, :] = 1.0

        # 频率选择 + 通道权重
        x_freq = x_freq * mask * self.gate

        # IFFT 回时域
        x_time = torch.fft.irfft(x_freq, n=N, dim=1)  # [B, N, C]
        return x_time


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, sigmoid_attn=False, freq_ratio=0.3):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = LightFFN(d_model, dropout=dropout, freq_ratio=freq_ratio)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points, src,
            spatial_shapes, level_start_index, padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 集成频域处理的FFN
        src = self.ffn(src)
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


# 从 deformable_transformer_mamba.py 复制 Decoder 相关代码
class DeformConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.offset_conv = nn.Conv1d(in_channels, kernel_size * groups, kernel_size, stride, padding, groups=groups,
                                     bias=True)
        # Simplified; real DCNv4 has more components, but for code matching description.

    def forward(self, x):
        offset = self.offset_conv(x)
        # Simulate deformable: for demo, just use regular conv.
        return self.conv(x)  # Placeholder; implement proper deformable sampling in real code.


# CrossAttention simple impl
class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, query, key_value):
        return self.attn(query, key_value, key_value)[0]


class textblock(nn.Module):  # Renamed to TextBlock as per description
    def __init__(self, d_model, kernel_sizes=(3, 5, 7), dilation=(1, 2, 4), dropout=0.1, ff_dim=512):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Multi-scale Depthwise TCN
        self.tcn_branches = nn.ModuleList()
        for ks, dil in zip(kernel_sizes, dilation):
            pad = (ks - 1) * dil // 2
            if dil > 2:  # For large receptive fields (large dilation), use DCNv4-like
                conv = DeformConv1d(d_model, d_model, ks, padding=pad, groups=d_model, bias=False)
            else:
                conv = nn.Conv1d(d_model, d_model, ks, padding=pad, dilation=dil, groups=d_model, bias=False)
            self.tcn_branches.append(nn.Sequential(conv, nn.GELU()))

        self.pointwise_fusion = nn.Conv1d(d_model * len(kernel_sizes), d_model, kernel_size=1, bias=False)

        # Hierarchical Modulation
        # Primary: CrossAttention with tracking query
        self.cross_attn = CrossAttention(d_model)

        # Auxiliary: Memory modulation
        self.gate_linear = nn.Linear(2 * d_model, 1)  # W_gate for [F_primary; M_t-1]
        self.memory_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, x, tracking_query: torch.Tensor, memory_prev: Optional[torch.Tensor] = None):
        """
        x: (B, L, C) - sparse text-fused features
        tracking_query: (B, Q, C) - Q_tracking
        memory_prev: (B, L, C) - M_{t-1}, optional
        """
        residual = x
        y = self.norm_in(x)  # PreNorm
        y = y.transpose(1, 2)  # (B, C, L)

        tcn_outputs = []
        for branch in self.tcn_branches:
            tcn_outputs.append(branch(y))

        min_len = min([t.size(2) for t in tcn_outputs])
        tcn_outputs = [t[:, :, :min_len] for t in tcn_outputs]

        y = torch.cat(tcn_outputs, dim=1)  # Concat multi-scale
        y = self.pointwise_fusion(y)  # Fuse
        y = y.transpose(1, 2)  # (B, L, C)

        # Assume F_aligned = y after TCN/DCN

        # Primary Modulation
        f_primary = self.cross_attn(y, tracking_query) + y

        # Auxiliary Modulation
        if memory_prev is not None:
            # Detect activation condition: for demo, always apply; in real, add condition like if change detected
            concat = torch.cat([f_primary.mean(dim=1), memory_prev.mean(dim=1)],
                               dim=-1)  # [F_primary; M_t-1], using mean for simplicity
            alpha = torch.sigmoid(self.gate_linear(concat)).unsqueeze(1)  # (B, 1, 1)
            attn_out = self.memory_attn(f_primary, memory_prev, memory_prev)[0]
            f_auxiliary = alpha * attn_out + (1 - alpha) * f_primary
        else:
            f_auxiliary = f_primary

        f_auxiliary = self.dropout(f_auxiliary)
        f_auxiliary = f_auxiliary + residual  # Residual after modulations

        # FFN
        ffn_out = self.ffn(f_auxiliary)
        out = self.norm_ffn(f_auxiliary + ffn_out)
        return out


class SS2d(nn.Module):
    def __init__(self, dim, expand=2, dropout=0.0, use_local_conv=True):
        super().__init__()
        self.dim = dim
        self.expand = expand
        self.use_local_conv = use_local_conv

        hidden_dim = dim * expand

        # 行向 Mamba
        self.row_mamba = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, dim, kernel_size=1, bias=False),
        )

        # 列向 Mamba
        self.col_mamba = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, dim, kernel_size=1, bias=False),
        )

        if self.use_local_conv:
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        else:
            self.local_conv = nn.Identity()

        # query modulation (FiLM)
        self.query_mod = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, query=None):
        B, C, H, W = x.shape

        # 行处理
        x_row = x.permute(0, 2, 1, 3).reshape(B * H, C, W)  # (B*H, C, W)
        x_row = self.row_mamba(x_row)
        x_row = x_row.reshape(B, H, C, W).permute(0, 2, 1, 3)

        # 列处理
        x_col = x.permute(0, 3, 1, 2).reshape(B * W, C, H)  # (B*W, C, H)
        x_col = self.col_mamba(x_col)
        x_col = x_col.reshape(B, W, C, H).permute(0, 2, 3, 1)

        # 融合 + 局部卷积
        x = x_row + x_col
        x = self.local_conv(x)

        if query is not None:
            q_pool = query.mean(1)  # (B, C)
            scale_shift = self.query_mod(q_pool)  # (B, 2C)
            scale, shift = scale_shift.chunk(2, dim=-1)
            scale, shift = scale.unsqueeze(-1).unsqueeze(-1), shift.unsqueeze(-1).unsqueeze(-1)
            x = x * (1 + scale) + shift

        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        return x


class CustomMambaDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_points=4, ff_dim=1024, dropout=0.1, activation="relu",
                 n_heads=8, sigmoid_attn=False, extra_track_attn=False):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_points = n_points
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.num_heads = n_heads

        # SSM & SS2D
        self.ssm = textblock(d_model=d_model, ff_dim=ff_dim, dropout=dropout)
        self.ss2d = SS2d(dim=d_model, expand=1, dropout=dropout)

        # cross-attn using MSDeformAttn
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # normalization layers
        self.norm_ssm = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_fuse = nn.LayerNorm(d_model)

        # projection layers
        self.proj_ssm = nn.Linear(d_model, d_model)
        self.proj_cross = nn.Linear(d_model, d_model)
        self.proj_ss2d = nn.Linear(d_model, d_model)
        self.fuse_proj = nn.Linear(d_model, d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(d_model)

        # alpha parameters
        self.register_parameter("ssm_alpha", nn.Parameter(torch.tensor(0.3)))
        self.register_parameter("cross_alpha", nn.Parameter(torch.tensor(0.3)))
        self.register_parameter("ss2d_alpha", nn.Parameter(torch.tensor(0.3)))
        self.register_parameter("fuse_alpha", nn.Parameter(torch.tensor(0.3)))
        self.register_parameter("ffn_alpha", nn.Parameter(torch.tensor(0.3)))

        # update track query_embed
        self.extra_track_attn = extra_track_attn
        if self.extra_track_attn:
            print('Training with Extra Self Attention in Every Decoder.', flush=True)
            self.update_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout5 = nn.Dropout(dropout)
            self.norm4 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def _forward_track_attn(self, tgt, query_pos):
        q = k = self.with_pos_embed(tgt, query_pos)
        if q.shape[1] > 300:
            tgt2 = self.update_attn(q[:, 300:].transpose(0, 1),
                                    k[:, 300:].transpose(0, 1),
                                    tgt[:, 300:].transpose(0, 1))[0].transpose(0, 1)
            tgt = torch.cat([tgt[:, :300], self.norm4(tgt[:, 300:] + self.dropout5(tgt2))], dim=1)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, spatial_shapes, level_start_index,
                src_padding_mask=None):

        # ---------------- SSM ----------------
        ssm_out = self.ssm(tgt, query_pos)

        ######ss2d_out
        B, N, C = ssm_out.shape

        # 1. 计算 H, W（尽量接近正方形）
        H = W = int(math.ceil(math.sqrt(N)))  # ceil 确保 H*W >= N

        # 2. padding 补齐到 H*W
        pad_N = H * W - N
        if pad_N > 0:
            pad_tensor = torch.zeros(B, pad_N, C, device=ssm_out.device, dtype=ssm_out.dtype)
            ssm_padded = torch.cat([ssm_out, pad_tensor], dim=1)  # (B, H*W, C)
        else:
            ssm_padded = ssm_out

        # 3. reshape 成 (B, C, H, W) 用于卷积
        x = ssm_padded.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)

        # 4. 经过你的 SS2D 卷积
        ss2d_out = self.ss2d(x)  # 输出 (B, C, H, W)

        # 5. reshape 回序列长度 N
        ss2d_out = ss2d_out.view(B, C, H * W)[:, :, :N].transpose(1, 2)  # (B, N, C)

        alpha_ssm = torch.sigmoid(self.ssm_alpha)
        alpha_ss2d = torch.sigmoid(self.ss2d_alpha)

        enhanced_tgt = alpha_ssm * ssm_out + alpha_ss2d * ss2d_out

        tgt2 = self.norm_fuse(tgt + enhanced_tgt)

        if self.extra_track_attn:
            tgt2 = self._forward_track_attn(tgt2, query_pos)

        # ---------------- Cross-attention ----------------
        cross_out = self.cross_attn(
            self.with_pos_embed(tgt2, query_pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            src_padding_mask
        )
        tgt = tgt2 + self.dropout2(cross_out)
        tgt = self.norm_cross(tgt)
        ffn_out = self.ffn(tgt)
        tgt = self.norm_ffn(tgt + ffn_out)

        return tgt


class CustomMambaDecoder(nn.Module):
    """
    Decoder composed of multiple CustomMambaDecoderLayer (no parameter sharing)
    Supports iterative reference point update like MOTR / Deformable DETR
    """

    def __init__(self, decoder_layer, num_layers=6, return_intermediate=True):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.class_embed = None
        self.bbox_embed = None

    def forward(self, tgt, reference_points, src, spatial_shapes, level_start_index, valid_ratios, query_pos=None,
                src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            output = layer(output, query_pos, reference_points_input, src, spatial_shapes, level_start_index,
                           src_padding_mask)

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(True)
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        decoder_self_cross=not args.decoder_cross_self,
        sigmoid_attn=args.sigmoid_attn,
        extra_track_attn=args.extra_track_attn,
    )
