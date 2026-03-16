import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Helper: depthwise separable conv 1D (temporal) used as lightweight SSM proxy
# -------------------------
class TemporalSSMProxyWithText(nn.Module):
    """1D SSM-like block with optional text-based channel gating.
       Input: x (B, L, C)
       text_features: t (B, C) -> generate channel-wise gate
       Output: (B, L, C)
    """
    def __init__(self, d_model, kernel_size=9, dilation=1, dropout=0.0):
        super().__init__()
        pad = ((kernel_size - 1) // 2) * dilation
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, padding=pad, dilation=dilation, groups=d_model)
        self.pw = nn.Conv1d(d_model, d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # text gating: small FC to generate channel gate
        self.text_gate_fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x, text_features: Optional[torch.Tensor] = None):
        # x: (B, L, C)
        B, L, C = x.shape
        y = x.transpose(1, 2)  # (B, C, L)
        y = self.dw(y)
        y = self.pw(y)
        y = y.transpose(1, 2)  # (B, L, C)

        # apply text gating if provided
        if text_features is not None:
            # text_features: (B, C)
            gate = self.text_gate_fc(text_features).unsqueeze(1)  # (B, 1, C)
            y = y * gate  # channel-wise modulation

        out = self.norm(x + self.dropout(y))
        return out


# -------------------------
# Helper: local spatial enhancement (SS2D primitive)
# -------------------------
class SpatialLocalEnhance(nn.Module):
    """Local 2D enhancement operating on spatial feature maps.
       Accepts flattened spatial tokens (B, H*W, C) or raw (B, C, H, W).
       Returns flattened tokens (B, H*W, C).
    """
    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or d_model
        # pointwise MLP + depthwise 2D conv
        self.mlp1 = nn.Linear(d_model, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, d_model)
        self.dw_conv = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x, spatial_shape: Optional[Tuple[int, int]] = None):
        """
        x: either (B, H*W, C) or (B, C, H, W)
        spatial_shape: (H, W) if x is flattened
        returns: (B, H*W, C)
        """
        if spatial_shape is not None:
            B, L, C = x.shape
            H, W = spatial_shape
            assert L == H * W
            y = self.mlp2(self.act(self.mlp1(x)))               # (B, L, C)
            y = y.transpose(1, 2).view(B, C, H, W)              # (B, C, H, W)
            y = self.dw_conv(y)                                 # (B, C, H, W)
            y = y.view(B, C, -1).transpose(1, 2)                # (B, L, C)
            out = self.norm(x + y)
            return out
        else:
            # assume (B, C, H, W)
            B, C, H, W = x.shape
            y = x
            y = self.dw_conv(y)
            y = y.view(B, C, -1).transpose(1, 2)  # (B, L, C)
            # apply MLP residual
            flat = x.view(B, C, -1).transpose(1, 2)
            out = self.norm(flat + y)
            return out


# -------------------------
# Gate fusion module
# -------------------------
class FusionGate(nn.Module):
    """Gate-based fusion for N sources.
       If all inputs have same length L, does token-wise gating.
       Else does global gating (pool each source, compute gates, broadcast).
    """
    def __init__(self, d_model, n_sources=2):
        super().__init__()
        self.n_sources = n_sources
        self.fc = nn.Linear(d_model * n_sources, n_sources)

    def forward(self, *sources):
        # sources: list of tensors, each (B, L_i, C)
        B = sources[0].shape[0]
        C = sources[0].shape[-1]
        same_length = all(s.shape[1] == sources[0].shape[1] for s in sources)
        if same_length:
            # token-wise gates
            cat = torch.cat(sources, dim=-1)                  # (B, L, n*C)
            logits = self.fc(cat)                             # (B, L, n_sources)
            gates = torch.softmax(logits, dim=-1)             # (B, L, n_sources)
            out = 0
            for i, s in enumerate(sources):
                out = out + gates[..., i:i+1] * s
            return out, gates
        else:
            # global gating: pool each source -> compute gates -> broadcast
            pooled = [s.mean(dim=1) for s in sources]        # list (B, C)
            cat = torch.cat(pooled, dim=-1)                  # (B, n*C)
            logits = self.fc(cat)                            # (B, n_sources)
            gates = torch.softmax(logits, dim=-1)            # (B, n_sources)
            out = 0
            for i, s in enumerate(sources):
                out = out + gates[:, i:i+1].unsqueeze(1) * s  # broadcast (B, 1, 1) -> (B, L_i, C) via broadcasting
            return out, gates


# -------------------------
# Mamba SSM block (for concat sequence tokens)
# -------------------------
class MambaSSMBlock(nn.Module):
    """Process sequence tokens (concat of sparse-image tokens + text tokens).
       Uses a small self-attention (optional) + TemporalSSMProxy + FFN.
    """
    def __init__(self, d_model=256, nhead=4, use_self_attn: bool = False, ssm_kernel=9, ff_dim=1024, dropout=0.1,text_emb=None):
        super().__init__()

        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.drop1 = nn.Dropout(dropout)

        self.ssm = TemporalSSMProxyWithText(d_model, kernel_size=ssm_kernel, dilation=1, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.text_emb = text_emb

    def forward(self, x, pos: Optional[torch.Tensor] = None):
        text_features = self.text_emb.expand(x.size(0), -1) if self.text_emb is not None else None
        residual = x
        if self.use_self_attn:
            q = k = x if pos is None else x + pos
            attn_out, _ = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), x.transpose(0, 1))
            attn_out = attn_out.transpose(0, 1)
            x = self.norm1(residual + self.drop1(attn_out))
            residual = x

        ssm_out = self.ssm(x, text_features=text_features)  # <- pass text features here
        x = self.norm2(residual + ssm_out)
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x


# -------------------------
# Mamba SS2D block (spatial)
# -------------------------
class MambaSS2DBlock(nn.Module):
    """Process spatial feature maps (flattened tokens or 4D) and return flattened tokens.
       Optionally supports light attention over spatial tokens.
    """
    def __init__(self, d_model=256, use_local=True, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.use_local = use_local
        if use_local:
            self.local = SpatialLocalEnhance(d_model)
        else:
            self.local = None

        # optional light spatial attention (channel-mixing)
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, spatial_tokens: torch.Tensor, spatial_shape: Optional[Tuple[int, int]] = None):
        """
        spatial_tokens: (B, Hw, C)  OR (B, C, H, W) if spatial_shape None
        spatial_shape: (H, W) if tokens are flattened
        returns: (B, Hw, C)
        """
        if spatial_shape is not None:
            # flattened tokens
            out = spatial_tokens
            if self.use_local:
                out = self.local(out, spatial_shape)
        else:
            # assume (B, C, H, W)
            out = self.local(spatial_tokens) if self.use_local else spatial_tokens.view(spatial_tokens.size(0), spatial_tokens.size(2)*spatial_tokens.size(3), -1)
        # channel mixing
        out2 = self.channel_mlp(out)
        out = self.norm(out + out2)
        return out


# -------------------------
# Custom Mamba Decoder Layer that uses SSM (seq) + SS2D (spatial) + FusionGate
# -------------------------
class CustomMambaDecoderLayer(nn.Module):
    """
    One decoder layer:
      - apply SSM to concat_seq (B, L_seq, C)
      - apply SS2D to spatial tokens (B, Hw, C)
      - optionally sample spatial tokens to match seq tokens via idx_map for token-wise fusion
      - fuse outputs via gate
      - final FFN on fused result -> return (B, L_seq, C)
    """
    def __init__(self, d_model=256, nhead=4, ssm_kernel=9, ff_dim=1024, dropout=0.1, use_self_attn=False):
        super().__init__()
        self.ssm = MambaSSMBlock(d_model=d_model, nhead=nhead, use_self_attn=use_self_attn, ssm_kernel=ssm_kernel, ff_dim=ff_dim, dropout=dropout)
        self.ss2d = MambaSS2DBlock(d_model=d_model, use_local=True, ff_dim=ff_dim, dropout=dropout)
        self.fuse = FusionGate(d_model, n_sources=2)
        # final small FFN
        self.final_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, concat_seq: torch.Tensor, spatial_tokens: torch.Tensor,
                seq_pos: Optional[torch.Tensor] = None, spatial_shape: Optional[Tuple[int, int]] = None,
                idx_map: Optional[torch.LongTensor] = None):
        """
        concat_seq: (B, L_seq, C)  -- tokens that were formed by concat sparse-image tokens + text tokens
        spatial_tokens: (B, Hw, C) -- full spatial tokens from backbone
        spatial_shape: (H, W) if spatial_tokens is flattened
        idx_map: Optional mapping from concat_seq tokens -> spatial token indices (B, L_seq) integer indices.
                 If provided, we will sample spatial tokens at idx_map to produce token-wise fusion.
        Returns:
           fused_seq: (B, L_seq, C)
           diagnostics: dict with gates (optional)
        """
        # SSM branch
        ssm_out = self.ssm(concat_seq, pos=seq_pos)   # (B, L_seq, C)

        # SS2D branch (process full spatial map)
        ss2d_out = self.ss2d(spatial_tokens, spatial_shape)  # (B, Hw, C)

        # If idx_map provided: sample spatial tokens per seq token for token-wise fusion
        if idx_map is not None:
            # idx_map: (B, L_seq) with indices in [0, Hw)
            B, L_seq, C = concat_seq.shape
            # gather: expand indices for channel gathering
            idx = idx_map.unsqueeze(-1).expand(-1, -1, C)  # (B, L_seq, C)
            sampled_spatial = torch.gather(ss2d_out, 1, idx)  # (B, L_seq, C)
            src_for_fuse = sampled_spatial
        else:
            # lengths mismatch -> use global gating (ss2d_out will be pooled inside FusionGate)
            src_for_fuse = ss2d_out  # FusionGate will auto-detect length mismatch

        fused, gates = self.fuse(ssm_out, src_for_fuse)  # either token-wise or global-mode

        out = fused + self.final_ffn(fused)
        return out, {'gates': gates}


# -------------------------
# Custom decoder (stack of layers) - returns hs (num_layers, B, L_seq, C)
# -------------------------
class CustomMambaDecoder(nn.Module):
    def __init__(self, layer: CustomMambaDecoderLayer, num_layers: int, return_intermediate: bool = True):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.norm = nn.LayerNorm(layer.ssm.ffn[0].in_features) if hasattr(layer.ssm, 'ffn') else None

    def forward(self, concat_seq: torch.Tensor, spatial_tokens: torch.Tensor,
                seq_pos: Optional[torch.Tensor] = None, spatial_shape: Optional[Tuple[int, int]] = None,
                idx_map: Optional[torch.LongTensor] = None):
        """
        concat_seq: (B, L_seq, C)  -- e.g., sparse tokens concatenated with text tokens (where text already fused)
        spatial_tokens: (B, Hw, C)  -- backbone full spatial tokens (projected to d_model)
        idx_map: (B, L_seq) optional mapping to spatial token indices for token-wise fusion
        """
        output = concat_seq
        intermediate = []
        gate_records = []
        for layer in self.layers:
            output, diag = layer(output, spatial_tokens, seq_pos=seq_pos, spatial_shape=spatial_shape, idx_map=idx_map)
            gate_records.append(diag.get('gates', None))
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            hs = torch.stack(intermediate)  # (num_layers, B, L_seq, C)
        else:
            hs = output.unsqueeze(0)
        if self.norm is not None:
            hs = self.norm(hs)

        return hs, gate_records

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
