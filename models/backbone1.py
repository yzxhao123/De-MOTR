# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Backbone modules.
"""
from collections import OrderedDict
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process
from .text_feature import DefaultTextEmb
from .position_encoding import build_position_encoding
from .demaba import  MambaSSMBlock

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
            
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, text_emb, num_topk=100, embed_dim=256):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.text_emb = text_emb
        self.num_topk = num_topk
        self.coord_proj = nn.Linear(2, embed_dim)
        self.text_proj = nn.Linear(text_emb.shape[-1], embed_dim)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels
        self.sparse_proj = nn.Linear(512, 256)  # **这里提前定义**
        self.fused_to_original_proj = nn.Linear(256, 512)

    def forward(self, tensor_list: NestedTensor):
        xs = self.backbone(tensor_list)
        out = []
        pos = []

        out_features = [xs[k] for k in sorted(xs.keys())]

        for i, feat in enumerate(out_features):
            tensor, mask = feat.decompose()
            B, C, H, W = tensor.shape

            if i == 0:
                feat_flat = tensor.view(B, C, -1).permute(0, 2, 1)  # B, HW, C
                scores = feat_flat.mean(-1)
                topk_vals, topk_idx = torch.topk(scores, self.num_topk, dim=1)

                sparse_feat = torch.gather(feat_flat, 1, topk_idx.unsqueeze(-1).expand(-1, -1, C))  # B, topk, C
                text_feat = self.text_proj(self.text_emb.reshape(1, 1, -1)).to(sparse_feat.device).expand(B,
                                                                                                          self.num_topk,
                                                                                                          -1)
                fused_sparse = self.sparse_proj(sparse_feat) + text_feat  # B, topk, 256
                fused_sparse = self.fused_to_original_proj(fused_sparse)  # B, topk, 512

                # 非 inplace scatter
                tensor_flat = tensor.view(B, C, -1).permute(0, 2, 1)  # B, HW, C
                tensor_flat = tensor_flat.scatter(1, topk_idx.unsqueeze(-1).expand(-1, -1, C), fused_sparse)
                tensor = tensor_flat.permute(0, 2, 1).view(B, C, H, W)

            out.append(NestedTensor(tensor, mask))
            pos.append(self.position_embedding(NestedTensor(tensor, mask)).to(tensor.dtype))

        return out, pos


def build_backbone(args):
    # 在 build_backbone 里
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)

    # 生成文本 embedding
    text_emb = DefaultTextEmb(model_dir="/root/autodl-tmp/data1/Bert").get()


    # 构建融合 backbone
    model = Joiner(backbone, position_embedding, text_emb)

    return model
