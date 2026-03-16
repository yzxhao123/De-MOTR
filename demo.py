# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from __future__ import print_function
import glob
import math
import os
import os.path as osp
import random
import time
from collections import OrderedDict
import torchvision.transforms.functional as F
import cv2
import numpy as np
import torch
import argparse
import torchvision.transforms.functional as F
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw
from models import build_model
from util.tool import load_model
from main import get_args_parser
from torch.nn.functional import interpolate
from typing import List
import shutil
from models.structures import Instances
import matplotlib.pyplot as plt
from thop import profile
from util.misc import NestedTensor


class MOTRWrapper(torch.nn.Module):
    """
    用于thop计算整个MOTR GFLOPs
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):

        # 构造 NestedTensor
        mask = torch.zeros((x.shape[0], x.shape[2], x.shape[3]), dtype=torch.bool).to(x.device)
        samples = NestedTensor(x, mask)

        # 不使用历史track
        track_instances = None

        outputs = self.model.inference_single_image(
            x,
            (x.shape[2], x.shape[3]),
            track_instances
        )

        return outputs["track_instances"].boxes

np.random.seed(2020)

COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
             (238, 130, 238),
             (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
             (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
             (144, 238, 144),
             (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
             (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
             (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
             (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
             (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
             (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]


def plot_one_box(x, img, color=None, label=None, score=None, line_thickness=None):
    # Plots one bounding box on image img

    # tl = line_thickness or round(
    #     0.002 * max(img.shape[0:2])) + 1  # line thickness
    tl = 2
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        if score is not None:
            cv2.putText(img, score, (c1[0], c1[1] + 30), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def draw_bboxes(ori_img, bbox, identities=None, offset=(0, 0), cvt_color=False):
    if cvt_color:
        ori_img = cv2.cvtColor(np.asarray(ori_img), cv2.COLOR_RGB2BGR)
    img = ori_img
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        if len(box) > 4:
            score = '{:.2f}'.format(box[4])
        else:
            score = None
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = COLORS_10[id % len(COLORS_10)]
        label = '{:d}'.format(id)
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        img = plot_one_box([x1, y1, x2, y2], img, color, label, score=score)
    return img


def draw_points(img: np.ndarray, points: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    assert len(points.shape) == 2 and points.shape[1] == 2, 'invalid points shape: {}'.format(points.shape)
    for i, (x, y) in enumerate(points):
        if i >= 300:
            color = (0, 255, 0)
        cv2.circle(img, (int(x), int(y)), 2, color=color, thickness=2)
    return img

class LoadVideo:  # for inference
    def __init__(self, path, img_size=(640, 640)):
        if not os.path.isfile(path):
            raise FileExistsError
        
        self.cap = cv2.VideoCapture(path)        
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.seq_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.seq_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        print('Lenth of the video: {:d} frames'.format(self.vn))

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration
        # Read image
        res, img = self.cap.read()  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB
        assert img is not None, 'Failed to load frame {:d}'.format(self.count)

        cur_img, ori_img = self.init_img(img)
        return self.count, cur_img, ori_img

    def init_img(self, img):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.width:
            scale = self.width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img

    def __len__(self):
        return self.vn  # number of files

class MOTR(object):
    def update(self, dt_instances: Instances):
        ret = []
        for i in range(len(dt_instances)):
            label = dt_instances.labels[i]
            if label == 0:
                id = dt_instances.obj_idxes[i]
                box_with_score = np.concatenate([dt_instances.boxes[i], dt_instances.scores[i:i+1]], axis=-1)
                ret.append(np.concatenate((box_with_score, [id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))

class Detector(object):
    def __init__(self, args):

        self.args = args

        # build model and load weights
        self.model, _, _ = build_model(args)
        checkpoint = torch.load(args.resume, map_location='cpu')
        self.model = load_model(self.model, args.resume)
        self.model = self.model.cuda()
        self.model.eval()
        self._calculate_model_stats()
        # ---------- 🔥 注册 FFN conv2 的 hook  ----------
        self._cam_activations = None   # forward 保存 activation (tensor)
        self._cam_grads = None         # backward 保存 grad (tensor)
        self._spatial_shapes = None    # encoder forward 保存 spatial_shapes (list of (H,W))
        self._cam_enabled = False       # 用于控制是否启用 CAM

        # choose last encoder layer id (最后一层)
        encoder_layer_id = -1   # 最后一层
        # get target conv2 (Conv1d)
        # 在Detector.__init__中修改target_conv的定位
        # 原来是:
        # target_conv = self.model.transformer.encoder.layers[encoder_layer_id].ffn.conv2

        # 改为hook整个LightFFN模块的输出:
        try:
            target_conv = self.model.transformer.encoder.layers[encoder_layer_id].ffn
        except Exception as e:
            raise RuntimeError("找不到 encoder.layers[...].ffn，请确认模型结构。") from e

        # 修改_forward_hook函数:
        def _forward_hook(module, inp, out):
            # out现在是LightFFN的最终输出 [B, L, C]
            # 需要转换为 [B, C, L] 格式以适配现有代码
            if isinstance(out, torch.Tensor) and len(out.shape) == 3:
                out_transposed = out.transpose(1, 2)  # [B, C, L]
                self._cam_activations = out_transposed
            else:
                self._cam_activations = out

        def _backward_hook(module, grad_in, grad_out):
            # grad_out[0] 对应 conv 输出的梯度
            self._cam_grads = grad_out[0]

        # encoder forward hook: 抓 spatial_shapes（传入参数里）
        def _encoder_forward_hook(module, inp, out):
            # encoder.forward signature: (src, spatial_shapes, level_start_index, valid_ratios, pos, padding_mask)
            # 当注册 forward_hook 时，inp 是传入的参数 tuple -> inp[1] 应该是 spatial_shapes
            try:
                spatial_shapes = inp[1]  # tensor (n_levels,2) or similar
                # convert to python list of tuples
                if isinstance(spatial_shapes, torch.Tensor):
                    sp = spatial_shapes.detach().cpu().tolist()
                else:
                    # already list/tuple
                    sp = list(spatial_shapes)
                self._spatial_shapes = [(int(x[0]), int(x[1])) for x in sp]
            except Exception:
                # fallback: don't crash; leave None
                self._spatial_shapes = None

        # 注册 hooks
        target_conv.register_forward_hook(_forward_hook)
        target_conv.register_full_backward_hook(_backward_hook)  # register_full_backward_hook 更可靠
        # 在 transformer.encoder 上注册 forward hook 捕获 spatial_shapes
        self.model.transformer.encoder.register_forward_hook(_encoder_forward_hook)

        print(">> Grad-CAM hooks registered on:", target_conv)


        # mkidr save_dir
        vid_name, prefix = args.input_video.split('/')[-1].split('.')
        self.save_root = os.path.join(args.output_dir, 'results', vid_name)
        Path(self.save_root).mkdir(parents=True, exist_ok=True)
        self.save_img_root = os.path.join(self.save_root, 'imgs')
        Path(self.save_img_root).mkdir(parents=True, exist_ok=True)
        self.txt_root = os.path.join(self.save_root, f'{vid_name}.txt')
        self.vid_root = os.path.join(self.save_root, args.input_video.split('/')[-1])

        # build dataloader and tracker
        self.dataloader = LoadVideo(args.input_video)
        self.tr_tracker = MOTR()

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        return dt_instances[keep]

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]

    @staticmethod
    def write_results(txt_path, frame_id, bbox_xyxy, identities):
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
        with open(txt_path, 'a') as f:
            for xyxy, track_id in zip(bbox_xyxy, identities):
                if track_id < 0 or track_id is None:
                    continue
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                line = save_format.format(frame=int(frame_id), id=int(track_id), x1=x1, y1=y1, w=w, h=h)
                f.write(line)

    @staticmethod
    def visualize_img_with_bbox(img_path, img, dt_instances: Instances, ref_pts=None, gt_boxes=None):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if dt_instances.has('scores'):
            img_show = draw_bboxes(img, np.concatenate([dt_instances.boxes, dt_instances.scores.reshape(-1, 1)], axis=-1), dt_instances.obj_idxes)
        else:
            img_show = draw_bboxes(img, dt_instances.boxes, dt_instances.obj_idxes)
        if ref_pts is not None:
            img_show = draw_points(img_show, ref_pts)
        if gt_boxes is not None:
            img_show = draw_bboxes(img_show, gt_boxes, identities=np.ones((len(gt_boxes), )) * -1)
        cv2.imwrite(img_path, img_show)
        return img_show

    from util.misc import NestedTensor

    def _calculate_model_stats(self):

        print("\n" + "=" * 60)
        print("Calculating Model Statistics...")
        print("=" * 60)

        # Params
        total_params = sum(p.numel() for p in self.model.parameters())
        self.model_params = total_params / 1e6

        print(f"\nModel Parameters:")
        print(f"Total Params: {total_params:,} ({self.model_params:.2f} M)")

        try:

            print("\nCalculating FULL Model GFLOPs...")

            wrapper = MOTRWrapper(self.model).cuda()

            dummy_input = torch.randn(1, 3, 640, 640).cuda()

            macs, params = profile(wrapper, inputs=(dummy_input,), verbose=False)

            flops = macs * 2
            gflops = flops / 1e9

            self.model_gflops = gflops

            print(f"Full Model GFLOPs: {gflops:.2f}")

        except Exception as e:

            print("GFLOPs calculation failed:", e)
            self.model_gflops = None

        print("=" * 60)
    # 在 Detector 类中添加以下方法：

    def visualize_feature_activation(self, original_img):
        """
        可视化 target_conv 的特征激活
        """
        if self._cam_activations is None or self._spatial_shapes is None:
            return None

        # 获取激活特征图 [B, C, L]
        activations = self._cam_activations.detach().cpu().numpy()[0]  # [C, L]

        # 对所有通道取平均得到激活强度
        activation_map = np.mean(activations, axis=0)  # [L]

        # 归一化
        activation_map = (activation_map - np.min(activation_map)) / (
                np.max(activation_map) - np.min(activation_map) + 1e-8)

        # 计算总的空间点数
        total_points = sum(h * w for h, w in self._spatial_shapes)

        # 截取或填充到正确长度
        if len(activation_map) > total_points:
            activation_map = activation_map[:total_points]
        elif len(activation_map) < total_points:
            padded = np.zeros(total_points)
            padded[:len(activation_map)] = activation_map
            activation_map = padded

        # 分配到不同的特征层
        heatmaps = []
        start_idx = 0
        for h, w in self._spatial_shapes:
            end_idx = start_idx + h * w
            level_activation = activation_map[start_idx:end_idx]

            # 重塑为二维热力图
            heatmap = level_activation.reshape(h, w)

            # 上采样到原始图像尺寸
            heatmap_upsampled = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]),
                                           interpolation=cv2.INTER_LINEAR)
            heatmaps.append(heatmap_upsampled)
            start_idx = end_idx

        # 融合多尺度热力图
        if len(heatmaps) > 0:
            final_heatmap = np.mean(heatmaps, axis=0)
        else:
            final_heatmap = np.zeros((original_img.shape[0], original_img.shape[1]))

        final_heatmap = (final_heatmap - np.min(final_heatmap)) / (
                np.max(final_heatmap) - np.min(final_heatmap) + 1e-8)

        # 提升对比度，使高值更高，低值更低
        # final_heatmap = np.power(final_heatmap, 3)  # 可调整指数参数控制收缩程度

        # 应用高斯滤波平滑边缘
        final_heatmap = cv2.GaussianBlur(final_heatmap, (21, 21), 5)  # 调整核大小(15,15)控制平滑程度

        # 归一化到0-255并应用颜色映射
        final_heatmap = (final_heatmap - np.min(final_heatmap)) / (
                np.max(final_heatmap) - np.min(final_heatmap) + 1e-8) * 255
        final_heatmap = final_heatmap.astype(np.uint8)

        # 应用热力图颜色
        heatmap_colored = cv2.applyColorMap(final_heatmap, cv2.COLORMAP_JET)

        # 叠加到原始图像
        overlay = cv2.addWeighted(cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR), 0.4, heatmap_colored, 0.6, 0)

        return overlay

    # 修改 run 方法中的可视化部分
    def run(self, prob_threshold=0.7, area_threshold=100, vis=True, dump=True):
        import time

        total_infer_time = 0
        total_frames = 0
        # save as video
        fps = self.dataloader.frame_rate
        videowriter = cv2.VideoWriter(self.vid_root, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                      (self.dataloader.seq_w, self.dataloader.seq_h))

        # 创建特征图保存目录
        feature_map_dir = os.path.join(self.save_root, 'feature_maps')
        if vis and self._cam_enabled:
            Path(feature_map_dir).mkdir(parents=True, exist_ok=True)

        track_instances = None
        fid = 0
        for _, cur_img, ori_img in tqdm(self.dataloader):
            if track_instances is not None:
                track_instances.remove('boxes')
                track_instances.remove('labels')

            # res = self.model.inference_single_image(cur_img.cuda().float(),
            #                                         (self.dataloader.seq_h, self.dataloader.seq_w), track_instances)
            torch.cuda.synchronize()
            start_time = time.time()

            res = self.model.inference_single_image(
                cur_img.cuda().float(),
                (self.dataloader.seq_h, self.dataloader.seq_w),
                track_instances)

            torch.cuda.synchronize()
            end_time = time.time()

            infer_time = end_time - start_time

            total_infer_time += infer_time
            total_frames += 1

            track_instances = res['track_instances']

            dt_instances = track_instances.to(torch.device('cpu'))

            # filter det instances by score.
            dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
            dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)

            if vis:
                vis_img_path = os.path.join(self.save_img_root, '{:06d}.jpg'.format(fid))
                vis_img = self.visualize_img_with_bbox(vis_img_path, ori_img, dt_instances)

                # 如果启用了CAM可视化，则生成特征激活热力图
                if self._cam_enabled:
                    feature_overlay = self.visualize_feature_activation(ori_img)
                    if feature_overlay is not None:
                        # 保存特征图
                        feature_map_path = os.path.join(feature_map_dir, '{:06d}_feature.jpg'.format(fid))
                        cv2.imwrite(feature_map_path, feature_overlay)


                    else:
                        videowriter.write(vis_img)
                else:
                    videowriter.write(vis_img)

            if dump:
                tracker_outputs = self.tr_tracker.update(dt_instances)
                self.write_results(txt_path=self.txt_root,
                                   frame_id=(fid + 1),
                                   bbox_xyxy=tracker_outputs[:, :4],
                                   identities=tracker_outputs[:, 5])
            fid += 1
        videowriter.release()
        avg_time = total_infer_time / total_frames
        fps = 1.0 / avg_time

        print("\n" + "=" * 60)
        print("Model Performance Summary")
        print("=" * 60)

        print(f"Input Size: 1536 x 800")
        print(f"Params: {self.model_params:.2f} M")

        if self.model_gflops is not None:
            print(f"GFLOPs: {self.model_gflops:.2f}")

        print(f"Average inference time: {avg_time * 1000:.2f} ms")
        print(f"FPS: {fps:.2f}")

        print("=" * 60)

    # def run(self, prob_threshold=0.7, area_threshold=100, vis=True, dump=True):
    #     # save as video
    #     fps = self.dataloader.frame_rate
    #     videowriter = cv2.VideoWriter(self.vid_root, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (self.dataloader.seq_w, self.dataloader.seq_h))
    #     track_instances = None
    #     fid = 0
    #     for _, cur_img, ori_img in tqdm(self.dataloader):
    #         if track_instances is not None:
    #             track_instances.remove('boxes')
    #             track_instances.remove('labels')
    #
    #
    #         res = self.model.inference_single_image(cur_img.cuda().float(), (self.dataloader.seq_h, self.dataloader.seq_w), track_instances)
    #
    #         track_instances = res['track_instances']
    #
    #         dt_instances = track_instances.to(torch.device('cpu'))
    #
    #
    #         # filter det instances by score.
    #         dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
    #         dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)
    #
    #         if vis:
    #             vis_img_path = os.path.join(self.save_img_root, '{:06d}.jpg'.format(fid))
    #             vis_img = self.visualize_img_with_bbox(vis_img_path, ori_img, dt_instances)
    #             videowriter.write(vis_img)
    #
    #         if dump:
    #             tracker_outputs = self.tr_tracker.update(dt_instances)
    #             self.write_results(txt_path=self.txt_root,
    #                             frame_id=(fid+1),
    #                             bbox_xyxy=tracker_outputs[:, :4],
    #                             identities=tracker_outputs[:, 5])
    #         fid += 1
    #     videowriter.release()

if __name__ == '__main__':

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    detector = Detector(args)
    detector.run()