import torch
import torch.nn.functional as F
import numpy as np
import cv2

class MOTRGradCAM:

    def __init__(self, model):
        self.model = model
        self.model.eval()

        # 用来存 forward 的特征
        self.feat_maps = {}
        # 用来存 backward 的梯度
        self.grads = {}

        # 注册 hook
        self._register_hooks()

    def _save_feat(self, name):
        def hook(module, inp, out):
            self.feat_maps[name] = out
        return hook

    def _save_grad(self, name):
        def hook(module, grad_input, grad_output):
            self.grads[name] = grad_output[0]
        return hook

    def _register_hooks(self):

        # --------------------------
        # ① Backbone 输出特征
        # --------------------------
        self.model.backbone[-1].register_forward_hook(
            self._save_feat("backbone")
        )
        self.model.backbone[-1].register_full_backward_hook(
            self._save_grad("backbone")
        )

        # --------------------------
        # ② Encoder 输出
        # --------------------------
        self.model.transformer.encoder.layers[-1].register_forward_hook(
            self._save_feat("encoder")
        )
        self.model.transformer.encoder.layers[-1].register_full_backward_hook(
            self._save_grad("encoder")
        )

        # --------------------------
        # ③ Decoder 输出
        # --------------------------
        self.model.transformer.decoder.layers[-1].register_forward_hook(
            self._save_feat("decoder")
        )
        self.model.transformer.decoder.layers[-1].register_full_backward_hook(
            self._save_grad("decoder")
        )

    # -----------------------------------------------------
    # 生成热力图的核心
    # -----------------------------------------------------
    def generate_cam(self, feature, gradient):
        # GAP 梯度 → 权重
        weights = gradient.mean(dim=(2, 3), keepdim=True)

        # 加权求和
        cam = (weights * feature).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        cam -= cam.min()
        cam /= (cam.max() + 1e-6)
        return cam

    # -----------------------------------------------------
    # 输入图像，输出 backbone / encoder / decoder 热力图
    # -----------------------------------------------------
    def __call__(self, img_tensor, output_idx=0):

        # forward
        out = self.model.inference_single_image(
            img_tensor,
            (img_tensor.shape[2], img_tensor.shape[3]),
            track_instances=None
        )

        # 取某个 track 的得分求梯度
        score = out["track_instances"].scores[output_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Backbone heatmap
        cam_backbone = self.generate_cam(
            self.feat_maps["backbone"],
            self.grads["backbone"]
        )

        # Encoder heatmap
        cam_encoder = self.generate_cam(
            self.feat_maps["encoder"],
            self.grads["encoder"]
        )

        # Decoder heatmap
        cam_decoder = self.generate_cam(
            self.feat_maps["decoder"],
            self.grads["decoder"]
        )

        return cam_backbone, cam_encoder, cam_decoder
