import torch
import torch.nn as nn
import torch.nn.functional as F


class LightFFN(nn.Module):
    """您的核心创新：频域-卷积协同增强的轻量级FFN"""

    def __init__(self, d_model, kernel_size=3, dropout=0.1,
                 use_freq_enhance=True, freq_ratio=0.4):
        super().__init__()

        self.d_model = d_model
        self.use_freq_enhance = use_freq_enhance
        self.freq_ratio = freq_ratio
        self.freq_dim = int(d_model * freq_ratio)


        self.conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size,
                               padding=kernel_size // 2, groups=d_model // 8)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(d_model * 2, d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        if use_freq_enhance:

            # 卷积引导的频域滤波 (Conv-Guided Frequency Filtering)
            self.conv_guided_freq_filter = ConvGuidedFreqFilter(self.freq_dim, kernel_size)


            # 双路径频域-空域交叉增强 (Dual-Path Freq-Spatial Cross Enhancement)
            #self.dual_path_enhancer = DualPathEnhancer(d_model, self.freq_dim)

            # 自适应频域门控 (Adaptive Frequency Gating)
            self.adaptive_freq_gate = AdaptiveFreqGate(d_model)


    def frequency_enhance(self, x):

        B, L, C = x.shape

        # 分离频域和空域部分
        freq_part = x[:, :, :self.freq_dim]
        spatial_part = x[:, :, self.freq_dim:]

        # 1. 卷积引导的频域滤波
        enhanced_freq = self.conv_guided_freq_filter(freq_part)

        # 2. 重组特征
        enhanced_feat = torch.cat([enhanced_freq, spatial_part], dim=-1)

        # 3. 自适应门控融合
        final_feat = self.adaptive_freq_gate(enhanced_feat, x)

        return final_feat

    def forward(self, x):

        residual = x


        if self.use_freq_enhance:
            x = self.frequency_enhance(x)

        # 转换为conv1d格式
        x = x.transpose(1, 2)

        # LightFFN处理
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)


        x = x.transpose(1, 2)

        # 残差连接
        x = residual + self.dropout(x)
        x = self.norm(x)

        return x


class ConvGuidedFreqFilter(nn.Module):
    """创新1: 卷积引导的频域滤波

    核心思想: 让卷积特征指导频域滤波，实现空域-频域的深度协同
    """

    def __init__(self, freq_dim, kernel_size):
        super().__init__()
        self.freq_dim = freq_dim

        # 卷积引导网络 - 从空域特征学习频域滤波策略
        self.guide_conv = nn.Sequential(
            nn.Conv1d(freq_dim, freq_dim // 2, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(freq_dim // 2, freq_dim, 1),
            nn.Sigmoid()  # 生成0-1的滤波权重
        )



    def forward(self, x):
        spatial_guide = self.guide_conv(x.transpose(1, 2)).transpose(1, 2)
        x_freq = torch.fft.fft(x, dim=1)

        # 直接用spatial_guide作为频域幅度权重

        magnitude = torch.abs(x_freq)
        phase = torch.angle(x_freq)

        # 用空域权重调制频域幅度
        filtered_magnitude = magnitude * spatial_guide  # 实数权重调制
        filtered_freq = filtered_magnitude * torch.exp(1j * phase)

        enhanced_x = torch.fft.ifft(filtered_freq, dim=1).real
        return enhanced_x




class AdaptiveFreqGate(nn.Module):
    """创新4: 自适应频域门控 - 您的轻量级创新

    核心思想: 根据输入内容自适应决定频域和空域的贡献比例
    """

    def __init__(self, d_model):
        super().__init__()

        # 内容感知门控生成器
        self.content_analyzer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model // 4, 1),
            nn.GELU(),
            nn.Conv1d(d_model // 4, 3, 1),  # 输出3个门控值
            nn.Softmax(dim=1)
        )

        # 特征融合权重
        self.fusion_weights = nn.Parameter(torch.tensor([0.4, 0.4, 0.2]))  # freq, spatial, cross

    def forward(self, enhanced_feat, original_feat):
        """
        简化的自适应门控融合
        enhanced_feat: 增强后的特征 [B, L, C]
        original_feat: 原始特征 [B, L, C]
        """
        B, L, C = enhanced_feat.shape

        # 全局特征分析
        gate_weights = self.content_analyzer(enhanced_feat.transpose(1, 2))  # [B, 3, 1]
        gate_weights = gate_weights.squeeze(-1).unsqueeze(1)  # [B, 1, 3]

        # 只用前两个权重 (enhanced vs original)
        enhanced_weight = gate_weights[:, :, 0:1]
        original_weight = gate_weights[:, :, 1:2]

        # 归一化确保和为1
        total_weight = enhanced_weight + original_weight + 1e-8
        enhanced_weight = enhanced_weight / total_weight
        original_weight = original_weight / total_weight

        # 加权融合
        fused_feat = enhanced_weight * enhanced_feat + original_weight * original_feat

        return fused_feat


# 测试函数
def test_light_ffn():
    """测试LightFFN"""
    print("=== 测试修复后的LightFFN ===")

    # 测试不同配置
    test_configs = [
        (2, 128, 256),  # (batch, seq_len, d_model)
        (4, 512, 512),
        (1, 1024, 768),
    ]

    for batch_size, seq_len, d_model in test_configs:
        print(f"\n测试配置: B={batch_size}, L={seq_len}, C={d_model}")

        try:
            # 创建模型
            model = LightFFN(
                d_model=d_model,
                kernel_size=3,
                dropout=0.1,
                use_freq_enhance=True,
                freq_ratio=0.4
            )

            # 测试输入
            x = torch.randn(batch_size, seq_len, d_model)

            # 前向传播
            with torch.no_grad():
                output = model(x)

            print(f"✅ 前向传播成功: {x.shape} -> {output.shape}")

            # 测试梯度
            x.requires_grad_(True)
            output = model(x)
            loss = output.sum()
            loss.backward()

            print(f"✅ 反向传播成功: 梯度范数 = {torch.norm(x.grad):.6f}")

            # 参数统计
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"📊 参数量: {total_params:,}")

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_light_ffn()
