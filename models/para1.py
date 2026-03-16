# parameters.py
import sys
import os
import torch
import torch.nn as nn
from collections import OrderedDict

# 添加路径解决导入问题
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from deformable_transformer_en import DeformableTransformerEncoderLayer
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保deformable_transformer_en.py和相关依赖文件在同一目录下")
    sys.exit(1)


class MOTREncoderAnalyzer:
    """MOTR Encoder Layer 参数分析工具（更新版）"""

    def __init__(self, encoder_layer):
        self.encoder_layer = encoder_layer

        # 根据新的架构更新组件映射
        self.component_mapping = {
            'Multi-Scale Deformable Attention': ['self_attn'],
            'Layer Normalization': ['norm1'],
            'Light Feed Forward Network': ['ffn'],
            'Frequency Enhancer': ['freq_enhancer'] if hasattr(encoder_layer, 'freq_enhancer') else [],
            'Dropout Layers': ['dropout1']
        }

    def analyze_parameters(self, verbose=True):
        """分析并输出参数信息"""
        if verbose:
            print("MOTR DeformableTransformerEncoderLayer 参数分析 (更新版):")
            print("=" * 70)

        results = OrderedDict()
        total_params = 0

        for comp_name, module_names in self.component_mapping.items():
            comp_params = 0
            comp_details = {}

            # 跳过空的组件列表
            if not module_names:
                continue

            if verbose:
                print(f"\n【{comp_name}】:")

            for module_name in module_names:
                if hasattr(self.encoder_layer, module_name):
                    module = getattr(self.encoder_layer, module_name)

                    if hasattr(module, 'parameters'):
                        params = sum(p.numel() for p in module.parameters())
                        comp_params += params
                        comp_details[module_name] = params

                        if verbose and params > 0:
                            print(f"  {module_name}: {params:,} 参数")

                            # 为特殊模块显示更多细节
                            if module_name == 'ffn' and verbose:
                                self._analyze_ffn_details(module)
                            elif module_name == 'freq_enhancer' and verbose:
                                self._analyze_freq_enhancer_details(module)

                    else:
                        comp_details[module_name] = 0
                        if verbose:
                            print(f"  {module_name}: 0 参数 (无可学习参数)")
                else:
                    comp_details[module_name] = 0
                    if verbose:
                        print(f"  {module_name}: 模块不存在")

            results[comp_name] = {
                'total': comp_params,
                'details': comp_details
            }

            if verbose:
                print(f"  小计: {comp_params:,} 参数")

            total_params += comp_params

        # 检查是否启用了频率增强器
        freq_status = "启用" if self.encoder_layer.use_freq_enhance else "未启用"

        results['总参数'] = total_params
        results['频率增强器状态'] = freq_status

        if verbose:
            print(f"\n频率增强器: {freq_status}")
            print(f"总参数数量: {total_params:,}")
            print("=" * 70)

        return results

    def _analyze_ffn_details(self, ffn_module):
        """分析LightFFN的详细结构"""
        print("    LightFFN详细结构:")
        for name, param in ffn_module.named_parameters():
            print(f"      {name}: {param.shape} -> {param.numel():,} 参数")

    def _analyze_freq_enhancer_details(self, freq_module):
        """分析FrequencyEnhancer的详细结构"""
        print("    FrequencyEnhancer详细结构:")
        for name, param in freq_module.named_parameters():
            print(f"      {name}: {param.shape} -> {param.numel():,} 参数")

    def get_architectural_info(self):
        """获取架构信息"""
        print("\n架构配置信息:")
        print("=" * 50)

        # 基本配置
        if hasattr(self.encoder_layer.self_attn, 'd_model'):
            print(f"模型维度 (d_model): {self.encoder_layer.self_attn.d_model}")

        if hasattr(self.encoder_layer.self_attn, 'n_heads'):
            print(f"注意力头数 (n_heads): {self.encoder_layer.self_attn.n_heads}")

        if hasattr(self.encoder_layer.self_attn, 'n_levels'):
            print(f"特征层级数 (n_levels): {self.encoder_layer.self_attn.n_levels}")

        if hasattr(self.encoder_layer.self_attn, 'n_points'):
            print(f"采样点数 (n_points): {self.encoder_layer.self_attn.n_points}")

        print(f"使用频率增强器: {'是' if self.encoder_layer.use_freq_enhance else '否'}")

        if self.encoder_layer.use_freq_enhance and hasattr(self.encoder_layer, 'freq_enhancer'):
            if hasattr(self.encoder_layer.freq_enhancer, 'freq_ratio'):
                print(f"频率比例: {self.encoder_layer.freq_enhancer.freq_ratio}")

    def compare_with_without_freq_enhancer(self):
        """比较启用和未启用频率增强器的参数差异"""
        print("\n频率增强器参数对比:")
        print("=" * 40)

        if self.encoder_layer.use_freq_enhance:
            freq_params = sum(p.numel() for p in self.encoder_layer.freq_enhancer.parameters())
            total_params = sum(p.numel() for p in self.encoder_layer.parameters())
            other_params = total_params - freq_params

            print(f"频率增强器参数: {freq_params:,}")
            print(f"其他组件参数: {other_params:,}")
            print(f"频率增强器占比: {freq_params / total_params * 100:.2f}%")
        else:
            print("频率增强器未启用")

    def export_detailed_summary(self, filename=None):
        """导出详细参数摘要"""
        results = self.analyze_parameters(verbose=False)

        summary_lines = []
        summary_lines.append("MOTR DeformableTransformerEncoderLayer 详细参数报告")
        summary_lines.append("=" * 65)
        summary_lines.append("")

        # 架构信息
        summary_lines.append("架构配置:")
        summary_lines.append(f"  频率增强器: {results['频率增强器状态']}")
        summary_lines.append("")

        # 参数详情
        summary_lines.append("组件参数详情:")
        for comp_name, comp_info in results.items():
            if comp_name not in ['总参数', '频率增强器状态']:
                summary_lines.append(f"\n{comp_name}: {comp_info['total']:,} 参数")
                for module_name, params in comp_info['details'].items():
                    if params > 0:
                        summary_lines.append(f"  - {module_name}: {params:,}")

        summary_lines.append(f"\n总参数数量: {results['总参数']:,}")

        summary = '\n'.join(summary_lines)

        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"\n详细报告已保存到: {filename}")

        return summary


def analyze_motr_encoder(encoder_layer, detailed=False):
    """便捷函数：分析MOTR encoder layer"""
    analyzer = MOTREncoderAnalyzer(encoder_layer)

    # 基本参数分析
    results = analyzer.analyze_parameters()

    if detailed:
        # 架构信息
        analyzer.get_architectural_info()

        # 频率增强器对比
        analyzer.compare_with_without_freq_enhancer()

    return results


if __name__ == "__main__":
    try:
        print("测试不同配置的encoder layer...")
        print()

        # 测试1: 不使用频率增强器
        print("配置1: 标准配置（无频率增强器）")
        encoder1 = DeformableTransformerEncoderLayer(
            d_model=256,
            d_ffn=1024,
            dropout=0.1,
            activation="relu",
            n_levels=4,
            n_heads=8,
            n_points=4,
            sigmoid_attn=False,
            use_freq_enhance=False
        )
        results1 = analyze_motr_encoder(encoder1, detailed=True)

        print("\n" + "=" * 80 + "\n")

        # 测试2: 使用频率增强器
        print("配置2: 增强配置（含频率增强器）")
        encoder2 = DeformableTransformerEncoderLayer(
            d_model=256,
            d_ffn=1024,
            dropout=0.1,
            activation="relu",
            n_levels=4,
            n_heads=8,
            n_points=4,
            sigmoid_attn=False,
            use_freq_enhance=True,
            freq_ratio=0.3
        )
        results2 = analyze_motr_encoder(encoder2, detailed=True)

        # 导出报告
        analyzer = MOTREncoderAnalyzer(encoder2)
        analyzer.export_detailed_summary("motr_encoder_report.txt")

    except Exception as e:
        print(f"运行错误: {e}")
        print("请检查所有依赖文件是否存在")