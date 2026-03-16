import torch
import torch.nn as nn
from collections import OrderedDict

from deformable_transformer_plus import DeformableTransformerEncoderLayer
class MOTREncoderAnalyzer:
    """MOTR Encoder Layer 参数分析工具"""

    def __init__(self, encoder_layer):
        self.encoder_layer = encoder_layer
        self.component_mapping = {
            'Multi-Scale Deformable Attention': ['self_attn'],
            'Layer Normalization': ['norm1', 'norm2'],
            'Feed Forward Network': ['linear1', 'linear2'],
            'Dropout Layers': ['dropout1', 'dropout2', 'dropout3']
        }

    def analyze_parameters(self, verbose=True):
        """分析并输出参数信息"""
        if verbose:
            print("MOTR DeformableTransformerEncoderLayer 参数分析:")
            print("=" * 65)

        results = OrderedDict()
        total_params = 0

        for comp_name, module_names in self.component_mapping.items():
            comp_params = 0
            comp_details = {}

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
                    else:
                        comp_details[module_name] = 0
                        if verbose:
                            print(f"  {module_name}: 0 参数 (无可学习参数)")

            results[comp_name] = {
                'total': comp_params,
                'details': comp_details
            }

            if verbose:
                print(f"  小计: {comp_params:,} 参数")

            total_params += comp_params

        results['总参数'] = total_params

        if verbose:
            print(f"\n总参数数量: {total_params:,}")
            print("=" * 65)

        return results

    def get_parameter_shapes(self):
        """获取所有参数的详细形状信息"""
        print("\n参数详细形状信息:")
        print("=" * 70)

        for name, param in self.encoder_layer.named_parameters():
            # 判断参数所属组件
            component = self._get_component_name(name)

            print(f"参数: {name}")
            print(f"  所属组件: {component}")
            print(f"  形状: {param.shape}")
            print(f"  参数数量: {param.numel():,}")
            print(f"  数据类型: {param.dtype}")
            print(f"  内存大小: {param.numel() * param.element_size() / 1024 / 1024:.3f} MB")
            print("-" * 50)

    def _get_component_name(self, param_name):
        """根据参数名确定所属组件"""
        if "self_attn" in param_name:
            return "Multi-Scale Deformable Attention"
        elif "norm" in param_name:
            return "Layer Normalization"
        elif "linear" in param_name:
            return "Feed Forward Network"
        else:
            return "Other"

    def compare_configurations(self, other_encoder):
        """比较两个encoder配置的参数差异"""
        print("\n配置对比:")
        print("=" * 50)

        results1 = self.analyze_parameters(verbose=False)
        analyzer2 = MOTREncoderAnalyzer(other_encoder)
        results2 = analyzer2.analyze_parameters(verbose=False)

        for comp_name in self.component_mapping.keys():
            params1 = results1[comp_name]['total']
            params2 = results2[comp_name]['total']
            diff = params2 - params1

            print(f"{comp_name}:")
            print(f"  配置1: {params1:,} 参数")
            print(f"  配置2: {params2:,} 参数")
            print(f"  差异: {diff:+,} 参数")
            print()

    def export_summary(self, filename=None):
        """导出参数摘要"""
        results = self.analyze_parameters(verbose=False)

        summary_text = []
        summary_text.append("MOTR DeformableTransformerEncoderLayer 参数摘要")
        summary_text.append("=" * 60)

        for comp_name, comp_info in results.items():
            if comp_name != '总参数':
                summary_text.append(f"\n{comp_name}: {comp_info['total']:,} 参数")
                for module_name, params in comp_info['details'].items():
                    summary_text.append(f"  - {module_name}: {params:,}")

        summary_text.append(f"\n总参数数量: {results['总参数']:,}")

        summary = '\n'.join(summary_text)

        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"参数摘要已保存到: {filename}")

        return summary


# 使用示例
def analyze_motr_encoder(encoder_layer):
    """便捷函数：分析MOTR encoder layer"""
    analyzer = MOTREncoderAnalyzer(encoder_layer)

    # 基本参数分析
    results = analyzer.analyze_parameters()

    # 详细形状信息（可选）
    # analyzer.get_parameter_shapes()

    # 导出摘要（可选）
    # analyzer.export_summary("motr_encoder_summary.txt")

    return results


# 调用方式示例：
if __name__ == "__main__":
    encoder_layer = DeformableTransformerEncoderLayer(
        d_model=256,
        d_ffn=1024,

        dropout=0.1,
        activation="relu",

        n_levels=4,
        n_heads=8,
        n_points=4,
        sigmoid_attn=False)
    # 假设你已经导入了MOTR的encoder layer
    # from motr.models.deformable_transformer import DeformableTransformerEncoderLayer

    # 创建encoder实例
    # encoder_layer = DeformableTransformerEncoderLayer(
    #     d_model=256,
    #     d_ffn=1024,
    #     dropout=0.1,
    #     activation="relu",
    #     n_levels=4,
    #     n_heads=8,
    #     n_points=4,
    #     sigmoid_attn=False
    # )

    # 分析参数
    analyze_motr_encoder(encoder_layer)

    # 或者使用完整的分析器
    # analyzer = MOTREncoderAnalyzer(encoder_layer)
    # analyzer.analyze_parameters()
    # analyzer.get_parameter_shapes()

    pass