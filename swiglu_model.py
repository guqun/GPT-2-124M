"""只替换 RoPE 模型的 MLP：GELU 两矩阵 → SwiGLU 三矩阵。

本地教学改编：无 bias，按最接近 8D/3 的整数选择宽度；不做硬件倍数对齐。
这不是完整 Llama：仍保留 MHA、共享 lm_head 和前课的固定 RoPE 配置。
"""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F
from rope_model import build_model as build_rope_model


def experiment_settings(width):
    """两矩阵共 8D² 个参数；三矩阵共 3DF 个参数 → F≈8D/3。"""
    hidden = (8 * width + 1) // 3  # 最近整数；D=128 → 341，而不是 512。
    return {'gelu_hidden': 4 * width, 'swiglu_hidden': hidden,
            'width_rule': 'nearest_integer_to_8D_over_3',
            'gelu_parameters_per_block': 8 * width * width,
            'swiglu_parameters_per_block': 3 * width * hidden,
            'new_mlp_seed_offset': 1000}


class SwiGLU(nn.Module):
    """输入 z[B,T,D]；门控和内容各 [B,T,F]；输出恢复 [B,T,D]。

这里不含归一化和残差相加，它们仍在原 Block.forward 中。
PyTorch Linear 保存 [out_features,in_features]，实际计算 z @ weight.T。
"""
    def __init__(self, config, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.resid_pdrop)
        # 两路独立随机权重；保持前课 .02 正态初始化及输出残差缩放规则。
        nn.init.normal_(self.gate_proj.weight, std=config.initializer_range)
        nn.init.normal_(self.up_proj.weight, std=config.initializer_range)
        nn.init.normal_(self.down_proj.weight,
                        std=config.initializer_range / math.sqrt(2 * config.n_layer))

    def forward(self, z):
        # [S1] 同一个 z 交给两套不同权重；不是把输入向量切成两半。
        gate_logits = self.gate_proj(z)  # [B,T,D] → [B,T,F]
        content = self.up_proj(z)       # [B,T,D] → [B,T,F]；不是注意力 V。
        # [S2] SiLU(a)=a*sigmoid(a)，门值可为负，也不限制在 [0,1]。
        gate = F.silu(gate_logits)
        # [S3] * 是同位置、同维度相乘，不是矩阵乘法。
        gated = gate * content
        # [S4] 混合 F 个门控特征，回到 D 维；Block 随后加回主干 R。
        return self.dropout(self.down_proj(gated))


def build_model(spec, variant):
    if variant not in ('gelu', 'swiglu'):
        raise ValueError('本课只支持 gelu / swiglu。')
    base = build_rope_model(spec, 'rope')
    if variant == 'swiglu':
        settings = experiment_settings(base.config.n_embd)
        # 隔离新模块初始化的随机数；主干参数来自相同的 RoPE 随机起点。
        # MLP 矩阵数量/形状不同，不能声称这部分也具有相同初值。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec['seed'] + settings['new_mlp_seed_offset'])
            for block in base.transformer.h:
                block.mlp = SwiGLU(base.config, settings['swiglu_hidden'])
    return base


def common_parameter_digest(model):
    """核对 MLP 以外所有可训练参数的初值；不把新旧 MLP 伪称为相同参数。"""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if '.mlp.' in name:
            continue
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
