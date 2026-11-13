"""本节结构差异集中在这里；旧 gpt2_model.py 保留原样。

A: GPT-2（LayerNorm + 线性 bias）。B: RMSNorm + 无线性 bias。
B 是两项改动的组合实验，不是完整 Llama/Qwen，也不是单因素消融。
来源：RMSNorm https://arxiv.org/abs/1910.07467；课程 GPT→Llama 配套材料。
"""
import hashlib

import torch
from torch import nn
from gpt2_model import GPT, GPTConfig


class RMSNorm(nn.Module):
    """输入/输出 [B,T,D]；只沿最后一维归一化，gamma[D] 从全 1 开始。"""

    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        # 【本项目】写出 RMSNorm 公式；【PyTorch】计算及自动求导。
        # 本实验固定 FP32。不减均值、没有 beta；eps 放在根号里面。
        inverse_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return x * inverse_rms * self.weight


def replace_norm_and_bias(module):
    """在模型构造后、优化器创建前改结构；保留全部共同权重初值。"""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            replacement = RMSNorm(child.normalized_shape[0], child.eps)
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
            setattr(module, name, replacement)
        elif isinstance(child, nn.Linear):
            # 把 bias 参数移除，不是把仍可训练的 bias 暂时设为零。
            # Linear.forward 自动使用 bias=None；权重矩阵不变。
            child.register_parameter('bias', None)
        else:
            replace_norm_and_bias(child)


def common_parameter_digest(model):
    """对共同的非 bias 参数初值取指纹；证明两入口共享相同起点。"""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name.endswith('.bias'):
            continue
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_model(spec, variant):
    """先用同一个 seed 构造 GPT；B 再替换模块，避免随机数消耗顺序不同。"""
    if variant not in ('gpt2', 'rmsnorm_no_bias'):
        raise ValueError('未知模型版本。')
    torch.manual_seed(spec['seed'])
    model = GPT(GPTConfig(**spec['model']), tie_embeddings=True, gpt2_init=True).float()
    if variant == 'rmsnorm_no_bias':
        replace_norm_and_bias(model)
    assert model.lm_head.weight is model.transformer.wte.weight
    return model
