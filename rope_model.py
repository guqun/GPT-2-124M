"""本课只替换位置方案：RMSNorm/no-bias + 绝对位置 → 相同结构 + RoPE。

邻接配对 (x0,x1)、(x2,x3)…；公式参考 RoFormer 及 Meta Llama 实现。
本机实验 theta=10000（教学选择），不是 Llama 3 默认的 500000。
保留 MHA/GELU/共享输出表；不含 KV cache、YaRN 或长度外推实验。
"""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F
from architecture_variants import build_model as build_absolute_model

ROPE_THETA = 10000.0


def apply_rotary(x, cos, sin):
    """x[B,H,T,d]、cos/sin[T,d/2] → 同形状；旋转头内相邻两维。"""
    # 【本项目】实现二维旋转；【PyTorch】广播乘加与自动求导。
    even, odd = x[..., 0::2], x[..., 1::2]
    # [T,d/2] 在 batch/head 轴上广播；同位置、同维对使用同一个角度。
    rotated_even = even*cos - odd*sin
    rotated_odd = even*sin + odd*cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """固定三角函数缓存不是可训练参数；所有头使用相同频率安排。"""

    def __init__(self, head_dim, max_length, theta=ROPE_THETA):
        super().__init__()
        if head_dim % 2 or head_dim < 2:
            raise ValueError('RoPE 每头维度必须是正偶数。')
        self.max_length = max_length
        # r=0…d/2−1；inv_freq[r]=theta^(−2r/d)，单位为弧度/位置。
        inv_freq = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32)/head_dim)
        angles = torch.arange(max_length, dtype=torch.float32)[:, None] * inv_freq[None, :]
        # buffer 随模型移动设备；不参与 AdamW，也不保存到权重，重建时由配置计算。
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    def forward(self, q, k):
        length = q.shape[-2]
        if length > self.max_length:
            raise ValueError('超过本课固定上下文容量；本节不做长度外推。')
        cos, sin = self.cos[:length], self.sin[:length]
        # 本课每条独立序列的位置都是 0…T−1。生成缓存的偏移量留到 KV cache 课。
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)


class RotaryAttention(nn.Module):
    """复用旧注意力的权重；唯一新增前向运算是拆头后的 Q/K 旋转。"""

    def __init__(self, original, max_length, theta):
        super().__init__()
        self.n_head, self.head_dim = original.n_head, original.head_dim
        self.c_attn, self.c_proj = original.c_attn, original.c_proj
        self.attn_dropout, self.resid_dropout = original.attn_dropout, original.resid_dropout
        self.register_buffer('causal_mask', original.causal_mask, persistent=False)
        self.rope = RotaryEmbedding(self.head_dim, max_length, theta)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.c_attn(x).split(width, dim=-1)
        q = q.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        # [R1] 每层都旋转：q/k 是当前输入投影后的激活值，不是 Wq/Wk 参数。
        q, k = self.rope(q, k)  # V 不旋转；Q/K 形状仍为 [B,H,T,d]。
        scores = (q @ k.transpose(-2, -1))/math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:, :, :length, :length], float('-inf'))
        weights = F.softmax(scores, dim=-1)
        heads = self.attn_dropout(weights) @ v
        merged = heads.transpose(1, 2).contiguous().view(batch, length, width)
        return self.resid_dropout(self.c_proj(merged))


class RotaryGPT(nn.Module):
    """只改变输入位置处理和每层注意力；旧 Block/MLP/残差直接复用。"""

    def __init__(self, base, theta):
        super().__init__()
        self.config = base.config
        self.transformer = base.transformer
        self.lm_head = base.lm_head
        # [R2] 真正移除学习式位置表，不是留下一组永远不用的参数。
        del self.transformer['wpe']
        for block in self.transformer.h:
            block.attn = RotaryAttention(block.attn, self.config.n_positions, theta)

    def forward(self, input_ids):
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError('输入须为 int64 的 [B,T] Token ID。')
        batch, length = input_ids.shape
        if batch < 1 or not 1 <= length <= self.config.n_positions:
            raise ValueError('输入长度超过本课固定上下文容量。')
        # [R3] 输入仅查 Token Embedding，不再加位置向量；位置进入每层 Q/K。
        x = self.transformer.drop(self.transformer.wte(input_ids))
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x))


def build_model(spec, variant):
    if variant not in ('absolute', 'rope'):
        raise ValueError('位置实验只支持 absolute / rope。')
    base = build_absolute_model(spec, 'rmsnorm_no_bias')
    return base if variant == 'absolute' else RotaryGPT(base, spec['position_experiment']['theta'])


def common_parameter_digest(model):
    """本轮共同参数不包含 A 独有的 wpe；其余初值必须逐项相同。"""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name == 'transformer.wpe.weight':
            continue
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
