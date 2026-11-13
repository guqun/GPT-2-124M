"""第22块：在 GQA 投影/拆头后、RoPE 前新增 QK-Norm；旧文件不修改。

限定本课 FP32；头内 RMSNorm 的顺序对照 Transformers v4.51.3 Qwen3Attention。
保留本地相邻配对 RoPE、共享 lm_head 等设置，不宣称完整 Qwen3/检查点兼容。
"""
import math
import torch
from torch import nn
from gqa_model import GroupedAttention, convert_attention
from rope_model import apply_rotary
from gqa_model import grouped_attention
from swiglu_model import build_model as build_swiglu


class HeadRMSNorm(nn.Module):
    """[N1] x[B,H,T,d] → 同形状；gamma[d]，只沿最后一维统计。

同层同类头共用 gamma；每个 batch/头/位置的均方根分别计算。
Q 与 K 使用不同实例，各层也独立。gamma 初值全 1，无 beta。
"""
    def __init__(self, head_dim, eps):
        super().__init__()
        if head_dim < 1 or not math.isfinite(eps) or eps <= 0:
            raise ValueError('头维度和 epsilon 必须为正。')
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        # PyTorch 提供平方、mean、rsqrt 和广播乘法，并为 gamma/输入求导。
        values = x.float()
        mean_square = values.square().mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(mean_square + self.eps)
        return normalized.to(dtype) * self.weight


class QKNormAttention(GroupedAttention):
    """复用已建好的 GQA 投影/输出参数；enabled=False 用 Identity 作为参照。

继承前课 forward：完整序列训练调用 self.step(x)[0]；缓存由 GroupedDecoder 管理。
本文件单独写 step 是为了在正确位置插入归一化，不改旧 GQA 的教学实现。
"""
    def __init__(self, original, eps, enabled=True):
        nn.Module.__init__(self)
        self.n_head, self.n_kv_heads = original.n_head, original.n_kv_heads
        self.head_dim = original.head_dim
        self.enabled = enabled
        for name in ('q_proj','k_proj','v_proj','c_proj','rope','attn_dropout','resid_dropout'):
            setattr(self, name, getattr(original, name))
        self.q_norm = HeadRMSNorm(self.head_dim, eps) if enabled else nn.Identity()
        self.k_norm = HeadRMSNorm(self.head_dim, eps) if enabled else nn.Identity()
        self.q_norm.to(device=self.q_proj.weight.device, dtype=self.q_proj.weight.dtype)
        self.k_norm.to(device=self.k_proj.weight.device, dtype=self.k_proj.weight.dtype)

    def step(self, x, past=None):
        batch, length, width = x.shape
        start = 0 if past is None else past[0].shape[2]
        if start + length > self.rope.max_length:
            raise ValueError('超过固定上下文容量。')
        # [N2] 输入为当前 Block 的入口 RMSNorm 输出；投影会再次改变尺度。
        q = self.q_proj(x).view(batch,length,self.n_head,self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(batch,length,self.n_kv_heads,self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(batch,length,self.n_kv_heads,self.head_dim).transpose(1,2)
        # [N3] 唯一新增的前向处理；不能对 V 做，也不能把 4 个 Q 头一起归一化。
        q, k = self.q_norm(q), self.k_norm(k)
        # [N4] 新 q/k 先归一化、再带位置偏移旋转；旧 K 不重复走这两步。
        cos, sin = self.rope.cos[start:start+length], self.rope.sin[start:start+length]
        q, k = apply_rotary(q,cos,sin), apply_rotary(k,cos,sin)
        if past is not None:
            k = torch.cat((past[0],k),dim=2)
            v = torch.cat((past[1],v),dim=2)
        # [N5] 沿用上一课的 GQA；缓存依然只有 Hkv 组。
        heads = grouped_attention(q,k,v,start,self.attn_dropout)
        merged = heads.transpose(1,2).contiguous().view(batch,length,width)
        return self.resid_dropout(self.c_proj(merged)), (k,v)


def add_qk_norm(model, enabled=True):
    """在建优化器之前添加新参数。epsilon 沿用本地配置 1e-5，不冒充官方默认。"""
    for block in model.transformer.h:
        block.attn = QKNormAttention(block.attn,model.config.layer_norm_epsilon,enabled)
    return model


def build_model(spec, enabled):
    # [N0] 两份入口的所有既有参数相同；只在 enabled=True 分支新增 gamma。
    model = convert_attention(build_swiglu(spec,'swiglu'),2)
    return add_qk_norm(model,enabled)
