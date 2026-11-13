"""第 21 块：显式分组的 GQA，与旧 MHA/KV Cache 文件分开。

PyTorch 提供 Linear/matmul/softmax/cat；本项目负责头分组、RoPE 偏移和缓存传递。
Hq=4,Hkv=2 时 Q0/Q1 读 KV0，Q2/Q3 读 KV1。缓存从不扩成 Hq 份。
只支持等长无 padding 序列、固定上下文；不是完整 Llama，也不是生产缓存管理器。
"""
import math
import torch
from torch import nn
from rope_model import apply_rotary


def grouped_attention(q, k, v, start, dropout=None):
    """[A4] Q[B,Hq,T,d] × K/V[B,Hkv,S+T,d] → H[B,Hq,T,d]。

把 Q 头轴拆成 [KV组数, 每组Q数]，K/V 插入长度为 1 的组内轴并广播。
这不是删除 Token，也没有再次投影 QKV；每个 Q 仍有独立的 A 和 H。
"""
    batch, hq, length, dim = q.shape
    hkv = k.shape[1]
    if hkv < 1 or hq % hkv:
        raise ValueError('Q 头数必须能被 KV 头数整除。')
    queries = q.reshape(batch, hkv, hq // hkv, length, dim)
    # [B,Hkv,G,T,d] @ [B,Hkv,1,d,S+T] → [B,Hkv,G,T,S+T]
    scores = (queries @ k.unsqueeze(2).transpose(-2, -1)) / math.sqrt(dim)
    query_pos = start + torch.arange(length, device=q.device)
    key_pos = torch.arange(k.shape[2], device=q.device)
    allowed = key_pos[None, :] <= query_pos[:, None]
    weights = torch.softmax(scores.masked_fill(~allowed, float('-inf')), dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    # 同组的两个 A 分别乘同一个 V；输出头数仍为 Hq。
    heads = weights @ v.unsqueeze(2)
    return heads.reshape(batch, hq, length, dim)


class GroupedAttention(nn.Module):
    """拆开原融合 QKV 投影，以看清 Q 不变、K/V 输出宽度变小的位置。

本实验从旧随机 MHA 起点复制 Q/输出投影，对组内 K/V 权重取均值。
均值只在构建模型时做一次，不是每次推理都平均历史缓存。
这种转换不保证保持原模型函数或语言质量；本课不执行后续适配训练。
"""
    def __init__(self, original, n_kv_heads):
        super().__init__()
        self.n_head = original.n_head
        self.head_dim = original.head_dim
        if not isinstance(n_kv_heads, int) or n_kv_heads < 1 or self.n_head % n_kv_heads:
            raise ValueError('KV 头数须为 Q 头数的正整数因子。')
        self.n_kv_heads = n_kv_heads
        width = self.n_head * self.head_dim
        source = original.c_attn.weight
        if original.c_attn.bias is not None:
            raise ValueError('本课基于前课 no-bias 模型。')
        self.q_proj = nn.Linear(width, width, bias=False, device=source.device, dtype=source.dtype)
        self.k_proj = nn.Linear(width, n_kv_heads*self.head_dim, bias=False, device=source.device, dtype=source.dtype)
        self.v_proj = nn.Linear(width, n_kv_heads*self.head_dim, bias=False, device=source.device, dtype=source.dtype)
        self.c_proj = original.c_proj
        self.rope = original.rope
        self.attn_dropout = original.attn_dropout
        self.resid_dropout = original.resid_dropout
        # [A0] Linear.weight 实际存 [out,in]，前向为 x @ weight.T。
        with torch.no_grad():
            qw, kw, vw = source.split(width, dim=0)
            self.q_proj.weight.copy_(qw)
            group_size = self.n_head // n_kv_heads
            for layer, weight in ((self.k_proj, kw), (self.v_proj, vw)):
                pooled = weight.reshape(n_kv_heads, group_size, self.head_dim, width).mean(dim=1)
                layer.weight.copy_(pooled.reshape(n_kv_heads*self.head_dim, width))

    def step(self, x, past=None):
        """本层新输入 x[B,T,D]，旧缓存各[B,Hkv,S,d] → 分支输出和新缓存。"""
        batch, length, width = x.shape
        start = 0 if past is None else past[0].shape[2]
        if start + length > self.rope.max_length:
            raise ValueError('超过固定上下文容量。')
        # [A1] 所有输出维都使用完整的输入向量；并不是先切 x 再投影。
        q = self.q_proj(x).view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # [A2] Q/K 头数可以不同；相同位置使用相同 RoPE 频率，V 不旋转。
        cos, sin = self.rope.cos[start:start+length], self.rope.sin[start:start+length]
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        # [A3] 只沿位置轴增长。旧 K 已经旋转，直接复用。
        if past is not None:
            k, v = torch.cat((past[0], k), dim=2), torch.cat((past[1], v), dim=2)
        heads = grouped_attention(q, k, v, start, self.attn_dropout)
        # [A5] 4 个输出头拼回 D；共享 KV 不会减少 H 的头数。
        merged = heads.transpose(1, 2).contiguous().view(batch, length, width)
        return self.resid_dropout(self.c_proj(merged)), (k, v)

    def forward(self, x):
        # 普通完整序列前向仍可自动求导；本节运行入口只做推理验证。
        return self.step(x)[0]


def convert_attention(model, n_kv_heads):
    """仅修改调用者新建的模型；不会改磁盘上的旧代码/检查点。"""
    for block in model.transformer.h:
        block.attn = GroupedAttention(block.attn, n_kv_heads)
    return model


class GroupedDecoder(nn.Module):
    """[D1] 每层独立接回缓存；接口兼容前课 generate()，可完整重算或增量推理。"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.inference_mode()
    def forward(self, input_ids, cache=None, *, use_cache=True, last_only=True):
        if any(m.training for m in self.modules()):
            raise ValueError('缓存实验必须先 eval()。')
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or min(input_ids.shape) < 1:
            raise ValueError('输入须为非空 int64 [B,T]。')
        if cache is not None and not use_cache:
            raise ValueError('完整重算不能传旧缓存。')
        model = self.model
        batch, length = input_ids.shape
        start = 0
        if cache is not None:
            if not isinstance(cache, (tuple, list)) or len(cache) != len(model.transformer.h):
                raise ValueError('缓存层数不匹配。')
            for index, (block, pair) in enumerate(zip(model.transformer.h, cache)):
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError('每层缓存应为 (K,V)。')
                if any(not isinstance(a, torch.Tensor) or a.ndim != 4 for a in pair):
                    raise ValueError('缓存应为四维张量。')
                if index == 0:
                    start = pair[0].shape[2]
                expected = (batch, block.attn.n_kv_heads, start, block.attn.head_dim)
                if any(tuple(a.shape) != expected or a.device != input_ids.device
                       or a.dtype != model.transformer.wte.weight.dtype for a in pair):
                    raise ValueError('缓存的 KV头数/batch/长度/维数/设备/精度不匹配。')
        if start + length > model.config.n_positions:
            raise ValueError('超出固定上下文容量；不自动截断缓存。')
        x = model.transformer.drop(model.transformer.wte(input_ids))
        present = []
        for index, block in enumerate(model.transformer.h):
            branch, pair = block.attn.step(block.ln_1(x), None if cache is None else cache[index])
            r = x + branch
            x = r + block.mlp(block.ln_2(r))
            if use_cache:
                present.append(pair)
        x = model.transformer.ln_f(x)
        if last_only:
            x = x[:, -1:, :]
        return model.lm_head(x), tuple(present) if use_cache else None
