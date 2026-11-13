"""第23块：直接构建现代稠密教学模型，不再先造 GPT 再替换模块。

本项目负责 Block/残差/缓存的组织；PyTorch 负责层运算与自动求导。
复用已学习的数学原语；不是官方 Qwen3 权重兼容实现。仅支持 FP32、等长无 padding。
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from qknorm_model import HeadRMSNorm
from rope_model import RotaryEmbedding, apply_rotary
from gqa_model import grouped_attention


@dataclass(frozen=True)
class ModernConfig:
    vocab_size: int = 50257
    max_length: int = 256
    width: int = 128
    layers: int = 2
    q_heads: int = 4
    kv_heads: int = 2
    intermediate: int = 341
    eps: float = 1e-5
    theta: float = 10000.0
    tie_embeddings: bool = True

    def __post_init__(self):
        for key in ('vocab_size','max_length','width','layers','q_heads','kv_heads','intermediate'):
            if type(getattr(self,key)) is not int or getattr(self,key) < 1:
                raise ValueError(key+' 必须为正整数')
        if self.width % self.q_heads or self.q_heads % self.kv_heads:
            raise ValueError('width 须整除为 Q 头，Q 头数须为 KV 头数的整数倍')
        if (self.width // self.q_heads) % 2:
            raise ValueError('相邻配对 RoPE 要求每头维度为偶数')
        if not math.isfinite(self.eps) or self.eps <= 0 or not math.isfinite(self.theta) or self.theta <= 0:
            raise ValueError('eps/theta 必须有限且为正数')
        if type(self.tie_embeddings) is not bool:
            raise ValueError('tie_embeddings 须为 bool')


class ModernAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.hq, self.hkv, self.d = c.q_heads, c.kv_heads, c.width//c.q_heads
        # PyTorch Linear.weight 保存 [out,in]；输入完整 D 维，输出宽度由头数决定。
        self.q_proj = nn.Linear(c.width,c.width,bias=False)
        self.k_proj = nn.Linear(c.width,self.hkv*self.d,bias=False)
        self.v_proj = nn.Linear(c.width,self.hkv*self.d,bias=False)
        self.q_norm, self.k_norm = HeadRMSNorm(self.d,c.eps), HeadRMSNorm(self.d,c.eps)
        self.rope = RotaryEmbedding(self.d,c.max_length,c.theta)
        self.c_proj = nn.Linear(c.width,c.width,bias=False)

    def forward(self, u, past=None):
        # [A1] u[B,T,D] 已通过 Block RMSNorm；每层每个位置重新投影。
        b,t,d = u.shape
        q = self.q_proj(u).view(b,t,self.hq,self.d).transpose(1,2)
        k = self.k_proj(u).view(b,t,self.hkv,self.d).transpose(1,2)
        v = self.v_proj(u).view(b,t,self.hkv,self.d).transpose(1,2)
        # [A2] 头内归一化 → 旋转；V 不归一化、不旋转。
        start = 0 if past is None else past[0].shape[2]
        cos, sin = self.rope.cos[start:start+t], self.rope.sin[start:start+t]
        q = apply_rotary(self.q_norm(q),cos,sin)
        k = apply_rotary(self.k_norm(k),cos,sin)
        # [A3] 旧 K 已归一化/旋转；只追加新行，缓存始终只有 Hkv 组。
        if past is not None:
            k, v = torch.cat((past[0],k),2), torch.cat((past[1],v),2)
        # [A4] A=softmax(QKᵀ/√d+mask)，H=AV；每个 Q 头有独立 A/H。
        heads = grouped_attention(q,k,v,start)
        merged = heads.transpose(1,2).contiguous().view(b,t,d)
        return self.c_proj(merged), (k,v)


class ModernMLP(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.gate_proj = nn.Linear(c.width,c.intermediate,bias=False)
        self.up_proj = nn.Linear(c.width,c.intermediate,bias=False)
        self.down_proj = nn.Linear(c.intermediate,c.width,bias=False)

    def forward(self,z):
        # [F1] 两路 [B,T,D]→[B,T,F]；逐元素门控，再混合回 D 维。
        return self.down_proj(F.silu(self.gate_proj(z))*self.up_proj(z))


class ModernBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.ln_1, self.ln_2 = HeadRMSNorm(c.width,c.eps), HeadRMSNorm(c.width,c.eps)
        self.attn, self.mlp = ModernAttention(c), ModernMLP(c)

    def forward(self,x,past=None):
        # [B1] 两条残差使用不同的主干输入；Norm 输出不覆盖保留的主干。
        branch, present = self.attn(self.ln_1(x),past)
        r = x + branch
        y = r + self.mlp(self.ln_2(r))
        return y, present


class ModernLM(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        c = config
        # [M1] 直接建所有模块，保留前课参数名称以便严格加载并核对数值。
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(c.vocab_size,c.width),
            h=nn.ModuleList([ModernBlock(c) for _ in range(c.layers)]),
            ln_f=HeadRMSNorm(c.width,c.eps)))
        self.lm_head = nn.Linear(c.width,c.vocab_size,bias=False)
        if c.tie_embeddings:
            self.lm_head.weight = self.transformer.wte.weight
        # named_parameters 自动去除共享参数的重复引用；共享 E 只初始化一次。
        for name,p in self.named_parameters():
            if p.ndim == 1:
                nn.init.ones_(p)
            else:
                std = .02/math.sqrt(2*c.layers) if name.endswith(('c_proj.weight','down_proj.weight')) else .02
                nn.init.normal_(p,std=std)

    def _validate(self,ids,past,use_cache):
        c = self.config
        if ids.ndim != 2 or ids.dtype != torch.long or min(ids.shape) < 1:
            raise ValueError('输入必须是非空 int64 [B,T]')
        if ids.device != self.transformer.wte.weight.device:
            raise ValueError('输入与模型必须在同一设备')
        if past is not None and not use_cache:
            raise ValueError('传入历史缓存时必须启用 use_cache')
        if use_cache and (any(m.training for m in self.modules()) or torch.is_grad_enabled()):
            raise ValueError('缓存仅用于 eval() + no_grad()/inference_mode()')
        start = 0
        if past is not None:
            if not isinstance(past,(tuple,list)) or len(past)!=c.layers:
                raise ValueError('缓存层数错误')
            for i,pair in enumerate(past):
                if not isinstance(pair,(tuple,list)) or len(pair)!=2 or any(not isinstance(a,torch.Tensor) or a.ndim!=4 for a in pair):
                    raise ValueError('每层缓存须为两个四维张量')
                if i==0: start=pair[0].shape[2]
                shape=(ids.shape[0],c.kv_heads,start,c.width//c.q_heads)
                if any(tuple(a.shape)!=shape or a.device!=ids.device or a.dtype!=self.transformer.wte.weight.dtype for a in pair):
                    raise ValueError('缓存形状、设备或精度不匹配')
        if start+ids.shape[1]>c.max_length:
            raise ValueError('超过固定上下文容量；不会自动丢弃历史')

    def forward(self,input_ids,past=None,*,use_cache=False,last_only=False):
        self._validate(input_ids,past,use_cache)
        # [M2] 无位置向量相加，位置在每层 A2 的 RoPE 中处理。
        x = self.transformer.wte(input_ids)
        present=[]
        for i,block in enumerate(self.transformer.h):
            x,pair=block(x,None if past is None else past[i])
            if use_cache: present.append(pair)
        # [M3] 所有 Block 结束后归一化；训练输出所有位置，生成只需末位置。
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x[:,-1:,:] if last_only else x)
        return (logits,tuple(present)) if use_cache else logits


@torch.inference_mode()
def generate(model,prompt,new_tokens,use_cache=True):
    """[G1] prefill → 选下一个 ID → decode；返回完整 ID 与缓存，不解码字符串。

本教学入口生成固定数量，不提前停止 EOS；独立请求从空缓存开始。
"""
    if type(new_tokens) is not int or new_tokens<1:
        raise ValueError('new_tokens 必须为正整数')
    if prompt.ndim!=2 or prompt.shape[1]+new_tokens>model.config.max_length:
        raise ValueError('提示词加生成长度超过容量')
    was_training=model.training
    model.eval()
    ids=prompt.clone();past=None
    try:
        for step in range(new_tokens):
            if use_cache:
                current=ids if step==0 else ids[:,-1:]
                logits,past=model(current,past,use_cache=True,last_only=True)
            else:
                logits=model(ids,last_only=True)
            ids=torch.cat((ids,logits[:,-1,:].argmax(-1,keepdim=True)),1)
    finally:
        model.train(was_training)
    return ids,past
