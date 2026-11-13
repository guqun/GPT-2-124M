"""本课只增加推理缓存，不增加或改变模型参数。

复用 RoPE/SwiGLU 模型的模块；缓存由调用者传入/接回，不藏在全局状态里。
每层缓存 (旋转后的 K, 原 V)，形状均 [B,H,S,d]。本课仍是 MHA。
仅支持相同长度、无 padding 的序列；固定权重、eval 推理；不实现滑窗/缓存淘汰。
"""
import math
import torch
from torch import nn
from rope_model import apply_rotary


def attention_step(attn, x, past=None):
    """新激活 x[B,T,D] + 本层旧 K/V → 新输出[B,T,D]、本层完整 K/V。"""
    batch, length, width = x.shape
    start = 0 if past is None else past[0].shape[2]
    # [K1] 只投影这次送来的位置。prefill 为完整提示，decode 通常 T=1。
    q, k, v = attn.c_attn(x).split(width, dim=-1)
    q, k, v = [a.view(batch,length,attn.n_head,attn.head_dim).transpose(1,2)
               for a in (q,k,v)]
    # [K2] 新 Token 的绝对位置是 start，而不是每轮从 0 重新旋转。
    cos = attn.rope.cos[start:start+length]
    sin = attn.rope.sin[start:start+length]
    q, k = apply_rotary(q,cos,sin), apply_rotary(k,cos,sin)
    if past is not None:
        # [K3] 旧 K 已旋转，不能再次旋转；沿位置轴追加。V 不经过 RoPE。
        k = torch.cat((past[0],k),dim=2)
        v = torch.cat((past[1],v),dim=2)
    # [K4] Q 只有 T 行，K 有 start+T 行，分数是长方形矩阵。
    scores = (q @ k.transpose(-2,-1))/math.sqrt(attn.head_dim)
    query_positions = start + torch.arange(length,device=x.device)
    key_positions = torch.arange(start+length,device=x.device)
    allowed = key_positions[None,:] <= query_positions[:,None]
    scores = scores.masked_fill(~allowed,float('-inf'))
    weights = torch.softmax(scores,dim=-1)
    heads = weights @ v  # 新 Q 仍读取全部可见的历史 V。
    merged = heads.transpose(1,2).contiguous().view(batch,length,width)
    return attn.resid_dropout(attn.c_proj(merged)), (k,v)


class CachedDecoder(nn.Module):
    """同一模型的两种执行方式；use_cache=False 为每轮重算完整前缀。

cache=None 开启新请求；已有 cache 必须来自同一权重、同一请求的连续前缀。
权重或上下文发生改变后，调用者必须丢弃旧 cache。本课不验证语义上的所属请求。
"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.inference_mode()
    def forward(self, input_ids, cache=None, *, use_cache=True, last_only=True):
        if any(m.training for m in self.modules()):
            raise ValueError('KV Cache 课只做 eval 推理，请先 decoder.eval()。')
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError('输入必须是 int64 的 [B,T] Token ID。')
        batch,length = input_ids.shape
        if batch < 1 or length < 1:
            raise ValueError('输入不能为空。')
        if cache is not None and not use_cache:
            raise ValueError('无缓存分支不能传入旧缓存。')
        model = self.model
        start = 0
        if cache is not None:
            if not isinstance(cache,(tuple,list)) or len(cache) != len(model.transformer.h):
                raise ValueError('需要每个 Block 各自的一对 K/V。')
            for index,(pair,block) in enumerate(zip(cache,model.transformer.h)):
                if not isinstance(pair,(tuple,list)) or len(pair)!=2:
                    raise ValueError('每层缓存必须是 (K,V)。')
                k,v=pair
                if not isinstance(k,torch.Tensor) or not isinstance(v,torch.Tensor) or k.ndim!=4:
                    raise ValueError('缓存张量须为 [B,H,S,d]。')
                if index==0: start=k.shape[2]
                expected=(batch,block.attn.n_head,start,block.attn.head_dim)
                if tuple(k.shape)!=expected or tuple(v.shape)!=expected:
                    raise ValueError('各层缓存的 batch/头数/长度/头维度不匹配。')
                if any(a.device!=input_ids.device or a.dtype!=model.transformer.wte.weight.dtype
                       for a in pair):
                    raise ValueError('缓存、模型和输入须使用相同设备及匹配的浮点精度。')
        if start+length > model.config.n_positions:
            raise ValueError('超出固定上下文容量；本课不裁剪缓存或重置位置。')
        # [K5] 每层都只处理本次输入的 T 个位置，主干/MLP 不重算旧位置。
        x=model.transformer.drop(model.transformer.wte(input_ids))
        new_cache=[]
        for index,block in enumerate(model.transformer.h):
            if use_cache:
                branch,present=attention_step(block.attn,block.ln_1(x),
                                             None if cache is None else cache[index])
                new_cache.append(present)
            else:
                branch=block.attn(block.ln_1(x))  # 直接调用前课未改动的注意力。
            r=x+branch
            x=r+block.mlp(block.ln_2(r))
        x=model.transformer.ln_f(x)
        # 两种模式都只为末位置算词表分数，避免把 lm_head 的差异混入测速。
        if last_only: x=x[:,-1:,:]
        logits=model.lm_head(x)
        return logits, tuple(new_cache) if use_cache else None


def cache_bytes(cache):
    """只统计存活 K/V 张量的有效字节；不含权重、工作区或分配器保留内存。"""
    return 0 if cache is None else sum(a.numel()*a.element_size() for pair in cache for a in pair)
