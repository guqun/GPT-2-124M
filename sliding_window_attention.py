"""3.9：同一份 Q/K/V 上，全局因果注意力与滑动窗口注意力的可观察对照。

本文件只接收投影、拆头和 RoPE 之后的特征，不创建 WQ/WK/WV。
窗口 W 包含当前位置：0 <= 查询位置 - 被读取位置 < W。
当前计算完整等长序列，不接 KV Cache，不处理 padding，不使用 Dropout。
为了展示 A，仍计算完整 T×T 分数；这是语义参考实现，不是稀疏加速内核。
"""
import math
from typing import NamedTuple, Optional

import torch


class AttentionTrace(NamedTuple):
    scores: torch.Tensor   # [B,Hq,T,T]，遮罩前的原始缩放分数
    allowed: torch.Tensor  # [T,T]，行是查询位置，列是 K/V 位置
    weights: torch.Tensor  # [B,Hq,T,T]，每行在允许位置上重新归一化
    heads: torch.Tensor    # [B,Hq,T,d]，每个查询位置、每个头的输出 H


def attention_with_window(q, k, v, window: Optional[int] = None):
    """[A1—A4] None 表示全局因果；正整数表示包含自身的窗口宽度。

    q=[B,Hq,T,d]；k/v=[B,Hkv,T,d]。支持 MHA、MQA 和 GQA，
    要求 Hq 能被 Hkv 整除。本节 T 相同，因为尚未接增量缓存。
    保留自动求导能力；调用者决定是否使用 no_grad。
    """
    if window is not None and (type(window) is not int or window < 1):
        raise ValueError('window 必须为正整数或 None（全局因果）')
    if any(x.ndim != 4 or not x.is_floating_point() or min(x.shape) < 1
           for x in (q, k, v)):
        raise ValueError('Q/K/V 必须是非空四维浮点张量')
    b, hq, t, d = q.shape
    hkv = k.shape[1]
    if k.shape != v.shape or (k.shape[0], k.shape[2], k.shape[3]) != (b, t, d):
        raise ValueError('本节要求 K/V 同形状，且 B、T、d 与 Q 一致')
    if hq % hkv:
        raise ValueError('Q 头数必须是 KV 头数的整数倍')
    if any(x.device != q.device or x.dtype != q.dtype for x in (k, v)):
        raise ValueError('Q/K/V 的设备和精度必须一致')

    # [A1] GQA 对应：把 Q 的头轴拆成 [KV组数, 每组Q头数]。
    # K/V 插入单例轴做广播，没有复制参数，也没有删除 Token。
    queries = q.reshape(b, hkv, hq // hkv, t, d)
    scores = (queries @ k.unsqueeze(2).transpose(-2, -1)) / math.sqrt(d)
    scores = scores.reshape(b, hq, t, t)

    # [A2] 同一层所有查询行并行建立可见关系。
    # distance[i,j]=i-j；未来 j>i 为负数，所以两种模式都禁止读未来。
    positions = torch.arange(t, device=q.device)
    distance = positions[:, None] - positions[None, :]
    allowed = distance >= 0
    if window is not None:
        # 唯一的模式差异：再挡住距离 >= W 的旧位置。
        # W=3 时，D(3) 可读 B(1)、C(2)、D(3)，不可读 A(0)。
        allowed = allowed & (distance < window)

    # [A3] 必须先遮罩，再 softmax。这样剩余位置重新分配总和为1的权重。
    # 正窗口保证对角线始终可见，不会出现全为 -inf 的查询行。
    weights = torch.softmax(scores.masked_fill(~allowed, float('-inf')), dim=-1)

    # [A4] 各 Q 头用自己的 A 读取所属组的 V，输出仍有 Hq 个头。
    grouped_weights = weights.reshape(b, hkv, hq // hkv, t, t)
    heads = (grouped_weights @ v.unsqueeze(2)).reshape(b, hq, t, d)
    return AttentionTrace(scores, allowed, weights, heads)
