"""第24块：同输入、同权重、同 Q/K/V，对照全局与滑动窗口注意力。

直接运行：.venv/bin/python compare_attention_windows.py
窗口包含自身，默认 W=3；只观察前向，不训练、不加载预训练权重。
"""
import argparse
import json
import math
from pathlib import Path

import torch

from modern_model import ModernAttention, ModernConfig
from rope_model import apply_rotary
from sliding_window_attention import attention_with_window


def project_once(layer, u):
    """[P1] 复用已有现代注意力的参数，投影/拆头/QK-Norm/RoPE 只算一次。

    u 已代表本层 Block 归一化后的隐藏向量，不是 Token ID。
    不在这里再次做 Block RMSNorm。每层 Q/K 头内归一化仍按已有模型执行。
    """
    b, t, _ = u.shape
    q = layer.q_proj(u).view(b, t, layer.hq, layer.d).transpose(1, 2)
    k = layer.k_proj(u).view(b, t, layer.hkv, layer.d).transpose(1, 2)
    v = layer.v_proj(u).view(b, t, layer.hkv, layer.d).transpose(1, 2)
    cos, sin = layer.rope.cos[:t], layer.rope.sin[:t]
    q = apply_rotary(layer.q_norm(q), cos, sin)
    k = apply_rotary(layer.k_norm(k), cos, sin)
    return q, k, v


def output_projection(layer, heads):
    """[P2] [B,Hq,T,d] → [B,T,Hq*d] → 共用的输出投影 W_O。"""
    b, h, t, d = heads.shape
    merged = heads.transpose(1, 2).contiguous().view(b, t, h * d)
    return layer.c_proj(merged)


def worked_example(window):
    """[E1] 单头 d=1 的已学算例；直接指定中间特征，不冒充投影实测值。"""
    q = torch.ones(1, 1, 5, 1)
    k = torch.tensor([0., 0., 0., math.log(2), 0.]).view(1, 1, 5, 1)
    v = torch.tensor([10., 2., 4., 6., 8.]).view(1, 1, 5, 1)
    global_trace = attention_with_window(q, k, v)
    local_trace = attention_with_window(q, k, v, window)
    return dict(global_D=global_trace.heads[0, 0, 3, 0].item(),
                window_D=local_trace.heads[0, 0, 3, 0].item(),
                global_D_weights=global_trace.weights[0, 0, 3].tolist(),
                window_D_weights=local_trace.weights[0, 0, 3].tolist())


def run(window=3):
    # [M1] 固定种子的随机教学输入与参数；A—E 仅是五个位置的名称。
    # 保留当前项目的 4Q/2KV 分组方式，将 D 缩为16以便观察。
    torch.manual_seed(42)
    config = ModernConfig(width=16, q_heads=4, kv_heads=2, max_length=16)
    layer = ModernAttention(config).eval()
    u = torch.randn(1, 5, config.width)
    before = {name: p.detach().clone() for name, p in layer.named_parameters()}

    with torch.no_grad():
        # [M2] 只创建一层、只投影一次；两种模式接收同一个 q/k/v 对象。
        q, k, v = project_once(layer, u)
        global_trace = attention_with_window(q, k, v)
        local_trace = attention_with_window(q, k, v, window)
        global_out = output_projection(layer, global_trace.heads)
        local_out = output_projection(layer, local_trace.heads)

        # [M3] 用旧 ModernAttention.forward 作全局分支的参考，不修改旧实现。
        reference, _ = layer(u)
        torch.testing.assert_close(global_out, reference, rtol=1e-5, atol=1e-6)
        full_window = attention_with_window(q, k, v, u.shape[1])
        torch.testing.assert_close(full_window.heads, global_trace.heads, rtol=0, atol=0)
        example = worked_example(window)

    for name, p in layer.named_parameters():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
    report = dict(scope='single attention layer; CPU/FP32; synthetic hidden vectors; no training/cache/speed claim',
                  seed=42, window_including_self=window,
                  shapes={name: list(x.shape) for name, x in [('U', u), ('Q', q), ('K', k), ('V', v),
                          ('A', global_trace.weights), ('H', global_trace.heads), ('O', global_out)]},
                  shared_projection_parameters=True, shared_qkv=True, parameters_unchanged=True,
                  global_matches_existing_layer=True, covering_window_matches_global=True,
                  raw_scores_equal=torch.equal(global_trace.scores, local_trace.scores),
                  global_mask=global_trace.allowed.int().tolist(),
                  window_mask=local_trace.allowed.int().tolist(),
                  global_head0_weights=global_trace.weights[0, 0].tolist(),
                  window_head0_weights=local_trace.weights[0, 0].tolist(),
                  global_head0_H=global_trace.heads[0, 0].tolist(),
                  window_head0_H=local_trace.heads[0, 0].tolist(),
                  output_max_difference=(global_out-local_out).abs().max().item(),
                  global_reference_max_difference=(global_out-reference).abs().max().item(),
                  worked_example=example)

    # [M4] 按位置显示“谁能读谁”，再显示第0个 Q 头的实际 A/H。
    labels = list('ABCDE')
    print('阶段二 · 单元3 · 3.9：全局因果 / 滑动窗口注意力')
    print(f'CPU/FP32；W={window}（包含当前位置）；同一层参数、同一份 Q/K/V；无训练')
    print('形状：', report['shapes'])
    for title, trace in [('全局因果', global_trace), ('滑动窗口', local_trace)]:
        print('\n'+title+'（每行对应一个查询位置）：')
        for i, label in enumerate(labels):
            visible = ''.join(labels[j] for j in range(5) if trace.allowed[i, j])
            weights = ', '.join(f'{x:.4f}' for x in trace.weights[0, 0, i].tolist())
            head = ', '.join(f'{x:.4f}' for x in trace.heads[0, 0, i].tolist())
            print(f'  Q_{label} 可读 {visible:<5}  A=[{weights}]  H=[{head}]')
    print(f'\n输出投影后 O 的最大差：{report["output_max_difference"]:.6f}')
    print(f'已学单头算例 D：全局={example["global_D"]:.4f}，窗口={example["window_D"]:.4f}')
    print('检查通过：原始分数相同；全局对齐旧层；覆盖全序列的窗口等于全局；参数未变。')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window', type=int, default=3, help='正整数，包含当前位置')
    parser.add_argument('--report', type=Path, help='可选 JSON 报告路径，拒绝覆盖已有文件')
    args = parser.parse_args()
    if args.window < 1:
        parser.error('window 必须为正整数')
    if args.report is not None and args.report.exists():
        parser.error('报告已存在，请使用新路径')
    report = run(args.window)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write('\n')
        print('报告：', args.report)


if __name__ == '__main__':
    main()
