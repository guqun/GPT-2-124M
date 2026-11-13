"""两份运行入口共用的短训练循环，避免数据/优化器/计时逻辑漂移。

阅读主线：main → run → evaluate / train_update → 保存 report/checkpoint。
【本项目】控制顺序、取批、损失归一化、实验协议与记录。
【PyTorch】模型运算、自动求导、梯度裁剪、AdamW 和设备执行。
本课只实现单进程 FP32，默认 CPU；无 compile、混合精度或 DDP。
"""
import argparse
import copy
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from architecture_variants import build_model, common_parameter_digest
from prepare_architecture_baseline import ROOT, load_spec, prepare_local_data, first_batch, sha


class SequentialBatches:
    """每取一批前进 B*T；当前 split 不足 B*T+1 时丢弃短尾并回到开头。"""

    def __init__(self, ids, batch_size, seq_len):
        self.ids, self.batch_size, self.seq_len = ids, batch_size, seq_len
        self.n = batch_size * seq_len
        if len(ids) < self.n + 1:
            raise ValueError('数据不足一个完整 batch。')
        self.cursor = 0
        self.wraps = 0

    def next_batch(self):
        if self.cursor + self.n + 1 > len(self.ids):
            self.cursor = 0
            self.wraps += 1
        start = self.cursor
        x, y = first_batch(self.ids, self.batch_size, self.seq_len, start)
        self.cursor += self.n
        return x, y, start


def learning_rate(update, total, warmup, peak, minimum):
    """update 从 1 开始。前 warmup 次线性预热，余下余弦下降，最后达到 minimum。"""
    if update <= warmup:
        return peak * update / warmup
    ratio = (update - warmup) / (total - warmup)
    return minimum + .5 * (1 + math.cos(math.pi * ratio)) * (peak - minimum)


def synchronize(device):
    # GPU 计算异步；计时边界等待完成。CPU 运算无需额外等待。
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def evaluate(model, ids, training, device):
    """每次重读同一验证窗口；不改变训练游标，不反传，恢复原 train/eval 状态。"""
    was_training = model.training
    model.eval()
    count = 0
    total_loss = 0.0
    n = training['batch_size'] * training['seq_len']
    try:
        with torch.no_grad():
            for batch in range(training['validation_batches']):
                x, y = first_batch(ids, training['batch_size'], training['seq_len'], batch*n)
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='sum')
                if not torch.isfinite(loss):
                    raise ValueError('验证损失非有限。')
                total_loss += loss.item()
                count += y.numel()
    finally:
        model.train(was_training)
    return {'loss': total_loss/count, 'prediction_positions': count}


def train_update(model, optimizer, loader, training, device):
    """一次完整更新：清梯度 → K 次 loss/K 反传 → 裁剪 → AdamW → 清梯度。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    k = training['accumulation_steps']
    losses, starts = [], []
    for micro in range(k):
        x, y, start = loader.next_batch()
        starts.append(start)
        x, y = x.to(device), y.to(device)
        logits = model(x)  # [B,T,V]；不在这里采样或生成新文本。
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            raise ValueError('训练损失非有限；停止更新。')
        (loss/k).backward()  # .grad 累加；这 K 轮中所有参数不变。
        losses.append(loss.item())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training['clip_norm'],
                                         error_if_nonfinite=True)
    optimizer.step()  # 这里才真正更新一次参数，也更新 AdamW 的 m/v。
    optimizer.zero_grad(set_to_none=True)
    return {'loss': sum(losses)/k, 'grad_norm_before_clip': float(norm),
            'microbatch_starts': starts, 'wraps': loader.wraps}


def resolve_device(name):
    if name == 'auto':
        name = ('cuda' if torch.cuda.is_available() else
                'mps' if torch.backends.mps.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('当前解释器无法使用 CUDA。')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('当前解释器无法使用 MPS。')
    return torch.device(name)


def run(variant, args):
    # [M1] 使用之前已固定的 local 配置；命令行覆盖值同时写入实际协议。
    spec = copy.deepcopy(load_spec('local'))
    t = spec['training']
    t['planned_updates'] = args.steps
    t['lr_warmup_updates'] = args.warmup_updates
    spec['seed'] = args.seed
    spec['budget_status'] = 'fixed_for_this_run'
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = resolve_device(args.device)
    # 独立输出目录，已有结果拒绝覆盖；可用 --output 指定新目录。
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f'输出目录非空，请用 --output 选择新目录：{output}')
    train, val, data = prepare_local_data(spec)

    # [M2] A/B 共同参数初值相同；先完成 B 的模块替换，再交给优化器。
    model = build_model(spec, variant)
    initial_digest = common_parameter_digest(model)
    parameter_count = sum(p.numel() for p in model.parameters())
    model.to(device)
    parameters = list(model.parameters())  # 共享 wte/lm_head 在这里只出现一次。
    optimizer = torch.optim.AdamW([
        {'params': [p for p in parameters if p.ndim >= 2], 'weight_decay': t['weight_decay']},
        {'params': [p for p in parameters if p.ndim < 2], 'weight_decay': 0.0},
    ], lr=t['peak_lr'], betas=tuple(t['betas']), eps=t['adam_eps'])
    loader = SequentialBatches(train, t['batch_size'], t['seq_len'])
    tokens_per_update = t['batch_size'] * t['seq_len'] * t['accumulation_steps']
    device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor()
    # 协议排除 variant/参数总数，但保留共同代码、数据、初值、硬件和全部训练设置。
    protocol = {'spec': spec, 'data': data, 'initial_common_parameters_sha256': initial_digest,
                'device': str(device), 'device_name': device_name, 'platform': platform.platform(),
                'torch': torch.__version__, 'threads': args.threads, 'precision': 'fp32',
                'tf32': False, 'timing_warmup_updates': args.timing_warmup,
                'sources': {name: sha(ROOT/name) for name in (
                    'gpt2_model.py', 'prepare_architecture_baseline.py',
                    'architecture_variants.py', 'architecture_training.py')}}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    report = {'variant': variant, 'protocol': protocol, 'protocol_sha256': fingerprint,
              'parameters': parameter_count, 'training': [], 'validation': [], 'completed': False,
              'scope': 'local small-model paired experiment; not full Llama or 124M/FineWeb',
              'timing_scope': 'batch fetch/transfer + forward/backward/clip/step; excludes validation/save'}
    print(f'{variant}：{parameter_count:,} 参数；{device}/FP32；seed={args.seed}', flush=True)
    print(f'每更新 {tokens_per_update} 个预测位置；计划 {args.steps} 次；输出 {output}', flush=True)

    # [M3] 固定窗口初始验证；它不改变训练的参数或梯度。
    before = evaluate(model, val, t, device)
    report['validation'].append({'update': 0, **before})
    print(f'验证 000：loss={before["loss"]:.6f}', flush=True)
    with (output/'metrics.jsonl').open('w', encoding='utf-8') as log:
        log.write(json.dumps({'event': 'validation', 'update': 0, **before})+'\n')
        log.flush()
        # [M4] 学习率按参数更新次数变化；每次包含 K 个微批次。
        for update in range(1, args.steps+1):
            lr = learning_rate(update, args.steps, args.warmup_updates, t['peak_lr'], t['min_lr'])
            for group in optimizer.param_groups:
                group['lr'] = lr
            synchronize(device)
            started = time.perf_counter()
            record = train_update(model, optimizer, loader, t, device)
            synchronize(device)
            elapsed = time.perf_counter()-started
            record.update(update=update, lr=lr, seconds=elapsed,
                          tokens=tokens_per_update, tokens_per_second=tokens_per_update/elapsed,
                          measured=update > args.timing_warmup)
            report['training'].append(record)
            log.write(json.dumps({'event': 'training', **record})+'\n')
            print(f'更新 {update:03d}：loss={record["loss"]:.6f} lr={lr:.7f} '
                  f'{elapsed*1000:.1f}ms {record["tokens_per_second"]:.1f} Token/s'
                  + ('（计时预热，仍计入训练量）' if not record['measured'] else ''), flush=True)
            # [M5] 验证在计时区间之外；结束后恢复 train，不动训练游标。
            if update % t['validation_every'] == 0 or update == args.steps:
                result = evaluate(model, val, t, device)
                report['validation'].append({'update': update, **result})
                log.write(json.dumps({'event': 'validation', 'update': update, **result})+'\n')
                print(f'验证 {update:03d}：loss={result["loss"]:.6f}', flush=True)
            log.flush()

    # [M6] 用总 Token 数/总时间算吞吐，不能直接平均每步 Token/s。
    measured = [r for r in report['training'] if r['measured']]
    report['summary'] = {'updates': args.steps, 'training_tokens': args.steps*tokens_per_update,
        'initial_val_loss': before['loss'], 'final_val_loss': report['validation'][-1]['loss'],
        'measured_updates': len(measured),
        'measured_tokens_per_second': sum(r['tokens'] for r in measured)/sum(r['seconds'] for r in measured)}
    report['completed'] = True
    # 保存每个版本自己的权重与优化器状态；本入口不实现断点恢复。
    torch.save({'variant': variant, 'spec': spec, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'updates': args.steps,
                'loader_cursor': loader.cursor, 'protocol_sha256': fingerprint}, output/'checkpoint.pt')
    report['checkpoint_sha256'] = sha(output/'checkpoint.pt')
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print('完成：'+str(output/'report.json'), flush=True)
    return report


def main(variant):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100, help='参数更新次数；默认完整本机短训练')
    parser.add_argument('--warmup-updates', type=int, default=None, help='学习率预热；默认 min(10,max(1,steps//10))')
    parser.add_argument('--timing-warmup', type=int, default=2, help='前几次实际更新不纳入稳定吞吐统计')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps', 'auto'], default='cpu')
    parser.add_argument('--output', type=Path, default=ROOT/'.cache/architecture-comparison'/variant)
    args = parser.parse_args()
    if args.warmup_updates is None:
        args.warmup_updates = min(10, max(1, args.steps//10))
    if args.steps < 2 or not 0 < args.warmup_updates < args.steps:
        parser.error('steps 至少为 2，且 0 < warmup-updates < steps。')
    if not 0 <= args.timing_warmup < args.steps or args.threads < 1:
        parser.error('计时预热次数须小于总更新次数；threads 必须为正整数。')
    try:
        run(variant, args)
    except (ValueError, OSError) as error:
        parser.exit(1, str(error)+'\n')
