"""阶段二 · 单元 2 · 2.4 · 知识块 11：DDP、分配数据与梯度累积。

阅读顺序：main() 的 M1—M5；新增主干为 R（进程）、D（数据）、T（更新）。
仍用 Tiny Shakespeare 和随机参数，先把多进程训练接通；FineWeb-Edu 留下一块。
默认 GPT-2 124M；--model-size tiny 只缩小模型作流程验证，不更换分词器。
原视频约 2:46:52—3:09:48、作者 ba2554a；改编与边界见 lesson11.md。
普通 python 运行单进程；torchrun 启动多进程。CPU 用 Gloo，CUDA 用 NCCL。
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys

try:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import GPT2TokenizerFast
    from gpt2_model import GPT, GPTConfig
    from train_initialization import initialize_like_course
    from train_weight_tying import tie_weights_like_course
    from train_mixed_precision import forward_and_loss, validate_amp
    from train_step import DATA_SHA256, DEFAULT_DATA_FILE, DEFAULT_MODEL_DIR, PROJECT_DIR
except ImportError as error:
    raise SystemExit(f'缺少课程依赖：{error}\n当前解释器：{sys.executable}\n'
                     f'请使用：{Path(__file__).resolve().parent / ".venv/bin/python"}') from error


COURSE_COMMIT = 'ba2554acce54487f07c13563247f2cce683e0d5e'


@dataclass
class Runtime:
    distributed: bool
    rank: int                   # 所有机器合起来的进程编号，用于分配数据。
    local_rank: int             # 当前机器内的进程编号，用于选择本机 GPU。
    world_size: int             # 参与训练的总进程数；单机一进程一卡时等于 GPU 数。
    device: torch.device
    backend: str


def setup_runtime(requested_device):
    # [R1] torchrun 启动的每个进程都会执行本函数，但读到的 rank 不同。
    launched = 'RANK' in os.environ
    rank = int(os.environ['RANK']) if launched else 0
    local_rank = int(os.environ['LOCAL_RANK']) if launched else 0
    world_size = int(os.environ['WORLD_SIZE']) if launched else 1
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError('无效的 RANK、LOCAL_RANK 或 WORLD_SIZE。')
    if requested_device == 'auto':
        requested_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested_device not in ('cpu', 'cuda'):
        raise ValueError('本节 DDP 实验支持 CPU/Gloo 或 CUDA/NCCL。')
    if requested_device == 'cuda':
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            raise ValueError('没有对应的 CUDA 设备；本机流程验证使用 --device cpu。')
        # [R2] 每个进程固定使用一张本机 GPU；CPU 进程没有独立的 GPU 显存。
        torch.cuda.set_device(local_rank)
        device, backend = torch.device('cuda', local_rank), 'nccl'
    else:
        device, backend = torch.device('cpu'), 'gloo'
    if launched:
        # [R3] env:// 还会读取 torchrun 设置的 MASTER_ADDR/MASTER_PORT，建立通信组。
        # 这一步只接通通信，不创建模型，也不自动划分数据。
        available = (dist.is_nccl_available() if backend == 'nccl'
                     else dist.is_gloo_available())
        if not available:
            raise ValueError(f'当前 PyTorch 未提供 {backend} 后端。')
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=120))
    return Runtime(launched, rank, local_rank, world_size, device,
                   backend if launched else 'none')


class DistributedDataLoaderLite:
    """每个进程有自己的加载器；通过相同轮次、不同 rank 取得不同输入位置。

    本地修正：所有 rank 用同一个全局块起点判断回绕，避免语料尾部错开轮次。
    这不是随机打乱；也没有把文件预切成 FineWeb-Edu 那样的磁盘分片。
    """

    def __init__(self, token_ids, batch_size, seq_len, rank=0, world_size=1):
        if min(batch_size, seq_len, world_size) < 1 or not 0 <= rank < world_size:
            raise ValueError('B、T、world_size 必须为正，且 0 <= rank < world_size。')
        self.B, self.T, self.rank, self.world_size = batch_size, seq_len, rank, world_size
        self.stride = batch_size * seq_len * world_size
        if len(token_ids) < self.stride + 1:
            raise ValueError('语料至少要能容纳所有进程各一批输入，以及最后一个答案。')
        self.tokens = torch.as_tensor(token_ids, dtype=torch.long)
        self.global_position = 0

    @property
    def current_position(self):
        # [D1] 全局块起点相同，但 rank 0/1 分别偏移 0 / B*T。
        return self.global_position + self.rank * self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        start = self.current_position
        # [D2] 仍然多取 1 个 ID 作最后的标签；x/y 仍为 [B,T]。
        buffer = self.tokens[start:start + B * T + 1]
        x, y = buffer[:-1].view(B, T), buffer[1:].view(B, T)
        # [D3] 跳过所有 rank 本轮共同消费的输入位置，而不是只跳过自己的 B*T。
        self.global_position += self.stride
        # [D4] 用全局块起点判断所有 rank 的下一轮能否放下；不足时一起回绕。
        # 恰好放得下仍应读取，因此是 >。每个 rank 必须调用相同次数 next_batch。
        if self.global_position + self.stride + 1 > len(self.tokens):
            self.global_position = 0
        return x, y


def accumulation_steps(total_tokens, batch_size, seq_len, world_size):
    """total_tokens 是一次 optimizer.step 的全局预测位置数，不是序列条数。"""
    per_round = batch_size * seq_len * world_size
    if min(total_tokens, batch_size, seq_len, world_size) < 1 or total_tokens % per_round:
        raise ValueError('total-tokens 必须是 B*T*world_size 的正整数倍。')
    return total_tokens // per_round


def learning_rate_at(step, max_lr, min_lr, warmup_steps, decay_steps):
    """复用已讨论的三段调度；step 数的是参数更新次数，不是微批次编号。"""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= decay_steps:
        return min_lr
    ratio = (step - warmup_steps) / (decay_steps - warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


def configure_optimizer(raw_model, device, learning_rate, weight_decay=0.1):
    # [O1] parameters() 对共享 Parameter 去重；输入表与 lm_head 只交给优化器一次。
    parameters = [p for p in raw_model.parameters() if p.requires_grad]
    groups = [
        {'params': [p for p in parameters if p.dim() >= 2], 'weight_decay': weight_decay},
        {'params': [p for p in parameters if p.dim() < 2], 'weight_decay': 0.0},
    ]
    # 矩阵（含 Embedding）衰减；偏置、LayerNorm 的一维参数不衰减。
    # 本项目固定 PyTorch 2.2.2：CUDA 使用 fused，CPU 使用普通 AdamW。
    return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95),
                             eps=1e-8, fused=device.type == 'cuda')


def train_updates(model, raw_model, optimizer, loader, runtime, *, steps, accum_steps,
                  precision, max_lr, min_lr, warmup_steps, decay_steps, clip_norm):
    records = []
    model.train()
    for step in range(steps):
        # [T1] 外层一次循环才对应一次参数更新；只在累积窗口开头清梯度。
        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=runtime.device,
                                 dtype=next(raw_model.parameters()).dtype)
        batch_starts = []
        for micro_step in range(accum_steps):
            batch_starts.append(loader.current_position)
            x, y = loader.next_batch()
            x, y = x.to(runtime.device), y.to(runtime.device)  # [B,T]；ID 仍是 int64。
            # [T2] 前 K-1 次只在本进程累积 .grad；最后一次允许 DDP 同步累计梯度。
            # no_sync 必须包住前向和反向。原视频直接改内部标志，本地采用公开 API。
            last_micro_step = micro_step == accum_steps - 1
            sync_context = (model.no_sync() if runtime.distributed and not last_micro_step
                            else nullcontext())
            with sync_context:
                # [T3] 前向 [B,T] -> [B,T,V]；交叉熵对 B*T 个位置平均，得到标量。
                logits, loss = forward_and_loss(model, x, y, runtime.device, precision)
                # backward 默认相加，所以除以 K，得到 K 个等大微批次的平均。
                # 默认 DDP 已经按 world_size 平均梯度，此处不要再除一次卡数。
                scaled_loss = loss / accum_steps
                loss_accum += scaled_loss.detach()  # 只供日志，不再参加反向。
                scaled_loss.backward()  # Parameter.grad 与 Parameter 的形状保持一致。
        # [T4] 这一通信只为了显示全局平均 loss；参数梯度已经由 DDP 在反向中同步。
        # 所有进程都要参与，不能把 all_reduce 放进 if rank == 0。
        if runtime.distributed:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
            loss_accum /= runtime.world_size  # Gloo 与 NCCL 均支持 SUM。
        if not torch.isfinite(loss_accum).item():
            raise RuntimeError('全局平均 loss 非有限数，停止更新。')
        # [T5] 累积并同步完成后统一裁剪，再设置本次学习率，最后只更新一次参数。
        # error_if_nonfinite 避免把 NaN/Inf 梯度交给优化器。
        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), clip_norm,
                                              error_if_nonfinite=True)
        lr = learning_rate_at(step, max_lr, min_lr, warmup_steps, decay_steps)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        record = {'step': step + 1, 'local_batch_starts': batch_starts,
                  'next_start': loader.current_position, 'input_shape': list(x.shape),
                  'logits_shape': list(logits.shape), 'global_mean_loss': loss_accum.item(),
                  'gradient_norm_before_clip': norm.item(), 'learning_rate': lr}
        records.append(record)
        if runtime.rank == 0:
            print(f"更新 {step + 1:02d}：全局平均 loss={record['global_mean_loss']:.6f}，"
                  f"lr={lr:.6g}，裁剪前 norm={norm.item():.4f}", flush=True)
    return records


def gather_records(local_record, runtime):
    """教学观察用：训练/预览结束后集中少量记录，不传模型或梯度，不参与训练计时。"""
    if not runtime.distributed:
        return [local_record]
    records = [None] * runtime.world_size
    dist.all_gather_object(records, local_record)
    return records


def main():
    # [M1] 各进程解析同样的参数。普通 python 不会自行变成多个进程。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--precision', choices=('auto', 'fp32', 'bf16'), default='auto')
    parser.add_argument('--model-size', choices=('gpt2', 'tiny'), default='gpt2')
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--data-file', type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument('--batch-size', type=int, default=2, help='每个进程每个微批次的 B。')
    parser.add_argument('--seq-len', type=int, default=32)
    parser.add_argument('--total-tokens', type=int, default=512,
                        help='一次参数更新，所有进程与所有微批次合计的预测位置数。')
    parser.add_argument('--steps', type=int, default=3, help='参数更新次数，不是微批次数。')
    parser.add_argument('--warmup-steps', type=int, default=1)
    parser.add_argument('--decay-steps', type=int, default=3)
    parser.add_argument('--max-lr', type=float, default=6e-4)
    parser.add_argument('--min-lr', type=float, default=6e-5)
    parser.add_argument('--clip-norm', type=float, default=1.0)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--preview-only', action='store_true', help='每个进程预览三批，不建模型。')
    parser.add_argument('--report', type=Path,
                        default=PROJECT_DIR / '.cache/runtime-check/lesson11-report.json')
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.total_tokens, args.steps, args.threads) < 1:
        parser.error('B、T、Token 总数、更新次数和线程数必须为正。')
    if not 0 <= args.warmup_steps < args.decay_steps:
        parser.error('调度要求 0 <= warmup-steps < decay-steps。')
    if (not all(math.isfinite(v) for v in (args.max_lr, args.min_lr, args.clip_norm))
            or not 0 < args.min_lr <= args.max_lr or args.clip_norm <= 0):
        parser.error('学习率和裁剪上限必须为有限正数，且 min-lr <= max-lr。')
    torch.set_num_threads(args.threads)
    runtime = setup_runtime(args.device)
    try:
        K = accumulation_steps(args.total_tokens, args.batch_size, args.seq_len,
                               runtime.world_size)
        if args.precision == 'auto':
            args.precision = ('bf16' if runtime.device.type == 'cuda'
                              and torch.cuda.is_bf16_supported() else 'fp32')
        validate_amp(runtime.device, args.precision)
        # [M2] 数据仍是同一份本地 Tiny Shakespeare；每个进程读整份 ID，取不同位置。
        data_file, model_dir = args.data_file.resolve(), args.model_dir.resolve()
        data = data_file.read_bytes()
        data_sha = hashlib.sha256(data).hexdigest()
        if data_file == DEFAULT_DATA_FILE and data_sha != DATA_SHA256:
            raise ValueError('默认 Tiny Shakespeare 与固定课程版本不一致。')
        tokenizer = GPT2TokenizerFast(tokenizer_file=str(model_dir / 'tokenizer.json'))
        token_ids = tokenizer.encode(data.decode('utf-8'), add_special_tokens=False)
        loader = DistributedDataLoaderLite(token_ids, args.batch_size, args.seq_len,
                                            runtime.rank, runtime.world_size)
        if runtime.rank == 0:
            print('阶段二 · 单元 2 · 2.4 · 知识块 11：DDP 与分配数据', flush=True)
            print(f'进程数 R={runtime.world_size}；每进程 B={args.batch_size}，T={args.seq_len}；'
                  f'累积 K={K}；每次更新 R*K*B*T={args.total_tokens} 个位置。', flush=True)
        report = {'lesson': '阶段二 · 单元 2 · 2.4 · 知识块 11',
                  'course_commit': COURSE_COMMIT, 'torch': torch.__version__,
                  'backend': runtime.backend, 'device_type': runtime.device.type,
                  'world_size': runtime.world_size, 'batch_size_per_rank': args.batch_size,
                  'seq_len': args.seq_len, 'accumulation_steps': K,
                  'tokens_per_optimizer_update': args.total_tokens,
                  'data_sha256': data_sha, 'encoded_tokens': len(token_ids),
                  'preview_only': args.preview_only, 'pretrained_weights_loaded': False,
                  'checkpoint_saved': False,
                  'scope': 'Training-flow check only; CPU/Gloo does not measure CUDA/NCCL performance.'}
        if args.preview_only:
            rows = []
            for _ in range(3):
                start = loader.current_position
                x, y = loader.next_batch()
                rows.append({'x_positions': [start, start + x.numel() - 1],
                             'y_positions': [start + 1, start + y.numel()],
                             'shape': list(x.shape), 'next_start': loader.current_position})
            report['ranks'] = gather_records({'rank': runtime.rank, 'batches': rows}, runtime)
            if runtime.rank == 0:
                print(json.dumps(report['ranks'], ensure_ascii=False, indent=2))
        else:
            # [M3] 各进程创建完整模型副本；tiny 是两层小模型，供本机多进程验证。
            config = GPTConfig.from_hf_config(json.loads((model_dir / 'config.json').read_text()))
            config.embd_pdrop = config.attn_pdrop = config.resid_pdrop = 0.0
            if args.model_size == 'tiny':
                config = replace(config, n_layer=2, n_embd=64, n_head=2, n_positions=128)
            if args.seq_len > config.n_positions or len(tokenizer) != config.vocab_size:
                raise ValueError('序列超过位置表容量，或 tokenizer 与模型词表不一致。')
            if min(token_ids) < 0 or max(token_ids) >= config.vocab_size:
                raise ValueError('语料中存在不属于模型词表的 Token ID。')
            torch.manual_seed(args.seed)
            raw_model = GPT(config, tie_embeddings=False, gpt2_init=False).float()
            tie_weights_like_course(raw_model)
            initialize_like_course(raw_model)
            raw_model.to(runtime.device)
            tie_weights_like_course(raw_model)
            # [M4] DDP 包装发生在建模/迁移之后；优化器继续管理 raw_model 的参数。
            # 默认构造还会从 rank 0 同步模型状态。它不会替我们划分输入数据。
            model = raw_model
            if runtime.distributed:
                model = DDP(raw_model, device_ids=[runtime.local_rank]
                            if runtime.device.type == 'cuda' else None)
            optimizer = configure_optimizer(raw_model, runtime.device, args.max_lr)
            if runtime.rank == 0:
                print(f'模型={args.model_size}；参数={sum(p.numel() for p in raw_model.parameters()):,}；'
                      f'{runtime.device} / {args.precision} / {runtime.backend}；随机新参数。', flush=True)
            # [M5] 每个进程都进入同一训练循环；DDP 在对应反向之间协调通信。
            rows = train_updates(model, raw_model, optimizer, loader, runtime, steps=args.steps,
                                 accum_steps=K, precision=args.precision, max_lr=args.max_lr,
                                 min_lr=args.min_lr, warmup_steps=args.warmup_steps,
                                 decay_steps=args.decay_steps, clip_norm=args.clip_norm)
            report.update(model_size=args.model_size, config=asdict(config), seed=args.seed,
                          precision=args.precision, execution='eager',
                          parameter_count=sum(p.numel() for p in raw_model.parameters()),
                          parameter_updates=args.steps, supervised_positions=args.steps * args.total_tokens,
                          weight_tied=raw_model.lm_head.weight is raw_model.transformer.wte.weight,
                          optimizer={'betas': [0.9, 0.95], 'eps': 1e-8,
                                     'weight_decay_matrix': 0.1, 'weight_decay_vector': 0.0,
                                     'fused': runtime.device.type == 'cuda'},
                          schedule={'max_lr': args.max_lr, 'min_lr': args.min_lr,
                                    'warmup_steps': args.warmup_steps, 'decay_steps': args.decay_steps},
                          clip_norm=args.clip_norm)
            report['ranks'] = gather_records({'rank': runtime.rank, 'steps': rows}, runtime)
        if runtime.rank == 0:
            path = args.report.resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print(f'报告：{path}', flush=True)
    finally:
        # 各进程都清理通信组；即使数据/训练抛错也会执行。
        if runtime.distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
