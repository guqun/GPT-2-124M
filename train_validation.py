"""阶段二 · 单元 2 · 2.4 · 知识块 13：把验证循环接入训练。

先读 evaluate_validation() 的 V1—V6，再读 train_with_validation() 的 T1—T3。
本项目 GPT.forward 返回 logits；forward_and_loss 负责另外计算交叉熵。
参数仍随机初始化；只复用缓存中的模型配置和分词器，不加载预训练权重。

原课约 3:23—3:25 / build-nanogpt 21d3d32；原课使用 FineWeb-Edu train/val 分片。
本机配套暂用已有 Tiny Shakespeare：前 90% Token 位置训练，后 10% 验证。
这是验证循环的独立实验，不是 FineWeb-Edu 实跑，也不意味着 2.4 已完成。
旧 train_ddp.py 保留；CPU/Gloo 与缩小模型用于流程检查，详见 lesson13.md。
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys

try:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import GPT2TokenizerFast
    from gpt2_model import GPT, GPTConfig
    from train_ddp import (
        DistributedDataLoaderLite, accumulation_steps, configure_optimizer,
        gather_records, learning_rate_at, setup_runtime,
    )
    from train_initialization import initialize_like_course
    from train_weight_tying import tie_weights_like_course
    from train_mixed_precision import forward_and_loss, validate_amp
    from train_step import DATA_SHA256, DEFAULT_DATA_FILE, DEFAULT_MODEL_DIR, PROJECT_DIR
except ImportError as error:
    raise SystemExit(f'缺少课程依赖：{error}\n当前解释器：{sys.executable}\n'
                     '请使用已安装 requirements.txt 依赖的课程解释器。') from error


COURSE_COMMIT = '21d3d32'


class ResettableDataLoaderLite(DistributedDataLoaderLite):
    """复用第 11 块的按 rank 取数规则，只增加回到本 split 起点的能力。"""

    def reset(self):
        # 各进程重置相同的全局起点；父类 current_position 再加 rank*B*T。
        # rank 0 从 0 开始，rank 1 从 B*T 开始，不会让两张卡都读取同一批。
        self.global_position = 0


def make_train_val_loaders(token_ids, batch_size, seq_len, runtime):
    """本机数据改编：按原始 Token 位置切两份，然后各自构造 x/y。

    分割位置 split：train=[0, split)，val=[split, N)。先切再错位，
    因而训练的最后一个 y 也不会越过分界线读到验证区。两份游标独立。
    这是位置隔离，不是文档级划分或重复文本去重；实际 FineWeb 入口待补。
    """
    split = len(token_ids) * 9 // 10
    train_loader = ResettableDataLoaderLite(
        token_ids[:split], batch_size, seq_len, runtime.rank, runtime.world_size)
    val_loader = ResettableDataLoaderLite(
        token_ids[split:], batch_size, seq_len, runtime.rank, runtime.world_size)
    return train_loader, val_loader, split


def evaluate_validation(model, val_loader, runtime, *, validation_batches, precision):
    """输入：当前模型、独立验证加载器、运行设备/进程信息、每卡验证批次数。
    输出：一个记录字典，含全局平均验证 loss、取数位置及预测位置数。

    本例每卡读取相同数量的完整 [B,T] 批次，没有 padding/ignore_index。
    因而“各批次均值再平均、各卡均值再平均”等于所有验证位置的平均。
    若未来有效位置数不同，必须改成总损失之和 / 总有效位置数。
    """
    available_batches = (len(val_loader.tokens) - 1) // val_loader.stride
    if not 1 <= validation_batches <= available_batches:
        raise ValueError(f'val-batches 应在 1..{available_batches} 之间，避免本次验证重复绕回数据。')

    # [V1] eval() 切换模块行为；它不删除参数、不清梯度，也不等于 no_grad。
    # 当前课程训练关闭了 Dropout；仍保留规范的 eval/train 边界。
    was_training = model.training
    model.eval()
    try:
        # [V2] 每次评估读取相同位置，降低“换了考题”对比较的影响。
        # 这里只重置 val_loader，绝不重置 train_loader。
        val_loader.reset()
        local_batch_starts = []
        val_loss_accum = torch.zeros((), device=runtime.device,
                                     dtype=next(model.parameters()).dtype)

        # [V3] no_grad 让下面的前向不记录反向计算图；参数原有 requires_grad 不变。
        # 验证使用当前参数，循环内既没有 backward，也没有 optimizer.step。
        with torch.no_grad():
            for _ in range(validation_batches):
                local_batch_starts.append(val_loader.current_position)
                x, y = val_loader.next_batch()             # 本卡 x/y: [B,T]，int64。
                x, y = x.to(runtime.device), y.to(runtime.device)
                # [V4] 我们组织调用；PyTorch 执行矩阵运算和交叉熵。
                # logits: [B,T,V]；loss: 对本批 B*T 个正确目标取平均的标量。
                logits, loss = forward_and_loss(model, x, y, runtime.device, precision)
                # 累加的是指标数值，不是 .grad。这里无计算图，不必再 detach。
                val_loss_accum += loss / validation_batches
                del logits, loss  # 不保留上一批的大词表输出。

        # [V5] 各卡先得到自己的均值，再求跨卡均值。单进程无需通信。
        # 这是主动汇总一个指标，不是 DDP 在 backward 中自动同步梯度。
        # 所有 rank 必须进入；只让 rank 0 打印，但不能只让 rank 0 all_reduce。
        if runtime.distributed:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.SUM)
            val_loss_accum /= runtime.world_size
        if not torch.isfinite(val_loss_accum).item():
            raise RuntimeError('验证 loss 出现 NaN/Inf，请检查数据、参数及数值配置。')
        return {
            'global_mean_loss': val_loss_accum.item(),
            'local_batch_starts': local_batch_starts,
            'batches_per_rank': validation_batches,
            'input_shape': [val_loader.B, val_loader.T],
            'global_predicted_positions': (
                validation_batches * val_loader.B * val_loader.T * runtime.world_size),
        }
    finally:
        # [V6] 正常返回或发生异常，都恢复调用前的模式。
        # 本函数没有接收 optimizer，不会改变 AdamW 的 m/v/step。
        model.train(was_training)


def train_with_validation(model, raw_model, optimizer, train_loader, val_loader, runtime,
                          *, steps, accum_steps, precision, max_lr, min_lr,
                          warmup_steps, decay_steps, clip_norm,
                          validation_every, validation_batches):
    """保留第 11 块的训练主干，增加初始/间隔/最后一次参数更新后的验证。

    steps 计参数更新；validation_batches 计一次评估读几批；accum_steps 计
    一次更新前累计几次反向。这三种计数彼此独立，验证不会推进学习率调度。
    """
    if min(steps, accum_steps, validation_every, validation_batches) < 1:
        raise ValueError('训练次数、累积次数和验证间隔/批次数必须为正。')
    train_records, val_records = [], []
    model.train()

    def record_validation(completed_updates):
        result = evaluate_validation(model, val_loader, runtime,
                                     validation_batches=validation_batches, precision=precision)
        result['after_updates'] = completed_updates
        val_records.append(result)
        if runtime.rank == 0:
            print(f"验证 · 已更新 {completed_updates:02d} 次 · "
                  f"val_loss={result['global_mean_loss']:.6f} · "
                  f"全局 {result['global_predicted_positions']} 个预测位置", flush=True)

    # [T1] 参数尚未更新时记录基线。验证结束后已恢复 train 模式。
    record_validation(0)
    for step in range(steps):
        # [T2] 以下仍是已学的训练：取 train 数据、累积梯度、同步、裁剪、更新。
        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=runtime.device,
                                 dtype=next(raw_model.parameters()).dtype)
        batch_starts = []
        for micro_step in range(accum_steps):
            batch_starts.append(train_loader.current_position)
            x, y = train_loader.next_batch()
            x, y = x.to(runtime.device), y.to(runtime.device)
            sync_context = (model.no_sync()
                            if runtime.distributed and micro_step < accum_steps - 1
                            else nullcontext())
            with sync_context:
                logits, loss = forward_and_loss(model, x, y, runtime.device, precision)
                scaled_loss = loss / accum_steps
                loss_accum += scaled_loss.detach()
                scaled_loss.backward()  # 只有训练分支计算梯度。
            del logits, loss, scaled_loss
        if runtime.distributed:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
            loss_accum /= runtime.world_size
        if not torch.isfinite(loss_accum).item():
            raise RuntimeError('训练 loss 非有限数，停止更新。')
        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), clip_norm,
                                              error_if_nonfinite=True)
        # 学习率只取决于更新序号；没有把 val_loss 传给调度器或优化器。
        lr = learning_rate_at(step, max_lr, min_lr, warmup_steps, decay_steps)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()               # 只有训练分支改变参数及 AdamW 状态。
        record = {'step': step + 1, 'local_batch_starts': batch_starts,
                  'next_start': train_loader.current_position,
                  'global_mean_loss': loss_accum.item(),
                  'gradient_norm_before_clip': norm.item(), 'learning_rate': lr}
        train_records.append(record)
        if runtime.rank == 0:
            print(f"训练 · 更新 {step + 1:02d} · loss={record['global_mean_loss']:.6f} · "
                  f'lr={lr:.6g}', flush=True)

        # [T3] 训练游标已向后推进；验证有自己的游标，不会打乱下一批训练取数。
        # 额外在最后一次更新后验证，便于本地短跑比较起点/终点。
        if (step + 1) % validation_every == 0 or step + 1 == steps:
            record_validation(step + 1)
    return {'training': train_records, 'validation': val_records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--precision', choices=('auto', 'fp32', 'bf16'), default='auto')
    parser.add_argument('--model-size', choices=('gpt2', 'tiny'), default='gpt2')
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--data-file', type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seq-len', type=int, default=32)
    parser.add_argument('--total-tokens', type=int, default=512,
                        help='每次参数更新的全局训练 Token 数，不包括验证。')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--val-every', type=int, default=2,
                        help='每更新几次参数验证一次；另外验证初始和最终模型。')
    parser.add_argument('--val-batches', type=int, default=2,
                        help='每次验证，每个进程读取的 batch 数；不是梯度累积次数。')
    parser.add_argument('--warmup-steps', type=int, default=1)
    parser.add_argument('--decay-steps', type=int, default=3)
    parser.add_argument('--max-lr', type=float, default=6e-4)
    parser.add_argument('--min-lr', type=float, default=6e-5)
    parser.add_argument('--clip-norm', type=float, default=1.0)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--report', type=Path,
                        default=PROJECT_DIR / '.cache/runtime-check/lesson13-report.json')
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.total_tokens, args.steps, args.threads,
           args.val_every, args.val_batches) < 1:
        parser.error('批次尺寸、次数和线程数必须为正。')
    if not 0 <= args.warmup_steps < args.decay_steps:
        parser.error('调度要求 0 <= warmup-steps < decay-steps。')
    if (not all(math.isfinite(v) for v in (args.max_lr, args.min_lr, args.clip_norm))
            or not 0 < args.min_lr <= args.max_lr or args.clip_norm <= 0):
        parser.error('学习率和裁剪上限必须为有限正数，且 min-lr <= max-lr。')
    torch.set_num_threads(args.threads)
    runtime = setup_runtime(args.device)
    try:
        K = accumulation_steps(args.total_tokens, args.batch_size, args.seq_len, runtime.world_size)
        if args.precision == 'auto':
            args.precision = ('bf16' if runtime.device.type == 'cuda'
                              and torch.cuda.is_bf16_supported() else 'fp32')
        validate_amp(runtime.device, args.precision)
        # [M1] 只读本地文本和分词器；不会自动下载数据或模型权重。
        data_file, model_dir = args.data_file.resolve(), args.model_dir.resolve()
        data = data_file.read_bytes()
        data_sha = hashlib.sha256(data).hexdigest()
        if data_file == DEFAULT_DATA_FILE and data_sha != DATA_SHA256:
            raise ValueError('默认 Tiny Shakespeare 与固定课程版本不一致。')
        tokenizer = GPT2TokenizerFast(tokenizer_file=str(model_dir / 'tokenizer.json'))
        token_ids = tokenizer.encode(data.decode('utf-8'), add_special_tokens=False)
        train_loader, val_loader, split = make_train_val_loaders(
            token_ids, args.batch_size, args.seq_len, runtime)
        if args.val_batches > (len(val_loader.tokens) - 1) // val_loader.stride:
            parser.error('验证数据不足：请减小 val-batches、B、T 或进程数。')

        # [M2] 继承上一节的随机初始化、共享权重和模型结构；tiny 只缩小层数/宽度。
        config = GPTConfig.from_hf_config(json.loads((model_dir / 'config.json').read_text()))
        config.embd_pdrop = config.attn_pdrop = config.resid_pdrop = 0.0
        if args.model_size == 'tiny':
            config = replace(config, n_layer=2, n_embd=64, n_head=2, n_positions=128)
        if args.seq_len > config.n_positions or len(tokenizer) != config.vocab_size:
            raise ValueError('序列长度超过位置表，或分词器与模型词表不一致。')
        if min(token_ids) < 0 or max(token_ids) >= config.vocab_size:
            raise ValueError('语料包含模型词表之外的 Token ID。')
        torch.manual_seed(args.seed)
        raw_model = GPT(config, tie_embeddings=False, gpt2_init=False).float()
        tie_weights_like_course(raw_model)
        initialize_like_course(raw_model)
        raw_model.to(runtime.device)
        tie_weights_like_course(raw_model)
        model = raw_model
        if runtime.distributed:
            model = DDP(raw_model, device_ids=[runtime.local_rank]
                        if runtime.device.type == 'cuda' else None)
        optimizer = configure_optimizer(raw_model, runtime.device, args.max_lr)
        if runtime.rank == 0:
            print('知识块 13：验证循环；本机数据改编为 Tiny Shakespeare 90%/10%。')
            print(f'原始位置 train=[0,{split})，val=[{split},{len(token_ids)})；'
                  f'B={args.batch_size}, T={args.seq_len}, R={runtime.world_size}, K={K}。')
            print(f'随机初始化 {args.model_size}；{runtime.device}/{args.precision}；'
                  '验证只记录指标，学习率按更新次数调度。', flush=True)

        # [M3] 所有进程同时进入，验证和训练使用同一份当前模型参数。
        records = train_with_validation(
            model, raw_model, optimizer, train_loader, val_loader, runtime,
            steps=args.steps, accum_steps=K, precision=args.precision,
            max_lr=args.max_lr, min_lr=args.min_lr, warmup_steps=args.warmup_steps,
            decay_steps=args.decay_steps, clip_norm=args.clip_norm,
            validation_every=args.val_every, validation_batches=args.val_batches)
        ranks = gather_records({'rank': runtime.rank, **records}, runtime)
        if runtime.rank == 0:
            report = {
                'lesson': '阶段二 · 单元 2 · 2.4 · 知识块 13', 'course_commit': COURSE_COMMIT,
                'torch': torch.__version__, 'device_type': runtime.device.type,
                'backend': runtime.backend, 'world_size': runtime.world_size,
                'precision': args.precision, 'execution': 'eager', 'seed': args.seed,
                'model_size': args.model_size, 'config': asdict(config),
                'parameter_count': sum(p.numel() for p in raw_model.parameters()),
                'pretrained_weights_loaded': False, 'checkpoint_saved': False,
                'data_sha256': data_sha, 'encoded_tokens': len(token_ids),
                'data_adaptation': 'Tiny Shakespeare, position-based 90/10 split; not FineWeb-Edu.',
                'train_range_exclusive': [0, split], 'val_range_exclusive': [split, len(token_ids)],
                'batch_size_per_rank': args.batch_size, 'seq_len': args.seq_len,
                'accumulation_steps': K, 'tokens_per_optimizer_update': args.total_tokens,
                'parameter_updates': args.steps, 'training_positions': args.steps * args.total_tokens,
                'validation_every': args.val_every, 'validation_batches_per_rank': args.val_batches,
                'validation_changes_training_policy': False,
                'schedule': {'max_lr': args.max_lr, 'min_lr': args.min_lr,
                             'warmup_steps': args.warmup_steps, 'decay_steps': args.decay_steps},
                'clip_norm': args.clip_norm, 'ranks': ranks,
                'scope': 'Validation-flow check; no CUDA/NCCL performance or full pretraining claim.',
            }
            path = args.report.resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print(f'报告：{path}', flush=True)
    finally:
        if runtime.distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
