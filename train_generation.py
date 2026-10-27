"""阶段二 · 单元 2 · 2.4 · 知识块 14：训练中定期生成文本。

先看 sample_current_model() 的 G1—G6，再看主循环的 T3/T4。
训练 / 验证 / 生成共用当前参数；只有训练执行 backward 和 optimizer.step。
每次启动仍随机初始化，读取缓存的配置与分词器，不加载预训练权重。

原课约 3:25—3:28 / build-nanogpt 4a63e6a。本机沿用 Tiny Shakespeare
90%/10%、短跑频率与 eager 模型；生成按原例使用 FP32、top-k=50、独立 RNG。
保留上一块 train_validation.py；原课 compile 的历史问题和改编见 lesson14.md。
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
    from torch.nn import functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import GPT2TokenizerFast
    from gpt2_model import GPT, GPTConfig
    from train_ddp import (
        accumulation_steps, configure_optimizer, gather_records,
        learning_rate_at, setup_runtime,
    )
    from train_initialization import initialize_like_course
    from train_weight_tying import tie_weights_like_course
    from train_mixed_precision import forward_and_loss, validate_amp
    from train_step import DATA_SHA256, DEFAULT_DATA_FILE, DEFAULT_MODEL_DIR, PROJECT_DIR
    from train_validation import evaluate_validation, make_train_val_loaders
except ImportError as error:
    raise SystemExit(f'缺少课程依赖：{error}\n当前解释器：{sys.executable}\n'
                     '请使用已安装 requirements.txt 依赖的课程解释器。') from error


COURSE_COMMIT = '4a63e6a599fea4f78d25e78a4970585ae5c6e27f'
DEFAULT_PROMPT = "Hello, I'm a language model,"


def sample_current_model(raw_model, prompt_ids, *, max_new_tokens=24,
                         num_sequences=4, top_k=50, seed=42):
    """当前 GPT + 一维提示词 ID -> 多条完整 ID 序列及观察记录。

    raw_model 是训练模型本体；DDP 时传 model.module，不复制或重新加载参数。
    每张卡在自己的本体上生成，不需要跨卡求平均；这里没有 DDP 通信。
    每次调用都从同一提示词重新开始，seed 固定便于同卡跨训练阶段比较。
    本例模型参数为 FP32，生成不用 autocast；训练精度由主入口另行指定。
    沿用原课固定长度生成，不因 EOS 提前结束；无 KV cache。
    """
    if min(max_new_tokens, num_sequences, top_k) < 1:
        raise ValueError('新增 Token 数、生成条数和 top-k 必须为正。')
    config = raw_model.config
    if top_k > config.vocab_size:
        raise ValueError('top-k 超过模型词表大小。')
    device = next(raw_model.parameters()).device
    prompt = torch.as_tensor(prompt_ids, dtype=torch.long, device=device)
    if prompt.ndim != 1 or not 1 <= prompt.numel() <= config.n_positions:
        raise ValueError('提示词必须是一维非空 ID，长度不能超过位置表。')
    if bool(((prompt < 0) | (prompt >= config.vocab_size)).any()):
        raise ValueError('提示词包含词表以外的 ID。')

    # [G1] 独立 RNG：只供生成抽样，不调用全局 torch.manual_seed。
    # sample_rng.manual_seed 只重置这个局部对象，训练 Dropout 的随机序列不受影响。
    sample_rng = torch.Generator(device=device).manual_seed(seed)
    # 同一提示词复制 S 条：[T0] -> [S,T0]。S 是生成条数，与训练的 B 无关。
    ids = prompt.unsqueeze(0).repeat(num_sequences, 1)
    prompt_length = ids.size(1)
    output_length = min(prompt_length + max_new_tokens, config.n_positions)
    steps = []

    # [G2] 暂时切到评估模式，关闭 Dropout；保存模式以便退出时恢复。
    # 这是同一个对象、同一份参数。eval() 不等于加载另一个“推理模型”。
    was_training = raw_model.training
    raw_model.eval()
    try:
        # no_grad 不记录本次前向的反向图；已有 .grad 和 AdamW 状态保持原样。
        with torch.no_grad():
            while ids.size(1) < output_length:
                # [G3] 完整前缀前向：[S,T] -> [S,T,V] -> 最后位置 [S,V]。
                # 本项目 forward 只返回 logits；不传 y，也不计算交叉熵。
                logits = raw_model(ids)
                last_logits = logits[:, -1, :]
                if not bool(torch.isfinite(last_logits).all()):
                    raise ValueError('生成分数含 NaN 或无穷大。')

                # [G4] 按原课先 softmax，再取 top-k。multinomial 内部按权重
                # 抽样，不要求截取后的这 k 个概率之和等于 1。
                probs = F.softmax(last_logits, dim=-1)          # [S,V]
                top_probs, top_ids = torch.topk(probs, top_k, dim=-1)  # [S,k]
                selected = torch.multinomial(top_probs, 1, generator=sample_rng)  # [S,1]
                next_ids = torch.gather(top_ids, -1, selected) # top-k 内下标 -> 词表 ID

                # [G5] 追加 ID 后重新前向：上下文在增长，参数没有更新。
                # Token ID 一直留在张量中，不需要每轮解码成文字再重新分词。
                input_length = ids.size(1)
                ids = torch.cat((ids, next_ids), dim=1)         # [S,T+1]
                steps.append({'input_length': input_length,
                              'logits_shape': list(logits.shape),
                              'next_token_ids': next_ids[:, 0].tolist()})
                del logits, last_logits, probs, top_probs, top_ids, next_ids
    finally:
        # [G6] 即使生成报错也恢复模式；不清梯度、不重新初始化、不更新参数。
        raw_model.train(was_training)

    return {'token_ids': ids.tolist(), 'steps': steps,
            'new_tokens_per_sequence': ids.size(1) - prompt_length,
            'stop_reason': ('context_limit' if output_length < prompt_length + max_new_tokens
                            else 'max_new_tokens')}


def train_with_generation(model, raw_model, optimizer, train_loader, val_loader, runtime,
                          *, steps, accum_steps, precision, max_lr, min_lr,
                          warmup_steps, decay_steps, clip_norm,
                          validation_every, validation_batches, tokenizer, prompt_ids,
                          generation_every=2, max_new_tokens=24, num_sequences=4,
                          top_k=50, sample_seed=42):
    """保留第 13 块训练与验证，增加定期生成；旧文件不变。

    generation_every 按已完成的参数更新次数触发，与生成 Token 的循环分开。
    不在随机初值时生成；在间隔和最终更新后生成，用于本地短跑观察。
    """
    if min(steps, accum_steps, validation_every, validation_batches) < 1:
        raise ValueError('训练次数、累积次数和验证间隔/批次数必须为正。')
    if min(generation_every, max_new_tokens, num_sequences, top_k) < 1:
        raise ValueError('生成间隔、长度、条数和 top-k 必须为正。')
    train_records, val_records, sample_records = [], [], []
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

    def record_generation(completed_updates):
        # 使用刚训练到的参数；各 rank 使用自己的本体，seed=固定基数+rank。
        # 不调用 train_loader / val_loader，所以它们的取数游标不会前进。
        result = sample_current_model(
            raw_model, prompt_ids, max_new_tokens=max_new_tokens,
            num_sequences=num_sequences, top_k=top_k, seed=sample_seed + runtime.rank)
        result.update(after_updates=completed_updates, sample_seed=sample_seed + runtime.rank,
                      texts=[tokenizer.decode(ids) for ids in result['token_ids']])
        sample_records.append(result)
        if runtime.rank == 0:
            print(f'生成 · 已更新 {completed_updates:02d} 次 · 当前参数 · 不反传、不更新', flush=True)
            for index, text in enumerate(result['texts'], start=1):
                print(f'  样本 {index}: {text!r}', flush=True)

    # [T1] 参数尚未更新时记录验证基线。原课跳过第 0 次生成，因为几乎是噪声。
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
        # [T4] 只在完成一次 optimizer.step 之后检查是否生成；每张卡独立生成。
        # raw_model 内部已恢复 train 模式，下一轮照常取新的训练 batch。
        if (step + 1) % generation_every == 0 or step + 1 == steps:
            record_generation(step + 1)
    return {'training': train_records, 'validation': val_records, 'generation': sample_records}


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
    parser.add_argument('--sample-every', type=int, default=2,
                        help='每几次参数更新后生成；也在最终更新后生成，跳过初始模型。')
    parser.add_argument('--prompt', default=DEFAULT_PROMPT)
    parser.add_argument('--num-sequences', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=24)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--sample-seed', type=int, default=42,
                        help='独立的采样种子；每张卡实际使用 sample-seed + rank。')
    parser.add_argument('--warmup-steps', type=int, default=1)
    parser.add_argument('--decay-steps', type=int, default=3)
    parser.add_argument('--max-lr', type=float, default=6e-4)
    parser.add_argument('--min-lr', type=float, default=6e-5)
    parser.add_argument('--clip-norm', type=float, default=1.0)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--report', type=Path,
                        default=PROJECT_DIR / '.cache/runtime-check/lesson14-report.json')
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.total_tokens, args.steps, args.threads,
           args.val_every, args.val_batches, args.sample_every, args.num_sequences,
           args.max_new_tokens, args.top_k) < 1:
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
        prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
        if not 1 <= len(prompt_ids) < config.n_positions or args.top_k > config.vocab_size:
            parser.error('提示词须非空并为生成留出位置；top-k 不能超过词表大小。')
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
            print('知识块 14：训练中定期生成；本机数据改编为 Tiny Shakespeare 90%/10%。')
            print(f'原始位置 train=[0,{split})，val=[{split},{len(token_ids)})；'
                  f'B={args.batch_size}, T={args.seq_len}, R={runtime.world_size}, K={K}。')
            print(f'随机初始化 {args.model_size}；{runtime.device}/{args.precision}；'
                  '验证与生成只观察，学习率按更新次数调度。', flush=True)

        # [M3] 所有进程同时进入，验证和训练使用同一份当前模型参数。
        records = train_with_generation(
            model, raw_model, optimizer, train_loader, val_loader, runtime,
            steps=args.steps, accum_steps=K, precision=args.precision,
            max_lr=args.max_lr, min_lr=args.min_lr, warmup_steps=args.warmup_steps,
            decay_steps=args.decay_steps, clip_norm=args.clip_norm,
            validation_every=args.val_every, validation_batches=args.val_batches,
            tokenizer=tokenizer, prompt_ids=prompt_ids, generation_every=args.sample_every,
            max_new_tokens=args.max_new_tokens, num_sequences=args.num_sequences,
            top_k=args.top_k, sample_seed=args.sample_seed)
        ranks = gather_records({'rank': runtime.rank, **records}, runtime)
        if runtime.rank == 0:
            report = {
                'lesson': '阶段二 · 单元 2 · 2.4 · 知识块 14', 'course_commit': COURSE_COMMIT,
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
                'generation': {'every_updates': args.sample_every, 'prompt': args.prompt,
                               'prompt_ids': prompt_ids, 'num_sequences': args.num_sequences,
                               'max_new_tokens': args.max_new_tokens, 'top_k': args.top_k,
                               'base_seed': args.sample_seed, 'precision': 'fp32',
                               'temperature': 1.0, 'stops_on_eos': False,
                               'generated_text_used_for_training': False},
                'schedule': {'max_lr': args.max_lr, 'min_lr': args.min_lr,
                             'warmup_steps': args.warmup_steps, 'decay_steps': args.decay_steps},
                'clip_norm': args.clip_norm, 'ranks': ranks,
                'scope': 'Periodic-generation flow check; no text-quality, CUDA performance or full pretraining claim.',
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
