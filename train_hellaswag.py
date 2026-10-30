"""2.5 / 知识块 15：在分片训练中定期做 HellaSwag 风格四选一评测。

新增入口，保留 train_shards.py 及所有旧代码。默认评测真实 HellaSwag 验证集
前 8 题的本地预览；--hellaswag-file 可传入完整验证集 JSONL。
参数仍随机初始化；默认 GPT-2 124M，tiny 用于本机流程验证。
阅读 M1—M4 → T1—T7 → hellaswag_eval.py 的 H1—H4。详见 lesson15.md。
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
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
    from train_ddp import (accumulation_steps, configure_optimizer, gather_records,
                           learning_rate_at, setup_runtime)
    from train_generation import DEFAULT_PROMPT, sample_current_model
    from train_initialization import initialize_like_course
    from train_mixed_precision import forward_and_loss, validate_amp
    from train_step import DEFAULT_MODEL_DIR, PROJECT_DIR
    from train_shards import (make_training_demo_shards, make_shard_loaders,
                              loader_cursor, evaluate_shards)
    from hellaswag_eval import DEFAULT_QUESTIONS, load_questions, evaluate_hellaswag
    from train_weight_tying import tie_weights_like_course
except ImportError as error:
    raise SystemExit(f'缺少课程依赖：{error}\n当前解释器：{sys.executable}\n'
                     '本机请使用 /Users/guqun/micrograd/venv/bin/python。') from error


COURSE_COMMIT = '8018ed2c82c78a0600b2f7c66b1e606a93c60584'


def train_with_hellaswag(model, raw_model, optimizer, train_loader, val_loader, runtime,
                      *, steps, accum_steps, precision, max_lr, min_lr, warmup_steps,
                      decay_steps, clip_norm, validation_every, validation_batches,
                      tokenizer, prompt_ids, generation_every=2, max_new_tokens=24,
                      num_sequences=4, top_k=50, sample_seed=42,
                      questions, hellaswag_every=2, question_source="teaching demo"):
    """保留训练/验证/生成三条分支，再增加 T7 四选一评测；steps 数 optimizer.step，不数分片或 batch。

    一次参数更新 = K 个微批次的前向/反向 + 一次裁剪/调度/AdamW。
    分片换了不重建模型、不重建优化器、不清 AdamW 历史，也不重置学习率。
    """
    if min(steps, accum_steps, validation_every, validation_batches, generation_every,
           max_new_tokens, num_sequences, top_k, hellaswag_every) < 1:
        raise ValueError('训练/验证/生成次数与尺寸必须为正。')
    training, validation, generation, hellaswag = [], [], [], []
    model.train()

    def record_validation(completed):
        before = loader_cursor(train_loader)
        result = evaluate_shards(model, val_loader, runtime,
                                 validation_batches=validation_batches, precision=precision)
        result.update(after_updates=completed, train_cursor_before=before,
                      train_cursor_after=loader_cursor(train_loader))
        validation.append(result)
        if runtime.rank == 0:
            print(f"验证 · 已更新 {completed:02d} 次 · val_loss={result['global_mean_loss']:.6f} · "
                  f"全局 {result['global_predicted_positions']} 个预测位置", flush=True)

    def record_hellaswag(completed):
        # T7：当前参数；传 raw_model，使题数不等的 rank 不在前向广播 buffer。
        before = {'train': loader_cursor(train_loader), 'val': loader_cursor(val_loader)}
        result = evaluate_hellaswag(raw_model, questions, runtime, precision=precision)
        result.update(after_updates=completed, cursors_before=before, cursors_after={
            'train': loader_cursor(train_loader), 'val': loader_cursor(val_loader)})
        hellaswag.append(result)
        if runtime.rank == 0:
            print(f"四选一评测 · {question_source} · 已更新 {completed:02d} 次 · "
                  f"{result['global_correct']}/{result['global_total']}="
                  f"{result['accuracy']:.2%} · 不反传、不更新", flush=True)

    # [T1] 更新 0 次时先记验证基线；不会推进 train_loader。
    record_validation(0)
    record_hellaswag(0)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)  # 一次累积窗口开始时才清梯度。
        loss_accum = torch.zeros((), device=runtime.device,
                                 dtype=next(raw_model.parameters()).dtype)
        batches = []
        for micro_step in range(accum_steps):
            # [T2] 只有数据入口换成分片。取得 x/y 后仍是之前的 [B,T] long Tensor。
            batches.append(loader_cursor(train_loader))
            x, y = train_loader.next_batch()
            x, y = x.to(runtime.device), y.to(runtime.device)
            # [T3] 前 K-1 次可只累积；最后一次 backward 才让 DDP 同步梯度。
            # no_sync 必须包住前向和反向。单进程 nullcontext 什么也不改变。
            sync_context = (model.no_sync() if runtime.distributed
                            and micro_step < accum_steps - 1 else nullcontext())
            with sync_context:
                logits, loss = forward_and_loss(model, x, y, runtime.device, precision)
                scaled_loss = loss / accum_steps
                loss_accum += scaled_loss.detach()  # 记录指标，与参数 .grad 的累积不同。
                scaled_loss.backward()
            del logits, loss, scaled_loss
        if runtime.distributed:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
            loss_accum /= runtime.world_size
        if not torch.isfinite(loss_accum).item():
            raise RuntimeError('训练 loss 出现 NaN/Inf，停止更新。')
        # [T4] 同步后的梯度 → 裁剪 → 设置本次 lr → AdamW 更新一次。
        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), clip_norm,
                                              error_if_nonfinite=True)
        lr = learning_rate_at(step, max_lr, min_lr, warmup_steps, decay_steps)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        record = {'step': step + 1, 'local_batches': batches,
                  'next_train_cursor': loader_cursor(train_loader),
                  'global_mean_loss': loss_accum.item(), 'learning_rate': lr,
                  'gradient_norm_before_clip': norm.item()}
        training.append(record)
        if runtime.rank == 0:
            positions = ', '.join(f"{b['file']}:{b['start']}" for b in batches)
            print(f"训练 · 更新 {step + 1:02d} · loss={record['global_mean_loss']:.6f} · lr={lr:.6g}",
                  flush=True)
            print(f'  本次累积使用：{positions}', flush=True)
        # [T5] 间隔/末次验证只推进 val_loader；下一轮训练接着原来的 train 游标。
        if (step + 1) % validation_every == 0 or step + 1 == steps:
            record_validation(step + 1)
        # [T7] 评测位于完整参数更新之后、下一次训练累积之前；末次必评。
        if (step + 1) % hellaswag_every == 0 or step + 1 == steps:
            record_hellaswag(step + 1)
        # [T6] 生成不取任何分片。复用第 14 块的当前参数、独立 RNG 与模式恢复。
        if (step + 1) % generation_every == 0 or step + 1 == steps:
            before = {'train': loader_cursor(train_loader), 'val': loader_cursor(val_loader)}
            result = sample_current_model(raw_model, prompt_ids, max_new_tokens=max_new_tokens,
                                          num_sequences=num_sequences, top_k=top_k,
                                          seed=sample_seed + runtime.rank)
            result.update(after_updates=step + 1, sample_seed=sample_seed + runtime.rank,
                          texts=[tokenizer.decode(ids) for ids in result['token_ids']],
                          cursors_before=before, cursors_after={
                              'train': loader_cursor(train_loader), 'val': loader_cursor(val_loader)})
            generation.append(result)
            if runtime.rank == 0:
                print(f'生成 · 已更新 {step + 1:02d} 次 · 当前参数 · 不取训练/验证数据', flush=True)
                for index, text in enumerate(result['texts'], start=1):
                    print(f'  样本 {index}: {text!r}', flush=True)
    return {'training': training, 'validation': validation, 'generation': generation,
            'hellaswag': hellaswag}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, help='已有 edufineweb_train/val_*.npy 目录')
    parser.add_argument('--hellaswag-file', type=Path, default=DEFAULT_QUESTIONS,
                        help='带 ctx/endings/label 的 JSONL；默认真实 HellaSwag 验证集前 8 题')
    parser.add_argument('--hellaswag-limit', type=int, default=8,
                        help='每轮固定取前 N 题；0 表示文件中的全部题目，全局共享上限')
    parser.add_argument('--hellaswag-every', type=int, default=2,
                        help='每 N 次参数更新评测；另外初始和末次也评测')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--precision', choices=('auto', 'fp32', 'bf16'), default='auto')
    parser.add_argument('--model-size', choices=('gpt2', 'tiny'), default='gpt2')
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seq-len', type=int, default=3)
    parser.add_argument('--total-tokens', type=int, default=12,
                        help='每次参数更新的全局预测位置数，=B*T*R*K；不包括验证/生成。')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--val-every', type=int, default=2)
    parser.add_argument('--val-batches', type=int, default=2)
    parser.add_argument('--sample-every', type=int, default=2)
    parser.add_argument('--prompt', default=DEFAULT_PROMPT)
    parser.add_argument('--num-sequences', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=24)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--sample-seed', type=int, default=42)
    parser.add_argument('--warmup-steps', type=int, default=1)
    parser.add_argument('--decay-steps', type=int, default=3)
    parser.add_argument('--max-lr', type=float, default=6e-4)
    parser.add_argument('--min-lr', type=float, default=6e-5)
    parser.add_argument('--clip-norm', type=float, default=1.0)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--report', type=Path,
                        default=PROJECT_DIR / '.cache/runtime-check/lesson15-report.json')
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.total_tokens, args.steps, args.threads,
           args.val_every, args.val_batches, args.sample_every, args.num_sequences,
           args.max_new_tokens, args.top_k, args.hellaswag_every) < 1:
        parser.error('尺寸、次数与线程数必须为正。')
    if not 0 <= args.warmup_steps < args.decay_steps:
        parser.error('调度要求 0 <= warmup-steps < decay-steps。')
    if (not all(math.isfinite(v) for v in (args.max_lr, args.min_lr, args.clip_norm))
            or not 0 < args.min_lr <= args.max_lr or args.clip_norm <= 0):
        parser.error('学习率和裁剪上限必须为有限正数，且 min-lr <= max-lr。')
    if args.hellaswag_limit < 0:
        parser.error('hellaswag-limit 不能为负；0 表示全部题目。')
    torch.set_num_threads(args.threads)
    runtime = setup_runtime(args.device)
    try:
        K = accumulation_steps(args.total_tokens, args.batch_size, args.seq_len, runtime.world_size)
        if args.precision == 'auto':
            args.precision = ('bf16' if runtime.device.type == 'cuda'
                              and torch.cuda.is_bf16_supported() else 'fp32')
        validate_amp(runtime.device, args.precision)
        # [M1] 正式数据直接传目录；默认教学数据只由 rank 0 写一次，再通知其他进程。
        if args.data_dir is None:
            prepared = [None, None]  # [路径, 错误]；让其他 rank 也能收到准备失败信息。
            if runtime.rank == 0:
                try:
                    prepared[0] = str(make_training_demo_shards())
                except (OSError, ValueError) as error:
                    prepared[1] = str(error)
            if runtime.distributed:
                dist.broadcast_object_list(prepared, src=0)
            if prepared[1]:
                raise ValueError(prepared[1])
            directory = Path(prepared[0])
        else:
            directory = args.data_dir.resolve()
        train_loader, val_loader, inventory = make_shard_loaders(
            directory, args.batch_size, args.seq_len, runtime)
        if args.val_batches > inventory['val']['batches_per_rank_per_pass']:
            raise ValueError('验证批数超过所有 val 分片一轮能提供的批数；请减小 val-batches/B/T。')

        # [M2] 模型和优化器仅创建一次。np.load 得到的数据 ID 不会成为随机参数。
        # 配置/分词器从本地缓存读；没有 from_pretrained 或 load_hf_weights。
        model_dir = args.model_dir.resolve()
        tokenizer = GPT2TokenizerFast(tokenizer_file=str(model_dir / 'tokenizer.json'))
        config = GPTConfig.from_hf_config(json.loads((model_dir / 'config.json').read_text()))
        config.embd_pdrop = config.attn_pdrop = config.resid_pdrop = 0.0
        if args.model_size == 'tiny':
            config = replace(config, n_layer=2, n_embd=64, n_head=2, n_positions=1024)
        if config.vocab_size != 50257 or len(tokenizer) != config.vocab_size:
            raise ValueError('本分片入口要求 GPT-2 的 50257 项词表及配套 Token ID。')
        if args.seq_len > config.n_positions:
            raise ValueError('seq-len 超过模型位置表。')
        prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
        if not 1 <= len(prompt_ids) < config.n_positions or args.top_k > config.vocab_size:
            raise ValueError('提示词须为生成留出位置，top-k 不能超过词表大小。')
        # [M2/H1] 全部 rank 先验证同一题集；超长或无标签直接报错，不悄悄跳题。
        questions, question_metadata = load_questions(
            args.hellaswag_file, tokenizer, limit=args.hellaswag_limit,
            max_length=config.n_positions, vocab_size=config.vocab_size)
        question_source = ('教学题，非 HellaSwag' if question_metadata['is_teaching_demo']
                           else 'HellaSwag 真实验证题预览，非完整基准'
                           if question_metadata['is_bundled_preview']
                           else '外部 JSONL，来源由提供者确认')
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
        source = ('two distinct teaching sentences; not FineWeb' if args.data_dir is None
                  else 'user-supplied GPT-2 shards; source not inferred from filenames')
        if runtime.rank == 0:
            print('2.5 / 知识块 15：分片训练 + 定期四选一评测。')
            print(f"评测：{question_source}；每轮 {len(questions)} 题；评分仅用候选结尾。")
            print('数据：' + ('教学文本；train=Hello world.，val=Good morning.；不是 FineWeb。'
                            if args.data_dir is None else str(directory)))
            print(f'随机初始化 {args.model_size}；{runtime.device}/{args.precision}；'
                  f'B={args.batch_size}, T={args.seq_len}, R={runtime.world_size}, K={K}。', flush=True)
        # [M3] 主循环始终复用上面这两个加载器、同一模型和同一 AdamW。
        records = train_with_hellaswag(
            model, raw_model, optimizer, train_loader, val_loader, runtime,
            steps=args.steps, accum_steps=K, precision=args.precision,
            max_lr=args.max_lr, min_lr=args.min_lr, warmup_steps=args.warmup_steps,
            decay_steps=args.decay_steps, clip_norm=args.clip_norm,
            validation_every=args.val_every, validation_batches=args.val_batches,
            tokenizer=tokenizer, prompt_ids=prompt_ids, generation_every=args.sample_every,
            max_new_tokens=args.max_new_tokens, num_sequences=args.num_sequences,
            top_k=args.top_k, sample_seed=args.sample_seed,
            questions=questions, hellaswag_every=args.hellaswag_every,
            question_source=question_source)
        ranks = gather_records({'rank': runtime.rank, **records}, runtime)
        # [M4] 保存观察报告；当前入口没有保存模型检查点或训练恢复逻辑。
        if runtime.rank == 0:
            report = {
                'lesson': '2.5 / periodic multiple-choice evaluation', 'course_commit': COURSE_COMMIT,
                'torch': torch.__version__, 'device_type': runtime.device.type,
                'backend': runtime.backend, 'world_size': runtime.world_size,
                'precision': args.precision, 'execution': 'eager', 'seed': args.seed,
                'model_size': args.model_size, 'config': asdict(config),
                'parameter_count': sum(p.numel() for p in raw_model.parameters()),
                'pretrained_weights_loaded': False, 'checkpoint_saved': False,
                'data_dir': str(directory.resolve()), 'data_source': source, 'inventory': inventory,
                'batch_size_per_rank': args.batch_size, 'seq_len': args.seq_len,
                'accumulation_steps': K, 'tokens_per_optimizer_update': args.total_tokens,
                'parameter_updates': args.steps, 'training_positions': args.steps * args.total_tokens,
                'validation_every': args.val_every, 'validation_batches_per_rank': args.val_batches,
                'validation_changes_training_policy': False,
                'hellaswag': {**question_metadata, 'every_updates': args.hellaswag_every,
                              'precision': args.precision, 'changes_training_policy': False},
                'generation': {'every_updates': args.sample_every, 'prompt': args.prompt,
                               'prompt_ids': prompt_ids, 'num_sequences': args.num_sequences,
                               'max_new_tokens': args.max_new_tokens, 'top_k': args.top_k,
                               'base_seed': args.sample_seed, 'precision': 'fp32',
                               'generated_text_used_for_training': False},
                'schedule': {'max_lr': args.max_lr, 'min_lr': args.min_lr,
                             'warmup_steps': args.warmup_steps, 'decay_steps': args.decay_steps},
                'clip_norm': args.clip_norm, 'ranks': ranks,
                'scope': 'Short control-flow check on selected questions; no full benchmark, FineWeb training or CUDA claim.',
            }
            path = args.report.resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print(f'报告：{path}', flush=True)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    finally:
        if runtime.distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
