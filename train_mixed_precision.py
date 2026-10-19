"""阶段二 · 单元 2 · 2.2 · 知识块 9：BF16 自动混合精度。

主线：main() 的 M1—M4 → train_amp_steps() 的 A1—A4。
与知识块 8 相比，核心变化只有 forward_and_loss() 中的 autocast。
模型、共享、初始化、数据加载和同步计时沿用前面的实现；旧脚本保留。
来源：Karpathy 视频约 1:39:28—1:47:34，历史提交 177e4cd；见 lesson09.md。
仍从随机参数训练，不加载预训练权重。默认完整 124M、B=4/T=32。
CPU 可验证 dtype 和更新流程；实际 GPU 加速必须在支持 BF16 的 CUDA 设备测量。
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

try:
    import torch
    from torch.nn import functional as F
    from transformers import GPT2TokenizerFast
    from gpt2_model import GPT, GPTConfig
    from train_dataloader import DataLoaderLite
    from train_initialization import initialize_like_course
    from train_weight_tying import tie_weights_like_course
    from train_step import DATA_SHA256, DEFAULT_DATA_FILE, DEFAULT_MODEL_DIR, PROJECT_DIR
    from benchmark_training import (
        configure_precision, select_device, summarize_timings, synchronize,
    )
except ImportError as error:
    project = Path(__file__).resolve().parent
    candidates = (project / '.venv/bin/python', project.parent / 'venv/bin/python')
    interpreter = next((p for p in candidates if p.is_file()), candidates[0])
    raise SystemExit(
        f'缺少课程依赖：{error}\n当前解释器：{sys.executable}\n'
        f'请使用课程环境：{interpreter}'
    ) from error


COURSE_COMMIT = '177e4cd5b4cc05df4bb637ed1a9e55911d6f1e2c'


def validate_amp(device, precision):
    """本实验支持 CPU/CUDA BF16；不把不支持的设备静默换成 FP16。

    CPU autocast 是本地观察 dtype 的补充路径，不代表具有原课 A100 的速度。
    当前课程 PyTorch 2.2.2 不提供 MPS BF16 autocast，本脚本明确限定实验范围。
    """
    if precision not in ('fp32', 'bf16'):
        raise ValueError('本节仅比较 fp32 与 bf16。')
    if precision == 'bf16':
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('本课 BF16 autocast 实验限定 CPU 或支持 BF16 的 CUDA 设备。')
        if device.type == 'cuda' and not torch.cuda.is_bf16_supported():
            raise ValueError('当前 CUDA 设备/环境不支持本课 BF16 实验。')


def forward_and_loss(model, x, y, device, precision):
    """本节核心：只让前向与 loss 进入自动混合精度区域。

    输入：FP32 参数的模型；x/y 是同设备的 int64 Token ID，形状 [B,T]。
    输出：logits [B,T,V] 和一个 loss 标量；具体 dtype 由算子的 autocast 规则决定。
    BF16 模式下，Linear 等运算使用较低精度，交叉熵保留 FP32。
    原 Parameter 不会被永久转成 BF16；也不把整数 ID 变成浮点数。
    """
    # 【Python】nullcontext 是不改变执行状态的 with 区域，供 FP32 对照使用。
    # 【PyTorch】autocast 按运算选择 dtype，不是给整个模型调用 .bfloat16()。
    context = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
               if precision == 'bf16' else nullcontext())
    with context:
        logits = model(x)
        # logits 可以是 BF16；在 autocast 内调用交叉熵，由 PyTorch 选择 FP32。
        # reshape 合并 B/T，V 个候选分数仍保留在最后一维。
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    # 离开 with 之后 logits 仍是已算出的 dtype，不会自动变回 FP32。
    return logits, loss


def describe_tensor(tensor):
    """辅助读数：只返回元信息，不保存计算图；字节数仅代表这一张量的逻辑大小。"""
    return {
        'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
        'element_bytes': tensor.element_size(),
        'logical_bytes': tensor.numel() * tensor.element_size(),
    }


def inspect_forward_dtypes(model, x, y, device, precision):
    """独立的观察模式：记录部分层输出，不更新参数，也不参与性能计时。

    hook 只读出 dtype/shape，返回 None，不修改层输出。退出时移除全部 hook。
    原有 model.training 状态会恢复。本项目所有模块原先使用同一 train/eval 状态。
    """
    observations, handles = [], []
    names = (
        'transformer.wte', 'transformer.wpe',
        'transformer.h.0.ln_1', 'transformer.h.0.attn.c_attn',
        'transformer.h.0.attn.c_proj', 'transformer.h.0.ln_2',
        'transformer.h.0.mlp.c_fc', 'transformer.h.0.mlp.c_proj',
        'transformer.h.0', 'transformer.ln_f', 'lm_head',
    )

    def make_hook(name):
        def record(module, inputs, output):
            observations.append({'name': name, **describe_tensor(output)})
        return record

    was_training = model.training
    try:
        for name in names:
            handles.append(model.get_submodule(name).register_forward_hook(make_hook(name)))
        model.eval()
        with torch.no_grad():
            logits, loss = forward_and_loss(model, x, y, device, precision)
        observations.append({'name': 'cross_entropy', **describe_tensor(loss)})
        result = {
            'precision': precision, 'input': describe_tensor(x),
            'parameter': describe_tensor(model.lm_head.weight),
            'observations': observations, 'loss': loss.item(),
        }
        # 只在独立观察模式下复制数值到 CPU，以比较同一输入的两种结果。
        return result, logits.detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)


def train_amp_steps(model, optimizer, loader, device, precision, warmup_steps, measured_steps):
    """同一套参数连续训练；autocast 之外执行 backward 和 optimizer.step。

    沿用知识块 8 的计时边界和预热规则。额外 dtype 读数在计时结束后记录。
    BF16 本节不使用 GradScaler；参数及其 .grad 保持 FP32。
    反向在 autocast 区域外执行，不等于所有反向中间运算都强制使用 FP32。
    """
    validate_amp(device, precision)
    if warmup_steps < 0 or measured_steps < 1:
        raise ValueError('预热步数不能为负，测量步数至少为 1。')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('本课使用 FP32 参数配合 autocast，请不要提前转换整个模型。')
    model.train()
    records = []
    shared_weight = model.lm_head.weight
    for index in range(warmup_steps + measured_steps):
        phase = 'warmup' if index < warmup_steps else 'measure'
        start = loader.current_position
        # [A1] 【沿用】等待、计时、取新 batch、放到设备与清梯度。
        synchronize(device)
        t0 = time.perf_counter()
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        # [A2] 【本节变化】这个函数只把前向和交叉熵包在 autocast 中。
        logits, loss = forward_and_loss(model, x, y, device, precision)

        # [A3] 【沿用】前向记录了实际使用的运算与转换，自动求导沿它们反传。
        # 叶子参数仍是 FP32，因此最终累积到这些参数的 .grad 也是 FP32。
        loss.backward()
        optimizer.step()

        # [A4] 【沿用】确认本步完成，再结束计时；读数和打印不放入计时区间。
        synchronize(device)
        seconds = time.perf_counter() - t0
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError('出现非有限 loss，停止本次实验。')
        state = optimizer.state[shared_weight]
        record = {
            'update': index + 1, 'phase': phase, 'batch_start': start,
            'loss_before_update': loss_value, 'seconds': seconds,
            'tokens_per_second': loader.B * loader.T / seconds,
            'input_dtype': str(x.dtype), 'parameter_dtype': str(shared_weight.dtype),
            'logits': describe_tensor(logits), 'loss_dtype': str(loss.dtype),
            'parameter_grad_dtype': str(shared_weight.grad.dtype),
            # AdamW 的一阶/二阶统计实际来自 FP32 参数，不是额外创建 BF16 主权重。
            'adam_m_dtype': str(state['exp_avg'].dtype),
            'adam_v_dtype': str(state['exp_avg_sq'].dtype),
        }
        records.append(record)
        label = '预热' if phase == 'warmup' else '测量'
        print(f'{label} · 更新 {index + 1:02d} · loss={loss_value:.6f} · '
              f'{seconds * 1000:.2f} ms · {record["tokens_per_second"]:.1f} Token/s', flush=True)
        if index == 0:
            print(f'  参数={shared_weight.dtype} · logits={logits.dtype} · '
                  f'loss={loss.dtype} · 参数梯度={shared_weight.grad.dtype}', flush=True)
        del logits, loss
    return records, summarize_timings(records, loader.B * loader.T)


def main():
    # [M1] 配置实验。本节 precision 选择 autocast；matmul-precision 沿用上一节，
    # 只控制仍采用 FP32 的矩阵乘法。做对照时，两次运行的这些条件要固定。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    parser.add_argument('--matmul-precision', choices=('highest', 'high'), default='highest')
    parser.add_argument('--probe-only', action='store_true',
                        help='同一参数/输入下比较 FP32 与 BF16 的 dtype；不反向、不更新、不测速。')
    parser.add_argument('--inspect-only', action='store_true', help='只检查运行环境。')
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--data-file', type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=32)
    parser.add_argument('--warmup-steps', type=int, default=2)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.steps, args.threads) < 1 or args.warmup_steps < 0:
        parser.error('B、T、测量步数、线程数必须为正整数；预热步数不能为负。')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('学习率必须为有限正数。')
    try:
        device = select_device(args.device)
        configure_precision(device, args.matmul_precision)
        validate_amp(device, 'bf16' if args.probe_only else args.precision)
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(args.threads)
    environment = {
        'python': platform.python_version(), 'torch': torch.__version__,
        'device': str(device), 'machine': platform.machine(), 'system': platform.system(),
        'cuda_available': torch.cuda.is_available(), 'mps_available': torch.backends.mps.is_available(),
        'cuda_runtime': torch.version.cuda, 'threads': torch.get_num_threads(),
        'matmul_precision': torch.get_float32_matmul_precision(),
    }
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        environment.update(gpu_name=props.name, bf16_supported=torch.cuda.is_bf16_supported(),
                           compute_capability=[props.major, props.minor],
                           allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    print('阶段二 · 单元 2 · 2.2 · 知识块 9：BF16 自动混合精度')
    print(json.dumps(environment, ensure_ascii=False, indent=2))
    if args.inspect_only:
        return
    if device.type == 'cpu':
        print('CPU 仅验证混合精度行为，BF16 可能更慢；不将此结果记为 GPU 加速。', flush=True)

    # [M2] 沿用相同配置/语料/随机起点，只读取配置和 tokenizer，不加载模型权重。
    model_dir, data_file = args.model_dir.expanduser().resolve(), args.data_file.expanduser().resolve()
    for path in (model_dir / 'config.json', model_dir / 'tokenizer.json', data_file):
        if not path.is_file():
            parser.error(f'缺少本地文件：{path}；准备方法见 README.md。')
    config = GPTConfig.from_hf_config(json.loads((model_dir / 'config.json').read_text()))
    config.embd_pdrop = config.attn_pdrop = config.resid_pdrop = 0.0
    if args.seq_len > config.n_positions:
        parser.error('T 超出模型的位置表长度。')
    raw = data_file.read_bytes()
    data_sha = hashlib.sha256(raw).hexdigest()
    if data_file == DEFAULT_DATA_FILE and data_sha != DATA_SHA256:
        parser.error('默认语料与固定课程版本不一致。')
    tokenizer = GPT2TokenizerFast(tokenizer_file=str(model_dir / 'tokenizer.json'))
    if len(tokenizer) != config.vocab_size:
        parser.error('分词器与模型词表大小不一致。')
    ids = tokenizer.encode(raw.decode('utf-8'), add_special_tokens=False)
    try:
        loader = DataLoaderLite(ids, args.batch_size, args.seq_len)
    except ValueError as error:
        parser.error(str(error))
    if loader.tokens.min().item() < 0 or loader.tokens.max().item() >= config.vocab_size:
        parser.error('语料中存在词表范围之外的 ID。')
    torch.manual_seed(args.seed)
    model = GPT(config, tie_embeddings=False, gpt2_init=False).cpu().float()
    tie_weights_like_course(model)
    initialize_like_course(model)
    model.to(device)
    tie_weights_like_course(model)
    report = {
        'lesson': '阶段二 · 单元 2 · 2.2 · 知识块 9', 'course_commit': COURSE_COMMIT,
        'mode': 'dtype_probe' if args.probe_only else 'training',
        'environment': environment, 'config': asdict(config), 'seed': args.seed,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'parameter_dtype': str(model.lm_head.weight.dtype),
        'shared_parameter': model.lm_head.weight is model.transformer.wte.weight,
        'data': {'file': str(data_file), 'sha256': data_sha, 'encoded_tokens': len(ids)},
        'batch_size': args.batch_size, 'seq_len': args.seq_len,
        'pretrained_weights_loaded': False, 'checkpoint_saved': False,
        'verification_scope': 'This run only. CPU results do not measure CUDA speedup. '
                              'Changing-batch training loss is not validation quality.',
    }
    # [M3] 观察类型是独立模式，避免 hook 和完整数值复制干扰正式性能计时。
    if args.probe_only:
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)
        fp32, reference = inspect_forward_dtypes(model, x, y, device, 'fp32')
        bf16, actual = inspect_forward_dtypes(model, x, y, device, 'bf16')
        delta = (actual - reference).abs()
        report.update(fp32=fp32, bf16=bf16,
                      logit_difference={'mean_abs': delta.mean().item(), 'max_abs': delta.max().item()},
                      parameter_updates=0)
        for left, right in zip(fp32['observations'], bf16['observations']):
            print(f'{left["name"]:<33} {left["dtype"]:<15} → {right["dtype"]}')
        print(f'同一输入的 loss：FP32={fp32["loss"]:.6f}；BF16={bf16["loss"]:.6f}。')
        print('本模式只观察前向结果，没有更新参数；张量逻辑字节数不是总显存。')
    else:
        # [M4] 优化器持有 FP32 参数；autocast 只在训练循环的前向/loss 区域开启。
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        records, summary = train_amp_steps(model, optimizer, loader, device, args.precision,
                                           args.warmup_steps, args.steps)
        report.update(
            precision=args.precision, warmup_steps=args.warmup_steps, measured_steps=args.steps,
            steps=records, summary=summary, parameter_updates=len(records),
            optimizer={'name': 'AdamW', 'lr': args.learning_rate,
                       'betas': list(optimizer.defaults['betas']),
                       'weight_decay': optimizer.defaults['weight_decay'],
                       'eps': optimizer.defaults['eps']},
            grad_scaler_used=False,
            peak_cuda_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                                       if device.type == 'cuda' else None),
            timing_scope='batch fetch + device transfer + zero_grad + forward/loss + backward '
                         '+ step + synchronize; excludes setup, probes, logging and reports',
        )
        print(f'排除预热：{summary["mean_step_ms"]:.2f} ms/步；'
              f'{summary["tokens_per_second"]:.1f} Token/s。')
    default_name = 'lesson09-dtypes.json' if args.probe_only else f'lesson09-{args.precision}.json'
    path = (args.report or PROJECT_DIR / '.cache/runtime-check' / default_name).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'报告：{path}')


if __name__ == '__main__':
    main()
