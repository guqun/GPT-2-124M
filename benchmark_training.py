"""阶段二 · 单元 2 · 2.2 · 知识块 8：GPU 执行、训练计时与 TF32。

阅读顺序：main() 的 M1—M4 -> benchmark_steps() 的 B1—B4。
来源：Karpathy 视频约 1:22—1:39，历史提交 5265b20；详见 lesson08.md。
复用上一块的模型、共享权重与初始化；仍从随机参数开始，不读取预训练权重。
默认完整 124M、B=4/T=32，2 步预热 + 5 步测量，便于当前 CPU 跟读。
GPU 上显式设置 --device cuda；--matmul-precision high 用于课程 TF32 对照。
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
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
    from train_step import (
        DATA_SHA256, DEFAULT_DATA_FILE, DEFAULT_MODEL_DIR, PROJECT_DIR,
    )
except ImportError as error:
    project = Path(__file__).resolve().parent
    candidates = (project / '.venv/bin/python', project.parent / 'venv/bin/python')
    interpreter = next((p for p in candidates if p.is_file()), candidates[0])
    raise SystemExit(
        f'缺少课程依赖：{error}\n当前解释器：{sys.executable}\n'
        f'请使用课程环境：{interpreter}'
    ) from error


COURSE_COMMIT = '5265b2042aaaa84943313002671b8b3ca7749b06'


def select_device(requested):
    """本项目选择规则：auto 优先 CUDA，其次 MPS，最后 CPU；显式指定不静默降级。"""
    cuda_ok = torch.cuda.is_available()
    mps_ok = torch.backends.mps.is_available()
    if requested == 'auto':
        return torch.device('cuda' if cuda_ok else 'mps' if mps_ok else 'cpu')
    if requested == 'cuda' and not cuda_ok:
        raise ValueError('当前 PyTorch 无可用 CUDA 设备；本机可用 --device cpu 跟读。')
    if requested == 'mps' and not mps_ok:
        raise ValueError('当前 PyTorch 无可用 MPS 设备；本机可用 --device cpu 跟读。')
    if requested not in ('cpu', 'cuda', 'mps'):
        raise ValueError(f'不支持的设备：{requested}')
    return torch.device(requested)


def configure_precision(device, precision):
    """设置矩阵乘法内部精度；参数和激活仍是 FP32，Token ID 仍是整数。

    本课 high 对照限定在支持 TF32 的 NVIDIA CUDA 设备；CPU/MPS 不冒充 TF32。
    high 是给后端的精度许可，不是把 Tensor 转成名为 tf32 的存储 dtype。
    """
    if precision not in ('highest', 'high'):
        raise ValueError('本节仅对照 highest 与 high。')
    if precision == 'high':
        if device.type != 'cuda':
            raise ValueError('本课 TF32 对照需要 CUDA；CPU/MPS 请使用 highest。')
        if torch.cuda.get_device_capability(device)[0] < 8:
            raise ValueError('本课 TF32 对照需要支持 TF32 的 NVIDIA GPU（计算能力至少 8.0）。')
    # 【PyTorch】highest 使用 FP32 内部精度；high 允许更低的内部计算精度。
    # 原课 A100 使用 TF32；具体内核由 GPU/PyTorch 后端选择，dtype 不会因此变化。
    torch.set_float32_matmul_precision(precision)


def synchronize(device):
    """等待所选加速设备完成已提交的工作；CPU 路径不调用 GPU API。"""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def summarize_timings(records, tokens_per_step):
    """只汇总正式测量步；吞吐率用总 Token 数 / 总时间，不能平均各步速率。"""
    times = [r['seconds'] for r in records if r['phase'] == 'measure']
    if not times or any(not math.isfinite(t) or t <= 0 for t in times):
        raise ValueError('需要至少一个耗时为有限正数的测量步骤。')
    elapsed = sum(times)
    return {
        'measured_steps': len(times), 'measured_tokens': len(times) * tokens_per_step,
        'total_measured_seconds': elapsed,
        'mean_step_ms': statistics.mean(times) * 1000,
        'median_step_ms': statistics.median(times) * 1000,
        'tokens_per_second': len(times) * tokens_per_step / elapsed,
    }


def benchmark_steps(model, optimizer, loader, device, warmup_steps, measured_steps):
    """输入已有模型、优化器和加载器；连续训练，返回各步耗时及 loss。

    预热也做真实更新和取下一批，只是不加入最终速度统计。这里不重新初始化。
    计时包含取 batch、传到设备、清梯度、前向/loss、反向、更新与末尾等待；
    不含文件读取、分词、模型创建、日志打印和报告写入。
    """
    if warmup_steps < 0 or measured_steps < 1:
        raise ValueError('预热步数不能为负，测量步数至少为 1。')
    model.train()
    records = []
    for index in range(warmup_steps + measured_steps):
        phase = 'warmup' if index < warmup_steps else 'measure'
        start = loader.current_position
        # [B1] 先排空前面排队的 GPU 工作，再开始计时，避免把上一轮工作算进来。
        synchronize(device)
        t0 = time.perf_counter()  # 【Python】单调高精度时钟，用于计算时间差。

        # [B2] 【本项目】CPU 上取下一批；【PyTorch】将 x/y 放到模型所在设备。
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        # [B3] 与上一节相同的训练计算；本节没有增加新的学习算法。
        logits = model(x)
        # 仍是全位置交叉熵。此处直接调用 F，避免旧教学包装中的
        # targets.min().item()/max().item() 在 GPU 计时中额外同步；ID 范围在 M2 检查。
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()

        # [B4] CPU 提交完命令不代表 GPU 已算完。必须等完成后再读结束时间。
        synchronize(device)
        seconds = time.perf_counter() - t0
        # item()/打印放在计时之后；GPU 上 item 本身也可能引起同步。
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError('出现非有限 loss，停止实验；不把本轮写成成功的速度结果。')
        record = {
            'update': index + 1, 'phase': phase, 'batch_start': start,
            'loss_before_update': loss_value, 'seconds': seconds,
            'tokens_per_second': loader.B * loader.T / seconds,
        }
        records.append(record)
        label = '预热' if phase == 'warmup' else '测量'
        print(f'{label} · 更新 {index + 1:02d} · loss={loss_value:.6f} · '
              f'{seconds * 1000:.2f} ms · {record["tokens_per_second"]:.1f} Token/s', flush=True)
        del logits, loss  # 不保留上一轮的整份 logits/计算图。
    return records, summarize_timings(records, loader.B * loader.T)


def main():
    # [M1] 先确定运行条件；各组对照要固定 B/T、种子、数据、预热与测量步数。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    parser.add_argument('--matmul-precision', choices=('highest', 'high'), default='highest')
    parser.add_argument('--inspect-only', action='store_true', help='只检查设备，不创建模型或训练。')
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--data-file', type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=32)
    parser.add_argument('--warmup-steps', type=int, default=2)
    parser.add_argument('--steps', type=int, default=5, help='正式测量步数，不含预热。')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--report', type=Path,
                        default=PROJECT_DIR / '.cache/runtime-check/lesson08-report.json')
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.steps, args.threads) < 1 or args.warmup_steps < 0:
        parser.error('B、T、测量步数、线程数必须为正整数；预热步数不能为负。')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('学习率必须是有限正数。')
    try:
        device = select_device(args.device)
        configure_precision(device, args.matmul_precision)
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(args.threads)
    environment = {
        'python': platform.python_version(), 'torch': torch.__version__,
        'system': platform.system(), 'machine': platform.machine(),
        'device': str(device), 'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available(),
        'threads': torch.get_num_threads(), 'cuda_runtime': torch.version.cuda,
        'matmul_precision': torch.get_float32_matmul_precision(),
    }
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        environment.update(gpu_name=props.name, device_memory_bytes=props.total_memory,
                           compute_capability=[props.major, props.minor],
                           allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    print('阶段二 · 单元 2 · 2.2 · 知识块 8：GPU 执行、训练计时与 TF32')
    print(json.dumps(environment, ensure_ascii=False, indent=2))
    if args.inspect_only:
        print('只检查设备；未创建模型或执行训练。')
        return

    # [M2] 文件读取、分词、创建模型都在计时区间之外。只读取配置与 tokenizer。
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
    token_ids = tokenizer.encode(raw.decode('utf-8'), add_special_tokens=False)
    try:
        loader = DataLoaderLite(token_ids, args.batch_size, args.seq_len)
    except ValueError as error:
        parser.error(str(error))
    if loader.tokens.min().item() < 0 or loader.tokens.max().item() >= config.vocab_size:
        parser.error('语料中存在超出模型词表的 ID。')
    torch.manual_seed(args.seed)
    model = GPT(config, tie_embeddings=False, gpt2_init=False).cpu().float()
    tie_weights_like_course(model)
    initialize_like_course(model)

    # [M3] 【PyTorch】把参数与注册 buffer 放到设备，之后再创建优化器。
    model.to(device)
    # 设备转换可能替换 Parameter；重新接好共享引用，不重新随机初始化。
    tie_weights_like_course(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    # [M4] 预热与测量使用连续的参数和 batch；输出训练耗时，不是推理生成速度。
    records, summary = benchmark_steps(model, optimizer, loader, device, args.warmup_steps, args.steps)
    report = {
        'lesson': '阶段二 · 单元 2 · 2.2 · 知识块 8', 'course_commit': COURSE_COMMIT,
        'environment': environment, 'config': asdict(config), 'seed': args.seed,
        'parameter_dtype': str(next(model.parameters()).dtype),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'sharing': {
            'same_parameter': model.transformer.wte.weight is model.lm_head.weight,
            'optimizer_occurrences': sum(p is model.transformer.wte.weight
                for group in optimizer.param_groups for p in group['params']),
        },
        'data': {'file': str(data_file), 'sha256': data_sha, 'encoded_tokens': len(token_ids)},
        'batch_size': args.batch_size, 'seq_len': args.seq_len,
        'warmup_steps': args.warmup_steps, 'measured_steps': args.steps,
        'optimizer': {'name': 'AdamW', 'lr': args.learning_rate,
                      'betas': list(optimizer.defaults['betas']),
                      'weight_decay': optimizer.defaults['weight_decay'],
                      'eps': optimizer.defaults['eps'], 'foreach': optimizer.defaults.get('foreach')},
        'steps': records, 'summary': summary,
        'timing_scope': 'batch fetch + device transfer + zero_grad + forward/loss + backward + step + synchronize; excludes setup/logging',
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        'pretrained_weights_loaded': False, 'checkpoint_saved': False,
        'verification_scope': 'This run only; CPU timing is not evidence of CUDA/TF32 speedup. Changing-batch loss is not validation loss.',
    }
    path = args.report.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'排除预热后的汇总：{summary["mean_step_ms"]:.2f} ms/步；'
          f'{summary["tokens_per_second"]:.1f} Token/s。报告：{path}')


if __name__ == '__main__':
    main()
