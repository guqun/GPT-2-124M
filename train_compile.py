"""阶段二 · 单元 2 · 2.2 · 知识块 10：在已有训练循环中使用 torch.compile。

阅读顺序：main() 的 C1 → C2 → C3 → C4；新增核心在 prepare_execution()。
沿用知识块 9 的 train_amp_steps / forward_and_loss，不改上一节文件。
参数仍随机初始化；配置和 tokenizer 来自本地缓存，不加载预训练权重。
原课来源：视频约 1:47:34—1:59:46，作者历史提交 fb8bd6e；见 lesson10.md。
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

try:
    import torch
    from transformers import GPT2TokenizerFast
    from gpt2_model import GPT, GPTConfig
    from train_dataloader import DataLoaderLite
    from train_initialization import initialize_like_course
    from train_weight_tying import tie_weights_like_course
    from train_step import DATA_SHA256, DEFAULT_DATA_FILE, DEFAULT_MODEL_DIR, PROJECT_DIR
    from benchmark_training import configure_precision, select_device
    from train_mixed_precision import train_amp_steps, validate_amp
except ImportError as error:
    project = Path(__file__).resolve().parent
    candidates = (project / '.venv/bin/python', project.parent / 'venv/bin/python')
    interpreter = next((p for p in candidates if p.is_file()), candidates[0])
    raise SystemExit(
        f'缺少课程依赖：{error}\n当前解释器：{sys.executable}\n请使用课程环境：{interpreter}'
    ) from error


COURSE_COMMIT = 'fb8bd6e'


def resolve_runtime(requested_device, requested_precision, execution):
    """把本课的设备/精度选项解析为实际配置。

    eager/compile 的 auto 都优先 CUDA，否则 CPU，保证默认对照条件相同。
    精度 auto 只在支持 BF16 的 CUDA 上选 BF16，其余使用 FP32。
    显式 MPS 保留给 eager；不将用户明确选择的设备或精度静默改掉。
    """
    if execution not in ('eager', 'compile'):
        raise ValueError('execution 只能是 eager 或 compile。')
    if requested_device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = select_device(requested_device)
    if execution == 'compile' and device.type not in ('cpu', 'cuda'):
        raise ValueError(
            '本课 PyTorch 2.2.2 的 Inductor 实验限定 CPU/CUDA；'
            '本机编译请使用 --device cpu --precision fp32，'
            'MPS 普通训练请使用 --device mps --precision fp32 --execution eager。'
        )
    if requested_precision == 'auto':
        precision = ('bf16' if device.type == 'cuda' and torch.cuda.is_bf16_supported()
                     else 'fp32')
    else:
        precision = requested_precision
    validate_amp(device, precision)
    return device, precision


def prepare_execution(raw_model, execution):
    """输入已在目标设备上的模型；输出训练循环要调用的模型入口。

    raw_model：我们写的 GPT 对象，持有已经初始化的 FP32 Parameter。
    eager：直接调用原模型，作为对照。
    compile：PyTorch 返回编译包装对象，仍引用原模型的参数，没有复制一套权重。
    包装对象创建时尚未完成实际编译；第一次用具体输入调用时才触发捕获/编译。
    """
    if execution == 'eager':
        return raw_model
    if execution != 'compile':
        raise ValueError('execution 只能是 eager 或 compile。')
    if next(raw_model.parameters()).device.type not in ('cpu', 'cuda'):
        raise ValueError('本课 PyTorch 2.2.2 的 Inductor 实验限定 CPU/CUDA。')
    # 【本节新增 · PyTorch】默认 backend="inductor"、mode="default"。
    # 它负责优化执行，不替我们定义网络，也不自动执行数据加载或 optimizer.step()。
    # 编译失败会报错；本脚本不把失败静默改成普通执行后继续宣称提速。
    return torch.compile(raw_model)


def main():
    # [C1] 【沿用】固定设备、精度、形状、随机种子和测量条件。
    # 新参数只有 execution：对照时两次运行只改变这一项。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', choices=('eager', 'compile'), default='compile')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    parser.add_argument('--precision', choices=('auto', 'fp32', 'bf16'), default='auto')
    parser.add_argument('--matmul-precision', choices=('highest', 'high'), default='highest')
    parser.add_argument('--inspect-only', action='store_true', help='只显示环境，不触发实际编译。')
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--data-file', type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=32)
    parser.add_argument('--warmup-steps', type=int, default=3)
    parser.add_argument('--steps', type=int, default=5)
    # 当前 Intel Mac / PyTorch 2.2.2 的完整模型编译反向曾卡在多线程 BLAS 等待。
    # 本节默认单线程；eager/compile 对照必须使用同样线程数。GPU 并行度不由此设置。
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.steps, args.threads, args.warmup_steps) < 1:
        parser.error('B、T、步数、线程数必须为正；至少预热 1 步，避免把首次编译计入稳态结果。')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('学习率必须为有限正数。')
    requested_runtime = {'device': args.device, 'precision': args.precision}
    try:
        device, args.precision = resolve_runtime(args.device, args.precision, args.execution)
        configure_precision(device, args.matmul_precision)
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(args.threads)
    # 【实验辅助】生成的编译产物放在忽略目录；它们不是模型权重或训练检查点。
    # setdefault 允许在外部指定其他缓存；不必在每次更新后删除或重建缓存。
    cache_dir = PROJECT_DIR / '.cache' / 'torch-compile'
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(cache_dir))
    environment = {
        'python': platform.python_version(), 'torch': torch.__version__,
        'device': str(device), 'system': platform.system(), 'machine': platform.machine(),
        'threads': torch.get_num_threads(),
        'matmul_precision': torch.get_float32_matmul_precision(),
        'cuda_available': torch.cuda.is_available(), 'mps_available': torch.backends.mps.is_available(),
        'cuda_runtime': torch.version.cuda,
        'inductor_cache': os.environ['TORCHINDUCTOR_CACHE_DIR'],
    }
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        environment.update(gpu_name=props.name, compute_capability=[props.major, props.minor],
                           bf16_supported=torch.cuda.is_bf16_supported())
    print('阶段二 · 单元 2 · 2.2 · 知识块 10：torch.compile')
    print(f'实际配置：device={device}，precision={args.precision}，execution={args.execution}。')
    print(json.dumps(environment, ensure_ascii=False, indent=2))
    if args.inspect_only:
        print('这里只检查环境；是否能编译该模型，需要实际运行前向和反向。')
        return

    # [C2] 【沿用】同一份语料、词表和随机起点；不用 from_pretrained 加载权重。
    model_dir = args.model_dir.expanduser().resolve()
    data_file = args.data_file.expanduser().resolve()
    for path in (model_dir / 'config.json', model_dir / 'tokenizer.json', data_file):
        if not path.is_file():
            parser.error(f'缺少本地文件：{path}；准备方法见 README.md。')
    config = GPTConfig.from_hf_config(json.loads((model_dir / 'config.json').read_text()))
    config.embd_pdrop = config.attn_pdrop = config.resid_pdrop = 0.0
    if args.seq_len > config.n_positions:
        parser.error('T 超出模型的位置表长度。')
    data = data_file.read_bytes()
    data_sha = hashlib.sha256(data).hexdigest()
    if data_file == DEFAULT_DATA_FILE and data_sha != DATA_SHA256:
        parser.error('默认语料与固定课程版本不一致。')
    tokenizer = GPT2TokenizerFast(tokenizer_file=str(model_dir / 'tokenizer.json'))
    if len(tokenizer) != config.vocab_size:
        parser.error('分词器与模型词表大小不一致。')
    ids = tokenizer.encode(data.decode('utf-8'), add_special_tokens=False)
    try:
        loader = DataLoaderLite(ids, args.batch_size, args.seq_len)
    except ValueError as error:
        parser.error(str(error))
    if loader.tokens.min().item() < 0 or loader.tokens.max().item() >= config.vocab_size:
        parser.error('语料中存在词表范围之外的 ID。')
    torch.manual_seed(args.seed)
    raw_model = GPT(config, tie_embeddings=False, gpt2_init=False).cpu().float()
    tie_weights_like_course(raw_model)
    initialize_like_course(raw_model)
    raw_model.to(device)
    tie_weights_like_course(raw_model)
    raw_model.train()

    # [C3] 【本节核心】只在循环外准备一次编译入口。
    # model 用于前向调用；raw_model 便于访问/保存原始参数。
    # AdamW 更新的仍是同一批 FP32 参数，下一次编译执行读取更新后的数值。
    model = prepare_execution(raw_model, args.execution)
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.learning_rate)

    # [C4] 【沿用知识块 9】传入新的入口，原训练循环无需重写。
    # train_amp_steps → forward_and_loss → model(x) → 编译后的 GPT.forward
    #                   外部交叉熵 → loss.backward() → optimizer.step()
    # 只编译模型：模型对应的反向由 AOTAutograd/Inductor 处理；
    # 本项目的外部交叉熵、取 batch、计时和 AdamW 仍普通执行。
    print(f'执行方式：{args.execution}；计算精度：{args.precision}。', flush=True)
    if args.execution == 'compile':
        print('首个预热更新含图捕获、前向/反向编译和训练；请等待。'
              '其耗时不等于纯编译时间。', flush=True)
    if device.type == 'cpu':
        print('当前是 CPU 实验；只能验证本机行为，GPU 提速仍需 CUDA 对照。', flush=True)
    records, summary = train_amp_steps(model, optimizer, loader, device, args.precision,
                                       args.warmup_steps, args.steps)
    report = {
        'lesson': '阶段二 · 单元 2 · 2.2 · 知识块 10', 'course_commit': COURSE_COMMIT,
        'execution': args.execution, 'precision': args.precision,
        'backend': 'inductor' if args.execution == 'compile' else None,
        'compile_scope': ('GPT.forward and its backward; loss and optimizer remain eager'
                          if args.execution == 'compile' else 'none; all operations run eagerly'),
        'requested_runtime': requested_runtime,
        'environment': environment, 'config': asdict(config), 'seed': args.seed,
        'parameter_count': sum(p.numel() for p in raw_model.parameters()),
        'shared_parameter': raw_model.lm_head.weight is raw_model.transformer.wte.weight,
        'same_parameter_objects': {id(p) for p in model.parameters()}
                                  == {id(p) for p in raw_model.parameters()},
        'data': {'file': str(data_file), 'sha256': data_sha, 'encoded_tokens': len(ids)},
        'batch_size': args.batch_size, 'seq_len': args.seq_len,
        'warmup_steps': args.warmup_steps, 'measured_steps': args.steps,
        'first_update_ms': records[0]['seconds'] * 1000,
        'first_update_note': 'Includes training and any compilation/cache loading; not pure compile time.',
        'steps': records, 'summary': summary, 'parameter_updates': len(records),
        'optimizer': {'name': 'AdamW', 'lr': args.learning_rate,
                      'betas': list(optimizer.defaults['betas']),
                      'weight_decay': optimizer.defaults['weight_decay'],
                      'eps': optimizer.defaults['eps']},
        'pretrained_weights_loaded': False, 'checkpoint_saved': False, 'grad_scaler_used': False,
        'timing_scope': 'batch fetch + transfer + zero_grad + forward/loss + backward + step '
                        '+ synchronize; excludes setup, logging and reports',
        'verification_scope': 'This run only; CPU measurements do not establish CUDA speedup. '
                              'Warmup length must be revisited if recompilation occurs.',
    }
    print(f'首次更新：{report["first_update_ms"]:.2f} ms；排除预热后的均值：'
          f'{summary["mean_step_ms"]:.2f} ms/步，{summary["tokens_per_second"]:.1f} Token/s。')
    path = (args.report or PROJECT_DIR / '.cache/runtime-check'
            / f'lesson10-{args.execution}-{args.precision}.json').expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'报告：{path}')


if __name__ == '__main__':
    main()
