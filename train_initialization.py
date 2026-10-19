"""阶段二 · 单元 2 · 2.1 · 知识块 7：参数初始化与残差分支的尺度。

阅读顺序：main() 的 M2 -> initialize_like_course() -> 训练循环。
对照视频约 1:13:03—1:22:11 和作者历史提交 9ac321e。
保留上一块的共享关系与顺序加载器；新增初始化规则在第一个 batch 之前执行。
--inspect-only 显示参数统计和视频中的残差累加小实验，不执行训练。
默认用完整 Tiny Shakespeare、B=4/T=32、CPU/FP32 训练 50 步；不读取预训练权重。
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

try:
    import torch
    from torch import nn
    from transformers import GPT2TokenizerFast
    from gpt2_model import GPT, GPTConfig
    from train_dataloader import DataLoaderLite
    from train_weight_tying import tie_weights_like_course
    from train_step import (
        DATA_SHA256, DEFAULT_DATA_FILE, DEFAULT_MODEL_DIR, PROJECT_DIR,
        next_token_loss,
    )
except ImportError as error:
    project = Path(__file__).resolve().parent
    candidates = (project / '.venv/bin/python', project.parent / 'venv/bin/python')
    interpreter = next((p for p in candidates if p.is_file()), candidates[0])
    raise SystemExit(
        f"缺少课程依赖：{error}\n当前解释器：{sys.executable}\n"
        f"请在 VS Code 的 Python: Select Interpreter 中选择课程环境：\n{interpreter}"
    ) from error


COURSE_COMMIT = '9ac321e299f06c0414bd577fd2776f4562283a8a'
BASE_STD = 0.02


def initialize_like_course(model: GPT) -> None:
    """输入：刚创建且已共享权重的模型；效果：按课程规则填入训练初值。

    【本项目】选择各类参数的分布和尺度。
    【PyTorch】apply 遍历子模块，nn.init 原地填写参数，不建立训练用的计算图。
    只在训练开始前调用；每个 batch 都调用会覆盖已经学到的参数。
    """
    # [I1] 每个 Block 有两条加回主干的分支，标记它们各自最后的 Linear。
    # 标记名称来自作者原课；它是普通 Python 属性，不是可训练参数。
    for block in model.transformer.h:
        block.attn.c_proj.NANOGPT_SCALE_INIT = 1
        block.mlp.c_proj.NANOGPT_SCALE_INIT = 1

    def init_module(module):
        # [I2] 一个 Linear 保存 weight 和可选的 bias；参数矩阵尺寸不变。
        if isinstance(module, nn.Linear):
            std = BASE_STD
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # L=12 时，共 2L=24 条残差分支，std=0.02/sqrt(24)。
                std *= (2 * model.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # [I3] 课程对 Token 表和 Position 表统一采用 std=0.02。
            nn.init.normal_(module.weight, mean=0.0, std=BASE_STD)
        # LayerNorm 保留构造时的 gamma=1、beta=0；无需在这里重新填写。

    # [I4] 传入函数，由 PyTorch 逐个调用；没有给前向过程加入新运算。
    # 与视频一致：共享表会在 Embedding 和 lm_head 两处各填一次，后一次保留。
    model.apply(init_module)


def initialization_summary(model: GPT):
    """读取训练前的实际统计；有限个随机数的样本均值/标准差不会精确等于目标。"""
    scaled_std = BASE_STD / math.sqrt(2 * model.config.n_layer)
    examples = (
        ('wte.weight（与 lm_head 共享）', model.transformer.wte.weight, BASE_STD),
        ('wpe.weight', model.transformer.wpe.weight, BASE_STD),
        ('block0.attn.c_attn.weight', model.transformer.h[0].attn.c_attn.weight, BASE_STD),
        ('block0.attn.c_proj.weight', model.transformer.h[0].attn.c_proj.weight, scaled_std),
        ('block0.mlp.c_fc.weight', model.transformer.h[0].mlp.c_fc.weight, BASE_STD),
        ('block0.mlp.c_proj.weight', model.transformer.h[0].mlp.c_proj.weight, scaled_std),
    )
    rows = []
    for name, weight, expected_std in examples:
        values = weight.detach()
        rows.append({'name': name, 'shape': list(values.shape),
                     'target_std': expected_std, 'mean': values.mean().item(),
                     'observed_std': values.std(unbiased=False).item()})
    return rows


def residual_accumulation_demo(additions=100, width=768, seed=42):
    """原视频的独立随机累加实验；展示方差规律，不是对真实 Transformer 的模拟。

    输入：累加次数 N 与每个向量的宽度。输出：缩放前后两组累加结果的标准差。
    两组使用同一批独立随机贡献；局部 Generator 不改变模型训练的随机数状态。
    """
    if additions < 1 or width < 2:
        raise ValueError('additions 必须为正整数，width 至少为 2。')
    generator = torch.Generator().manual_seed(seed)
    unscaled = torch.zeros(width)
    scaled = torch.zeros(width)
    for _ in range(additions):
        contribution = torch.randn(width, generator=generator)
        unscaled += contribution
        scaled += contribution / math.sqrt(additions)
    return {'additions': additions, 'width': width, 'seed': seed,
            'unscaled_expected_std': math.sqrt(additions),
            'unscaled_observed_std': unscaled.std(unbiased=False).item(),
            'scaled_expected_std': 1.0,
            'scaled_observed_std': scaled.std(unbiased=False).item()}


def main():
    # [M1] 沿用上一块的数据与训练设置；本节新增内容在 M2 初始化处。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--data-file', type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=32)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--inspect-only', action='store_true',
                        help='只检查初始化和残差累加，不读取语料或执行训练。')
    parser.add_argument('--report', type=Path,
                        default=PROJECT_DIR / '.cache/runtime-check/lesson07-report.json')
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.steps) < 1:
        parser.error('batch-size、seq-len 和 steps 必须为正整数。')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('learning-rate 必须是有限正数。')
    model_dir = args.model_dir.expanduser().resolve()
    config_file = model_dir / 'config.json'
    if not config_file.is_file():
        parser.error(f'缺少配置：{config_file}；准备方法见 README.md。')
    config = GPTConfig.from_hf_config(json.loads(config_file.read_text(encoding='utf-8')))
    config.embd_pdrop = config.attn_pdrop = config.resid_pdrop = 0.0
    if args.seq_len > config.n_positions:
        parser.error('seq-len 超过模型的位置表容量。')

    # [M2] 创建参数 -> 建立共享 -> 填写课程初值。全部发生在训练循环之前。
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    # 关闭公共 GPT 内置初始化，在本文件显式按作者当前顺序执行，方便逐句阅读。
    model = GPT(config, tie_embeddings=False, gpt2_init=False).cpu().float()
    tie_weights_like_course(model)
    initialize_like_course(model)
    summary = initialization_summary(model)
    demo = residual_accumulation_demo(seed=args.seed)
    print('阶段二 · 单元 2 · 2.1 · 知识块 7：参数初始化')
    for row in summary:
        print(f"{row['name']}: 目标 std={row['target_std']:.6f}；"
              f"实测 mean={row['mean']:.6f}, std={row['observed_std']:.6f}")
    print(f"残差累加教学实验（100 次）：不缩放 std={demo['unscaled_observed_std']:.4f}；"
          f"除以 sqrt(100) 后 std={demo['scaled_observed_std']:.4f}")
    if args.inspect_only:
        print('初始化检查完成；未执行训练、未写训练报告。')
        return

    data_file = args.data_file.expanduser().resolve()
    for required in (data_file, model_dir / 'tokenizer.json'):
        if not required.is_file():
            parser.error(f'缺少本地文件：{required}；准备方法见 README.md。')
    raw = data_file.read_bytes()
    data_sha256 = hashlib.sha256(raw).hexdigest()
    if data_file == DEFAULT_DATA_FILE and data_sha256 != DATA_SHA256:
        parser.error('默认语料与固定课程版本不一致。')
    text = raw.decode('utf-8')
    tokenizer = GPT2TokenizerFast(tokenizer_file=str(model_dir / 'tokenizer.json'))
    if len(tokenizer) != config.vocab_size:
        parser.error('分词器与模型词表大小不一致。')
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    try:
        loader = DataLoaderLite(token_ids, args.batch_size, args.seq_len)
    except ValueError as error:
        parser.error(str(error))

    # [M3] 初始化已结束。优化器和加载器都只创建一次，随后连续训练。
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    records = []
    # [M4] 每个 batch 都沿用上一步的参数，不再调用初始化函数。
    for step in range(args.steps):
        start = loader.current_position
        x, y = loader.next_batch()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = next_token_loss(logits, y)
        if not torch.isfinite(loss).item():
            raise RuntimeError('loss 非有限数，停止更新。')
        loss.backward()
        optimizer.step()
        records.append({'step': step + 1, 'batch_start': start,
                        'next_start': loader.current_position,
                        'loss_before_update': loss.item()})
        print(f'第 {step + 1:02d} 步：数据起点 {start:5d}；loss={loss.item():.6f}', flush=True)

    report = {
        'lesson': '阶段二 · 单元 2 · 2.1 · 知识块 7：参数初始化',
        'course_commit': COURSE_COMMIT,
        'initialization': {'base_std': BASE_STD,
                           'residual_std': BASE_STD / math.sqrt(2 * config.n_layer),
                           'measurements_before_training': summary,
                           'pretrained_weights_loaded': False},
        'residual_accumulation_demo': demo,
        'sharing': {'same_parameter': model.transformer.wte.weight is model.lm_head.weight,
                    'optimizer_occurrences': sum(p is model.transformer.wte.weight
                        for group in optimizer.param_groups for p in group['params'])},
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'data': {'file': str(data_file), 'sha256': data_sha256,
                 'total_characters': len(text), 'encoded_tokens': len(token_ids),
                 'sampling': 'sequential; reset to zero; no shuffle'},
        'input_shape': [args.batch_size, args.seq_len], 'seed': args.seed,
        'environment': {'torch': torch.__version__, 'device': 'cpu', 'dtype': 'float32'},
        'optimizer': {'name': 'AdamW', 'lr': args.learning_rate,
                      'betas': list(optimizer.defaults['betas']),
                      'eps': optimizer.defaults['eps'], 'weight_decay': optimizer.defaults['weight_decay']},
        'steps': records,
        'processed_supervised_positions': args.steps * args.batch_size * args.seq_len,
        'distinct_batch_starts': len({r['batch_start'] for r in records}),
        'metric_scope': 'pre-update loss on changing training batches; not held-out validation',
        'checkpoint_saved': False,
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'报告：{report_path}')


if __name__ == '__main__':
    main()
