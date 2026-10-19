"""阶段二 · 单元 2 · 2.1 · 知识块 6：让输入词向量表与输出评分表共享参数。

阅读顺序：main() 的 M2 -> tie_weights_like_course() -> M3/M4 的训练循环。
原课依据：视频约 1:05:55—1:13:03，作者历史提交 ec39245。
沿用上一块的完整 Tiny Shakespeare、顺序加载器、B=4/T=32、50 次更新。
每次运行新建随机模型；只读取配置和分词器，不加载 model.safetensors。
--inspect-only 只检查共享关系和参数量；配套讲义：lesson06.md。
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    import torch
    from transformers import GPT2TokenizerFast
    from gpt2_model import GPT, GPTConfig
    from train_dataloader import DataLoaderLite
    from train_step import (
        DATA_SHA256, DEFAULT_DATA_FILE, DEFAULT_MODEL_DIR, PROJECT_DIR,
        next_token_loss,
    )
except ImportError as error:
    raise SystemExit(
        f"缺少课程依赖：{error}\n请使用已安装 requirements.txt 中依赖的解释器。"
    ) from error


COURSE_COMMIT = "ec39245425b2c82a38c0787f27e5f9d7082c973b"


def tie_weights_like_course(model: GPT) -> None:
    """输入：尚未创建优化器的 GPT；效果：两个模块引用同一个 Parameter。

    【本项目接线，方向与原课一致】保留 lm_head 已有的随机参数，替换 wte 的引用。
    这不是复制数值，不会新建 Tensor，也不是把两张表相加或平均。
    【PyTorch】nn.Module 注册这个 Parameter；autograd 自动累加两条路径的梯度。
    函数原地修改 model 的参数引用，没有额外返回一个新模型。
    """
    model.transformer.wte.weight = model.lm_head.weight


def main():
    # [M1] 与上一块使用相同数据和超参数。本节只加入共享，不提前做专用初始化。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect-only", action="store_true",
                        help="只创建模型并检查共享关系；不读取语料、不执行训练。")
    parser.add_argument("--report", type=Path,
                        default=PROJECT_DIR / ".cache/runtime-check/lesson06-report.json")
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.steps) < 1:
        parser.error("batch-size、seq-len 和 steps 必须为正整数。")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate 必须是有限正数。")

    model_dir = args.model_dir.expanduser().resolve()
    config_file = model_dir / "config.json"
    if not config_file.is_file():
        parser.error(f"缺少配置：{config_file}；准备方法见 README.md。")
    config = GPTConfig.from_hf_config(json.loads(config_file.read_text(encoding="utf-8")))
    config.embd_pdrop = config.attn_pdrop = config.resid_pdrop = 0.0
    if args.seq_len > config.n_positions:
        parser.error("seq-len 超过模型的位置表容量。")

    # [M2] 先建立两张独立表，再显式执行本课的共享语句，便于看清这次的变化。
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    # tie_embeddings=False：暂时不执行公共模型内的默认共享，下一行按本节方向共享。
    # gpt2_init=False：保留各层自身的随机初始化，GPT-2 专用初始化在下一知识块讲。
    model = GPT(config, tie_embeddings=False, gpt2_init=False).cpu().float()
    untied_count = sum(p.numel() for p in model.parameters())
    tie_weights_like_course(model)
    tied_count = sum(p.numel() for p in model.parameters())
    sharing = {
        "shape": list(model.transformer.wte.weight.shape),
        "same_parameter": model.transformer.wte.weight is model.lm_head.weight,
        "parameters_before": untied_count,
        "parameters_after": tied_count,
        "saved_parameters": untied_count - tied_count,
        "retained_initial_values": "lm_head.weight (nn.Linear default initialization)",
    }
    print("阶段二 · 单元 2 · 2.1 · 知识块 6：权重共享")
    print(f"wte.weight 与 lm_head.weight 的形状：{sharing['shape']}")
    print(f"是否同一个 Parameter：{sharing['same_parameter']}")
    print(f"独立参数量：{untied_count:,} -> {tied_count:,}；减少 {untied_count - tied_count:,}")
    if args.inspect_only:
        print("这里只检查结构，没有执行训练，也没有生成训练报告。")
        return

    # 沿用上一块的数据准备。配置/分词器可以复用，预训练模型权重没有被读入。
    data_file = args.data_file.expanduser().resolve()
    for required in (data_file, model_dir / "tokenizer.json"):
        if not required.is_file():
            parser.error(f"缺少本地文件：{required}；准备方法见 README.md。")
    raw = data_file.read_bytes()
    data_sha256 = hashlib.sha256(raw).hexdigest()
    if data_file == DEFAULT_DATA_FILE and data_sha256 != DATA_SHA256:
        parser.error("默认语料与固定课程版本不一致。")
    text = raw.decode("utf-8")
    tokenizer = GPT2TokenizerFast(tokenizer_file=str(model_dir / "tokenizer.json"))
    if len(tokenizer) != config.vocab_size:
        parser.error("分词器与模型词表大小不一致。")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    try:
        loader = DataLoaderLite(token_ids, args.batch_size, args.seq_len)
    except ValueError as error:
        parser.error(str(error))

    # [M3] 必须先共享、后创建优化器。parameters() 默认对同一个参数对象去重。
    # 否则旧优化器可能仍握着已经被替换的 wte 参数引用。
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    sharing["optimizer_occurrences"] = sum(
        p is model.transformer.wte.weight
        for group in optimizer.param_groups for p in group["params"]
    )
    print(f"优化器中的共享参数对象出现 {sharing['optimizer_occurrences']} 次")
    print(f"完整语料 {len(token_ids):,} 个 Token；CPU / FP32；seed={args.seed}")
    records = []
    # [M4] 训练主干不变：共享关系已经建立，无需在循环中额外复制或同步两张表。
    for step in range(args.steps):
        start = loader.current_position
        x, y = loader.next_batch()  # 输入 ID 和下一 Token 答案，均为 [B,T]。
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)  # 查共享表 -> Transformer -> 用共享表给全词表打分。
        loss = next_token_loss(logits, y)
        if not torch.isfinite(loss).item():
            raise RuntimeError("loss 非有限数，停止更新。")
        # 【PyTorch autograd】共享表在输入/输出两处使用，梯度贡献自动相加。
        loss.backward()
        # 【PyTorch AdamW】使用合计梯度，对同一份参数做一次优化器更新。
        optimizer.step()
        value = loss.item()  # 更新前、当前 batch 的 loss；不是更新后的重新评估。
        records.append({"step": step + 1, "batch_start": start,
                        "next_start": loader.current_position,
                        "loss_before_update": value})
        print(f"第 {step + 1:02d} 步：数据起点 {start:5d}；loss={value:.6f}", flush=True)

    report = {
        "lesson": "阶段二 · 单元 2 · 2.1 · 知识块 6：权重共享",
        "course_commit": COURSE_COMMIT,
        "sharing": sharing,
        "data": {"file": str(data_file), "sha256": data_sha256,
                 "total_characters": len(text), "encoded_tokens": len(token_ids),
                 "sampling": "sequential; reset to zero; no shuffle"},
        "input_shape": [args.batch_size, args.seq_len],
        "target_shape": [args.batch_size, args.seq_len],
        "initialization": "PyTorch module defaults; tied by retaining lm_head; no pretrained weights loaded",
        "seed": args.seed,
        "environment": {"torch": torch.__version__, "device": "cpu", "dtype": "float32"},
        "optimizer": {"name": "AdamW", "lr": args.learning_rate,
                      "betas": list(optimizer.defaults["betas"]),
                      "eps": optimizer.defaults["eps"], "weight_decay": optimizer.defaults["weight_decay"]},
        "steps": records,
        "processed_supervised_positions": args.steps * args.batch_size * args.seq_len,
        "distinct_batch_starts": len({r["batch_start"] for r in records}),
        "metric_scope": "pre-update loss on changing training batches; not held-out validation",
        "checkpoint_saved": False,
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("每步数据不同；这些 loss 不用于单独证明共享权重改善了泛化。")
    print(f"报告：{report_path}")


if __name__ == "__main__":
    main()
