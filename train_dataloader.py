"""阶段二 · 单元 2 · 2.1 · 知识块 5：按顺序提供新 batch 的数据加载器。

阅读顺序：main() -> DataLoaderLite.__init__() -> 训练循环中的 next_batch()。
沿用知识块 4 的模型和 loss；新内容集中在数据读取位置的保存、推进和回绕。
原课依据：build-nanogpt 的 631f7d6 提交，视频约 1:01:52—1:05:55。
默认读取完整 Tiny Shakespeare，以 B=4/T=32 顺序训练 50 步；不加载预训练权重。
运行 --preview-only 可只观察前三批数据，不创建模型。配套讲义：lesson05.md。
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
    # 只复用上一块的路径、数据校验值与 loss 函数；导入不会运行它的 main()。
    from train_step import (
        DATA_SHA256, DEFAULT_DATA_FILE, DEFAULT_MODEL_DIR, PROJECT_DIR,
        next_token_loss,
    )
except ImportError as error:
    raise SystemExit(
        f"缺少课程依赖：{error}\n请使用解释器："
        f"{Path(__file__).resolve().parent / '.venv/bin/python'}"
    ) from error


COURSE_COMMIT = "631f7d6f080effc4342b5e2f2f41879642c76fd1"


class DataLoaderLite:
    """保存整段 Token 数据与读取位置，每次返回下一组 x/y。

    这是普通 Python 类，名称沿用原课，不是 torch.utils.data.DataLoader。
    本地把文件读取/分词放在 main()，此类只接收编码后的整数 ID，便于单独观察。
    """

    def __init__(self, token_ids, batch_size, seq_len):
        if batch_size < 1 or seq_len < 1:
            raise ValueError("batch_size 和 seq_len 必须为正整数。")
        if len(token_ids) < batch_size * seq_len + 1:
            raise ValueError("语料不足以组成一批输入及其最后一个目标 Token。")
        self.B = batch_size  # 一批有多少条输入序列。
        self.T = seq_len    # 每条序列有多少个 Token。
        self.tokens = torch.tensor(token_ids, dtype=torch.long)  # [N]，保存在 CPU。
        # 这是整个语料中的位置下标，不是 Token ID，也不是模型的位置编码。
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        # [L1] 从上次停下的位置取 B*T+1 个 ID；多取的一个供最后位置作答案。
        start = self.current_position
        buffer = self.tokens[start:start + B * T + 1]  # [B*T+1]
        # [L2] 这部分和上一块相同：输入和答案错开一位，各排成 B 行、T 列。
        x = buffer[:-1].view(B, T)  # [B,T]
        y = buffer[1:].view(B, T)   # [B,T]
        # [L3] 只前进 B*T：本批最后的答案 Token，还要作为下一批的第一个输入。
        self.current_position += B * T
        # [L4] 若下一批连同其答案装不下，就跳过不足一批的尾部，从头再读。
        # “刚好装得下”仍然要读，所以这里是 >，而不是 >=。当前版本不打乱顺序。
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0
        return x, y


def main():
    # [M1] 沿用 B/T、优化器与参数起点；本块改为读取完整原文并逐批提供数据。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-only", action="store_true",
                        help="只展示前三批数据，不创建模型或执行训练。")
    parser.add_argument("--report", type=Path,
                        default=PROJECT_DIR / ".cache/runtime-check/lesson05-report.json")
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.steps) < 1:
        parser.error("batch-size、seq-len 和 steps 必须为正整数。")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate 必须是有限正数。")

    data_file = args.data_file.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    for path in (data_file, model_dir / "config.json", model_dir / "tokenizer.json"):
        if not path.is_file():
            parser.error(f"缺少本地文件：{path}；准备方法见 README.md。")
    raw = data_file.read_bytes()
    data_sha256 = hashlib.sha256(raw).hexdigest()
    if data_file == DEFAULT_DATA_FILE and data_sha256 != DATA_SHA256:
        parser.error("默认语料与固定课程版本不一致，来源见 data/tiny_shakespeare/README.md。")
    text = raw.decode("utf-8")  # 完整文件；不再截取前 1000 字符。
    config = GPTConfig.from_hf_config(json.loads((model_dir / "config.json").read_text()))
    config.embd_pdrop = config.attn_pdrop = config.resid_pdrop = 0.0
    if args.seq_len > config.n_positions:
        parser.error("seq-len 超过模型的位置表容量。")
    tokenizer = GPT2TokenizerFast(tokenizer_file=str(model_dir / "tokenizer.json"))
    if len(tokenizer) != config.vocab_size:
        parser.error("分词器与模型词表大小不一致。")
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # [M2] 加载器只创建一次，才能在多次 next_batch() 之间记住读取位置。
    try:
        train_loader = DataLoaderLite(token_ids, args.batch_size, args.seq_len)
    except ValueError as error:
        parser.error(str(error))
    tokens_per_batch = args.batch_size * args.seq_len
    # 最后一批也要有额外的答案 Token，因此完整批数按 N-1 计算。
    batches_per_pass = (len(token_ids) - 1) // tokens_per_batch
    print("阶段二 · 单元 2 · 2.1 · 知识块 5：顺序数据加载器")
    print(f"完整原文 {len(text):,} 字符 -> {len(token_ids):,} 个 Token")
    print(f"B={args.batch_size}，T={args.seq_len}；每批 {tokens_per_batch} 个训练位置；"
          f"每轮可取 {batches_per_pass} 个完整 batch，随后回到开头")
    if args.preview_only:
        for batch_index in range(3):
            start = train_loader.current_position
            x, y = train_loader.next_batch()
            print(f"第 {batch_index + 1} 批：x 使用语料位置 {start}..{start + tokens_per_batch - 1}；"
                  f"y 使用 {start + 1}..{start + tokens_per_batch}；下一起点 {train_loader.current_position}")
            print(f"  x/y 形状 {list(x.shape)}；第一条输入：{tokenizer.decode(x[0].tolist())!r}")
        return

    # [M3] 每次启动脚本都创建新模型；上一块没有保存检查点，因此这里不承接其参数。
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    # 输入/输出表仍各自独立；初始化仍用各层默认规则，后两块再分别调整。
    model = GPT(config, tie_embeddings=False, gpt2_init=False).cpu().float()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    records = []
    print(f"随机参数 / CPU / FP32；seed={args.seed}；AdamW lr={args.learning_rate}")
    # [M4] 模型与优化器在循环外创建；数据获取放到循环内，每步才会换到下一批。
    for step in range(args.steps):
        start = train_loader.current_position
        x, y = train_loader.next_batch()  # 新增的关键调用；x/y 的形状仍然是 [B,T]。
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = next_token_loss(logits, y)
        if not torch.isfinite(loss).item():
            raise RuntimeError("loss 非有限数，停止更新。")
        loss.backward()
        optimizer.step()

        # 只记录数据位置与 loss，避免参数观察日志遮住本节主干。
        value = loss.item()  # 本轮参数更新前，对当前 batch 计算的 loss。
        records.append({"step": step + 1, "batch_start": start,
                        "next_start": train_loader.current_position,
                        "loss_before_update": value})
        print(f"第 {step + 1:02d} 步：数据起点 {start:5d} -> {train_loader.current_position:5d}；"
              f"当前批次更新前 loss={value:.6f}", flush=True)

    report = {
        "lesson": "阶段二 · 单元 2 · 2.1 · 知识块 5：顺序数据加载器",
        "course_commit": COURSE_COMMIT,
        "data": {"file": str(data_file), "sha256": data_sha256,
                 "total_characters": len(text), "encoded_tokens": len(token_ids),
                 "tokens_per_batch": tokens_per_batch, "batches_per_pass": batches_per_pass,
                 "unused_target_positions_per_pass": (len(token_ids) - 1) % tokens_per_batch,
                 "sampling": "sequential; reset to zero; no shuffle", "fixed_batch": False},
        "input_shape": [args.batch_size, args.seq_len],
        "target_shape": [args.batch_size, args.seq_len],
        "initialization": "PyTorch module defaults; untied embeddings; no pretrained weights loaded",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "seed": args.seed,
        "environment": {"torch": torch.__version__, "device": "cpu", "dtype": "float32"},
        "optimizer": {"name": "AdamW", "lr": args.learning_rate,
                      "betas": list(optimizer.defaults["betas"]),
                      "eps": optimizer.defaults["eps"], "weight_decay": optimizer.defaults["weight_decay"]},
        "steps": records,
        "processed_supervised_positions": args.steps * tokens_per_batch,
        "distinct_batch_starts": len({r["batch_start"] for r in records}),
        "metric_scope": "pre-update loss on changing training batches; not held-out validation",
        "checkpoint_saved": False,
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("每步数据不同，loss 不要求单调下降，也不以接近零作为本块的成功标准。")
    print(f"报告：{report_path}")


if __name__ == "__main__":
    main()
