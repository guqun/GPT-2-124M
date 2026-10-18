"""阶段二 · 单元 2 · 2.1 · 知识块 4：按视频用 Tiny Shakespeare 做单批次过拟合。

阅读顺序：main() -> make_batch() -> GPT() -> training_step() -> next_token_loss()。
默认取原文开头 1000 字符，B=4/T=32，从随机参数开始反复训练同一批 50 步。
对齐作者 7822fce 阶段：尚未共享 embedding/lm_head，保留 PyTorch 默认初始化。
只读取项目内的语料、config.json、tokenizer.json；不加载预训练模型权重。
配套讲解：lesson04.md。源码中的 M / D / T 编号对应学习单。
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

try:
    import torch
    from torch.nn import functional as F
    from gpt2_model import GPT, GPTConfig
except ImportError as error:
    raise SystemExit(
        f"缺少课程依赖：{error}\n请使用解释器："
        f"{Path(__file__).resolve().parent / '.venv/bin/python'}"
    ) from error


PROJECT_DIR = Path(__file__).resolve().parent
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
DEFAULT_MODEL_DIR = PROJECT_DIR / ".cache/gpt2" / REVISION
DEFAULT_DATA_FILE = PROJECT_DIR / "data/tiny_shakespeare/input.txt"
COURSE_COMMIT = "7822fce8ebf46ba3907ee8f7cf0356f8ec1c63fa"
DATA_URL = f"https://raw.githubusercontent.com/karpathy/build-nanogpt/{COURSE_COMMIT}/input.txt"
DATA_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def make_batch(token_ids, batch_size, seq_len):
    """连续 B*T+1 个 ID -> 输入 x [B,T] 和右移一位的答案 y [B,T]。

    每个位置的训练目标都来自原文。这里固定取文本开头的一个 batch，
    反复训练同一批；遍历语料、切换 batch 留到视频后面的数据加载器知识块。
    """
    if batch_size < 1 or seq_len < 1:
        raise ValueError("batch_size 和 seq_len 必须为正整数。")
    needed = batch_size * seq_len + 1
    if len(token_ids) < needed:
        raise ValueError(f"本批需要 {needed} 个 Token，文本只有 {len(token_ids)} 个。")

    # [D1] 额外取一个 Token：最后一个输入位置也需要一个后继答案。
    buffer = torch.tensor(token_ids[:needed], dtype=torch.long)  # [B*T+1]
    # [D2] 两个切片错开一格，再按相同顺序排成 B 行、每行 T 个位置。
    # 例：[A,B,C,D,E] -> x=[A,B,C,D]，y=[B,C,D,E]。
    # x/y 的 ID 是整数索引；它们不需要梯度，需要学习的是模型的浮点参数。
    x = buffer[:-1].reshape(batch_size, seq_len)  # [B,T]
    y = buffer[1:].reshape(batch_size, seq_len)   # [B,T]
    return x, y


def next_token_loss(logits, targets):
    """全位置交叉熵：[B,T,V] + [B,T] -> 标量 loss，形状 []。

    targets 已由 make_batch 向后错开一格，此处不要再次 shift。
    本课每个位置都有答案，没有 padding，也没有忽略位置。
    """
    if (logits.ndim != 3 or targets.ndim != 2
            or logits.shape[:2] != targets.shape
            or targets.dtype != torch.long or targets.numel() == 0):
        raise ValueError("需要 logits [B,T,V] 和同 B/T 的非空 long targets [B,T]。")
    vocab_size = logits.shape[-1]
    if vocab_size < 1 or targets.min().item() < 0 or targets.max().item() >= vocab_size:
        raise ValueError("答案 ID 必须落在词表范围内。")

    # [T3] 把 B*T 个位置看作 B*T 道词表分类题；词表维 V 保持不动。
    # 两边以相同顺序展开，所以第 i 行分数仍然对应第 i 个正确 ID。
    scores = logits.reshape(-1, vocab_size)  # [B*T,V]
    labels = targets.reshape(-1)            # [B*T]
    # 直接传未归一化的 logits。cross_entropy 内部完成稳定的 log_softmax
    # 与目标类别负对数似然，并默认取平均；不要先做 softmax 或 argmax。
    return F.cross_entropy(scores, labels)  # []，对 B*T 个位置取平均


def training_step(model, optimizer, x, y):
    """清旧梯度 -> 前向 -> loss -> 反传 -> 更新，返回用于观察的普通数值。

    backward() 写入 Parameter.grad；optimizer.step() 才修改 Parameter。
    返回值均已脱离计算图，记录多步日志时不会把整条求导图一直保存下来。
    """
    if not torch.is_grad_enabled():
        raise ValueError("训练需要自动求导，不能放在 no_grad / inference_mode 中。")
    if x.ndim != 2 or x.numel() == 0 or x.dtype != torch.long or x.shape != y.shape:
        raise ValueError("x/y 必须是同形状的非空 [B,T] Token ID。")

    # [T0] train() 设置模块的训练行为；它本身不会更新参数或启用自动求导。
    # 原视频这个模型没有 Dropout；本项目将各 Dropout 的 p 设为 0 与之对齐。
    model.train()
    # 观察第一个输入 Token 的前 8 个 embedding 分量，不复制整个大模型。
    probe_row = int(x[0, 0].item())
    embedding = model.transformer.wte.weight
    before = embedding[probe_row, :8].detach().clone()

    # [T1] 清掉上一轮累计的 .grad。set_to_none=True 将它们置为 None；
    # 下一次 backward 会重新建立梯度，不会把上一步的梯度意外加进来。
    optimizer.zero_grad(set_to_none=True)
    # [T2] 沿用已验证的 GPT.forward，计算所有位置；本次保留自动求导图。
    logits = model(x)                      # [B,T] -> [B,T,V]
    loss = next_token_loss(logits, y)      # [B,T,V] + [B,T] -> []
    if not torch.isfinite(loss).item():
        raise RuntimeError("loss 非有限数，停止更新。")

    # [T4] 从标量 loss 反向应用链式法则，给每个参与计算的参数累计梯度。
    # 每个 parameter.grad 都与 parameter 同形状；此时参数值尚未改变。
    # 当前阶段 wte 与 lm_head 各自有梯度；视频稍后才讲共享权重。
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    grad_norm = torch.stack([g.detach().norm() for g in gradients]).norm().item()
    if not math.isfinite(grad_norm):
        raise RuntimeError("梯度非有限数，停止更新。")
    unchanged_after_backward = torch.equal(before, embedding[probe_row, :8].detach())
    probe_gradient = embedding.grad[probe_row, :8].detach().clone()

    # [T5] 优化器读取梯度，原地更新已有 Parameter；不会改变参数张量的形状。
    # AdamW 还使用历史梯度的一阶/二阶矩，具体更新规则留在优化器专题。
    # .step() 不会清空 .grad，所以下一轮仍需执行 [T1]。
    optimizer.step()
    after = embedding[probe_row, :8].detach().clone()
    # 只用于观察“给定真实前缀时，猜对了多少个下一 Token”。不参与 loss/反传。
    # logits 来自本轮更新前的前向；参数改了，也不会自动重算这个张量。
    correct = (logits.detach().argmax(dim=-1) == y).sum().item()
    return {
        "loss_before_update": loss.detach().item(),
        "correct_before_update": correct,
        "token_accuracy_before_update": correct / y.numel(),
        "logits_shape": list(logits.shape),
        "loss_shape": list(loss.shape),
        "gradient_l2_norm": grad_norm,
        "embedding_gradient_shape": list(embedding.grad.shape),
        "probe": {
            "parameter": "transformer.wte.weight",
            "row": probe_row,
            "columns": list(range(before.numel())),
            "before": before.tolist(),
            "gradient": probe_gradient.tolist(),
            "after": after.tolist(),
            "unchanged_after_backward": unchanged_after_backward,
            "max_abs_update": (after - before).abs().max().item(),
        },
    }


def main():
    # [M1] 原视频的实验设置。steps 是更新次数，不是新样本数量。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--text-chars", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report", type=Path,
                        default=PROJECT_DIR / ".cache/runtime-check/lesson04-report.json")
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.steps, args.text_chars) < 1:
        parser.error("batch-size、seq-len、steps 和 text-chars 必须为正整数。")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate 必须是有限正数。")
    model_dir = args.model_dir.expanduser().resolve()
    for name in ("config.json", "tokenizer.json"):
        if not (model_dir / name).is_file():
            parser.error(f"缺少本地文件：{model_dir / name}；准备方法见 README.md。")
    try:
        from transformers import GPT2TokenizerFast
    except ImportError as error:
        raise SystemExit(f"缺少课程依赖：{error}\n请使用解释器：{PROJECT_DIR / '.venv/bin/python'}") from error

    # [M2] 先准备固定数据，和视频一样先取 1000 字符，再编码成 Token ID。
    # 1000 字符不等于 1000 个 Token；其中只用开头 B*T+1=129 个 ID 构造本批。
    data_file = args.data_file.expanduser().resolve()
    if not data_file.is_file():
        parser.error(f"缺少训练语料：{data_file}；来源和准备方法见 data/tiny_shakespeare/README.md。")
    data_bytes = data_file.read_bytes()
    data_sha256 = hashlib.sha256(data_bytes).hexdigest()
    if data_file == DEFAULT_DATA_FILE and data_sha256 != DATA_SHA256:
        parser.error("默认语料的 SHA-256 与课程版本不一致；请按数据说明恢复，或用 --data-file 显式选择自定义实验。")
    full_text = data_bytes.decode("utf-8")
    text = full_text[:args.text_chars]
    raw_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    config = GPTConfig.from_hf_config(raw_config)
    config.embd_pdrop = config.attn_pdrop = config.resid_pdrop = 0.0
    if args.seq_len > config.n_positions:
        parser.error(f"seq-len 不能超过位置表容量 {config.n_positions}。")
    tokenizer = GPT2TokenizerFast(tokenizer_file=str(model_dir / "tokenizer.json"))
    if len(tokenizer) != config.vocab_size:
        parser.error("分词器与模型词表大小不一致。")
    # 视频使用 tiktoken 的 gpt2 编码；这里复用已有的同词表 GPT-2 tokenizer。
    # 没有额外加 BOS/EOS；也没有把原文改成问答格式。
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    try:
        x, y = make_batch(token_ids, args.batch_size, args.seq_len)
    except ValueError as error:
        parser.error(str(error))
    # [M3] 复用已经学过的网络，按视频此刻的参数设置建立一个新模型。
    # config 只规定结构；此处没有调用 load_hf_weights，不读取 safetensors。
    # seed 控制随后初始化参数用到的随机数；数据没有随机抽样。
    # 视频的这一步没有固定训练 seed，42 是本地用于重跑比较的记录设置。
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    # tie_embeddings 和 gpt2_init 是本项目 GPT 类定义的两个布尔选项：
    # tie_embeddings=False：输入端 wte 与输出端 lm_head 各用一张独立的权重表。
    #   wte 负责把 Token ID 查成向量；lm_head 负责把隐藏向量转成词表分数。
    #   两张表都会参与训练；设为 True 时才会让它们共用同一个 Parameter。
    # gpt2_init=False：跳过本项目额外编写的 GPT-2 初始化规则，保留各层默认初值。
    #   nn.Embedding / nn.Linear 创建时已经按 PyTorch 默认规则随机初始化权重；
    #   设为 True 时会再按 _init_weights 和残差投影缩放规则重新设置初值。
    #   因此 False 不表示“不初始化”，这个选项也不负责加载预训练权重。
    # 当前先按视频做单批次过拟合；权重共享和初始化策略分别在知识块 6、7 学习。
    model = GPT(config, tie_embeddings=False, gpt2_init=False).cpu().float()
    # 因 lm_head 暂时独立，比后面的 124M 多一张 [50257,768] 表，共约 163M 参数。
    # 优化器只建一次，保留跨步状态。除 lr 外使用 AdamW 默认值，与原代码一致：
    # betas=(0.9,0.999), eps=1e-8, weight_decay=0.01。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print("阶段二 · 单元 2 · 2.1 · 知识块 4：Tiny Shakespeare 单批次过拟合")
    print(f"原文 {len(full_text):,} 字符；取前 {len(text)} 字符 -> {len(token_ids)} 个 Token；本批用 {x.numel() + 1} 个 ID")
    print(f"随机参数 / PyTorch 默认初始化 / 暂不共享；参数量：{sum(p.numel() for p in model.parameters()):,}")
    print(f"CPU / FP32；seed={args.seed}；同一 batch 更新 {args.steps} 次；AdamW lr={args.learning_rate}")
    print(f"x 形状：{list(x.shape)}；y 形状：{list(y.shape)}")
    print(f"对齐方式：x[b,t] 所在前缀，预测 y[b,t]；共 {x.numel()} 个训练位置")
    print(f"均匀词表预测的理论 loss：ln({config.vocab_size}) = {math.log(config.vocab_size):.6f}")
    print(f"第一条输入序列：{tokenizer.decode(x[0].tolist(), clean_up_tokenization_spaces=False)!r}")
    print("第一条序列的前 8 个位置：输入 Token → 正确下一 Token（显示值仅用于观察）：")
    for position in range(min(8, x.shape[1])):
        decode = lambda value: tokenizer.decode([int(value)], clean_up_tokenization_spaces=False)
        print(f"  t={position}: {decode(x[0, position])!r} → {decode(y[0, position])!r}")

    # [M4] 默认重复训练 50 次：x/y 在循环外只创建一次，循环里不换数据。
    # 每轮重新前向、建立新图，参数沿用上一轮更新后的值；共处理 50*128
    # 个训练位置，但独特的 (序列,位置) 仍只有这 128 个，不是 6400 个新样本。
    # 这一步用于检查训练链路能否把固定小批次学好；下一块再引入数据加载器。
    records = []
    start_time = time.perf_counter()
    for step in range(args.steps):
        record = training_step(model, optimizer, x, y)
        record["step"] = step + 1
        records.append(record)
        print(f"第 {step + 1:02d} 步更新前：loss={record['loss_before_update']:.6f}；"
              f"给定真实前缀，下一 Token 预测正确 {record['correct_before_update']}/{y.numel()}", flush=True)
        if step == 0:
            print(f"  logits={record['logits_shape']} -> loss={record['loss_shape']}；"
                  f"embedding.grad={record['embedding_gradient_shape']}")
            print(f"  backward 后观察参数未变：{record['probe']['unchanged_after_backward']}；"
                  f"step 后观察分量最大变化：{record['probe']['max_abs_update']:.8f}")

    # [M5] 再次前向，才能得到更新后的 loss；之前的 loss 张量不会自动刷新。
    # eval 控制模块行为；no_grad 只用于这次测量，不能包住训练主循环。
    model.eval()
    with torch.no_grad():
        final_logits = model(x)
        final_loss = next_token_loss(final_logits, y).item()
        final_correct = (final_logits.argmax(dim=-1) == y).sum().item()
    elapsed_seconds = time.perf_counter() - start_time
    print(f"完成 {args.steps} 次更新后再测：loss={final_loss:.6f}；下一 Token 正确 {final_correct}/{y.numel()}")
    print(f"训练循环及最终测量耗时：{elapsed_seconds:.1f} 秒（含教学观察记录）")
    print("以上是给定真实前缀的同批次训练指标；不能代表自由生成、验证集表现或泛化能力。")

    # [M6] 报告只写普通数值；不保存模型权重或求导图。
    report = {
        "lesson": "阶段二 · 单元 2 · 2.1 · 知识块 4：Tiny Shakespeare 单批次过拟合",
        "course_commit": COURSE_COMMIT,
        "model_dir": str(model_dir),
        "initialization": "PyTorch module defaults; untied embeddings; no pretrained weights loaded",
        "environment": {"torch": torch.__version__, "device": "cpu", "dtype": "float32",
                        "threads": torch.get_num_threads()},
        "data": {"file": str(data_file), "course_source_url": DATA_URL,
                 "sha256": data_sha256, "matches_course_file": data_sha256 == DATA_SHA256,
                 "total_characters": len(full_text), "selected_characters": len(text),
                 "encoded_tokens": len(token_ids), "batch_buffer_tokens": x.numel() + 1,
                 "supervised_positions_per_step": x.numel(), "fixed_batch": True,
                 "first_input_text": tokenizer.decode(x[0].tolist(), clean_up_tokenization_spaces=False)},
        "config": asdict(config),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "seed": args.seed,
        "optimizer": {"name": "AdamW", "lr": args.learning_rate,
                      "betas": list(optimizer.defaults["betas"]),
                      "eps": optimizer.defaults["eps"], "weight_decay": optimizer.defaults["weight_decay"]},
        "input_ids": x.tolist(), "target_ids": y.tolist(),
        "input_shape": list(x.shape), "target_shape": list(y.shape),
        "uniform_loss_theory": math.log(config.vocab_size),
        "steps": records, "loss_after_updates": final_loss,
        "correct_after_updates": final_correct,
        "token_accuracy_after_updates": final_correct / y.numel(),
        "elapsed_seconds": elapsed_seconds,
        "embedding_head_tied": model.transformer.wte.weight is model.lm_head.weight,
        "checkpoint_saved": False,
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"报告：{report_path}")


if __name__ == "__main__":
    main()
