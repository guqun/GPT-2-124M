"""知识块 12 实践（二）：读取 .npy 分片，逐批返回 x/y。

阅读顺序：load_tokens → ShardDataLoaderLite.__init__ → next_batch → main。
默认只打印 CPU 上四次取批，不创建模型、不更新参数、不下载 FineWeb。
前一块只有 5+3 个 Token，本块另建容量 16 的教学文件以演示换片。
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parent


def load_tokens(filename):
    """[S1] 磁盘上的一个 uint16 .npy → CPU 上一维 torch.long Tensor。"""
    # NumPy 负责解析 .npy 文件头与数据；此处将整片读进 RAM，不放入 GPU。
    array = np.load(filename, allow_pickle=False)
    if array.ndim != 1 or array.dtype != np.uint16:
        raise ValueError(f"{filename} 必须保存一维 uint16 Token ID 数组。")
    if array.size and int(array.max()) >= 50257:
        raise ValueError(f"{filename} 包含 GPT-2 词表范围之外的 ID。")
    # 本机 PyTorch 2.2.2 对 NumPy uint16 的直接转换支持有限，先显式转 int64。
    # from_numpy 复用转换后的 CPU 数组内存；ID 数值没有改变，也没有查 Embedding。
    return torch.from_numpy(array.astype(np.int64))


class ShardDataLoaderLite:
    """普通 Python 类：管理文件列表、当前分片与片内游标，提供 next_batch()。

    单进程先看 rank=0/world_size=1，即每次前进 B*T。
    多进程沿用 train_ddp.py 的全局块起点修正，所有 rank 同轮换片。
    本类只管理数据位置，不启动 DDP、不执行梯度通信、不迁移数据到 GPU。
    """

    def __init__(self, data_dir, batch_size, seq_len, split="train", rank=0, world_size=1):
        if min(batch_size, seq_len, world_size) < 1 or not 0 <= rank < world_size:
            raise ValueError("B、T、world_size 必须为正，且 0 <= rank < world_size。")
        if split not in {"train", "val"}:
            raise ValueError("split 只能是 train 或 val。")
        self.B, self.T = batch_size, seq_len
        self.rank, self.world_size = rank, world_size
        self.stride = batch_size * seq_len * world_size

        # [S2] 只选本用途的文件，按编号排序；train 不会读到 val 文件。
        self.shards = sorted(Path(data_dir).glob(f"edufineweb_{split}_*.npy"))
        if not self.shards:
            raise ValueError(f"{data_dir} 没有 {split} 分片。")
        self.reset()

    def _load_usable_shard(self, first_index):
        """[S3] 读取下一片；整片也不足一轮时跳过，最多检查一遍文件列表。"""
        for attempt in range(len(self.shards)):
            index = (first_index + attempt) % len(self.shards)
            tokens = load_tokens(self.shards[index])
            # 每个 rank 都必须有完整的一批以及最后一个标签。
            # 单进程条件就是 N >= B*T+1；短尾片不会跨文件拼接补齐。
            if len(tokens) >= self.stride + 1:
                self.current_shard = index
                self.tokens = tokens         # [N]，仍在 CPU 内存。
                self.global_position = 0     # 新分片从头开始。
                return
        raise ValueError(f"所有分片都不足 {self.stride + 1} 个 Token，无法构成一轮完整 batch。")

    def reset(self):
        """回到本 split 的第一片可用数据；可供验证循环每次从相同位置开始。"""
        self._load_usable_shard(0)

    @property
    def current_position(self):
        # 单进程 rank=0，这就是当前片内的读取起点。
        # 多进程每张卡分到同一全局块中的不同 B*T 个输入位置。
        return self.global_position + self.rank * self.B * self.T

    @property
    def current_file(self):
        return self.shards[self.current_shard]

    def next_batch(self):
        B, T = self.B, self.T
        # [S4] 取 B*T+1 个连续 ID；最后那个 ID 用来做最后输入位置的正确答案。
        start = self.current_position
        buf = self.tokens[start:start + B * T + 1]  # [B*T+1]，在 CPU。
        # [S5] 先整体错开一位，再分别排成 B 行、T 列。
        x = buf[:-1].view(B, T)  # [B,T]：每个位置当前看到的 Token ID。
        y = buf[1:].view(B, T)   # [B,T]：每个位置应该预测的下一个 Token ID。

        # [S6] 前进 B*T*R；单进程只前进 B*T，而不是 B*T+1。
        # 同一片内，上一批最后的标签可以作为下一批的第一个输入。
        self.global_position += self.stride
        # [S7] 检查“下一轮”能否放下。这里是 >，恰好相等时仍能读完整批次。
        # 与作者历史多卡版本不同：用所有 rank 相同的全局起点判断换片。
        if self.global_position + self.stride + 1 > len(self.tokens):
            self._load_usable_shard((self.current_shard + 1) % len(self.shards))
        # S7 即使已经加载下一片，x/y 仍引用刚才取出的旧片数据，不会被改写。
        return x, y


def make_demo_shards():
    """复用前一节已归档的编码结果，另建小文件；不需要重新联网分词。"""
    from prepare_fineweb import write_shards

    previous = json.loads((PROJECT_DIR / "artifacts/lesson12-data-preparation-demo.json").read_text())
    if previous["source"]["is_fineweb"] or previous["total_tokens"] != 8:
        raise ValueError("前一节 demo 报告发生变化，请重新核对本节教学数据。")
    # 两句正文每句含 EOT 共 4 个 ID；重复 5 组得到 40 个 ID。
    docs = [np.array(row["first_ids"], dtype=np.uint16) for row in previous["documents"]]
    if any(len(tokens) != row["tokens"] for tokens, row in zip(docs, previous["documents"])):
        raise ValueError("demo 报告的 first_ids 没有完整记录文章。")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = PROJECT_DIR / ".cache/shard-batches" / ("demo-" + stamp)
    prepared = write_shards((tokens for _ in range(5) for tokens in docs), directory, 16)
    (directory / "manifest.json").write_text(json.dumps({
        "source": "previous two-sentence demo repeated five times; not FineWeb",
        "shard_size": 16, **prepared,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="已有分片目录；不传则另建小型 demo")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--report", type=Path,
                        default=PROJECT_DIR / ".cache/runtime-check/lesson12-shard-batches.json")
    args = parser.parse_args()
    if args.batches < 1:
        parser.error("batches 必须为正整数。")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("本机 PyTorch 没有可用 CUDA 设备，可使用 --device cpu。")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("本机 PyTorch 没有可用 MPS 设备，可使用 --device cpu。")
    device = torch.device(args.device)
    try:
        # [M1] 准备/选择磁盘文件，再创建一次 loader，之后复用其读取位置。
        directory = args.data_dir if args.data_dir is not None else make_demo_shards()
        loader = ShardDataLoaderLite(directory, args.batch_size, args.seq_len,
                                     args.split, args.rank, args.world_size)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    records = []
    print("知识块 12 实践（二）：读取分片并组成 batch；不创建模型、不训练。")
    print("数据：" + (str(directory) if args.data_dir else "两句教学文本重复 5 组，40 个 ID；不是 FineWeb。"))
    print(f"B={loader.B}, T={loader.T}, R={loader.world_size}, rank={loader.rank}；split={args.split}；device={device}")
    for batch in range(args.batches):
        filename, start, size = loader.current_file.name, loader.current_position, len(loader.tokens)
        # [M2] 每次拿到一批 CPU Tensor；随后仅将这一批 x/y 送到计算设备。
        x, y = loader.next_batch()
        record = dict(batch=batch + 1, file=filename, start=start, shard_tokens=size,
                      input_shape=list(x.shape), cpu_dtype=str(x.dtype), cpu_device=str(x.device),
                      x=x.tolist(), y=y.tolist(),
                      next_file=loader.current_file.name, next_start=loader.current_position)
        x, y = x.to(device), y.to(device)  # GPU 训练时才在这里搬运本批数据。
        record["batch_device"] = str(x.device)
        records.append(record)
        print(f"批次 {batch + 1}：{filename}，取 [{start},{start + loader.B * loader.T + 1})")
        print(f"  x={record['x']}\n  y={record['y']}")
        print(f"  下一次：{record['next_file']}，起点 {record['next_start']}；本批设备 {x.device}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "lesson": "2.4 / block 12 / shard loading and batches",
        "data_kind": "user supplied directory" if args.data_dir else "repeated teaching demo; not FineWeb",
        "data_dir": str(directory.resolve()), "split": args.split,
        "batch_size": loader.B, "seq_len": loader.T,
        "rank": loader.rank, "world_size": loader.world_size,
        "loader_device": str(loader.tokens.device), "batch_device": str(device),
        "model_created": False, "training_run": False, "ddp_launched": False,
        "records": records,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"报告：{args.report.resolve()}")


if __name__ == "__main__":
    main()
