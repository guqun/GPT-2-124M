"""知识块 12 实践（一）：正文 → GPT-2 ID → uint16 → .npy 分片。

本文件独立于训练入口，不构造模型、不修改任何模型参数。
默认使用两句教学文本，方便逐个 ID 检查；--source fineweb-preview 则读取
官方 Dataset Viewer 的 20 条真实 FineWeb-Edu 记录，不下载整个 10BT 数据集。
原课依据与本地改编见 lesson12-data-preparation.md。
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import tiktoken


PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / ".cache" / "fineweb-preparation"
DEMO_DOCUMENTS = ["Hello world.", "Good morning."]
DATASET = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"


def read_documents(source, cache_dir):
    """P1：取得正文。返回字符串列表与来源；缓存原文不进入 Git。"""
    if source == "demo":
        return DEMO_DOCUMENTS, {"kind": "invented teaching text", "is_fineweb": False}

    # 这是本地小样本改编。原课使用 datasets.load_dataset 下载整个 sample-10BT。
    # 官方预览接口每次最多 100 行；本节固定 20 行，避免无意下载完整语料。
    query = urlencode(dict(dataset=DATASET, config=CONFIG, split="train", offset=0, length=20))
    url = "https://datasets-server.huggingface.co/rows?" + query
    cache_path = cache_dir / "fineweb-rows-0-19.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        raw = cache_path.read_bytes()
    else:
        # 限定响应大小与等待时间。失败时向用户报错，不悄悄改用演示文本。
        with urlopen(url, timeout=30) as response:
            raw = response.read(10 * 1024 * 1024 + 1)
        if len(raw) > 10 * 1024 * 1024:
            raise ValueError("公开样本响应超过 10 MiB，已停止。")

    payload = json.loads(raw)
    rows = payload.get("rows", [])
    if len(rows) != 20 or [row["row_idx"] for row in rows] != list(range(20)):
        raise ValueError("预览接口没有返回完整的第 0～19 行。")
    documents = []
    for row in rows:
        if "text" in row.get("truncated_cells", []):
            raise ValueError("预览接口截断了正文，不能把截断文本当作完整文档。")
        text = row["row"].get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("记录缺少非空 text 正文。")
        documents.append(text)
    if not cache_path.exists():
        cache_path.write_bytes(raw)
    return documents, {
        "kind": "official dataset-viewer rows", "is_fineweb": True,
        "dataset": DATASET, "config": CONFIG, "split": "train", "url": url,
        "row_range": [0, 20], "cache_file": str(cache_path.relative_to(PROJECT_DIR)),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "revision": None,
        "revision_note": "当前公开预览快照；未固定到原视频的数据版本。",
    }


def encode_document(text, tokenizer):
    """P2：一篇正文 → [EOT, 正文 ID...]；P3：检查后转 uint16。"""
    # encode_ordinary 把正文当普通文本处理，即使正文恰好包含特殊 Token 的拼写。
    # eot_token 是公开属性；GPT-2 中它为 50256。没有在文末再额外插一次。
    ids = [tokenizer.eot_token] + tokenizer.encode_ordinary(text)
    if min(ids) < 0 or max(ids) >= 2**16:
        raise ValueError("Token ID 超出 uint16 的 0～65535 范围。")
    # 这里转换的是“整数的存储类型”，不是 Embedding，也不是 BF16 浮点计算。
    return np.asarray(ids, dtype=np.uint16)


def write_shards(encoded_documents, output_dir, shard_size):
    """P4：顺序填满容量 S 的缓冲区；P5：写文件，最后保存不足 S 的尾片。

    输入：逐篇的一维 uint16 数组。输出：文件清单与文档在总流中的区间。
    不在内存中拼出完整数据集；一个文档可跨任意多个分片。
    """
    if shard_size <= 0:
        raise ValueError("shard_size 必须大于 0。")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("输出目录非空，请换一个新目录；本脚本不覆盖旧数据。")
    output_dir.mkdir(parents=True, exist_ok=True)
    buffer = np.empty(shard_size, dtype=np.uint16)
    used = 0                       # 当前分片已填多少个 ID
    total = 0                      # 整条数据流已有多少个 ID
    shards, documents = [], []

    def save_shard(count):
        # 仿照原课：第 0 片为 val，其余为 train。按容量切，不按文章划分。
        index = len(shards)
        split = "val" if index == 0 else "train"
        path = output_dir / f"edufineweb_{split}_{index:06d}.npy"
        np.save(path, buffer[:count], allow_pickle=False)
        start = sum(item["tokens"] for item in shards)
        shards.append({
            "file": path.name, "split": split, "tokens": count,
            "stream_start": start, "stream_end": start + count,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })

    for doc_index, tokens in enumerate(encoded_documents):
        if tokens.ndim != 1 or tokens.dtype != np.uint16 or tokens.size == 0:
            raise ValueError("每篇文档必须是一维、非空的 uint16 数组。")
        count = len(tokens)
        documents.append({
            "document": doc_index, "tokens": count,
            "stream_start": total, "stream_end": total + count,
            "first_ids": tokens[:10].tolist(),
        })
        total += count
        offset = 0                 # 当前文档已经写到了哪里
        while offset < count:
            # 规则：本轮写入量 = min(当前片剩余空间, 当前文章剩余 ID 数)。
            take = min(shard_size - used, count - offset)
            buffer[used:used + take] = tokens[offset:offset + take]
            used += take
            offset += take
            if used == shard_size:
                save_shard(used)
                used = 0           # 复用缓冲区；不是丢掉文档的剩余部分。

    if used > 0:
        save_shard(used)            # 只保存有效位置，不把未写入的空位当 Token。
    return {"total_tokens": total, "documents": documents, "shards": shards}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["demo", "fineweb-preview"], default="demo")
    parser.add_argument("--shard-size", type=int, help="每片 Token 数；demo 默认 5，FineWeb 小样本默认 2048")
    parser.add_argument("--output-dir", type=Path, help="必须为空；默认每次创建一个新目录")
    args = parser.parse_args()
    shard_size = args.shard_size if args.shard_size is not None else (5 if args.source == "demo" else 2048)
    if shard_size <= 0:
        parser.error("--shard-size 必须大于 0。")

    # 分词规则缓存也在项目内；第一次使用 tiktoken 可能需要下载这些小文件。
    os.environ["TIKTOKEN_CACHE_DIR"] = str(PROJECT_DIR / ".cache" / "tiktoken")
    try:
        texts, source = read_documents(args.source, CACHE_DIR)
        tokenizer = tiktoken.get_encoding("gpt2")
        created = datetime.now(timezone.utc)
        output_dir = args.output_dir or CACHE_DIR / (args.source + "-" + created.strftime("%Y%m%dT%H%M%S%fZ"))
        # 生成器逐篇编码，交给分片器；没有加载 GPT-2 模型权重。
        encoded = (encode_document(text, tokenizer) for text in texts)
        result = write_shards(encoded, output_dir, shard_size)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"数据准备未完成：{exc}\n可先不带参数运行两句 demo；失败不会回退成 FineWeb 成功。\n")

    report = dict(
        created_at=created.isoformat(), source=source,
        tokenizer="tiktoken/gpt2", tiktoken_version=tiktoken.__version__,
        eot_id=tokenizer.eot_token, dtype="uint16", shard_size=shard_size,
        training_run=False, **result,
    )
    report_path = output_dir / "manifest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("知识块 12 实践（一）：数据准备，不运行模型。")
    print("来源：" + ("FineWeb-Edu 公开预览 20 篇" if source["is_fineweb"] else "两句教学文本，不是 FineWeb-Edu"))
    print(f"{len(texts)} 篇 → {result['total_tokens']} 个 Token（含每篇开头的 EOT）→ {len(result['shards'])} 片")
    for doc in result["documents"][:2]:
        print(f"文档 {doc['document']}：{doc['tokens']} 个 ID；开头 {doc['first_ids']}")
    for shard in result["shards"]:
        print(f"{shard['file']} · {shard['tokens']} 个 Token · 总流区间 [{shard['stream_start']},{shard['stream_end']})")
    print(f"文件目录：{output_dir.resolve()}")
    print(f"清单：{report_path.resolve()}")


if __name__ == "__main__":
    main()
