# 2.4 数据准备实践（二）：读取分片并组成 batch

接在 [正文到分片文件](lesson12-data-preparation.md)之后。本节新建 [load_shards.py](load_shards.py)，前面的准备脚本、训练入口与配图保留。

完整复习见[数据处理全流程笔记](docs/fineweb-edu-sharding-notes.md)第十三节：从正文写盘到预测位置、微批次累积和模型训练。

## 1. 目的、输入与输出

上一块把正文保存为 `.npy` 文件；这一块让文件里的 Token ID 持续成为训练输入。

- 输入：分片目录、用途 train/val、batch 大小 B、序列长度 T。
- 输出：每次调用返回两个 `torch.long` 张量 `x/y`，形状都是 `[B,T]`。
- 需要记住的状态：当前是哪一个分片、当前在片内读到哪里。
- 本节默认只展示取数与设备位置，不创建模型、不计算 loss、不更新参数。

```text
磁盘 .npy → np.load → 内存中的 NumPy uint16 数组
    → 转成 CPU torch.long → 从当前起点取 B×T+1 个 ID
    → 整体错开一位 → 各自排成 [B,T] → 返回 x、y
    → x.to(device)、y.to(device) → 后续模型计算
```

对应[原视频约 3:17:58—3:18:58](https://www.youtube.com/watch?v=l8pRSuU81PU&t=11878s)，已核对[非官方字幕副本](https://github.com/nolimitkun/ai-kol-wiki/blob/main/sources/karpathy/20240609-l8pRSuU81PU/transcript.md)与作者 [`69cb21f` 的 load_tokens / DataLoaderLite](https://github.com/karpathy/build-nanogpt/blob/69cb21f0f9439a8804741912045b90879a573591/train_gpt2.py#L211)。未直接观看视频画面。沿用 EOT、普通因果注意力、尾部不足时换片；没有添加文档级遮罩或跨文件拼 batch。

## 2. 本次实际使用哪些文件

上一块的 5+3 示例太短：唯一 train 文件只有 3 个 ID，无法提供本节需要的 `2×3+1=7` 个 ID。保持这些旧文件原样，本节 `make_demo_shards()` 复用已归档的两句编码，重复 5 组得到 40 个 ID，按每片 16 个另存：

```text
edufineweb_val_000000.npy    16 个 ID
edufineweb_train_000001.npy  16 个 ID
edufineweb_train_000002.npy   8 个 ID
```

这是重复教学文本，**不是 FineWeb 实际语料**。真实 FineWeb 下载仍未成功。本次不重新分词、不联网；重复相同句子只用于跟踪数组位置，因此不同文件可以出现相同数值。

`split="train"` 只选择两个 train 文件。排序后的列表为：

```python
[
    "edufineweb_train_000001.npy",  # loader 内部索引 0
    "edufineweb_train_000002.npy",  # loader 内部索引 1
]
```

内部 `current_shard=0` 表示当前列表的第 0 项，不表示文件名一定是 `000000`。

## 3. 先把一个文件读到内存

对应 `load_tokens()`，标记 S1：

```python
array = np.load(filename, allow_pickle=False)
tokens = torch.from_numpy(array.astype(np.int64))
```

实际代码在两句之间检查一维 uint16 格式与 GPT-2 ID 范围。

| 对象 | 类型与形状 | 位置 | 内容 |
| --- | --- | --- | --- |
| `.npy` 文件 | NumPy 文件格式 | 磁盘 | 数组头与 uint16 ID 数据 |
| `array` | NumPy uint16 `[N]` | CPU 内存 | 当前整个分片的 ID |
| `tokens` | PyTorch long/int64 `[N]` | CPU 内存 | 同一组整数 ID |

`astype(np.int64)` 转换整数存储类型；`from_numpy` 复用转换后的数组内存。这样显式适配本机 PyTorch 的输入类型。它没有将 ID 变成 Embedding，也没有搬到 GPU。

加载器在换片时读取下一个文件。由于本轮返回的 x/y 仍引用旧 Tensor，换片时新旧分片可能短暂同时存在；不是从磁盘删除旧文件，也不是把所有分片永久放进内存。

## 4. 取一批数据：核心是四行代码

对应 `ShardDataLoaderLite.next_batch()` 的 S4/S5：

```python
start = self.current_position
buf = self.tokens[start:start + B * T + 1]
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)
```

本节 `B=2、T=3`：一批 2 条输入序列，每条 3 个 Token，一共 6 个预测位置。为了让第 6 个位置也有下一个 Token 作为答案，需要取 **7 个 ID**。

第一次 `start=0`，实际读取：

```text
片内位置：  0      1      2    3      4      5      6
buf：    50256  15496   995   13   50256  10248  3329
```

先错开一位：

```text
x 的 6 个数：[50256, 15496, 995, 13, 50256, 10248]
y 的 6 个数：[15496, 995, 13, 50256, 10248, 3329]
```

再由 `.view(2,3)` 排成两行三列：

```python
x = [[50256, 15496,   995],
     [   13, 50256, 10248]]

y = [[15496,   995,   13],
     [50256, 10248, 3329]]
```

每个 `y[b,t]` 都是相应 `x[b,t]` 后面紧接着的 Token。例如第一行 `x[0,2]=995`，答案是 `y[0,2]=13`。训练时每个位置预测自己的下一 Token；模型实际输入是 x，y 交给损失函数作监督。

这里一行是长度 T 的连续窗口，并不保证是一篇完整文章；EOT 作为普通的特殊 Token ID 留在窗口中。不同 batch 行是独立输入序列，注意力不会跨 batch 行读取。

## 5. 为什么读 7 个，却只前进 6 个

对应 S6。单进程时：

```python
self.global_position += B * T  # 本节前进 6
```

第一批最后的答案位于片内位置 6。第二批从位置 6 开始，使它也有机会作为输入，学习预测位置 7。

```text
第一批：x 用位置 0..5， y 用位置 1..6
第二批：x 用位置 6..11，y 用位置 7..12
```

所以两批的输入位置没有重叠；重用的是“上一批末尾标签 → 下一批开头输入”。前进 7 会漏掉位置 6 作为输入的那次预测。

代码写成 `self.global_position += self.stride`，因为它还兼容前面学过的多进程。`stride=B*T*world_size`；现在 world_size=1，正好是 6。

## 6. 什么条件下换片

S7 检查**下一轮**是否装得下，单进程相当于：

```python
if next_start + B * T + 1 > len(self.tokens):
    # 切到下一片，从它的起点开始。
```

对于第一个 train 分片，`N=16`：

| 已完成的取批 | 下一起点 | 判定下一批需要的右边界 | 判断 | 接下来 |
| --- | ---: | --- | --- | --- |
| 第一次，从 0 读取 | 6 | `6+6+1=13` | `13 <= 16` | 留在当前片 |
| 第二次，从 6 读取 | 12 | `12+6+1=19` | `19 > 16` | 切到下一片，起点归 0 |

此时从位置 12 往后只有 4 个 ID，不足 7；这些 ID 不与下一片拼接。位置 12 已作为上一批最后的标签出现过，所以不能把这 4 个位置全称为“从未使用过的 Token”。

判断用 `>`，不是 `>=`。如果右边界恰好等于 N，Python 左闭右开的切片仍能取足数量，必须保留这一批。

若某个文件整片都短于 `B*T+1`，本地实现也会跳过它；如果所有文件都太短，则明确报错，避免无限跳片。读完最后一片后按文件列表循环回第一片，不打乱顺序。

本次四次取批已经实跑：

| 调用 | 来自哪个 train 文件 | 本次起点 | 下一次文件 / 起点 |
| --- | --- | ---: | --- |
| 1 | `000001`，N=16 | 0 | `000001` / 6 |
| 2 | `000001`，N=16 | 6 | `000002` / 0 |
| 3 | `000002`，N=8 | 0 | `000001` / 0 |
| 4 | `000001`，N=16 | 0 | `000001` / 6 |

虽然 S7 在 `return` 前可能已经换片，返回的 x/y 仍然是刚才那一批的旧片数据；换片是在为下一次调用做准备。

## 7. 怎么接到已有训练流程

已有 [train_generation.py](train_generation.py) 的训练循环仍然是：

```python
x, y = train_loader.next_batch()
x, y = x.to(runtime.device), y.to(runtime.device)
logits, loss = forward_and_loss(model, x, y, runtime.device, precision)
```

新的 loader 保留同样的 `next_batch() → x,y` 接口。数据来源从整段文本换成磁盘分片，后面的前向、loss、梯度累积与 AdamW 仍可以接收相同的 `[B,T]` 整数张量。

本次新增的是独立读取预览入口，**没有修改旧训练脚本的 loader 构造或把 FineWeb 接入完整训练**。这段旧代码用于说明衔接位置，不代表已经完成端到端分片训练。

`load_shards.py` 的 M2 实际执行 `x.to(device)` 和 `y.to(device)`，默认 device=cpu。若选择可用 CUDA/MPS，才把当前 batch 搬入该设备；分片 Tensor 留在 CPU。正式训练还需要模型参数在同一计算设备。

## 8. 多进程与原课差异

先掌握单进程即可；这里保留与已有 DDP 代码一致的接口：

```text
本 rank 起点 = global_position + rank × B × T
所有 rank 下一轮全局起点 = global_position + B × T × R
换片条件 = 下一轮全局起点 + B × T × R + 1 > N
```

各进程有各自的 loader，必须读取相同的有序文件列表，并调用相同次数 `next_batch()`。它们因而同轮换片；不需要为游标再做一次 All-Reduce。

作者 `69cb21f` 的历史实现使用各 rank 的局部游标判断，多卡在边界可能不同轮换片。本地延续知识块 11 已说明的全局游标修正，不把修正后的边界行为称为原课逐行复刻。文件过滤、ID 校验、跳过整片过短数据和 `reset()` 也是本地接口补充；单进程取 `B*T+1`、错位、推进及普通片内尾部跳过规则保持。

## 9. 运行与已验证范围

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/load_shards.py
```

默认新建 demo 目录，打印四次完整 x/y。已有真实分片时可指定目录：

```bash
python load_shards.py --data-dir /实际的分片目录 --split train --batch-size 2 --seq-len 32
```

CPU demo 报告见 [lesson12-shard-batches-cpu.json](artifacts/lesson12-shard-batches-cpu.json)。5 项检查通过：错位/步长/换片/回绕、恰好够一批、train/val 隔离与 reset、双 rank 的取数与单进程全局 batch 对照，以及类型与异常数据。

双 rank 检查使用 CPU 上两个 loader 对象核对数据位置，没有启动真实 DDP，也没有进行 CUDA/NCCL 或模型训练。本次没有重跑前面训练课的完整模型实验，也不把协作验证计作用户独立完成。
