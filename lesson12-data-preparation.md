# 2.4 数据准备实践（一）：正文怎样变成分片文件

承接知识块 12 的[数据处理全流程笔记](docs/fineweb-edu-sharding-notes.md)（第十三节已补齐本实践至模型训练的串联）。验证循环和定期生成已经学习；进入 2.5 前，先补上此前尚未落到代码的数据准备。它是知识块 12 的实践补充，不新增已完成段落。

配套[代码流转图与源码定位](docs/diagrams/gpt2-data-preparation-flow.md)：左侧 `main()` 调用与返回，右侧逐篇编码、两层循环和写盘判断；底部用本次 8 个 ID 展开三轮搬运。图中 P1—P5、函数名和行号均对应当前代码。

[打开 PNG](docs/diagrams/gpt2-data-preparation-flow.png) · [打开可缩放 SVG](docs/diagrams/gpt2-data-preparation-flow.svg)

![数据准备：正文到 Token 分片的完整代码流转](docs/diagrams/gpt2-data-preparation-flow.png)

## 1. 现在解决什么问题

之前的训练脚本读取一份 Tiny Shakespeare 文本，编码后在内存中切出 `x/y`。面对更大的语料，我们先把文字编码好并保存成多个文件，让后续训练读取现成 ID。

本节输入是多篇文章的正文；输出是多个 `.npy` 文件，每个文件里是一维整数数组。学完应能跟踪一篇文章的 ID 写到了哪个文件，以及分片满了之后剩余 ID 去哪里。

```text
准备阶段，本节实现：
多篇正文 → 逐篇 GPT-2 编码、前置 EOT → uint16 → 按容量写 .npy
                                                        ↓
训练阶段，下一小块接入：
读取分片 → 转成 torch.long → 取 B×T+1 个 ID → x/y → GPT → loss → 更新
```

本节没有模型参数起点：不创建 GPT，不读取预训练权重，也不运行反向传播。使用的 GPT-2 分词器规则已经固定，不重新训练分词器。

## 2. 与原课对应，以及本次运行边界

依据 [视频约 3:13:52—3:18:58](https://www.youtube.com/watch?v=l8pRSuU81PU&t=11632s)，交叉核对 [非官方字幕副本](https://github.com/nolimitkun/ai-kol-wiki/blob/main/sources/karpathy/20240609-l8pRSuU81PU/transcript.md)与作者历史提交 [`69cb21f` 的 fineweb.py](https://github.com/karpathy/build-nanogpt/blob/69cb21f0f9439a8804741912045b90879a573591/fineweb.py)。本次核对了字幕与源码，未直接观看视频画面。

| 项目 | 原课做法 | 本地实践 |
| --- | --- | --- |
| 正文来源 | `datasets.load_dataset` 读取 FineWeb-Edu 的 `sample-10BT` | 默认两句教学文本；可选官方预览接口的前 20 篇 |
| 分词 | `tiktoken` GPT-2，逐篇 `encode_ordinary`，前置 EOT | 相同规则，EOT 使用公开的 `eot_token` 属性 |
| 存储 | 一维 `uint16` NumPy 数组 | 相同 |
| 分片容量 | 100,000,000 个 Token | demo 默认 5；真实小样本入口默认 2048 |
| 文件划分 | 第 0 片为 val，其余为 train | 相同；没有按文档额外切分 |
| 编码执行 | 多进程并行编码、按输入顺序返回 | 顺序编码，方便看清控制流 |
| 长文章 | 大容量分片，处理一次跨界 | `while` 允许一篇文章跨任意多片，适配缩小的容量 |

2026-10-28 已实跑 **两句教学文本**并完成边界测试。官方预览接口在本机多次连接重置，**20 篇真实 FineWeb 数据尚未下载、分片或用于训练**。代码不会在下载失败后自动切换成 demo 并宣称成功。预览 API 读取的是当时公开数据，未固定到视频的数据版本；成功时保存原始响应的 SHA-256，便于识别本次输入。

这不是完整的 10BT 预处理程序；预览接口只用于小规模学习。[数据集说明](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)与[官方 rows 接口说明](https://huggingface.co/docs/dataset-viewer/rows)可用于核对来源。

## 3. 跟着两个真实编码结果走一遍

这里的句子是教学文本，以下 ID 来自实际 GPT-2 tokenizer 运行。

| 文档 | 原文 | 前置 EOT 后的 ID |
| --- | --- | --- |
| 0 | `Hello world.` | `[50256, 15496, 995, 13]` |
| 1 | `Good morning.` | `[50256, 10248, 3329, 13]` |

`50256` 是 EOT。原课把它放在每篇文章开头，用来分隔文档。因此连续读这些 ID，相当于：

```text
EOT  Hello world.  EOT  Good morning.
```

两篇合计 8 个 ID。概念上的总流为：

```text
总流位置：   0      1     2   3      4      5     6   7
Token ID：50256  15496   995  13  50256  10248  3329  13
文档：     └────── 文档 0 ─────┘  └────── 文档 1 ─────┘
分片：     └────────── 第 0 片 ─────────┘  └── 第 1 片 ──┘
```

假设一片最多 **5 个 Token**，得到：

```text
edufineweb_val_000000.npy   = [50256, 15496, 995, 13, 50256]
edufineweb_train_000001.npy = [10248, 3329, 13]
```

第二篇的 EOT 在前一片，其余 ID 在后一片。这就是“文档边界与文件边界不同”。切文件不额外插 EOT，也不把余下的 ID 丢掉。这个只有 8 个 Token 的例子仅用于检查存储，不能拿来衡量模型质量。

原课按 Token 位置划分第 0 片与其他片，因此不能声称 train/val 一定按完整文档隔离。一篇文章可以跨越它们的分界；这里如实保留原课选择。

## 4. 每一轮究竟怎么判断

`write_shards()` 中记住三个数：

- `used`：当前分片已经放了多少个 ID。
- `offset`：当前文章已经写出了多少个 ID。
- `count`：当前文章一共多少个 ID，包含前置 EOT。

每一轮的取数规则是：

```python
take = min(shard_size - used, count - offset)
buffer[used:used + take] = tokens[offset:offset + take]
used += take
offset += take
```

意思是：**剩余空间和文章剩余长度，谁更小，这一轮就写多少。**

| 本轮 | 当前片已有 used | 当前文章尚余 | 可用空间 | take | 接下来 |
| --- | ---: | ---: | ---: | ---: | --- |
| 文档 0，第 1 轮 | 0 | 4 | 5 | 4 | 当前片变为 4，继续读下一篇 |
| 文档 1，第 1 轮 | 4 | 4 | 1 | 1 | 当前片恰好为 5，保存第 0 片；used 清零 |
| 文档 1，第 2 轮 | 0 | 3 | 5 | 3 | 文档写完，当前片为 3 |
| 所有文档结束 | 3 | — | — | — | 保存 `buffer[:3]` 作为最后一片 |

两个判断分别是：

```python
while offset < count:       # 当前文章还没写完，继续搬运。
    ...
    if used == shard_size: # 当前片写满了，立即保存，开始下一片。
        ...

if used > 0:               # 所有文章结束后，保存不足一整片的有效尾部。
    ...
```

因此分片准备阶段不会跳过尾部。之前讨论的“尾部不足一个 batch 就跳过”，发生在**后续训练读取分片**时，二者不是同一个判断。

## 5. 为什么用 uint16，文件里面是什么

GPT-2 的 Token ID 最大为 50256，而 `uint16` 能表示 `0～65535`，所以一个 ID 可以用 2 字节保存。对应代码：

```python
ids = [tokenizer.eot_token] + tokenizer.encode_ordinary(text)
tokens = np.asarray(ids, dtype=np.uint16)
np.save(path, tokens, allow_pickle=False)
```

实际代码先检查取值范围，再转换类型，避免溢出。`.npy` 保存 dtype、shape 等文件头和数组数据，所以文件大小会比 `2 × Token 数` 稍大。

`uint16` 与之前 BF16 不同：这里是存储整数 ID，不是神经网络的浮点精度。下一块读取时会用：

```python
ids = np.load(shard_path, allow_pickle=False)   # uint16，一维
tokens = torch.tensor(ids.astype(np.int64), dtype=torch.long)
```

ID 的数值不变，只把整数存储类型变成后续 PyTorch 层通常使用的 `long`。真正的 Embedding 查表在模型前向中发生。

## 6. 代码阅读顺序与运行

独立文件：[prepare_fineweb.py](prepare_fineweb.py)。上一节所有训练文件保持原样。

| 标记 | 函数 | 输入 → 输出 |
| --- | --- | --- |
| P1 | `read_documents()` | 来源选择 → 正文字符串列表、来源信息 |
| P2/P3 | `encode_document()` | 一篇正文、分词器 → 前置 EOT 的一维 uint16 数组 |
| P4/P5 | `write_shards()` | 逐篇数组、容量、目录 → `.npy` 文件与清单 |
| 入口 | `main()` | 命令行选项 → 串起上述步骤、打印结果、保存 manifest |

先运行两句 demo：

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/prepare_fineweb.py
```

预期核心输出：

```text
来源：两句教学文本，不是 FineWeb-Edu
2 篇 → 8 个 Token（含每篇开头的 EOT）→ 2 片
edufineweb_val_000000.npy · 5 个 Token · 总流区间 [0,5)
edufineweb_train_000001.npy · 3 个 Token · 总流区间 [5,8)
```

每次默认在项目 `.cache/fineweb-preparation/` 下新建运行目录，终端打印绝对路径。可以用 `--output-dir` 指定空目录，程序拒绝覆盖非空目录。`manifest.json` 保存来源、dtype、文档区间、分片区间与文件校验值；它不是模型 checkpoint。

网络可用时再运行真实样本入口：

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/prepare_fineweb.py --source fineweb-preview
```

第一次会请求 20 条公开记录，后续复用项目内缓存；如果接口截断正文，程序拒绝继续。第一次准备 GPT-2 分词规则也可能需要联网。本机已具备 `tiktoken`；新环境在基础依赖外安装 [requirements-data.txt](requirements-data.txt)：

```bash
python -m pip install -r requirements-data.txt
```

本次 [demo 报告](artifacts/lesson12-data-preparation-demo.json)保存实跑结果；三项边界测试通过，包括单篇跨多片、恰好填满、不足一片、范围检查与保护旧文件。真实 FineWeb 的读取与训练尚未验证；下一小块才把分片文件接到 batch 加载器，完成 `B×T+1 → x/y` 与换片。
