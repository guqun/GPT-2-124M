# 知识块 13：验证循环如何接入训练

所属：阶段二 → 单元 2「预训练工程：复现 GPT-2」→ 2.4「多卡与数据」。

## 0. 承接、来源与本次目标

已经讨论过验证 loss 的意义和用途。本块回答工程问题：**用当前参数检查留出的数据，同时让下一次训练仍按原计划继续。** 学完应能找到验证取数、前向、平均、跨卡汇总和模式恢复的位置，并解释为什么验证不更新参数。

依据 [原视频约 3:23—3:25](https://www.youtube.com/watch?v=l8pRSuU81PU&t=12185s)、[非官方自动字幕副本](https://github.com/nolimitkun/ai-kol-wiki/blob/main/sources/karpathy/20240609-l8pRSuU81PU/transcript.md)和作者 [21d3d32：增加 validation split](https://github.com/karpathy/build-nanogpt/blob/21d3d32/train_gpt2.py)。已核对字幕与这份历史代码，未直接观看视频画面。自动字幕中“accumulate the gradients”应结合随后的“没有 backward”及源码理解为**累计验证损失**。

本次新建 [train_validation.py](train_validation.py)，保留 [train_ddp.py](train_ddp.py) 和前面的讲义。

先配合[验证循环代码流转图](docs/diagrams/gpt2-validation-loop-flow.md)阅读：[PNG 全图](docs/diagrams/gpt2-validation-loop-flow.png) / [可缩放 SVG](docs/diagrams/gpt2-validation-loop-flow.svg)。左侧对应主训练循环与验证触发条件，右侧展开同一个验证函数，底部用实际运行记录串起更新 0、2、3 次时的验证。图中标出了 M / T / V 源码标记、行号、张量形状，以及验证结束后返回训练的位置。

| 项目 | 原课此处 | 本机配套 |
|---|---|---|
| 参数起点 | 随机参数 | 同样随机初始化；只读缓存的配置、分词器，不加载权重 |
| 数据 | FineWeb-Edu train/val 分片 | 已有 Tiny Shakespeare，按 Token 位置前 90%/后 10% 切分 |
| 原因与边界 | 大规模预训练 | 立即运行验证循环；不代表完成 FineWeb 分片准备或实跑 |
| 模型 | GPT-2 124M，词表对齐至 50304 | 默认完整 124M、词表 50257；`--model-size tiny` 缩为 2 层/64 维 |
| 执行 | CUDA/NCCL、BF16、compile/SDPA | 复用本项目 eager 模型；CPU/Gloo 或 CUDA/NCCL，精度沿用既有选项 |
| 验证时机 | 初始及每 100 次更新，20 批/卡 | 默认初始、每 2 次更新、最终，2 批/卡；短跑默认训练 3 次 |
| 最后一次验证 | 此历史版本未单独保证 | 本地额外保证最后一次参数更新后评估 |
| 指标归约 | `AVG` | `SUM / world_size`，兼容 Gloo/NCCL |

原视频后续还会修改评估频率。本块不把后来的频率倒填到当前版本。FineWeb 独立加载器、实际数据运行及 CUDA 对照继续保留为待办。

## 1. 先运行这个小实验

本机已授权复用的解释器：

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/train_validation.py --device cpu --model-size tiny
```

在其他机器上使用安装了 `requirements.txt` 依赖的解释器。脚本只读取已有本地文件，不下载模型或语料。`tiny` 保留 GPT-2 分词器及词表，仅缩小网络，便于看控制流。

默认观察顺序是：

```text
验证：已更新 0 次，记录随机模型的起点
训练：更新 1 次
训练：更新 2 次
验证：已更新 2 次
训练：更新 3 次
验证：已更新 3 次，记录最后一次更新后的结果
```

短跑用于观察流程，不能凭三次更新判断泛化质量。默认输出报告在 `.cache/runtime-check/lesson13-report.json`，不保存模型 checkpoint。

## 2. 先把两条数据流分开

阅读 `make_train_val_loaders()`：

```python
split = len(token_ids) * 9 // 10
train_loader = ResettableDataLoaderLite(token_ids[:split], ...)
val_loader = ResettableDataLoaderLite(token_ids[split:], ...)
```

这里的 `...` 省略 B/T/rank/world_size，完整可运行代码在脚本中。

我们**先切分位置，再各自在自己的区间构造 x/y**。假设原始序列有 200 个位置，训练只能使用位置 0—179，验证从位置 180 开始。训练的最后一个目标也不会跨过边界。两边可以出现相同 Token ID（同一个词会在很多位置出现），隔离的是数据位置，不是词表。

这是小语料的位置切分，不是文档级划分或文本去重。原课用不同的磁盘分片实现 train/val 分开。

每个加载器保存自己的 `global_position`。重置 `val_loader` 不影响 `train_loader`；验证结束后训练接着原来的位置读取。

## 3. V1—V6：一次验证内部做什么

核心函数：`evaluate_validation(model, val_loader, runtime, ...)`。

| 标号 | 代码 | 输入、输出与目的 |
|---|---|---|
| V1 | 保存 `model.training`，调用 `model.eval()` | 切换模块行为，例如关闭 Dropout；不改参数的 `requires_grad` |
| V2 | `val_loader.reset()` | 每次验证读取相同的数据位置，便于比较当前参数的变化 |
| V3 | `with torch.no_grad():` | 前向不记录反向计算图；不计算新梯度 |
| V4 | 取 x/y，调用 `forward_and_loss()` | `[B,T]` ID → `[B,T,V]` logits → 本批平均交叉熵标量 |
| V5 | 跨卡 `all_reduce(SUM)`，除以卡数 | 各卡的平均 loss → 所有参与位置的平均 loss |
| V6 | `finally: model.train(was_training)` | 正常返回或报错，都恢复调用前的训练/评估模式 |

`reset()` 不是让所有卡从同一批读取。父类会计算 `current_position = global_position + rank * B * T`。例如 B=2、T=32、两卡：rank 0 起点为 0，rank 1 起点为 64；下一轮分别为 128、192。

注意这两个 API 各司其职：[PyTorch 的 `eval()` 与 `no_grad()` 说明](https://docs.pytorch.org/docs/2.2/notes/autograd.html#locally-disabling-gradient-computation)。当前课程把 Dropout 配置为 0，但仍保留模式切换；有非零 Dropout 时它会影响输出。

## 4. 一批数据如何变成 loss

本项目的 `GPT.forward()` 只返回 logits。复用的 [forward_and_loss()](train_mixed_precision.py) 实际执行：

```python
logits = model(x)
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    y.reshape(-1),
)
```

以 B=2、T=32 为例：

```text
x / y： [2,32]，每卡 64 个预测位置
logits：[2,32,50257]
交叉熵：[64,50257] 个候选分数，对照 [64] 个正确目标 ID
loss：  64 个位置损失的均值，一个标量
```

仍保留因果注意力、x/y 错位以及同一模型结构。我们编写控制流、决定数据/模式/汇总规则；PyTorch 执行张量运算和交叉熵。

## 5. 多批次与多卡如何平均

代码在 `no_grad()` 内累计：

```python
val_loss_accum += loss / validation_batches
```

假设每卡读取 2 个相同尺寸的批次，下面是**教学数字**：

| 所在卡 | 第一批 loss | 第二批 loss | 本卡均值 |
|---|---:|---:|---:|
| GPU 0 | 3.6 | 3.4 | 3.5 |
| GPU 1 | 3.2 | 3.0 | 3.1 |

所有卡调用：

```python
dist.all_reduce(val_loss_accum, op=dist.ReduceOp.SUM)
val_loss_accum /= runtime.world_size
```

每卡最终都得到 `(3.5 + 3.1) / 2 = 3.3`。若 B=2、T=32，则它代表 `2 批 × 2 条 × 32 个位置 × 2 张卡 = 256` 个预测位置的平均损失。

这里传的是一个指标标量，**没有反向传播，也没有参数梯度的 All-Reduce**。所有 rank 都要参与通信，仅打印限制在 rank 0。当前每批/每卡有效位置数相同，均值再平均成立；若以后引入不等批次或忽略位置，改用总损失之和除以总有效位置数。

## 6. T1—T3：返回训练后发生什么

`train_with_validation()` 的三个位置对应：

1. T1：`record_validation(0)`，记录初始模型的验证指标。
2. T2：训练分支按原流程做 `zero_grad → 取 train 数据 → backward → 裁剪 → 设置学习率 → optimizer.step`。
3. T3：更新次数达到间隔或训练结束时，再调用同一个验证函数。

`evaluate_validation()` 不接受优化器，内部不调用 `backward()`、`zero_grad()` 或 `optimizer.step()`。已有 `.grad` 保留原值；下一轮训练开头才清空。AdamW 的历史状态不推进，学习率仍由训练更新序号决定。

| 计数 | 默认单进程小实验 | 用途 |
|---|---:|---|
| `steps` | 3 | 参数总共更新几次 |
| `accum_steps` | 8 | 512 / (B2×T32×R1)，累计几次反向再更新 |
| `val-every` | 2 | 每更新几次参数检查一次 |
| `val-batches` | 2 | 一次检查每卡读取几批，仅汇总指标 |

验证记录进入打印输出和 JSON 报告；没有自动早停、保存最佳参数或按 val loss 调整学习率。前面的讨论解释的是这些规则可以另外实现，本块不默认添加。

## 7. 双进程运行与复习

本机在项目根目录运行：

```bash
/Users/guqun/micrograd/venv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29513 train_validation.py --device cpu --model-size tiny --report .cache/runtime-check/lesson13-ddp-report.json
```

该命令运行两个 CPU/Gloo 进程；不是双 GPU 实验。每卡 B/T 不变时，两进程下训练累积次数 K 从 8 变成 4，单次验证的全局位置数从 128 变成 256。

复习时按此顺序读：`make_train_val_loaders → evaluate_validation → train_with_validation`，最后再看 `main()` 中的环境准备。优先回答：数据有没有混用、验证有没有更新参数、loss 有没有多除一次卡数、训练游标与调度有没有被验证推进。

配套行为检查在 [tests/test_validation.py](tests/test_validation.py)。后续接训练中定期生成文本，再进入 2.5 的 HellaSwag 评测。2026-10-26 用户已亲自完成完整 GPT-2 的单进程 CPU 短跑并反馈“大概了解了”，详见第 9 节；整个 2.4 尚未结算为完成。

## 8. 已运行的检查（2026-10-26）

5 项行为测试通过：数据及错位标签隔离；验证 loss 与联合 batch 一致且保持参数、已有梯度、AdamW 和随机状态；异常时恢复模式；非零 Dropout 下插入验证不改变后续训练；真实双进程 CPU/Gloo 与单进程联合 batch 对照。

双进程小模型的 loss 误差约 `8.88e-16`，验证之后再训练两次的参数误差约 `1.87e-16`，见 [数值对照报告](artifacts/lesson13-validation-equivalence.json)。测试使用 float64、小词表，以检查数值等价关系。

实际文本短跑使用 GPT-2 分词器、3,324,736 参数 tiny 模型与 FP32：

| 运行方式 | 更新 0 次时 val loss | 更新 2 次时 | 更新 3 次时 | 每次验证的全局位置数 |
|---|---:|---:|---:|---:|
| [单进程 CPU](artifacts/lesson13-cpu-report.json) | 10.799194 | 10.646638 | 10.610084 | 128 |
| [双进程 CPU/Gloo](artifacts/lesson13-ddp-cpu-report.json) | 10.768393 | 10.608240 | 10.574162 | 256 |

两行的验证数据量不同，loss 不能直接用来判断多进程是否更好。严格等价比较使用前面的联合 batch 测试。以上不是 CUDA/NCCL、FineWeb-Edu 或完整 124M 训练结果。

## 9. 用户实际运行与答疑归档（2026-10-26）

用户未传 `--model-size tiny`，实际运行默认的完整 GPT-2：124,439,808 个参数、12 层、768 维、12 个注意力头；CPU/FP32、R=1，仍从随机参数开始，只复用模型配置与分词器。

| 动作 | 已完成更新数 | loss | 学习率 |
|---|---:|---:|---:|
| 验证 | 0 | 11.036055 | 不更新 |
| 训练 | 1 | 10.958789 | 0.0006 |
| 训练 | 2 | 9.674578 | 0.0006 |
| 验证 | 2 | 9.143974 | 不更新 |
| 训练 | 3 | 8.898489 | 0.00033 |
| 验证 | 3 | 9.071917 | 不更新 |

每次训练更新处理 B×T×K×R=2×32×8×1=512 个预测位置，三次共 1,536 个；每次验证固定读取 128 个位置。第 2 次命中验证间隔，第 3 次因为是最后一次而额外验证。训练 loss 来自更新前，随后验证使用更新后参数，且两者数据不同，不要求数值相等。

原始报告已原样保存为 [lesson13-user-gpt2-cpu-report.json](artifacts/lesson13-user-gpt2-cpu-report.json)，避免后续运行覆盖缓存报告。流程图底部仍使用第 8 节的 tiny 实测数值；模型规模不同，不能要求与本次 loss 一致。

本次说明训练/验证流程已跑通、固定验证片段上的预测改善，不表示完成正式 GPT-2 预训练或证明泛化质量。指标报告未保存模型权重；重新启动该脚本会重新初始化。开头的 `NotOpenSSLWarning` 没有阻止本次本地训练完成。

完整答疑、参数起点、验证如何影响训练决策与学习状态见[验证循环复习笔记](docs/validation-loop-notes.md)。本知识块记为已学习、初步理解、完整模型 CPU 实践完成；下一步训练中定期生成文本。FineWeb、用户 DDP 多进程实践、CUDA/NCCL 等待办和 1/5、5/32 完整段落计数保留。
