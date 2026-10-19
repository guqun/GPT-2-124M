# 阶段二 · 单元 2 · 2.2 · 知识块 8：GPU 执行、训练计时与 TF32

配套：[benchmark_training.py](benchmark_training.py)。阅读顺序：先读本页的目标与输入输出，再看 `main()` 的 M1—M4，最后看 `benchmark_steps()` 的 B1—B4。

**2026-10-19：开始学习。** 用户已反馈参数初始化“大概清楚了”，并确认 2.1 整节学习完成；七块内容已讲解并核对配套实验，自检保留作复习。现在进入 2.2 提速，当前开始第一个知识块，不把 CPU 跑通记成 GPU 提速实验已经完成。

**同日学习跟进：** 已讨论 device/MPS、预热与测量、每步及总体吞吐率、TF32，用户反馈“大概了解了”。现在接续 [知识块 9：BF16 自动混合精度](lesson09.md)。本页保留知识块 8 的内容与实验，GPU 实测仍待进行。

**本轮归档：** 已学的设备、计时、TF32 与 BF16 讨论合并在 [GPU 训练提速专题笔记](docs/gpu-training-notes.md)，含日志复算和精度流程图。本页继续保留知识块 8 的原始操作讲义。

## 0. 来源、承接和目标

主线是 Karpathy [Let's reproduce GPT-2 (124M)，约 1:22—1:39](https://www.youtube.com/watch?v=l8pRSuU81PU&t=4938s)，实现核对作者 [5265b20：加入计时和 TF32](https://github.com/karpathy/build-nanogpt/commit/5265b2042aaaa84943313002671b8b3ca7749b06)。本次核对了[非官方自动字幕副本](https://github.com/nolimitkun/ai-kol-wiki/blob/main/sources/karpathy/20240609-l8pRSuU81PU/transcript.md)对应时间段与官方历史提交，没有直接观看视频画面。字幕可能转录错数字和术语；计算用代码和公式核验。

视频在约 1:22 从初始化转到硬件能力，1:24—1:32 介绍精度、带宽和 Tensor Core，1:32—1:36 加入训练计时与 Token/s，1:36—1:39 开启 TF32 并比较。后面才是 BF16、`torch.compile`、FlashAttention 和词表尺寸调整。这些会逐块学习。

2.1 已接通“取数据 → 前向 → loss → 反向 → 更新”。本节研究同样的计算怎样在硬件上执行，以及怎样判断运行速度是否改善。

学完应能：

1. 说明模型、x/y 分别在哪里，`.to(device)` 做什么。
2. 说明 GPU 异步执行为何需要同步计时。
3. 计算每步耗时、训练吞吐率，分清预热和正式测量。
4. 解释 TF32 改变矩阵乘法内部精度，而 Tensor 的存储 dtype 仍可为 FP32。
5. 在相同硬件、B/T、数据、种子和步数下做对照，记录速度及数值影响。

参数起点仍是随机初始化的完整 GPT-2 124M，使用上一块的共享与初始化规则；语料仍为完整 Tiny Shakespeare。不读取预训练权重，不承接上一脚本训练后的参数。当前 Mac 的 PyTorch 2.2.2 检测到 CUDA=False、MPS=False，因此本机仅验证 CPU 流程。GPU 实验保留，待 GPU 环境实际运行。

## 1. GPU 加速的是哪些计算

以已学过的 lm_head 为例：

```text
H [B,T,768] × Eᵀ [768,50257] → logits [B,T,50257]
```

对一个输出元素，需要多个乘积再求和。不同输出元素有大量可以并行的计算；QKV、注意力和 MLP 中也有矩阵乘法。PyTorch 调用设备后端执行这些数值运算，Python 仍在 CPU 上组织训练流程。

GPU 不意味着必须手写逐个线程。现阶段模型代码中的矩阵乘法和线性层由框架负责调度。视频里的 Tensor Core 是 NVIDIA GPU 中专门加速支持格式的矩阵乘加的计算单元。

硬件至少要区分三件事：

| 能力 | 含义 | 对训练的影响 |
| --- | --- | --- |
| 计算吞吐 | 每秒能完成多少浮点运算 | 矩阵乘法能算多快 |
| 显存容量 | 能放多少数据 | 参数、梯度、优化器状态和激活是否装得下 |
| 显存带宽 | 每秒能在显存与计算单元之间搬多少数据 | 计算单元是否在等数据 |

标称算力提高几倍，不代表整个训练步骤一定提高相同比例。取数据、传输、非矩阵运算、内存访问和 CPU 调度也占时间。具体瓶颈需要测量，不能只根据模型名字判断。

## 2. 输入和输出，如何放到设备上

```python
model.to(device)             # 参数和注册 buffer
x, y = x.to(device), y.to(device)
logits = model(x)            # [B,T,V]
loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
loss.backward()
optimizer.step()
```

`device='cuda'` 表示使用 NVIDIA CUDA 后端；`mps` 是 PyTorch 的 Apple GPU 后端；`cpu` 使用 CPU。是否可用取决于机器与安装的 PyTorch，本文件会检查；明确指定不可用设备会报错，不会静默换成 CPU。

| 数据 | 本节形状/类型 | 去向 |
| --- | --- | --- |
| 输入 x | `[B,T]`，整数 ID | 与模型相同设备，供 Embedding 查表 |
| 标签 y | `[B,T]`，整数 ID | 与 logits 相同设备，供交叉熵计算 |
| 模型参数 | 各层矩阵/向量，FP32 | 所选设备，参与前向与更新 |
| logits | `[B,T,50257]`，FP32 | 交叉熵 |
| loss | 一个浮点标量 | 反向传播；随后读数用于日志 |

`.to(device)` 改变存储和执行的位置，不会把 Token ID 变成 Embedding，也不会训练参数。模型迁移后再创建优化器。代码会在迁移后重新确认共享引用，不重新填写随机初值。

## 3. 为什么 GPU 上不能只在两边读时钟

CPU 向 GPU 提交计算请求后，通常可以继续执行后续 Python 代码；GPU 此时可能还在计算。于是：

```text
CPU：提交前向 → 提交反向 → 提交更新 → 读时钟
GPU：             实际执行这些计算………………
```

如果直接结束计时，可能只测到提交命令的时间。因此本节计时顺序是：

```text
等待之前的工作完成 → 开始时间
    ↓
取 batch → 搬到设备 → 清梯度 → 前向/loss → 反向 → 更新
    ↓
等待本步 GPU 工作完成 → 结束时间
    ↓
读取 loss、打印日志
```

CUDA 使用 `torch.cuda.synchronize(device)`，MPS 使用 `torch.mps.synchronize()`，CPU 路径不调用 GPU API。当前版本语义可核对 [PyTorch 2.2.2 CUDA 实现](https://github.com/pytorch/pytorch/blob/v2.2.2/torch/cuda/__init__.py)。计时使用 `time.perf_counter()` 的时间差。

本节计时**包含**取批、设备传输、前向、loss、反向、优化器更新和末尾等待；**不包含**文件读取、分词、模型构建、日志和写报告。loss 与 logits 不提前 `.item()`，避免给计时区间增加不必要的等待。旧 `next_token_loss` 的 ID 范围检查使用 `.item()`；本文件把 ID 检查移到计时前的 CPU 数据上，计时内直接调用相同的交叉熵公式。

## 4. 看每步耗时，也看 Token/s

每一步训练处理 B 条序列，每条有 T 个监督位置，因此：

\[
\text{每步 Token 数}=B\times T,\qquad
\text{Token/s}=\frac{B\times T}{\text{每步耗时（秒）}}
\]

手算示例：B=4、T=32，一步耗时 0.25 秒，那么吞吐率是 `128/0.25=512 Token/s`。这是为了说明公式的假设数据，不是本机测量结果。

这是**训练**吞吐率：一个 batch 的所有位置参与预测和 loss；不表示模型自由生成了这么多个新 Token。x/y 是相互错位的监督对，也不把输入和标签各算一遍而乘 2。多步汇总使用“总 Token 数 ÷ 总耗时”，不是各步 Token/s 的算术平均。

首次执行可能包含运行时准备、优化器状态分配等开销。本节默认先做 2 步预热，再统计 5 步。预热仍然更新参数并前进到下一批，不回滚；两组对照必须采用相同的预热条件。

## 5. TF32：保留 FP32 Tensor，允许更快的内部乘法

基线设置：

```python
torch.set_float32_matmul_precision('highest')
```

原课 A100 上允许 TF32 的设置：

```python
torch.set_float32_matmul_precision('high')
```

FP32 的小数部分显式存 23 位；TF32 的乘法输入使用较少的有效位，指数范围保持与 FP32 类似，并使用 FP32 累加。因此矩阵乘法可能更快，但结果会有数值误差，不能要求与基线逐位相同。权重和输出仍可显示 `torch.float32`，也不会因此把整张参数表的存储空间减半。

`high` 是对内部精度的许可，API 还允许后端选择 BF16 组合算法；它不是一个新的 Tensor 存储 dtype。精确行为依设备和版本而定。本项目按当前已安装的 [PyTorch 2.2.2 文档与实现](https://github.com/pytorch/pytorch/blob/v2.2.2/torch/__init__.py)核对，不把原课某次 A100 的效果承诺给所有硬件。

本文件的 high 实验限定 CUDA 且计算能力至少 8.0。CPU/MPS 会明确拒绝此对照，避免把无效开关测出的波动误称为 TF32 提速。BF16 混合精度将在下一块讨论，此处不执行 `model.half()` 或 `autocast`。

## 6. 如何运行

当前 Mac 使用已经有依赖的解释器，先看设备，再运行基线：

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/benchmark_training.py --inspect-only
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/benchmark_training.py --device cpu
```

GPU 机器在仓库目录下，用该机器已安装依赖的 Python，执行两次独立的新运行。例如按原课 B=16/T=1024 设置：

```bash
python benchmark_training.py --device cuda --batch-size 16 --seq-len 1024 --warmup-steps 5 --steps 50 --matmul-precision highest --report artifacts/lesson08-cuda-highest.json
python benchmark_training.py --device cuda --batch-size 16 --seq-len 1024 --warmup-steps 5 --steps 50 --matmul-precision high --report artifacts/lesson08-cuda-high.json
```

如果当前 GPU 装不下，**两组一起**减小 B，再重新运行；保持其他条件一致。实际 GPU 时序还会受其他进程、温度、频率等影响，正式对照需重复测量。不要拿本机 B=4/T=32 与视频 B=16/T=1024 的 ms/步直接算加速比。

代码中的输入输出和计算沿用前面课程。新内容集中在 M1 设备/精度、M3 设备迁移、B1/B4 同步计时和汇总。参数统计、JSON 报告是辅助观察，不是新的训练算法。

## 7. 本次验证与下一步

已归档 [本机 CPU 运行报告](artifacts/lesson08-cpu-baseline.json)：完整 124,439,808 个参数、共享保持、2 步预热加 5 步测量。报告给出实际耗时、设备、版本、B/T、种子和每步 loss。这个短实验验证计时与训练流程，不能证明 GPU/TF32 加速，也不能证明模型已完成预训练。

新增测试核验：计时包装与普通循环的初值/数据相同时，5 次更新结果一致；预热真的执行更新但不计入吞吐统计；计时两端都等待；吞吐使用总量/总时间；无 GPU 和 CPU 请求 TF32 时明确报错。连同已有测试共 25 项通过。

GPU 对照记录时至少填写以下表格，再判断是否值得采用：

| 运行条件 | 平均 ms/步 | 总 Token/s | loss 是否有限/初始差异 | GPU 与版本 |
| --- | --- | --- | --- | --- |
| FP32 highest | 待 GPU 实跑 | 待 GPU 实跑 | 待记录 | 待记录 |
| 允许 TF32 high | 待 GPU 实跑 | 待 GPU 实跑 | 待记录 | 与上行相同 |

短期 loss 只能发现明显数值问题；长期质量还需要相同验证数据与训练预算下的比较。当前先理解硬件执行、正确计时与 TF32；后续按视频顺序进入 BF16 混合精度。
