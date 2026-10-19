# 阶段二 · 单元 2 · 2.1 · 知识块 7：参数初始化

配套代码：[train_initialization.py](train_initialization.py)。保留上一块的 [train_weight_tying.py](train_weight_tying.py) 与 [lesson06.md](lesson06.md)。上一块决定哪些位置共用一份参数；本块决定训练开始时，这些参数先填什么数值。

**学习状态（2026-10-19）：已学习、初步理解。** 用户反馈“大概清楚了”，并已讨论 lm_head 的创建、共享与初始化位置。2.1 内容和实验已做[收尾回顾](docs/section-2.1-review.md)，用户确认整节学习完成；自检保留作复习，当前开始 [2.2 提速的第一块](lesson08.md)。代码验证与用户独立掌握分别记录。

## 0. 本块学什么、为什么学

我们已经能连续读取 batch、计算 loss、反传和更新，而且输入/输出表已经共享。接下来学习三个问题：普通权重的随机数应有多大；哪些参数从零或一开始；深层残差网络为什么需要更小的分支初值。

预期能力是：读懂 `normal_`、`zeros_`、`apply` 在本项目中的作用，知道初始化与训练更新分别发生在什么时候，解释 `0.02 / sqrt(2L)` 中的 2 和 L，并能检查实际初值是否符合规则。

来源为 [Karpathy 视频约 1:13:03—1:22:11](https://www.youtube.com/watch?v=l8pRSuU81PU&t=4383s)，实现锁定作者 [9ac321e：gpt-2 initialization](https://github.com/karpathy/build-nanogpt/commit/9ac321e299f06c0414bd577fd2776f4562283a8a)。本次核对[非官方自动字幕副本](https://github.com/nolimitkun/ai-kol-wiki/blob/main/sources/karpathy/20240609-l8pRSuU81PU/transcript.md)与历史源码，没有直接观看视频画面；时间用于导航。

本课继续从随机参数开始，不加载预训练权重。完整 Tiny Shakespeare、B=4/T=32、顺序加载器、AdamW lr=3e-4 和 50 步训练沿用上一块。数据与模型形状不变，主要变化是初始化规则。

## 1. 初始化在什么时候发生

```text
读取模型配置：规定词表大小、层数、维度
    ↓
创建模型：建立各层参数，PyTorch 先填入默认初值
    ↓
建立 wte 与 lm_head 的共享关系
    ↓
【本块】按课程规则填写参数初值
    ↓
创建优化器
    ↓
batch 1：前向 → loss → backward → step
    ↓ 保留更新后的参数
batch 2：前向 → loss → backward → step
    ↓
继续训练……
```

初始化需要的输入是**模型结构与初始化规则**，输出是**已经具有初始数值的可训练参数**。这时不需要训练文本，也不根据标签计算梯度。Token ID 和位置编号仍是整数索引；随机填写的是它们所查询的向量表，以及各层权重矩阵。

训练过程中，参数由梯度和优化器继续更新。若在每个 batch 前重新初始化，就会覆盖之前学到的结果。本文件的初始化调用放在 `[M2]`，位于训练循环外。

## 2. 随机初始化，具体在随机什么

之前使用 `nn.Linear`、`nn.Embedding` 的各自默认初值。本节明确指定大多数权重元素从以下分布独立采样：

\[
w\sim\mathcal N(0,\sigma^2),\qquad \sigma=0.02
\]

均值 0：数值围绕零分布；标准差 0.02：控制散布尺度。不是把每个权重都设为 0.02，也不是限定在 [-0.02, 0.02] 范围内。样本可以有正有负，某些数值的绝对值可以超过 0.02。

```python
nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

`module.weight` 已经有规定的形状。`normal_` 在原来的存储中填写随机数，返回后形状不变。下划线表示原地修改；这里的 `std` 参数直接接收标准差，无需再平方。

随机差异有助于不同单元从不同起点学习；尺度则影响前向激活和反向梯度的数值。尺度选择需要结合网络结构，不能认为随机数越小越好。当前的 0.02 是复现这节课程采用的规则，不是所有模型统一的最优值。

## 3. 哪些参数按什么规则填写

本课使用 12 个 Transformer Block，记 L=12。

| 参数 | 初值规则 | 作用 |
| --- | --- | --- |
| Token Embedding；与它共享的 lm_head | 正态分布，均值 0、标准差 0.02 | 输入表示与输出读出共同的起点 |
| Position Embedding | 正态分布，均值 0、标准差 0.02 | 各位置可学习向量的起点 |
| QKV 投影、MLP 的扩展层权重 | 正态分布，均值 0、标准差 0.02 | 各分支内部的初始变换 |
| Attention 输出投影 `attn.c_proj.weight` | 标准差 `0.02 / sqrt(2L)` | 减小写回残差主干的初始贡献 |
| MLP 输出投影 `mlp.c_proj.weight` | 标准差 `0.02 / sqrt(2L)` | 同上 |
| 各 Linear 的 bias（如果存在） | 全部为 0 | 从零偏置开始 |
| LayerNorm 的 gamma / beta | 分别为 1 / 0 | 保留 PyTorch 构造时的默认值 |

LayerNorm 的 gamma=1、beta=0 只表示可学习仿射部分从单位缩放、零平移开始；LayerNorm 仍然执行标准化。线性层偏置为零，也不代表其随机权重为零。

本块沿用 Karpathy 的实现：Token / Position 两类 Embedding 都用 0.02。[OpenAI 原始公开源码](https://github.com/openai/gpt-2/blob/master/src/model.py)中的位置表曾用 0.01，视频明确说明在课程代码中统一为 0.02。当前复现的是这节课程的初始化设置，不混称为所有历史实现完全相同。

## 4. 残差相加为什么需要考虑尺度

已学过一个 Block 中两次加回主干：

\[
r_l=x_l+\mathrm{Attention}(\mathrm{LN}_1(x_l))
\]

\[
x_{l+1}=r_l+\mathrm{MLP}(\mathrm{LN}_2(r_l))
\]

12 个 Block 共带来 24 次分支贡献相加。为说明多次累加的尺度，视频先看一个独立随机实验：令每次贡献 \(\epsilon_k\) 均值为 0、方差为 1，彼此独立。

\[
s_N=\epsilon_1+\epsilon_2+\cdots+\epsilon_N
\]

在这些假设下，方差相加：

\[
\operatorname{Var}(s_N)=N,\qquad
\operatorname{Std}(s_N)=\sqrt N
\]

所以加 100 次，标准差约为 10。若每次贡献先除以 \(\sqrt N\)：

\[
\widetilde{s}_N=\sum_{k=1}^{N}\frac{\epsilon_k}{\sqrt N}
\]

\[
\operatorname{Var}(\widetilde{s}_N)=N\times\frac1N=1
\]

这是初始化缩放的动机。真实网络中各分支的输出不一定独立，输入与分支也有相关性；这个理想实验不保证真实网络每层的标准差精确不变。

## 5. 为什么缩放的是分支最后的投影

Attention 拼接各头之后，要经过输出投影；MLP 激活后，要经过压回 C 维的线性层。它们的结果随后加回残差主干。因此，将这两类投影权重的初始标准差缩小，能控制分支一开始写回的数值尺度。

课程的具体规则是：

\[
\sigma_{\text{残差输出投影}}=\frac{0.02}{\sqrt{2L}}
\]

这里 **L 是 Block 数量，2 表示每个 Block 有 Attention、MLP 两条残差分支**。与 batch 数、Token 数和注意力头数无关。

对于 L=12：

\[
\frac{0.02}{\sqrt{24}}\approx0.00408248
\]

这项调整发生在**填写初值时**。前向仍使用已经学过的矩阵乘法与残差相加，代码没有在每次相加后再除以 \(\sqrt{24}\)。也没有缩小主干 x 本身。训练开始后，这些权重照常更新，标准差不被强制固定。

注意区分：Attention 打分里的 \(1/\sqrt{d_{head}}\) 在每次前向中缩放 QK 分数；这里的 \(1/\sqrt{2L}\) 在训练开始前缩放残差输出投影的初值。

## 6. 对照代码的四个动作

在 `initialize_like_course()` 中按 `[I1]`—`[I4]` 阅读：

1. 给每个 Block 的两个 `c_proj` 标记 `NANOGPT_SCALE_INIT`。这个名字来自作者代码，是普通属性，标识哪些层需要较小初值。
2. 对 Linear：选择标准差，调用 `normal_` 填写 weight；若有 bias，就用 `zeros_` 填零。
3. 对 Embedding：用标准差 0.02 填写 weight。LayerNorm 保留默认 gamma=1、beta=0。
4. 调用 `model.apply(init_module)`，由 PyTorch 遍历子模块并执行我们传入的函数。

`apply` 遍历模块。共享表分别被 Embedding 模块和 Linear 模块持有，因此按视频原实现，会被填两次，最终保留后一次的值。这两次都采用标准差 0.02，仍是同一个 Parameter。整个过程仅发生在训练准备阶段；它与优化器按参数对象去重、每步更新一次，是不同的遍历。

公共 `gpt2_model.py` 已经有默认初始化功能。本节先关闭它，再在独立文件中按照作者这一提交的遍历顺序显式填写。公共实现先统一初始化、再单独处理残差投影，分布规则相同，但随机数消耗顺序不同。这样安排便于本节逐句核对，保留之前各知识块的运行行为。

## 7. 运行、观察与自检

先观察初值，不执行训练：

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/train_initialization.py --inspect-only
```

重点看两类数：普通权重的实测标准差接近 0.02；两个残差输出投影接近 0.004082。有限样本的均值、标准差有随机偏差，不要求恰好等于目标。

脚本还展示视频中的 100 次随机累加实验。本机 seed=42、向量宽度 768 时，不缩放的实测标准差为 `9.8440`，缩放后为 `0.9844`，对应理论尺度 10 与 1。这是独立小实验，不是对真实 Transformer 激活的测量，也不会消耗模型使用的全局随机数状态。

默认执行完整 50 步训练：

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/train_initialization.py
```

当前 Mac 复用已授权的课程解释器。在其他机器按 README 创建项目 `.venv` 后，可在项目目录使用 `.venv/bin/python train_initialization.py`。VS Code 需选择安装了 PyTorch 的解释器。本节使用 PyTorch 项目中的新文件；此前查看的 `gpt-2/src/sample.py` 属于另一份 TensorFlow 实现。

默认报告为 `.cache/runtime-check/lesson07-report.json`；本次完整模型的归档为 [lesson07-initialization-report.json](artifacts/lesson07-initialization-report.json)。50 步使用不同 batch，loss 可以波动；这些训练记录不单独证明初始化方案的泛化收益。

验证记录：新增 3 项初始化测试，与已有 19 项合计 22 项通过。另用作者历史模型，在小配置、CPU/FP32、单头宽度 64、相同 ID 的条件下，对照 seed=42 与 1337：初始参数、logits/loss、梯度和一次 AdamW 更新均完全一致，见[源码对齐报告](artifacts/lesson07-course-alignment.json)。这不扩展为与视频 GPU 全部训练轨迹逐位相同。

本地保持 seed=42；作者这一提交的训练种子是 1337，设备自动选择 CUDA/MPS/CPU，分词库使用 tiktoken。本地仍用已验证的 HF GPT-2 tokenizer 与 CPU/FP32。这些是运行条件差异，不改变本块的初始化规则。

开始时先掌握两点：初始化给训练提供起点；下一批继续使用上一次更新后的参数。随后再能解释标准差、零偏置、两个 `c_proj` 的缩放，以及 `apply` 的执行时机。当前实验已由助手核验，用户独立运行与复述另行记录。
