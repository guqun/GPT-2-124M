# 把 DDP 连上神经网络：从 lm_head 到梯度同步

这张图承接 [DDP / NCCL / All-Reduce 概念图](gpt2-ddp-allreduce-flow.md)，回答“本地梯度到底从哪个神经网络计算中来、两张 GPU 分别算什么、如何合并”。它是阶段二 · 单元 2 · 2.4 的补充例子，不增加课程段落。

完整答疑见 [DDP、平均损失与梯度同步笔记](../ddp-batch-gradient-notes.md)：补充整批矩阵求导、为什么定义平均损失，以及平均系数如何沿计算图传递。

![lm_head 前向、反向、跨卡梯度同步与参数更新](gpt2-ddp-lm-head-example.png)

[原尺寸 PNG（2600×3700）](gpt2-ddp-lm-head-example.png) · [可缩放 SVG](gpt2-ddp-lm-head-example.svg)

反向部分的 [GPU 归属标注局部图](gpt2-ddp-backward-gpu-labels.png) / [局部 SVG](gpt2-ddp-backward-gpu-labels.svg)可单独查看。每个操作数与结果都标出 GPU 编号；完整图的每个阶段也重复标注设备，通信箭头标明发送卡、接收卡和数据。

## 每份数据保存在哪张卡

| 数据 | GPU 0 显存 | GPU 1 显存 |
|---|---|---|
| 本卡样本特征 | h_A = `[1,2]` | h_B = `[3,6]` |
| 本卡权重副本 | W = `[[1,0],[0,0.5]]` | W = `[[1,0],[0,0.5]]` |
| 本卡 logits | z_A = `[1,1]` | z_B = `[3,3]` |
| 本卡概率与目标 | p_A = `[0.5,0.5]`，t_A = `[1,0]` | p_B = `[0.5,0.5]`，t_B = `[0,1]` |
| 本卡 loss | L_A = `0.693147…` | L_B = `0.693147…` |
| 本卡 logits 梯度 | δ_A = `[-0.5,0.5]` | δ_B = `[0.5,-0.5]` |
| 同步前的本地权重梯度 | G_A = `[[-0.5,-1],[0.5,1]]` | G_B = `[[1.5,3],[-1.5,-3]]` |
| 传给本卡上游的梯度 | ∂L_A/∂h_A = `[-0.5,0.25]` | ∂L_B/∂h_B = `[0.5,-0.25]` |
| 同步后的本卡平均梯度 | 一份 `[[0.5,1],[-0.5,-1]]` | 另一份 `[[0.5,1],[-0.5,-1]]` |

两卡的 W 是**数值相同、分别存储的两个模型副本**。同步后的平均梯度也是每卡各存一份相同数值。图中 `W[0,0]` 指某个权重的行列索引，GPU 编号才指设备；高亮同一位置不表示跨 GPU 共用同一块显存。

截图中的反向乘法，左侧的 δ_A、h_A、G_A 全在 GPU 0，右侧的 δ_B、h_B、G_B 全在 GPU 1；两个乘法分别在各自卡上执行。传回上游的 ∂L/∂h 继续留在原卡。随后进入通信示意，才将需要汇总的 G 行发送到另一张卡；接收完成后，收到的那份数值已经位于接收卡可用的缓冲区，接收卡再执行对应元素加法。

## 先看清分给 GPU 的是什么

DDP 的每张 GPU 都持有完整模型：Embedding、所有 Transformer Block、最终 LayerNorm 和 lm_head。一次总 batch 的不同样本分给不同 GPU，各卡为自己的样本运行整条前向和反向路线。

图中总 batch 有 A、B 两条样本，每卡一条，每条只展示一个参与损失的预测位置。每张卡都计算两个输出词的分数，并算出这张权重矩阵的全部四个梯度。之后的 Reduce-Scatter 才把梯度汇总任务分给各卡；该划分与样本目标类别无关。

实际训练由数据加载代码划分样本，DDP 负责梯度同步。计算在每张 GPU 本地执行，NCCL 通信把另一张卡的对应梯度送过来；同名参数、同一元素的梯度才能相加。

## 这次具体观察哪个环节

项目 [gpt2_model.py](../../gpt2_model.py) 的对应位置为：

```python
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

# GPT.forward 的最后两个计算步骤
x = self.transformer.ln_f(x)
logits = self.lm_head(x)
```

Linear 的权重按 `[输出维度, 输入维度]` 保存，前向为 `z = h @ W.T`。真实 GPT-2 的输出权重为 `[50257,768]`；本例缩成 `[2,2]`，候选词仅用“猫 / ID 0”“狗 / ID 1”表示。

上游特征 `h_A=[1,2]`、`h_B=[3,6]` 是教学给定值，不是本次运行完整 GPT-2 得到的输出。从这两个特征及给定 W 开始，图中 logits、概率、loss、梯度和更新值均实际计算并核验。

**本例隔离 lm_head 的输出投影路径，固定上游特征并独立训练 W。** 完整 GPT-2 的 lm_head 与 Token Embedding 共享参数，真实共享参数的 `.grad` 还包含输入 Embedding 路径的贡献。这两条路径的贡献先在本卡相加，再参加跨卡同步。本图的数值不冒充该完整共享参数的实际梯度。

## 前向：相同 W，不同样本

两张卡都有相同初始权重：

\[
W=\begin{bmatrix}1&0\\0&0.5\end{bmatrix}.
\]

第 0 行对应猫，第 1 行对应狗；第 0、1 列对应输入特征的两个维度。

| 项目 | GPU 0：样本 A | GPU 1：样本 B |
|---|---|---|
| 给定特征 h | `[1,2]` | `[3,6]` |
| 正确目标 | 猫 / ID 0 | 狗 / ID 1 |
| 目标向量 t | `[1,0]` | `[0,1]` |
| logits = hWᵀ | `[1,1]` | `[3,3]` |
| p = softmax(logits) | `[0.5,0.5]` | `[0.5,0.5]` |
| L = −ln p[正确目标] | `0.693147…` | `0.693147…` |

两个概率向量相同来自本例专门选择的数值，便于手算；DDP 并不要求不同样本有相同 logits、概率或 loss。

## 反向：本地权重梯度从哪里来

单个预测位置、无类别加权或标签平滑的 softmax 交叉熵，对 logits 的梯度为：

\[
\delta=\frac{\partial L}{\partial z}=p-t.
\]

Linear 的权重梯度按链式法则计算：

\[
G=\frac{\partial L}{\partial W}=\delta^T h,
\qquad G[j,k]=\delta[j]h[k].
\]

于是 GPU 0 得到：

\[
G_A=\begin{bmatrix}-0.5\\0.5\end{bmatrix}
\begin{bmatrix}1&2\end{bmatrix}
=\begin{bmatrix}-0.5&-1\\0.5&1\end{bmatrix}.
\]

GPU 1 得到：

\[
G_B=\begin{bmatrix}0.5\\-0.5\end{bmatrix}
\begin{bmatrix}3&6\end{bmatrix}
=\begin{bmatrix}1.5&3\\-1.5&-3\end{bmatrix}.
\]

图中橙色格始终追踪 **同一个 W[0,0]**。这个参数决定特征第 0 维对“猫”的 logit 的贡献：

- 样本 A：`∂L_A/∂W[0,0] = (0.5−1)×1 = −0.5`。
- 样本 B：`∂L_B/∂W[0,0] = (0.5−0)×3 = 1.5`。

本卡还会得到输入特征梯度 `∂L/∂h = δW`。在完整网络中，这个梯度继续沿本卡的计算图传给 LN_f 和更前面的层；每一层再求出各自的参数梯度。

## 通信：合并同一参数的两份梯度

1. GPU 0 负责第一行的汇总，已有 `[-0.5,-1]`，必须收到 GPU 1 的 `[1.5,3]`，才能在本卡相加得到 `[1,2]`。
2. GPU 1 负责第二行的汇总，已有 `[-1.5,-3]`，收到 GPU 0 的 `[0.5,1]`，在本卡相加得到 `[-1,-2]`。
3. 交换已经汇总的两行后，每张卡都有完整梯度和。两阶段对应一种 Reduce-Scatter + All-Gather 分解。
4. 求平均后，每张卡的 W.grad 都是：

\[
\bar G=\frac{G_A+G_B}{2}
=\begin{bmatrix}0.5&1\\-0.5&-1\end{bmatrix}.
\]

对于被高亮的 W[0,0]，就是 `(-0.5+1.5)/2=0.5`。这与联合损失 `L=(L_A+L_B)/2` 对该参数求导的结果一致。

两卡损失各自含一个有效位置，故等权平均正确。真实任务如果每卡有效 Token 数不同，要匹配全局目标的权重。图中“先 SUM 再除卡数”展示数学含义，实际可以先缩放或融合执行；按行分块仅是通信教学分解，不代表 NCCL 固定采用此安排。

## 参数更新：每卡本地做同一次 AdamW

本例学习率 `η=0.01`，权重衰减 `λ=0.1`，`β1=0.9`、`β2=0.95`、`ε=1e−8`。为了只观察跨卡同步，本例不裁剪梯度；若启用裁剪，应在同步完成后、更新前对完整参数梯度统一执行。

两卡都从相同参数、零 m/v 状态开始。第一次更新经过偏差修正，`m_hat=g`，`v_hat=g²`。以 W[0,0] 为例：

\[
W'_{0,0}=(1-0.01\times0.1)\times1
-0.01\frac{0.5}{\sqrt{0.25}+10^{-8}}
\approx0.989.
\]

两张卡各自得到相同的新权重：

\[
W'\approx\begin{bmatrix}0.9890&-0.0100\\0.0100&0.5095\end{bmatrix}.
\]

之后清梯度、保留新参数和优化器状态，读取下一批不同样本继续计算。DDP 通信可按 bucket 与反向计算重叠，图中分阶段展开是为了理解数据依赖。

## 数值核对、来源与重绘

[验证报告](../../artifacts/ddp-lm-head-diagram-check.json)在 CPU/float64 上建立两个独立 `nn.Linear(2,2,bias=False)`，验证 logits、交叉熵、logits 梯度、权重梯度、输入特征梯度。随后用 CPU 加法模拟跨卡平均，与直接计算两条样本的平均交叉熵比较：梯度最大差为 0；各自一次 AdamW 后与整批更新的权重最大差也为 0。未启动 DDP / NCCL，未运行完整 GPT-2 或 CUDA 测速。

- 课程承接：[原视频 DDP 段落](https://www.youtube.com/watch?v=l8pRSuU81PU&t=10012s)，此前已核对自动字幕及作者历史实现；本图的两词分类模型是额外教学缩小例子。
- [PyTorch Linear 文档](https://docs.pytorch.org/docs/main/generated/torch.nn.Linear.html)：权重方向和线性变换。
- [PyTorch CrossEntropyLoss 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)：从 logits 与目标计算交叉熵。本次实际运行依赖版本记录在验证报告中。
- [PyTorch DDP 文档](https://docs.pytorch.org/docs/2.2/generated/torch.nn.parallel.DistributedDataParallel.html)：梯度同步与输入分配边界。
- [NCCL 集合通信定义](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)：All-Reduce、Reduce-Scatter 和 All-Gather。

重绘脚本：[render_ddp_lm_head_example.py](render_ddp_lm_head_example.py)，复用本目录前一张 DDP 图的 SVG 绘制与 AppKit 导出工具。保留此前概念图和所有训练入口。

```bash
python docs/diagrams/render_ddp_lm_head_example.py --png --verify-torch
```
