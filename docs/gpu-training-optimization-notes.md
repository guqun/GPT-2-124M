# GPT-2 训练提速（二）：torch.compile、FlashAttention 与词表对齐

> 本篇接续 GPU、TF32 与 BF16，整理阶段二 · 单元 2 · 2.2 的后半段。核心问题是：计算目标基本不变时，怎样通过编译、减少显存读写和调整矩阵尺寸，提高训练效率；哪些改动仍会影响数值或训练目标。

归档日期：2026-10-20。已讨论 `torch.compile`、GPU kernel、FlashAttention、词表 50257 → 50304，以及额外 logits 的梯度影响。**2.2 理论主线已覆盖，部分工程与 CUDA 对照仍待补；下一节为 2.3「超参数与优化」。**

[Notion 归档版本](https://app.notion.com/p/3ffeaea3141481f9bd20ca4745fa8b80)，已上传本篇 FlashAttention 流程图。

项目：[guqun/GPT-2-124M](https://github.com/guqun/GPT-2-124M)。前篇：[GPU 执行、性能测量、TF32 与 BF16](gpu-training-notes.md)（[Notion](https://app.notion.com/p/3feeaea3141481af8e6fd509bee27fc7)）。本篇对应 [lesson10](../lesson10.md) 与 [train_compile.py](../train_compile.py)；FlashAttention 和词表对齐目前记录为原理讨论与教学示例。

**来源边界**：课程顺序来自 [Karpathy 原视频约 1:47:34—2:14:55](https://www.youtube.com/watch?v=l8pRSuU81PU&t=6454s)，此前通过[非官方自动字幕副本](https://github.com/nolimitkun/ai-kol-wiki/blob/main/sources/karpathy/20240609-l8pRSuU81PU/transcript.md)定位，并核对作者历史提交；未直接观看视频画面。以下分别标明课程做法、教学补充和本地实测。API 与设备结论限定项目已验证的 PyTorch 2.2.2，不能直接推广到所有版本。

## 一、这一段要学到什么程度

学习目标是能解释瓶颈、正确启用优化、测量收益并核对数值。当前不要求手写 CUDA/Triton 内核，也不展开寄存器分配、warp 调度和编译器内部实现。

前面已有的训练过程继续成立：

```text
取 batch → 前向 → 交叉熵 → backward → AdamW 更新参数
    ↑                                      │
    └──────── 下一个 batch 使用新参数 ────────┘
```

本段围绕“怎样执行这些计算”展开：

| 方案 | 主要改变什么 | 需要关注的边界 |
| --- | --- | --- |
| TF32 / BF16（前篇） | 运算精度、部分张量的存储类型 | 舍入误差、支持的硬件与算子 |
| torch.compile | 捕获计算图，优化、融合、生成执行代码 | 首次编译、graph break、重编译、实际收益 |
| FlashAttention | 注意力计算的分块、融合与存储方式 | 因果遮罩、数值接近、实际使用的后端 |
| 词表尺寸对齐 | Embedding / lm_head 的行数和 logits 宽度 | Tokenizer 不变，但额外输出类会影响未遮罩的交叉熵 |

最后一项尤其不能简单概括为“完全不改变训练”：若增加输出列后直接计算交叉熵，softmax 的分母确实变了。

## 二、GPU kernel 是什么，为什么会有调度成本

这里的 kernel 指**提交给 GPU 执行的一段计算程序**。CPU 侧运行 Python、框架和调度逻辑，GPU 上的线程协作完成具体数值运算。

例如：

```python
y = (x + bias) * scale
```

在一个便于理解的未融合实现中，可以先执行加法 kernel，写出中间结果，再执行乘法 kernel，读取它并产生 y。融合后可以在一次执行中完成加法和乘法，减少启动与中间张量读写。

这只是教学例子；不能按“一行 Python = 一个 kernel”计数。一个算子可能调用多个 kernel，多行计算也可能融合成一个 kernel；矩阵乘法还可能调用已经优化的底层库。

GPU 速度快，不代表所有环节都没有开销：

- **计算受限**：大量时间花在乘加等数值运算，低精度硬件吞吐可能有帮助。
- **带宽受限**：时间主要花在读写数据，减少中间结果或读写次数可能更有效。
- **调度开销明显**：许多很小的运算，单次计算短，启动和 Python 调度占比变大。

同一个模型的不同操作可能受不同因素限制，不能只看 GPU 的理论算力。

## 三、torch.compile 做了什么

可以先理解为：**观察模型的计算，生成可重复使用的优化执行程序。**

```text
原模型与一次输入
    ↓ 捕获能够处理的计算图
图变换、自动求导相关处理、后端编译
    ↓
优化后的执行程序
    ↓ 以后用当前输入和当前参数继续计算
logits → loss → 梯度 → 参数更新
```

这里复用的是程序，不是上一个 batch 的输出，也不是冻结了参数。每次 `optimizer.step()` 更新原参数后，下一步仍用更新后的数值。

潜在收益包括减少 Python 调度、融合部分算子、减少中间张量的分配和读写。并非所有算子都会被融合，也不保证每个模型、设备和 batch 都更快。参见 [PyTorch 2.2 torch.compile 接口](https://docs.pytorch.org/docs/2.2/generated/torch.compile.html)和[原课提交 fb8bd6e](https://github.com/karpathy/build-nanogpt/commit/fb8bd6e)。

有两种情况要区分：

| 情况 | 含义 |
| --- | --- |
| graph break | 一段逻辑不能继续留在同一张图内，执行被拆开，可能降低优化效果 |
| recompile | 输入形状、dtype、设备或其他保护条件变化，已有编译版本不再适用，需要新版本 |

普通训练中的参数数值更新通常不会触发重编译；参数形状、设备等条件变化则可能影响缓存使用。项目曾把 `enumerate(..., start=1)` 等价改为 `enumerate(..., 1)`，解决已安装 PyTorch 2.2.2 中遇到的 graph break；这是版本相关的兼容处理。

## 四、在我们项目里怎样使用

[train_compile.py](../train_compile.py) 的 `prepare_execution` 保留 eager/compile 两种入口。关键关系可缩写为：

```python
# 示意结构；完整初始化、设备与精度检查见 train_compile.py。
raw_model = GPT(config).to(device).float()

# 放在训练循环前，只包装一次。
train_model = torch.compile(raw_model)

# 仍然管理 raw_model 中的同一批 Parameter。
optimizer = torch.optim.AdamW(raw_model.parameters(), lr=learning_rate)

for x, y in batches:
    optimizer.zero_grad(set_to_none=True)
    logits = train_model(x)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )
    loss.backward()
    optimizer.step()
```

调用 `torch.compile(...)` 建立入口后，编译工作通常在首次执行相应路径时发生，因此第一次前向/反向可能很慢。不要把包装操作放进每个 batch 的循环。

**本项目的编译范围**是模型前向及其对应反向。模型外的交叉熵、取数据和 AdamW 仍按普通方式执行；不能说“整条训练脚本都编译了”。Embedding 与 lm_head 的共享 Parameter、优化器引用和更新关系保持成立。

使用 BF16 时，仍可沿用前篇的 `autocast`，把前向和 loss 放入对应区域；FP32 原参数、梯度与 AdamW 状态的关系不因 compile 消失。编译与混合精度解决不同问题，可以组合验证。

本机复现 CPU 对照时，除执行模式外保持参数相同：

```bash
/Users/guqun/micrograd/venv/bin/python train_compile.py \
  --device cpu --precision fp32 --execution eager \
  --batch-size 1 --seq-len 8 --threads 1 --warmup-steps 3 --steps 5

/Users/guqun/micrograd/venv/bin/python train_compile.py \
  --device cpu --precision fp32 --execution compile \
  --batch-size 1 --seq-len 8 --threads 1 --warmup-steps 3 --steps 5
```

命令在项目根目录执行。已安装依赖的解释器为上面的 venv；另一个没有 torch 的 Python 解释器不能直接运行课程。

当前脚本默认先选 CUDA，否则选 CPU；精度 `auto` 在支持 BF16 的 CUDA 上使用 BF16，其余 FP32。显式 `--device mps --execution eager --precision fp32` 可运行本机 GPU 普通训练。**本机 MPS 可用，当前 PyTorch 2.2.2 的本课 Inductor 编译对照使用 CPU/CUDA。** 不能把“这个编译后端未支持 MPS”理解为“Mac 没有 GPU”。

## 五、如何看 compile 的实测结果

完整 124M 模型、随机初始化、相同 Tiny Shakespeare 数据顺序、CPU/FP32、B=1/T=8、1 线程，各运行 3 次预热和 5 次测量：

| 执行方式 | 首次更新 | 预热后平均每步 | 总 Token / 总时间 |
| --- | --- | --- | --- |
| eager | 1527.60 ms | 848.85 ms | 9.42 Token/s |
| compile | 30972.57 ms | 846.40 ms | 9.45 Token/s |

来源：[eager 报告](../artifacts/lesson10-cpu-eager.json)、[compile 报告](../artifacts/lesson10-cpu-compile.json)。首次更新含训练以及可能发生的编译/缓存加载，不等于纯编译耗时。

这次小批次 CPU 对照**没有显示明显提速**。8 步更新前 loss 的最大绝对差约 `0.0000314713`；说明本次两条训练轨迹接近，不能据此证明长期质量完全相同。

预热仍执行真实更新，参数和数据游标都向前推进。测量时应同步设备，固定设备、初值、数据、B/T、线程、精度、优化器和测量窗口，再逐项改设置。如果测量期间还在重编译，要另外记录，不能都称为“稳定性能”。

另有 [MPS/FP32/eager 短跑](../artifacts/lesson10-mps-eager-smoke.json)：B=1/T=8，1 次预热、2 次测量，平均 234.11 ms、34.17 Token/s。它证明完整模型能在本机 MPS 更新；测量窗口与上表不同，不拿来直接计算 CPU→GPU 加速比。早期工具隔离环境的 MPS=False 不能代表普通终端的设备能力。

## 六、FlashAttention 优化的是哪一步

它位于每个 Transformer Block 的注意力内部：

```text
输入 X → LN1 → Q/K/V 投影
                  ↓
      匹配分数 → 因果遮罩 → softmax → 汇总 V
                  ↑
          FlashAttention 优化这一段
                  ↓
        多头拼接 → 输出投影 → 第一次残差
                  ↓
           LN2 → MLP → 第二次残差
```

![FlashAttention：普通注意力与分块累计的对照流程](diagrams/gpt2-flash-attention-flow.png)

[可缩放 SVG](diagrams/gpt2-flash-attention-flow.svg) · [读图、公式与来源](diagrams/gpt2-flash-attention-flow.md)。

阅读顺序：上方确定在 Block 的位置；中间比较完整中间矩阵和分块累计；下方看数字例子、稳定归一化与反向重算。

单个注意力头的计算仍是：

```text
输入：Q、K、V，形状均为 [T,d]；因果遮罩 M 为 [T,T]
S = QKᵀ / √d
A = softmax(S + M)      沿每个查询对应的上下文位置归一化
O = AV
输出：O，形状 [T,d]
```

整批多头是 `[B,H,T,d]`。优化没有改变“用 Q/K 决定比例，按比例汇总 V”的含义，也没有取消未来位置遮罩。

## 七、为什么分块能够更快

直接实现通常要形成完整的分数 S 和权重 A，二者均有 T×T 个元素。T=1024 时，**每个样本、每个头的一张矩阵就有 1,048,576 个元素**。写入、重新读取这些中间结果，会产生显存流量。

FlashAttention 把数据分块带到容量较小但更快的片上存储，连续完成局部分数、归一化统计与输出累计，避免将完整 S/A 存入 HBM。这里的 HBM 与片上存储都在 GPU 的存储层级内，不是在每个小步骤都把数据传回 CPU。

普通矩阵乘法本身也会分块。FlashAttention 的关键是**把注意力的多项操作一起按块组织，并用在线 softmax 跨块累计**，而不只是把一个大矩阵切成小矩阵。

密集注意力的主要计算量仍随 T² 增长；减少的是大中间结果的存储和读写。Q/K/V、输出及必要统计量仍需保存，不能说“不用显存”或“每份数据一定只读一次”。具体循环顺序、并行与内存访问依实现而定。依据：[FlashAttention 原论文 §3.1](https://arxiv.org/html/2205.14135v2#S3.SS1)。

## 八、不先算完整 A，怎么得到相同的输出

**教学例子**：只看一个查询、V 的一个维度，四个位置均允许访问，Dropout 为 0。设缩放分数的指数为 `[1,2,3,4]`，V 的这一维为 `[10,20,30,40]`。

普通计算：

```text
总权重 = 1 + 2 + 3 + 4 = 10
A = [0.1, 0.2, 0.3, 0.4]
o = 0.1×10 + 0.2×20 + 0.3×30 + 0.4×40 = 30
```

先累计未归一化的分子 U、分母 L，也能得到同一个结果：

| 处理到哪里 | 新增分母 | 新增分子 | 累计 L | 累计 U |
| --- | --- | --- | --- | --- |
| 初始 | — | — | 0 | 0 |
| 第 1 块：位置 0、1 | 1+2=3 | 1×10+2×20=50 | 3 | 50 |
| 第 2 块：位置 2、3 | 3+4=7 | 3×30+4×40=250 | 10 | 300 |

最后 `o=U/L=300/10=30`。完整 V 有 d 维时，U 是 d 维向量，同一个查询的各维共用分母 L。多个查询分别保存自己的累计状态。

实际实现还需要避免指数溢出：维护已处理分数的最大值 m，新块改变最大值时，将此前累计值换到新的尺度：

```text
m_new = max(m, 新块中允许位置的最大分数)
alpha = exp(m - m_new)
L_new = alpha × L + Σ新块 exp(s_j - m_new)
U_new = alpha × U + Σ新块 exp(s_j - m_new) × v_j
o = U_new / L_new
```

**为什么旧的 L/U 要同时乘 alpha？** 因为旧累计值使用的是 `exp(s−m)`，新的使用 `exp(s−m_new)`；二者必须统一尺度。分子和分母一起换尺度，最终比值保持不变。

上述公式说明累计原理；真实内核还要正确处理全被遮罩的块等边界。反向时可根据保存的 Q/K/V、输出及统计量局部重算注意力权重，以计算换取更少的中间存储。数学目标相同，浮点运算顺序变化可能产生小误差。

## 九、PyTorch 里如何使用 FlashAttention

原课 [7ee630c 提交](https://github.com/karpathy/build-nanogpt/commit/7ee630c)用 SDPA 接口替换显式注意力：

```python
# q/k/v: [B, H, T, d]
# 本课训练配置的 attention dropout 为 0。
heads = F.scaled_dot_product_attention(
    q, k, v,
    dropout_p=0.0,
    is_causal=True,
)
# heads 仍为 [B,H,T,d]，后面照常合并头并输出投影。
```

`is_causal=True` 保留“当前查询不能读取未来位置”的规则。若改用非零 Dropout，需要按训练/推理状态明确传入概率；不要假定这个函数会自行读取模型的 `eval()` 状态。

**SDPA 是统一接口，FlashAttention 是可能使用的实现之一。** 后端会根据设备、dtype、形状等条件选择实现；调用成功不等于已使用 CUDA FlashAttention。正式对照时，需要记录实际后端并验证输出/梯度，再测端到端速度。参见 [PyTorch 2.2 SDPA 文档](https://docs.pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html)。

当前本地模型仍保留显式注意力作为学习基线；本段尚未新增独立 FlashAttention 训练入口，也没有 CUDA FlashAttention 提速结果。上图是算法说明，不是 GPU 实跑截图。

## 十、为什么把词表从 50257 改成 50304

课程在 [约 2:06:54](https://www.youtube.com/watch?v=l8pRSuU81PU&t=7614s)调整模型的 vocab_size；对应[历史提交 7230096](https://github.com/karpathy/build-nanogpt/commit/7230096)。

```text
50257 = 392 × 128 + 81
50304 = 393 × 128
增加 47 行
```

GPU 矩阵内核按块组织计算，合适的维度对齐可能减少边缘处理、改善硬件利用率。虽然逻辑计算量增加，实际耗时仍可能下降；并非所有 GPU 都固定“每块 128”，也不是任何扩维都有收益。依据：[NVIDIA 矩阵乘法性能指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)。

| 对象 | 原尺寸 / 范围 | 对齐后 |
| --- | --- | --- |
| Tokenizer 的真实词条 | 50257 个，ID 0—50256 | 不变 |
| 共享 Embedding / lm_head 权重 E | `[50257,768]` | `[50304,768]` |
| 隐藏状态 h | `[B,T,768]` | 不变 |
| logits = hEᵀ | `[B,T,50257]` | `[B,T,50304]` |
| 新增行的语义 | — | 47 个计算用输出类，没有新增 Tokenizer 词条 |

共享权重多出 `47×768=36,096` 个参数，只计一次。输入 ID 和目标 ID 仍来自原 Tokenizer，不会因为扩表自动出现新编号。推理选择输出前，要限制为有效的 50257 列，避免选到没有文本映射的额外 ID。

## 十一、额外 logits 会不会影响梯度

**会，若直接对全部 50304 列计算交叉熵。** 这是本次最重要的追问。

对一个位置、目标 ID 为 y，不使用 label smoothing 时：

```text
p_j = exp(z_j) / Σ所有输出列 exp(z_k)
L = -ln(p_y)
∂L/∂z_j = p_j - 1[j=y]
```

新增列虽然从不是真实标签，仍在分母里分走概率；它们的梯度为正的 `p_j`，并通过 lm_head 影响隐藏状态及参数。共享权重的额外行即使没有被输入查表选中过，也可以从输出端获得梯度。

原课扩表提交直接使用扩宽后的交叉熵，**没有在训练 loss 前遮罩这 47 列**。模型训练会受到压低这些无效类概率的驱动，但不能把它们的概率视为天然等于 0。

这也意味着：不遮罩扩表与原 50257 类目标不严格相同。下面是根据用户追问补充的规避方案，不能倒写成“视频原本就是这么做”。

## 十二、怎样保留扩宽计算，又排除无效类

可以先用完整 lm_head 计算 logits，再在交叉熵前把额外列设为负无穷：

```python
valid_vocab_size = 50257

# 布尔向量长度为模型输出宽度；可在形状固定时预先建立。
padding_columns = (
    torch.arange(logits.size(-1), device=logits.device)
    >= valid_vocab_size
)
loss_logits = logits.masked_fill(padding_columns, float("-inf"))

loss = F.cross_entropy(
    loss_logits.reshape(-1, loss_logits.size(-1)),
    targets.reshape(-1),
    label_smoothing=0.0,
)
loss.backward()
```

前提：targets 都在有效 ID 范围内。这里无需先手动 softmax，`cross_entropy` 接收 logits。

- 完整 `hEᵀ` 的输出宽度仍为 50304，保留该矩阵乘法的尺寸对齐。
- 无效列的 `exp(-∞)=0`，不进入有效概率分母。
- 它们不是真实目标，且 `masked_fill` 对被替换位置不回传梯度，因此这些原始 logits 的损失梯度为 0。
- **给定相同的有效 logits**，loss 与有效列梯度在数学上等于仅使用原词表的计算；不同实现仍可能有浮点舍入差异。

设为 0 不行，因为 `exp(0)=1`。`ignore_index` 忽略的是某些目标位置，不是从每个位置的 softmax 中排除输出列。

另一种做法是 `F.cross_entropy(logits[..., :50257].reshape(-1, 50257), ...)`。它也排除无效类，而且前面的完整 lm_head 乘法已完成；但切片、后续是否复制以及损失计算的尺寸都可能影响性能。两种方案都应重新测量，不能默认零开销。

注意两个边界：

1. 上面的遮罩写法明确使用 `label_smoothing=0.0`。若平滑标签把目标质量分给被设为 `-∞` 的列，会产生问题；需只在有效类别上进行平滑，例如先切片。
2. **loss 梯度为 0，不代表参数必然不变。** AdamW 的权重衰减或已有动量仍可能改变额外行；这里保证的是无效类不影响当前交叉熵目标，不是冻结全部额外参数。

此方案目前是教学补充，尚未集成到项目完整模型的训练入口。

## 十三、用小例子核对 loss 与梯度

这是独立教学实验：有效类只有 3 个，logits 为 `[1,2,3]`，正确标签是 ID 2；再添加两个无效输出 `[4,5]`。使用 CPU、float64、无 label smoothing 的交叉熵。

| 情况 | loss | 原 3 个 logits 的梯度 | 额外 2 个梯度 |
| --- | --- | --- | --- |
| 仅有效类 `[1,2,3]` | 0.407606 | `[0.090031, 0.244728, -0.334759]` | — |
| 加 2 列，不遮罩 | 2.451914 | `[0.011656, 0.031685, -0.913871]` | `[0.234122, 0.636409]` |
| 加 2 列，loss 前遮罩 | 0.407606 | `[0.090031, 0.244728, -0.334759]` | `[0,0]` |

第三行恢复了第一行的结果，第二行则明显改变了梯度。这个例子只核对数学与自动求导行为，不能证明扩表提速，也没有训练一个 GPT。[实际复算记录](../artifacts/vocab-padding-gradient-check.json)保存了未舍入数值；可在已安装 torch 的环境中复算：

```python
import torch
import torch.nn.functional as F

target = torch.tensor([2])
for mode in ("valid_only", "extra_unmasked", "extra_masked"):
    values = [1.0, 2.0, 3.0] if mode == "valid_only" else [1., 2., 3., 4., 5.]
    z = torch.tensor([values], dtype=torch.float64, requires_grad=True)
    loss_logits = z
    if mode == "extra_masked":
        loss_logits = z.masked_fill(torch.arange(5) >= 3, float("-inf"))
    loss = F.cross_entropy(loss_logits, target, label_smoothing=0.0)
    loss.backward()
    print(mode, loss.item(), z.grad.tolist())
```

## 十四、学习进度与后续衔接

| 内容 | 理解与工程状态 |
| --- | --- |
| GPU 执行、计时、TF32、BF16 | 已讨论并反馈初步理解；前篇已归档 |
| torch.compile / kernel | 已讲原理和项目使用；CPU eager/compile 完整模型对照已保存 |
| 本机 GPU | MPS/FP32/eager 完整模型短跑成功；不能代替 CUDA 后端实验 |
| FlashAttention | 原理、输入输出、分块累计、反向重算与接口已讨论；流程图已归档；独立训练入口/正式对照待补 |
| 词表尺寸对齐 | 50257→50304、共享表尺寸、额外 logits 与梯度已讨论；完整模型对照待补 |
| 无效类遮罩 | 小型 loss/梯度例子已核对；属于教学补充，完整训练尚未集成 |

**2.2 的理论主线已覆盖，可以接着学习 2.3；2.2 计划中的部分实验仍保留待办。** 完整段落计数暂为单元 2 的 1/5、阶段二 5/32，同时单独记录本次理论进展；没有把“代码已验证”写成“用户已独立掌握”。自检供复习，不新增学习门槛。

需要补的工程工作是：保留显式实现，新增独立 SDPA / 对齐对照入口；在条件一致的 CUDA 环境中，逐项测量 TF32、BF16、compile、FlashAttention 和尺寸对齐，并核对 loss/梯度、实际后端及峰值内存。这里保留原计划的工作，不把本机 CPU 结果当作替代证据。

下一节按原视频 [2:14:55—2:46:52](https://www.youtube.com/watch?v=l8pRSuU81PU&t=8095s)进入 **2.3「超参数与优化」**：AdamW、梯度裁剪、学习率预热与余弦衰减、权重衰减、fused AdamW 和梯度累积。此前已见过 optimizer 的调用，下一节要理解如何选择与组织参数更新；尚未记为已开始。

## 十五、复习自检

1. compile 缓存了什么？下一批更新了参数，为什么还能复用？
2. kernel 与一行 Python 是一一对应的吗？
3. 为什么应分别记录首次更新和预热后的平均耗时？
4. 调用了 SDPA，就能确认使用 FlashAttention 吗？
5. FlashAttention 不保存完整 A，怎样跨块求同一个输出？
6. 最大值 m 改变时，为什么旧 L 和 U 要同时缩放？
7. 词表对齐有没有给 Tokenizer 增加 47 个真实词条？
8. 无效类从不作为目标，为什么仍然有梯度？
9. 在 loss 前遮罩，与只在生成时排除无效类，解决的是同一问题吗？
10. 无效类的损失梯度为 0，是否等于它的参数永远不更新？

<details>
<summary>展开简要答案</summary>

1. 缓存优化后的执行程序；它读取当前输入和当前参数，普通数值更新不等于结构变化。
2. 不对应。一个算子可能多个 kernel，多项操作也可融合。
3. 首次包含编译等启动成本；长期运行收益需另看稳定阶段。
4. 不能。统一接口会按设备、类型和形状等选择后端。
5. 累计未归一化分子与分母，最后取比值；实际使用稳定在线归一化。
6. 统一指数使用的尺度，保持最终比值和已有贡献正确。
7. 没有。输入/标签仍是原 Tokenizer ID，增加的是模型参数行与输出列。
8. 无效类仍在未遮罩 softmax 的分母里，梯度为其预测概率。
9. 不同。训练前遮罩影响 loss/梯度；生成时过滤只限制选择输出。
10. 不等于。权重衰减和历史动量仍可能改变参数。

</details>
