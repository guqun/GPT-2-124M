> 本篇整理阶段二 · 单元 2 · 2.2 已学的知识块 8、9。核心问题是：同一套训练计算放在哪里执行，怎样测量速度，以及怎样在部分计算使用低精度时，继续更新 FP32 参数。

归档日期：<mention-date start="2026-10-19"/>。设备、计时、TF32 和 BF16 已讨论，当前反馈为“大概理解”；GPU 性能对照仍待实跑。2.2 后续的 `torch.compile`、FlashAttention 和词表尺寸调整尚未学习，整节继续进行中。

承接已归档内容：<mention-page url="https://app.notion.com/p/3fbeaea31414815294cdffb15d7404b0"/>。

项目：[guqun/GPT-2-124M](https://github.com/guqun/GPT-2-124M)。配套讲义：lesson08（项目内 `lesson08.md`）、lesson09（项目内 `lesson09.md`）。承接 2.1：从检查点到随机初始化训练（项目内 `docs/section-2.1-review.md`）。

**来源边界**：课程顺序来自 Karpathy [Let's reproduce GPT-2 (124M)，约 1:22—1:47:34](https://www.youtube.com/watch?v=l8pRSuU81PU&t=4938s)；此前用非官方字幕副本与作者历史提交交叉核对，未直接观看视频画面。本篇把课程内容、讨论中的教学推导、项目代码及已归档的实验结果串起来。具体类型行为按本项目 PyTorch 2.2.2 记录，不代表所有版本和后端。

## 一、PyTorch 已经实现 GPU 运算，我们还要学什么 {color="blue"}

前面的训练主干已经是：

```plain text
取一个 batch → 前向 → 交叉熵 loss → backward → optimizer.step
    ↑                                               │
    └──────────── 下一批使用更新后的参数 ────────────────┘
```

现在保持这个训练目标，学习执行条件与测量方法。

<table header-row="true">
<tr>
<td>我们决定什么</td>
<td>PyTorch 与设备后端负责什么</td>
</tr>
<tr>
<td>把模型、输入、标签放在哪个设备</td>
<td>管理 Tensor 存储，调度相应设备上的算子</td>
</tr>
<tr>
<td>哪个区域开启混合精度</td>
<td>按运算规则选择 dtype、转换输入并执行</td>
</tr>
<tr>
<td>取多少数据、何时清梯度、何时更新</td>
<td>自动求导与 AdamW 的数值计算</td>
</tr>
<tr>
<td>如何同步、排除预热、汇总吞吐</td>
<td>提供同步、计时配合及读取 Tensor 信息的接口</td>
</tr>
<tr>
<td>怎样比较速度、显存和数值变化</td>
<td>提供观察工具，不自动证明一项优化有效</td>
</tr>
</table>

现阶段需要能解释、配置和核对这些行为，不需要自己编写 GPU 矩阵乘法内核。知道怎么调用之后，还要能判断调用是否生效、计时是否正确、结果是否仍符合预期。

本课仍从随机参数训练完整 GPT-2 124M，沿用权重共享和初始化规则，使用 Tiny Shakespeare。读取的是配置和 tokenizer，没有载入预训练参数；每个独立脚本运行都从新模型开始。

## 二、device 是什么，在哪里指定使用 GPU {color="blue"}

`device` 表示 Tensor 存放和执行运算的设备；`dtype` 表示每个元素的数值类型。这是两个独立属性。

<table header-row="true">
<tr>
<td>设置</td>
<td>含义</td>
</tr>
<tr>
<td>`torch.device('cpu')`</td>
<td>使用 CPU</td>
</tr>
<tr>
<td>`torch.device('cuda')` / `'cuda:0'`</td>
<td>本课程中使用 NVIDIA GPU 的 CUDA 后端；后者明确指定编号 0</td>
</tr>
<tr>
<td>`torch.device('mps')`</td>
<td>使用 Apple GPU 的 MPS 后端</td>
</tr>
<tr>
<td>`torch.float32` / `torch.bfloat16`</td>
<td>浮点数的存储类型，不是设备名称</td>
</tr>
</table>

MPS 是 **Metal Performance Shaders**。在 PyTorch 中它是使用 Apple GPU 的后端入口；是否可用取决于硬件、系统与 PyTorch 构建。当前已归档环境检测到 `CUDA=False`、`MPS=False`，因此本地报告实际使用 CPU。不能只看到“Mac”就认定这次已经用了 MPS。

本项目在 `benchmark_training.py` 的 `select_device()` 中选择设备；`auto` 按 CUDA → MPS → CPU 检查。显式指定不可用设备时会报错。真正放入设备发生在以下位置：

```python
model.to(device)                       # 移动参数与注册的 buffer
tie_weights_like_course(model)         # 设备转换后确认共享引用
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

x, y = loader.next_batch()             # 这里取得 CPU 上的整数 ID
x, y = x.to(device), y.to(device)      # Tensor.to 返回放到目标设备的 Tensor
logits = model(x)                      # 调用设备后端执行各层运算
```

`model.to(device)` 会递归处理模型管理的参数和 buffer；输入不会因此自动移动，所以 x/y 也要处理。`x.to(device)` 不改变变量绑定，才需要接回 `x`。模型迁移后再创建优化器，使优化器持有将要训练的参数对象。

<table header-row="true">
<tr>
<td>数据</td>
<td>形状示例</td>
<td>本课含义</td>
</tr>
<tr>
<td>x / y</td>
<td>`[B,T]`，int64</td>
<td>输入 ID / 错开一位的目标 ID</td>
</tr>
<tr>
<td>Embedding 后的特征 X</td>
<td>`[B,T,768]`，浮点数</td>
<td>参与后续矩阵计算的样本表示</td>
</tr>
<tr>
<td>参数 W</td>
<td>各层矩阵或向量，FP32</td>
<td>长期保存并逐批更新的数值</td>
</tr>
<tr>
<td>logits</td>
<td>`[B,T,50257]`，浮点数</td>
<td>各位置对词表的预测分数</td>
</tr>
</table>

因此，“样本是 FP32”指的是这里的浮点特征，并非最初输入的 Token ID。换设备不会把 ID 变成向量，也不会改变它的编号。

## 三、一次训练 step 计时，到底测了什么 {color="blue"}

在当前代码中，**一个 batch 对应一次前向、一次反向和一次优化器更新**，这是一条日志里的一个 step。以后加入梯度累积时，batch 与更新次数的关系可能改变；本篇采用当前脚本的口径。

GPU 运算常以异步方式提交：CPU 已经执行到下一行 Python，GPU 可能还在处理前面的任务。如果只在两端读取 CPU 时钟，可能把“命令提交完”误认为“数值计算完”。

本项目采用以下边界：

```plain text
等待此前设备工作完成
开始计时
    取 batch → 搬到设备 → 清梯度
    → 前向 / loss → backward → optimizer.step
    → 等待本步设备工作完成
结束计时
读取 loss → 打印日志 → 保存报告
```

CUDA 的等待由 `torch.cuda.synchronize(device)` 完成，MPS 路径使用 `torch.mps.synchronize()`。CPU 路径不调用 GPU 同步接口。计时来自 Python 的 `time.perf_counter()`。

**包含**：取批、传输、清梯度、前向、loss、反向、更新和等待。**不包含**：预先读文件、分词、构建模型、打印日志和写报告。这是在当前串行循环中测量一步训练的经过时间，不能称为“纯 GPU 矩阵乘法耗时”。

`.item()` 要把设备上的标量读回 Python，可能等待 GPU，因此放在计时结束之后。吞吐率本身只是 CPU 根据“已知的 Token 数”和“同步后的时间差”做一次除法，不需要 GPU 另算一遍模型。

## 四、benchmark、预热、测量与吞吐率 {color="blue"}

**benchmark** 在这里指按固定条件执行的性能测试。相同模型和数据，换一种执行设置，再比较真实耗时及数值结果。

**预热**仍执行真实训练。首次运行可能有运行时准备、内存分配、AdamW 状态首次创建等额外工作，所以先运行若干步，让后续测量更能代表持续运行的状态。当前默认预热 2 步，然后测量 5 步；预热会更新参数、推进加载器，不会自动回滚。

**测量**继续执行相同训练，但只有这些步骤进入最终速度汇总。“预热”与“测量”的区别是统计口径，不是前者不学习、后者才学习。

每步有 B 条序列，每条有 T 个监督位置：

```plain text
单步 Token 数 = B × T
单步吞吐率   = B × T / 本步秒数
总体吞吐率   = 所有测量步的 Token 数之和 / 所有测量步的秒数之和
平均每步耗时 = 所有测量步的秒数之和 / 测量步数
```

这是训练监督位置的处理速度。x 和 y 是配对的输入、目标，不乘 2；也不是模型每秒自由生成的新 Token 数。

### 用讨论中贴出的日志复算

按当时的 B=4、T=32，每步处理 128 个监督位置：

<table header-row="true">
<tr>
<td>阶段</td>
<td>更新编号</td>
<td>loss</td>
<td>耗时 ms</td>
<td>Token/s</td>
</tr>
<tr>
<td>预热</td>
<td>01</td>
<td>11.008783</td>
<td>1161.67</td>
<td>110.2</td>
</tr>
<tr>
<td>预热</td>
<td>02</td>
<td>9.924229</td>
<td>241.52</td>
<td>530.0</td>
</tr>
<tr>
<td>测量</td>
<td>03</td>
<td>8.877958</td>
<td>234.19</td>
<td>546.6</td>
</tr>
<tr>
<td>测量</td>
<td>04</td>
<td>9.184955</td>
<td>229.48</td>
<td>557.8</td>
</tr>
<tr>
<td>测量</td>
<td>05</td>
<td>8.670364</td>
<td>237.11</td>
<td>539.8</td>
</tr>
<tr>
<td>测量</td>
<td>06</td>
<td>8.403952</td>
<td>251.72</td>
<td>508.5</td>
</tr>
<tr>
<td>测量</td>
<td>07</td>
<td>9.085043</td>
<td>225.46</td>
<td>567.7</td>
</tr>
</table>

```plain text
第 3 步：128 / 0.23419 ≈ 546.6 Token/s
5 步测量总时间：1.17796 秒
平均每步：1.17796 / 5 = 0.235592 秒 = 235.592 ms
总体吞吐：5 × 128 / 1.17796 ≈ 543.3 Token/s
```

这是根据用户贴出的、已经四舍五入的日志复算；日志片段没有设备元数据，不能据此判断使用了哪种 GPU。项目另存的 CPU 报告属于另一轮运行，数值不同，不混用。

当前 `loss` 是**该 batch 更新前**的损失，虽然日志在更新后打印，仍是刚才那次前向算出的值。不同步骤输入不同，loss 可以上升；这些值不能直接当作同一验证集上的质量曲线。

总体吞吐是一个速率，不是“一批 batch 的时间”；总时间与平均时间是另外两个量。也不要直接平均各步的 Token/s：应该先加 Token 数、再加耗时、最后相除。

## 五、GPU 的算力、容量、带宽，各解决什么问题 {color="blue"}

熟悉的输出层可以写成：

```plain text
H [B,T,768] × Eᵀ [768,50257] → logits [B,T,50257]
```

许多输出元素可以并行计算。QKV、Attention 和 MLP 也包含大量矩阵运算，适合 GPU。Python 在 CPU 上组织流程，数值工作由相应设备后端执行。

<table header-row="true">
<tr>
<td>概念</td>
<td>具体问题</td>
</tr>
<tr>
<td>计算吞吐量</td>
<td>设备每秒能完成多少运算</td>
</tr>
<tr>
<td>显存容量</td>
<td>参数、梯度、激活、优化器状态是否放得下</td>
</tr>
<tr>
<td>显存带宽</td>
<td>数据能以多快的速度供给计算单元</td>
</tr>
<tr>
<td>Tensor Core</td>
<td>NVIDIA GPU 中用于特定格式矩阵乘加的专用计算单元</td>
</tr>
</table>

降低精度可能同时提高矩阵计算吞吐、减少部分数据搬运，但训练还包含传输、非矩阵运算和调度。因此硬件标称吞吐提高几倍，并不意味着整个 step 同比例加快。最终要看同条件测量。

## 六、TF32：存储仍是 FP32，内部乘法可以用较少有效位 {color="blue"}

本课从这两种设置开始对照：

```python
torch.set_float32_matmul_precision('highest')  # FP32 内部精度基线
torch.set_float32_matmul_precision('high')     # 允许较低内部精度的快速实现
```

在原课 A100 的 TF32 路径中，输入 Tensor 与参数仍按 FP32 保存；矩阵乘法内部使用较短的小数精度进行乘法，并以 FP32 累加。TF32 使用 8 位指数、10 位显式小数精度，FP32 的显式小数位为 23 位。

所以 `.dtype` 仍显示 `torch.float32`，输出 Tensor 也可为 FP32，但结果不必与 `highest` 逐位相同。开启这个选项不会把原参数表的存储空间减半。

`high` 是一项计算精度许可，后端也可以采用 BF16 组合算法；它不是一个叫 `torch.tf32` 的 Tensor 存储类型。实际路径由硬件和版本决定。参见 [PyTorch 2.2.2 的 set_float32_matmul_precision 说明](https://github.com/pytorch/pytorch/blob/v2.2.2/torch/__init__.py)与[原课 TF32 历史提交](https://github.com/karpathy/build-nanogpt/commit/5265b2042aaaa84943313002671b8b3ca7749b06)。

本项目把 `high` 对照限定在支持该课程路径的 CUDA 设备，CPU/MPS 会明确拒绝，避免把没有生效的选项测成“TF32 提速”。

## 七、BF16 是什么，与 FP16、TF32 有何不同 {color="blue"}

BF16 即 **Brain Floating Point 的 16 位格式**。指数位主要决定可表示的数量级，小数位决定同一数量级下可以区分多细的差别。

<table header-row="true">
<tr>
<td>浮点存储格式</td>
<td>符号位</td>
<td>指数位</td>
<td>显式小数位</td>
<td>每元素字节数</td>
</tr>
<tr>
<td>FP32</td>
<td>1</td>
<td>8</td>
<td>23</td>
<td>4</td>
</tr>
<tr>
<td>BF16</td>
<td>1</td>
<td>8</td>
<td>7</td>
<td>2</td>
</tr>
<tr>
<td>FP16</td>
<td>1</td>
<td>5</td>
<td>10</td>
<td>2</td>
</tr>
</table>

BF16 的数值范围接近 FP32，但舍入更粗；FP16 的小数位多于 BF16，指数范围却小得多。两种 16 位格式不能互换理解。格式依据见 [Google 的 BF16 说明](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)。

<table header-row="true">
<tr>
<td>本课方案</td>
<td>长期参数存储</td>
<td>矩阵运算方式</td>
<td>本课 logits</td>
</tr>
<tr>
<td>FP32 highest</td>
<td>FP32</td>
<td>基线内部精度</td>
<td>FP32</td>
</tr>
<tr>
<td>允许 TF32 high</td>
<td>FP32</td>
<td>允许后端用更低内部乘法精度</td>
<td>FP32</td>
</tr>
<tr>
<td>BF16 autocast</td>
<td>FP32</td>
<td>符合规则的运算使用 BF16 输入</td>
<td>BF16</td>
</tr>
</table>

BF16 autocast 真正产生了一部分 BF16 Tensor，因此这些张量按 2 字节/元素保存。**本课没有把整个模型永久改成 BF16。**

FP16 训练常配合 GradScaler，缓解小梯度下溢；本课 BF16 的指数范围较大，沿用原课方案不使用 GradScaler。是否需要缩放与数值范围、具体方案有关，不能仅按“16 位”判断。参见 [PyTorch 2.2.2 AMP 文档](https://github.com/pytorch/pytorch/blob/v2.2.2/docs/source/amp.rst)。

## 八、一张图看前向、反向和更新的精度 {color="blue"}

<image src="file-upload://3feeaea3-1414-818c-bc57-00b21801ce3d"></image>
图：一次训练更新中的精度变化，从前向、反向到 AdamW 参数更新。

可缩放 SVG（项目内 `docs/diagrams/gpt2-bf16-precision-flow.svg`） · 读图与来源说明（项目内 `docs/diagrams/gpt2-bf16-precision-flow.md`）。图从左到右分为三个阶段；每列自上往下读，底部回到下一个 batch。

**本项目 CPU / PyTorch 2.2.2 的实际记录：**

<table header-row="true">
<tr>
<td>位置或数据</td>
<td>类型</td>
<td>输入、输出或目的</td>
</tr>
<tr>
<td>x / y</td>
<td>int64</td>
<td>编号，供查表和计算正确目标损失</td>
</tr>
<tr>
<td>wte、wpe 参数与查表结果</td>
<td>FP32</td>
<td>ID → 浮点特征</td>
</tr>
<tr>
<td>LN1 / LN2 / 最终 LN 输出</td>
<td>FP32</td>
<td>调整特征尺度</td>
</tr>
<tr>
<td>QKV、Attention 输出投影、MLP 线性层输出</td>
<td>BF16</td>
<td>计算新的特征表示</td>
</tr>
<tr>
<td>残差相加后的 Block 输出</td>
<td>FP32</td>
<td>本例 FP32 主干与 BF16 分支相加</td>
</tr>
<tr>
<td>lm_head 输出 logits</td>
<td>BF16</td>
<td>`[B,T,V]`，全词表打分</td>
</tr>
<tr>
<td>交叉熵 loss</td>
<td>FP32</td>
<td>在 autocast 区域内计算的标量</td>
</tr>
<tr>
<td>反向中间梯度</td>
<td>BF16 与 FP32 并存</td>
<td>跟随实际计算图的类型</td>
</tr>
<tr>
<td>原参数的 `.grad`</td>
<td>FP32</td>
<td>给 FP32 参数累积梯度</td>
</tr>
<tr>
<td>原参数、AdamW 的 m / v</td>
<td>FP32</td>
<td>保存参数和历史统计，执行更新</td>
</tr>
</table>

前向类型见 完整模型 dtype 报告（项目内 `artifacts/lesson09-cpu-dtypes.json`）；参数梯度与优化器状态见 3 次更新报告（项目内 `artifacts/lesson09-cpu-bf16-smoke.json`）。这张表限定当前模型与实现，不把某个设备的所有算子规则推广到其他后端。

## 九、输入 FP32、权重 BF16，乘完到底是什么精度 {color="blue"}

先分开三个问题：**运算接收什么类型、内部怎样累加、最终 Tensor 用什么类型保存。**

以本课被 autocast 选择为 BF16 的线性层为例：

```plain text
长期保存的 W32（FP32） ──按需转换──→ W16（BF16）
本次浮点特征 X32（FP32）──按需转换──→ X16（BF16）
                                        │
                              Y16 = X16 × W16ᵀ + b
                                        │
                                 输出 Tensor：BF16
```

这里的 X32、W32 对象并不会被永久改成 BF16；算子使用了转换后的计算表示。如果上一步已经产生 BF16 输入，也无需把它先还原成 FP32 再转换。

例如一行输入与一个输出神经元的权重：

```plain text
X = [1.25, 2.5]          原 Tensor：FP32
W = [0.5, 0.75]          计算用权重：BF16
Y = 1.25×0.5 + 2.5×0.75 = 2.5
```

在 BF16 autocast 的矩阵乘法中，X 也会转成 BF16 参与计算，Y 保存为 BF16。例子的数恰好可精确表示；一般数值会发生舍入。没有 autocast 时，直接做不匹配类型的 FP32 `@` BF16 矩阵乘法通常会报类型错误，不能套用“总是自动提升到较高精度”的规则。

**内部累加是另一层含义。** A100 Tensor Core 支持 BF16 输入、FP32 累加；这不要求输出 Tensor 必须保存成 FP32，最终还可以舍入为 BF16。内部归约行为仍取决于后端、内核与配置。参见 [NVIDIA A100 的 BF16 运算说明](https://developer.nvidia.com/blog/accelerating-tensorflow-on-a100-gpus/)。

## 十、梯度怎样回到原来的 FP32 参数 {color="blue"}

计算用的 BF16 权重与原 FP32 参数之间保留自动求导关系。以一个线性层为例：

```plain text
前向：W32 ──浮点类型转换──→ W16 ──线性运算──→ 输出 ──→ loss
反向：W32.grad ←─转回 FP32─ 梯度 ←─────────── 沿图回传
```

反向先经过实际前向中建立的运算和类型转换节点。FP32 参数参与多条路径时，这些贡献最终累积到同一个 FP32 `.grad`；共享的 Embedding / lm_head 就是前面已经学过的例子。转换反向规则可以核对 [PyTorch 的 `_to_copy` 梯度定义](https://github.com/pytorch/pytorch/blob/v2.2.2/tools/autograd/derivatives.yaml)。

**`.grad` 是 FP32，表示最终保存梯度的类型；它不代表梯度曾经全部按 FP32 计算，也不能恢复此前低精度运算丢失的信息。**

`loss.backward()` 之后得到梯度，参数尚未因这个调用完成优化更新；`optimizer.step()` 才使用它们改变原参数。AdamW 的流程为：

```plain text
输入：原参数 W、当前梯度 g、上一轮 m / v、步数和超参数
    m ← β₁m + (1−β₁)g
    v ← β₂v + (1−β₂)g²
    m̂、v̂ ← 按当前步数做偏差修正
    W ← (1−ηλ)W − ηm̂ / (√v̂ + ε)
输出：更新后的 W、m、v，留给下一轮
```

η 是学习率，λ 是权重衰减，ε 是稳定分母的小常数。这是针对施加该 weight decay 的参数写出的逐元素公式；本课参数、梯度、m/v 均按 FP32 保存与更新。

下一批继续使用这套新参数；需要 BF16 运算时再取得相应计算表示。不是先把 BF16 权重更新一遍，再用它覆盖 FP32 参数。也不能说每一步每个标量都必然改变：实际更新取决于梯度、优化器状态、衰减和舍入。

### 为什么仍保留 FP32 参数：一个简化 SGD 例子

为突出精度，先用单权重线性层、x=1、目标 0、损失 `L=y²`，把优化器简化为 SGD，学习率 0.0001。它不是本课完整 GPT 的 AdamW 数值：

```plain text
长期 FP32 参数：       W ≈ 1.0001000
BF16 计算表示：        W16 = 1.0
本次前向：            y = 1.0，L = 1.0
传回参数的梯度：       g = 2.0，保存成 FP32
在原 FP32 参数上更新： W_new ≈ 1.0001000 − 0.0001×2 = 0.9999000
```

若把这个更新值立刻按 BF16 保存，它仍可能舍入为 1.0。保留 FP32 参数，能在其精度范围内积累这样的小更新；即使某几步的 BF16 计算表示暂时相同，长期参数也在变化。

## 十一、BF16 转回 FP32，多出来的位放什么 {color="blue"}

对普通有限数值，BF16 的 7 位显式小数可以放入 FP32 的 23 位小数中；扩展出的 16 个低位补零，数值保持不变。

```plain text
原 FP32：       1.2345670461654663
转成 BF16：     1.234375
再转回 FP32：   1.234375
```

转回去能更精细地表示**以后**的计算结果，却不会知道原来被舍去的数字。随后用 FP32 做运算，低位可以产生新的非零信息；不是“转成 FP32 后永远只能带零”。

这也解释了：FP32 参数梯度可以来自混合精度的反向过程，但不能因此声称它等同于一次全 FP32 前向/反向求出的梯度。

## 十二、来回转换也耗时，为什么总体还可能更快 {color="blue"}

转换确实有代价。收益来自支持 BF16 的硬件更高的运算吞吐，以及部分张量更少的读写；能否抵消代价，要看工作负载。

一个教学量级比较：两张 n×n 矩阵的输入转换涉及约 `2n²` 个元素，普通矩阵乘法涉及约 `n³` 次乘法和相应加法。当 n=1024 时，前者约 210 万次元素转换，后者约 10.7 亿次乘法。大矩阵有机会让更快的乘法覆盖转换成本；这只是运算量解释，不是实际耗时比例。

还要注意：

- 转换发生在张量所在设备，不意味着每层都在 CPU 和 GPU 之间搬一次数据。
- 已经是目标 dtype 的输入，不必重复转换。
- autocast 在合适条件下缓存权重转换结果；本课每次前向退出外层 autocast 后会清理相应缓存，不能依赖旧副本跨参数更新长期复用。规则可核对 [自动转换实现](https://github.com/pytorch/pytorch/blob/v2.2.2/aten/src/ATen/autocast_mode.cpp)和 [autocast 上下文实现](https://github.com/pytorch/pytorch/blob/v2.2.2/torch/amp/autocast_mode.py)。
- 小矩阵、CPU 路径、缺少相应硬件能力时，转换与调度可能抵消收益，甚至更慢。

**内存也不能直接按一半估计。** 本课参数、参数梯度、AdamW m/v 仍为 FP32；部分激活也是 FP32，还有临时张量和转换缓存。只有真正改用 BF16 的相同元素张量，其逻辑大小才从每元素 4 字节降到 2 字节。

例如本次 `[1,8,50257]` logits 的逻辑大小：

```plain text
元素数：1 × 8 × 50257 = 402,056
FP32：402,056 × 4 = 1,608,224 字节
BF16：402,056 × 2 =   804,112 字节
```

这不是整次训练的峰值显存。GPU 实验还需要另外观察峰值已分配内存及具体配置。

## 十三、把混合精度流程对应到代码 {color="blue"}

以下是当前核心结构的缩写，前提是模型参数为 FP32、x/y 为同设备的整数 ID：

```python
optimizer.zero_grad(set_to_none=True)

with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    logits = model(x)  # 本课为 BF16，形状 [B,T,V]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )                 # 本课交叉熵使用 FP32，输出一个标量

loss.backward()       # 沿真实计算图反传，原 FP32 参数的 .grad 为 FP32
optimizer.step()      # 用原 FP32 参数、梯度与 AdamW 状态更新
```

`with` 控制自动转换规则的启用范围，离开后恢复原设置。已经生成的 BF16 logits 不会因离开 `with` 自动变成 FP32。反向放在区域外，也不意味着整个反向的中间结果都强制为 FP32；见 [PyTorch autocast 的使用约定](https://github.com/pytorch/pytorch/blob/v2.2.2/torch/amp/autocast_mode.py)。

本方案不用 `model.bfloat16()`，也不把 ID 转成浮点数。与原课的差别只是作者在 `model(x,y)` 内算 loss，本项目在模型外算，但二者都放在 autocast 区域内。见[原课 BF16 提交](https://github.com/karpathy/build-nanogpt/commit/177e4cd5b4cc05df4bb637ed1a9e55911d6f1e2c)。

<table header-row="true">
<tr>
<td>文件 / 函数</td>
<td>复习时关注什么</td>
</tr>
<tr>
<td>`benchmark_training.py / select_device`</td>
<td>怎么选择设备，指定不可用设备怎么办</td>
</tr>
<tr>
<td>`configure_precision`</td>
<td>`highest` / `high` 与硬件检查</td>
</tr>
<tr>
<td>`synchronize`、`benchmark_steps`</td>
<td>一个 step 的计时边界与预热规则</td>
</tr>
<tr>
<td>`summarize_timings`</td>
<td>总 Token / 总时间的汇总口径</td>
</tr>
<tr>
<td>`train_mixed_precision.py / forward_and_loss`</td>
<td>autocast 覆盖前向和 loss</td>
</tr>
<tr>
<td>`train_amp_steps`</td>
<td>backward、更新与计时的衔接</td>
</tr>
<tr>
<td>`inspect_forward_dtypes`</td>
<td>用临时 hook 观察类型；不训练，不计入性能测量</td>
</tr>
</table>

## 十四、已经核对的结果，和仍待完成的实验 {color="blue"}

以下都是已有报告，本次整理笔记未重新训练完整模型。

<table header-row="true">
<tr>
<td>已有记录</td>
<td>运行条件</td>
<td>可以支持的结论</td>
</tr>
<tr>
<td>知识块 8 CPU 基线（项目内 `artifacts/lesson08-cpu-baseline.json`）</td>
<td>124M、FP32、B=4/T=32、2 步预热 + 5 步测量</td>
<td>计时和训练连通；平均 786.19 ms/步，162.8 Token/s</td>
</tr>
<tr>
<td>知识块 9 dtype 对照（项目内 `artifacts/lesson09-cpu-dtypes.json`）</td>
<td>CPU、同一初值/输入、B=1/T=8、不更新</td>
<td>FP32/BF16 前向 loss 分别为 10.530962 / 10.526217；输出类型和逻辑字节数符合报告</td>
</tr>
<tr>
<td>知识块 9 BF16 短跑（项目内 `artifacts/lesson09-cpu-bf16-smoke.json`）</td>
<td>CPU、B=1/T=8、1 步预热 + 2 步测量</td>
<td>3 次实际参数更新；参数/梯度/loss/m/v 为 FP32，logits 为 BF16</td>
</tr>
<tr>
<td>已有 29 项行为测试</td>
<td>小模型，包含新旧训练相关行为</td>
<td>FP32 轨迹保持、真实 BF16 更新、类型边界、共享与观察工具行为已核验</td>
</tr>
</table>

BF16 CPU 短跑平均约 3863.43 ms/步、2.07 Token/s。这与上一行 FP32 基线的 B/T 不同，而且都不是 GPU 对照，所以不计算两份报告的“BF16 加速比”。不同 batch 的短期 loss 也不能证明长期质量改善。

GPU 实验应固定设备、模型/初值、数据顺序、B/T、优化器、预热与测量步数和软件版本，再逐项更改设置：

<table header-row="true">
<tr>
<td>对照</td>
<td>基线</td>
<td>变化项</td>
<td>记录</td>
</tr>
<tr>
<td>TF32</td>
<td>FP32 + highest</td>
<td>FP32 + high</td>
<td>同条件耗时、吞吐与数值变化</td>
</tr>
<tr>
<td>BF16</td>
<td>FP32 + high</td>
<td>BF16 autocast + high</td>
<td>同条件耗时、吞吐、峰值内存与数值变化</td>
</tr>
</table>

保留 `high` 一致，是为了沿视频“已开启 TF32 → 再加 BF16”的顺序单独观察 BF16。具体可运行命令分别在 lesson08（项目内 `lesson08.md`）、lesson09（项目内 `lesson09.md`）。当前 GPU/TF32/BF16 性能实验尚未执行。

本机复习可以先运行设备检查或 dtype 观察，不需要重跑所有训练：

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/benchmark_training.py --inspect-only
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/train_mixed_precision.py --device cpu --probe-only --batch-size 1 --seq-len 8
```

## 十五、复习自检与学习衔接 {color="blue"}

1. `device='cuda'` 与 `dtype=torch.bfloat16` 各决定什么？
2. 为什么只移动模型还不够，输入 ID 又为什么不能当作 BF16 特征？
3. 为什么预热会改变参数，但不加入最终吞吐统计？
4. 一个 batch 有 4×32 个监督位置，一步 0.25 秒，吞吐是多少？
5. `dtype` 显示 FP32，为什么仍可能使用了 TF32 内部计算？
6. 输入 FP32、权重 FP32，为什么 autocast 下的线性层能输出 BF16？
7. 内部 FP32 累加，是否意味着输出 Tensor 一定是 FP32？
8. 计算用 BF16 权重怎么与原 FP32 参数连接，谁真正更新参数？
9. BF16 扩成 FP32，能不能恢复被舍去的低位？
10. logits 大小减半，是否能推出训练显存减半、速度翻倍？

<details>
<summary>展开简要答案</summary>

	1. device 决定设备，dtype 决定元素数值类型。
	2. 本课算子需要模型与输入处于兼容设备；ID 是索引，Embedding 后才有浮点特征。
	3. 预热执行完整更新，只排除启动阶段的测量干扰，不回滚模型或取批位置。
	4. `128 / 0.25 = 512 Token/s`；训练监督位置只算一次。
	5. TF32 改变内部运算精度，FP32 的 Tensor 存储类型可以保持不变。
	6. 符合规则的算子按需转换输入和权重，原 Tensor 的类型不被永久更改。
	7. 不能。内部累加与输出存储是不同层次。
	8. 浮点转换保留自动求导连接；backward 累积梯度，AdamW step 更新原 FP32 参数。
	9. 不能。只精确表示已有的 BF16 值，新增低位补零，后续 FP32 运算可以产生新低位。
	10. 不能。其他训练状态、激活、临时数据和转换成本仍要计算，收益必须实测。

</details>

目前已学习设备执行、正确计时、TF32 和 BF16，BF16 按本人反馈记录为“初步理解”。下一知识块按视频进入 `torch.compile`（约 1:47:34 起），本篇只记录衔接，不提前算作已学习。2.1 已完成；2.2 仍在进行；单元 2 保持 1/5，阶段二保持 5/32。自检供复习，不另设完成门槛。
