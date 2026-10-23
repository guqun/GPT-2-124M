> 已经算出了梯度，为什么还需要优化器？本篇沿着“问题 → 历史记录 → 参数更新 → 偏差修正 → 归一化 → 期望”逐层说明，再接上梯度裁剪、学习率调度与梯度累积，保留学习中反复追问的推导、手算、曲线和全流程图。

归档日期：<date start="2026-10-21" />；梯度累积补充：<date start="2026-10-23" />。学习位置：<b>阶段二 · 单元 2 · 2.3「超参数与优化」：已讨论 AdamW、梯度裁剪、学习率调度与梯度累积，理解中。</b> 本篇是讨论记录和数学补充，2.3 尚未完成。

项目：[guqun/GPT-2-124M](https://github.com/guqun/GPT-2-124M)。前篇：训练提速（二）（项目内 <code>docs/gpu-training-optimization-notes.md</code>）（<mention url="https://app.notion.com/p/3ffeaea3141481f9bd20ca4745fa8b80" />）。代码对应：<code>train_compile.py</code>（项目内 <code>train_compile.py</code>）、<code>train_mixed_precision.py</code>（项目内 <code>train_mixed_precision.py</code>）。

<b>来源与范围</b>：主线接续 [Karpathy 原视频约 2:14:55 起的优化器讨论](https://www.youtube.com/watch?v=l8pRSuU81PU&t=8095s)。通过非官方自动字幕定位 AdamW、约 2:17—2:20 的梯度裁剪及约 2:21—2:25 的学习率调度，核对作者 [105f117](https://github.com/karpathy/build-nanogpt/commit/105f117) 与 [90e5d15](https://github.com/karpathy/build-nanogpt/commit/90e5d15) 历史提交，未直接观看视频画面。完整的偏差修正推导、期望例子及配图属于答疑补充，不能当成原视频逐项讲过的内容。实现按项目 PyTorch 2.2.2 核对；手算数值是独立例子，不是 GPT 的实测梯度。

<table-of-contents />

## 一、当前问题：梯度给出了什么，优化器还要决定什么

当前主线仍是随机初始化 GPT-2、使用 Tiny Shakespeare 学习预训练。已有训练入口已经使用 AdamW；这一部分是理解它的更新机制，后续再学习配置。

一次普通训练更新的职责如下：

| 环节 | 输入 | 输出 | 目的 |
| --- | --- | --- | --- |
| 前向与损失 | 当前参数、当前 batch 的输入和标签 | loss 标量 | 衡量当前预测与目标的差距 |
| 反向传播 | loss 对应的计算图 | 各参数的梯度 | 计算参数变化对损失的局部影响 |
| AdamW | 参数、当前梯度、历史 m/v、超参数 | 新参数、新 m/v、更新次数 | 决定具体如何修改参数 |
| 下一批前向 | 新参数、下一批数据 | 新的预测和 loss | 继续学习 |

对某个标量参数 w，梯度是：

$$
g_t=\frac{\partial L_t}{\partial w}
$$

普通 SGD 可以直接使用：

$$
w_t=w_{t-1}-\eta_t g_t
$$

例如旧参数为 1，梯度为 0.2，学习率为 0.01，新参数就是 0.998。这是为了对比规则的教学例子，不代表项目此前使用的是 SGD。

实际训练中，梯度会随 batch 波动，不同参数的梯度尺度也不一样。Adam 用历史方向和历史大小调整更新；AdamW 再加入独立的权重衰减。改善的是参数更新的行为，不能仅凭一次步幅更大就断言更好，也不保证每个 batch 的 loss 都下降。

## 二、它作用于哪些参数，什么时候执行

<code>torch.optim.AdamW(model.parameters(), ...)</code> 将模型参数交给优化器管理。当前模型中包括 Token/Position Embedding、Attention 的投影权重与偏置、MLP、LayerNorm 的缩放与偏移，以及输出层。Embedding 与 lm\_head 共享时是同一份 Parameter，常规 <code>model.parameters()</code> 会去重。

需要区分：

| 对象 | 是否直接由 AdamW 更新 |
| --- | --- |
| Embedding 表、W\_Q/W\_K/W\_V、MLP 权重、LayerNorm γ/β | 是，前提是被优化器管理且本轮有梯度 |
| 前向产生的 Q/K/V、隐藏向量、logits | 是中间计算结果，下一批用新参数重新计算 |
| Token ID、位置编号、固定因果遮罩 | 是索引或固定数据，不是本例的可训练参数 |
| AdamW 的 m、v | 由优化器按递推公式维护，不通过反向传播训练 |

<b>每个参数元素都有自己的 m 和 v。</b> 对一个 \[768,768\] 的参数矩阵，m、v 也是 \[768,768\]；并非整个模型共用两个标量。

<code>optimizer.step()</code> 在反向传播完成后执行，批量更新各层参数，并不是每完成一个前向层就更新一次。常规训练中冻结参数不产生梯度；PyTorch AdamW 跳过本轮 <code>.grad is None</code> 的参数。梯度数值为 0 与没有梯度不同：历史动量和权重衰减仍可能使零梯度参数改变。

## 三、先认清符号，再看完整公式

以下对一个参数元素推导；矩阵实现逐元素计算。t 表示该参数参与优化器更新的次数，通常与训练更新步一致，不是 Token 位置或 epoch 编号。

| 符号 | 含义 | 来自哪里 |
| --- | --- | --- |
| w | 模型中一个可训练参数 | 随机初始化或加载权重 |
| g\_t | 本次更新使用的梯度 | backward 写入 .grad |
| m\_t | 梯度的指数移动平均记录 | 优化器维护，初始为 0 |
| v\_t | 梯度平方的指数移动平均记录 | 优化器维护，初始为 0 |
| β₁、β₂ | 对历史记录的保留系数 | 超参数，范围 \[0,1) |
| m̂\_t、v̂\_t | 初始化偏差修正后的记录 | 用于计算当前更新 |
| η\_t | 学习率 | 人为设置或后续调度器给出 |
| λ | 权重衰减系数 | 当前参数组的超参数 |
| ε | 分母中的很小正数 | 数值稳定项，例如 10⁻⁸ |

β₁、β₂ 是优化器超参数，与 LayerNorm 中常记作 β 的可训练偏移不是同一个对象。

标准 AdamW（默认不使用 AMSGrad，做损失最小化）的公式：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

$$
w_t=(1-\eta_t\lambda)w_{t-1}
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

这里的幂、平方根、乘除都按参数元素执行；不是矩阵乘法或矩阵开方。公式与 [PyTorch 2.2.2 AdamW](https://github.com/pytorch/pytorch/blob/v2.2.2/torch/optim/adamw.py) 对应。

## 四、m 与 v 分别解决什么问题

以 β₁=0.9 为例：

```txt
新的 m = 90% 的旧 m + 10% 的当前梯度
```

m 保留符号，反映近期的调整方向。它减弱只跟随一个 batch 的波动，但也会滞后；一次梯度翻转不一定立即改变更新方向。

以 β₂=0.95 为例：

```txt
新的 v = 95% 的旧 v + 5% 的当前梯度平方
```

v 使用平方，记录近期梯度的大小尺度。它是<b>未中心化的二阶矩记录</b>，不是二阶导数，也不是减去均值之后得到的方差。

更新时，用 m̂ 除以 √v̂ + ε，可按这个参数自身的历史梯度尺度调整步幅。不能简化成“任何大梯度参数一定比小梯度参数走得更慢”，因为分子和分母会一起变化。学习率 η 仍然控制整体步幅。[Adam 原论文](https://arxiv.org/abs/1412.6980)

<b>下一批递推继续使用原始 m、v。</b> 带帽子的修正量用于本次更新，不能直接覆盖原始记录再套同一个递推公式。清梯度也不会清除 m/v；每个 batch 都重建优化器则会丢失历史。

## 五、先完整算一次参数更新

本例为便于手算选择：

```txt
w₀ = 1
g₁ = 0.2
m₀ = 0，v₀ = 0
η = 0.01，β₁ = 0.9，β₂ = 0.95，λ = 0.1，ε = 10⁻⁸
```

这些是假设值，尤其 η=0.01 不是 GPT 训练建议。

先得到历史记录：

$$
m_1=0.9\times0+0.1\times0.2=0.02
$$

$$
v_1=0.95\times0+0.05\times0.2^2=0.002
$$

再修正：

$$
\hat m_1=0.02/0.1=0.2,
\qquad
\hat v_1=0.002/0.05=0.04
$$

暂时忽略很小的 ε：

$$
w_1=(1-0.01\times0.1)\times1
-0.01\times\frac{0.2}{\sqrt{0.04}}
=0.999-0.01=0.989
$$

其中权重衰减使参数减少 0.001，梯度历史更新使它减少约 0.01。

下一批梯度假设变成 −0.1：

$$
m_2=0.9\times0.02+0.1\times(-0.1)=0.008
$$

当前梯度虽然为负，m₂ 仍然为正，参数在这个例子中继续减少。历史记录可能使更新方向暂时与当前梯度的负方向不同。

| 更新 t | 当前 g | m | v | m̂ | v̂ | 更新后的 w |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.2 | 0.020000 | 0.002000 | 0.200000 | 0.040000 | 0.9890000005 |
| 2 | −0.1 | 0.008000 | 0.002400 | 0.042105 | 0.024615 | 0.9853273062 |

表格使用 ε=10⁻⁸；已用 CPU/float64 与 PyTorch AdamW 逐步核对，结果在 10⁻¹² 容差内一致。数值复算记录（项目内 <code>artifacts/adamw-bias-correction-check.json</code>）

## 六、初始化偏差：为什么要除以一个数

模型参数 w 可以随机初始化，但这里要修正的是<b>优化器的 m/v 从 0 开始</b>所带来的影响，不是修正模型参数的随机初值。

先换成一个独立例子：每次梯度固定为 0.2，β₁=0.9。

| 更新 t | 未修正 m\_t | 实际梯度系数总和 | 修正后的 m̂\_t |
| --- | --- | --- | --- |
| 1 | 0.0200 | 0.100 | 0.2 |
| 2 | 0.0380 | 0.190 | 0.2 |
| 3 | 0.0542 | 0.271 | 0.2 |

例如第一步：

$$
m_1=0.9\times0+0.1\times0.2=0.02
$$

梯度一直为 0.2，但初始记录为 0，实际梯度只占了 10% 的系数。除以 0.1 后得到 0.2。

这并不表示修正后永远等于当前梯度；本例特意让所有梯度相同。梯度变化时，修正后的量仍是历史加权平均。

## 七、数学推导：1 − βᵗ 从哪里来

先令 β=β₁，且 m₀=0。把递推式逐次展开：

$$
m_1=(1-\beta)g_1
$$

$$
m_2=\beta(1-\beta)g_1+(1-\beta)g_2
$$

$$
m_3=\beta^2(1-\beta)g_1+\beta(1-\beta)g_2+(1-\beta)g_3
$$

每经过一次更新，旧梯度的系数就再乘一次 β。因此：

$$
m_t=(1-\beta)\sum_{i=1}^{t}\beta^{t-i}g_i
$$

第 i 次梯度前面的系数为：

$$
a_i=(1-\beta)\beta^{t-i}
$$

把所有实际梯度的系数加起来：

$$
C_t=\sum_{i=1}^{t}a_i
=(1-\beta)(1+\beta+\cdots+\beta^{t-1})
$$

等比求和不必硬记。先记 S=1+β+⋯+βᵗ⁻¹，两边乘 β 后相减：

$$
S-\beta S
=(1+\beta+\cdots+\beta^{t-1})
-(\beta+\beta^2+\cdots+\beta^t)
=1-\beta^t
$$

所以：

$$
S=\frac{1-\beta^t}{1-\beta}
$$

代回：

$$
C_t=(1-\beta)\frac{1-\beta^t}{1-\beta}
=\boxed{1-\beta^t}
$$

原始递推还可以写成“βᵗm₀ + 所有梯度的加权和”。当 m₀=0 时，初始化的 0 仍占 βᵗ 的系数，真正的梯度合计占 1−βᵗ。

于是：

$$
\boxed{\hat m_t=\frac{m_t}{1-\beta^t}}
$$

这是从初始条件和递推式推出的分母，不是随意增加的经验常数。若采用其他初始条件，不能照搬这一推导。

## 八、为什么“梯度加权和 ÷ 权重总和”能归一化

<b>归一化的是梯度前面的系数，使系数之和等于 1。梯度值本身不用变成 1，也不用落在 0 到 1 之间。</b> 此处“权重”指历史梯度的系数，不是网络参数 w。

取 β=0.9，第二次记录为：

$$
m_2=0.09g_1+0.10g_2
$$

两个系数的总和是 0.19。分别除以 0.19：

| 项目 | 原权重 | 新权重 |
| --- | --- | --- |
| g₁ | 0.09 | 9/19 |
| g₂ | 0.10 | 10/19 |
| 总和 | 0.19 | 1 |

权重的比例仍为 9:10。整体除法与逐项除法相同：

$$
\begin{aligned}
\frac{m_2}{0.19}
&=\frac{0.09g_1+0.10g_2}{0.19}\\
&=\frac{0.09g_1}{0.19}+\frac{0.10g_2}{0.19}\\
&=\left(\frac{0.09}{0.19}\right)g_1
+\left(\frac{0.10}{0.19}\right)g_2
\end{aligned}
$$

因此，“先归一化每份权重再加权求和”和“先计算原来的加权和再整体除以权重总和”完全等价。

再单独取 g₁=2、g₂=4：

$$
m_2=0.09\times2+0.10\times4=0.58
$$

$$
\hat m_2=\frac{0.58}{0.19}
=\frac9{19}\times2+\frac{10}{19}\times4
\approx3.052632
$$

结果是 2 和 4 的加权平均，更偏向权重稍大的 4。总和为 1 的是 <b>9/19 与 10/19</b>，不是结果 3.052632。

一般形式就是：

$$
\hat m_t
=\frac{\sum_i a_i g_i}{C_t}
=\sum_i\left(\frac{a_i}{C_t}\right)g_i,
\qquad
\sum_i\frac{a_i}{C_t}=1
$$

把 a\_i 和 C\_t 换回上一节的表达式，每次历史梯度的新系数就是：

$$
\frac{(1-\beta)\beta^{t-i}}{1-\beta^t}
$$

## 九、梯度期望 μ 究竟是哪个数值

期望是<b>可能结果按发生概率加权的平均</b>。它是概率分布的性质，不是某次训练一定得到的值，也不是程序额外需要输入的参数。

这里换成一个随机梯度的独立教学例子：在相同条件下随机取样，梯度可能为 0.1 或 0.3，各占 50%。

$$
\mu=\mathbb E[g_i]
=0.5\times0.1+0.5\times0.3
=0.2
$$

一次实际取样得到 0.1 或 0.3；0.2 是期望。需要分开以下四个量：

| 量 | 含义 | 第一次更新的本例数值 |
| --- | --- | --- |
| g₁ | 实际梯度 | 0.1 或 0.3 |
| E\[g₁\] = μ | 梯度的期望 | 0.2 |
| m₁ | 实际历史记录 | 0.01 或 0.03 |
| E\[m₁\] | 历史记录的期望 | 0.02 |

因为 m₀=0、β=0.9：

```txt
抽到 g₁=0.1 → m₁=0.1×0.1=0.01
抽到 g₁=0.3 → m₁=0.1×0.3=0.03

E[m₁] = 50%×0.01 + 50%×0.03 = 0.02
```

假设各次梯度的期望都保持 0.2：

| 更新 t | 梯度期望 E\[g\_t\] | 记录期望 E\[m\_t\] = (1−0.9ᵗ)×0.2 | 修正后的期望 E\[m̂\_t\] |
| --- | --- | --- | --- |
| 1 | 0.2 | 0.0200 | 0.2 |
| 2 | 0.2 | 0.0380 | 0.2 |
| 3 | 0.2 | 0.0542 | 0.2 |

<b>(1−βᵗ)μ 是 m\_t 的期望，不是 g\_t 的期望。</b> 只有给定 β、t 和梯度分布对应的 μ，才能算出具体值。不能把表中的 E\[m₂\]=0.038 当作每次实际运行的 m₂。

例如实际先抽到 0.3、再抽到 0.1，则 m₁=0.03、m₂=0.037，m̂₂=0.037/0.19≈0.194737，并不恰好是 0.2。将所有可能结果按概率平均，修正后的期望才是 0.2。

AdamW 执行时无需知道 μ，只读取实际 g\_t、维护实际 m\_t，再除以已知的 1−βᵗ。

## 十、为什么称为“偏差修正”：期望推导及条件

假设各次梯度的期望相同，即 E\[g\_i\]=μ。使用期望的线性性质：

$$
\begin{aligned}
\mathbb E[m_t]
&=(1-\beta)\sum_{i=1}^{t}\beta^{t-i}\mathbb E[g_i]\\
&=(1-\beta)\sum_{i=1}^{t}\beta^{t-i}\mu\\
&=(1-\beta^t)\mu
\end{aligned}
$$

相对于想估计的 μ，偏差为：

$$
\mathbb E[m_t]-\mu=-\beta^t\mu
$$

因此记录的期望被拉向 0；若 μ 为负，是绝对值偏小，不能笼统理解成数值总是更小。

修正之后：

$$
\mathbb E[\hat m_t]
=\frac{\mathbb E[m_t]}{1-\beta^t}
=\frac{(1-\beta^t)\mu}{1-\beta^t}
=\mu
$$

在这个假设下，修正后的估计是无偏的。期望的这一步推导只需要各次期望相同，不要求梯度相互独立。

实际训练不断更新参数，梯度分布通常也会变化。一般情况下：

$$
\mathbb E[\hat m_t]
=\sum_{i=1}^{t}
\frac{(1-\beta)\beta^{t-i}}{1-\beta^t}\mathbb E[g_i]
$$

它是各次期望的归一化加权平均，不保证等于当前的 E\[g\_t\]。所以严格区分：

- <b>任意梯度序列都成立</b>：修正后，已有梯度的系数之和为 1。
- <b>需要额外假设</b>：当各次梯度期望相同时，m̂\_t 是该共同期望的无偏估计。

t 增大后，βᵗ 趋近于 0，分母趋近于 1，初始化修正影响逐渐减小。它不意味着随机波动被消除。

## 十一、v 的修正为何一样，为什么不能把 E\[g²\] 当成 μ²

v 的递推只把 g 换成了 g²：

$$
v_t=(1-\beta_2)\sum_{i=1}^{t}\beta_2^{t-i}g_i^2
$$

实际平方梯度的系数总和是 1−β₂ᵗ，所以：

$$
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

若各次平方梯度的期望保持为 ν：

$$
\mathbb E[g_i^2]=\nu
\quad\Longrightarrow\quad
\mathbb E[v_t]=(1-\beta_2^t)\nu,
\qquad
\mathbb E[\hat v_t]=\nu
$$

仍用上一节 0.1/0.3 各半的随机例子：

$$
\nu=\mathbb E[g_i^2]
=0.5\times0.1^2+0.5\times0.3^2
=0.05
$$

但：

$$
(\mathbb E[g_i])^2=0.2^2=0.04
$$

<b>先平方再平均，与先平均再平方不相同。</b> 这个随机例子中 v̂ 的期望是 0.05；第六节每次梯度都固定为 0.2 的例子中，v̂ 是 0.04，因为两个例子的梯度分布不同。

m̂、v̂ 的相应期望结论，也不能直接推出它们的比值 m̂/(√v̂+ε) 是某个目标的无偏估计；比值和平方根是非线性运算。本篇只用它们说明 Adam 如何估计历史方向与尺度。

## 十二、AdamW 名称的含义与独立权重衰减

Adam 来自 <b>Adaptive Moment Estimation（自适应矩估计）</b>，由 Kingma 和 Jimmy Ba 提出。“自适应”表示参考每个参数自己的梯度历史调整更新；“矩”是统计概念，m 估计梯度的一阶矩，v 估计梯度的二阶矩；“估计”表示使用训练中实际遇到的梯度累计统计量。v 记录梯度平方的平均，不是方差。[Adam 原论文](https://arxiv.org/abs/1412.6980)

W 来自 <b>Weight Decay（权重衰减）</b>。AdamW 是 Adam with decoupled Weight Decay，即带解耦权重衰减的 Adam。解耦表示衰减直接作用于参数，独立于梯度的 m/v 统计。[AdamW 原论文](https://arxiv.org/abs/1711.05101)

完整更新可以拆成两项：

```txt
新参数 = 旧参数
       − 学习率 × 权重衰减系数 × 旧参数
       − 学习率 × m̂ / (√v̂ + ε)
```

权重衰减使正参数略微减小、负参数略微增大，即朝 0 收缩。通常 ηλ 很小；衰减是一项收缩作用，不保证叠加梯度更新后的参数绝对值每步都变小。

与把 L2 正则项并入梯度的 Adam 不同，AdamW 的衰减项不进入 m/v 的梯度统计；其动机是将衰减与自适应梯度更新分离。[AdamW 原论文](https://arxiv.org/abs/1711.05101)

优化器可以设置参数组：一些参数使用衰减，一些参数的 λ=0。后续课程会讲矩阵权重与一维偏置/LayerNorm 参数的分组；λ=0 的参数仍执行 Adam 的 m/v 更新。

<b>当前项目入口的实际配置与课程设置要分开：</b>

| 配置来源 | betas | eps | weight\_decay |
| --- | --- | --- | --- |
| 当前 <code>train_compile.py</code>，只显式传入 lr | PyTorch 2.2.2 默认 (0.9,0.999) | 10⁻⁸ | 默认 0.01，当前没有分组 |
| 原课 105f117 的 AdamW 调用 | 显式 (0.9,0.95) | 显式 10⁻⁸ | 该调用未显式覆盖 |
| 本篇两步标量手算 | (0.9,0.95) | 10⁻⁸ | 0.1，仅为手算 |

本篇记录原理，不修改旧课程脚本来冒充已经完成后续配置实验。

## 十三、对应到我们项目的代码

创建优化器发生在 <code>train_compile.py</code>（项目内 <code>train_compile.py</code>） 的 main 中，准备好模型之后：

```py
# raw_model 中的参数才是长期被更新的对象。
# compile 包装后的前向依然读取这些参数的当前值。
optimizer = torch.optim.AdamW(
    raw_model.parameters(),
    lr=args.learning_rate,
)
```

训练循环在 <code>train_mixed_precision.py</code>（项目内 <code>train_mixed_precision.py</code>） 的 train\_amp\_steps 中：

```py
# 下面摘出更新主线；设备、计时、报告等代码见原文件。
optimizer.zero_grad(set_to_none=True)  # 清本轮之前的梯度；m/v 继续保留
logits, loss = forward_and_loss(model, x, y, device, precision)
loss.backward()                       # 产生当前梯度；此时还没执行参数更新
optimizer.step()                      # 递推 m/v，执行 AdamW 参数更新
```

| 数学中的量 | PyTorch 中的对应 |
| --- | --- |
| w | Parameter 内的数值 |
| g\_t | parameter.grad |
| m\_t | optimizer.state\[parameter\]\["exp\_avg"\] |
| v\_t | optimizer.state\[parameter\]\["exp\_avg\_sq"\] |
| 更新次数 | optimizer.state\[parameter\]\["step"\] |
| η、β₁/β₂、λ、ε | optimizer.param\_groups 中的配置 |

当前采用 FP32 原参数配合 BF16 autocast 时，原参数及其梯度、m/v 仍按此前的 FP32 方案维护。这一机制接在前向和反向之后。

将来保存可恢复的训练检查点，需要同时考虑优化器状态：只有模型参数，不能恢复原来的 m/v 与更新次数。这里记录这个关系，不扩展为新保存功能。

## 十四、梯度裁剪：先处理梯度，再交给 AdamW

<b>问题与目的。</b> 某个 batch 可能产生异常大的梯度，突然影响优化器的历史统计与后续更新。梯度裁剪给本次梯度的整体长度设置上限，减少这种冲击。AdamW 的自适应处理并没有自动实施这个上限；同时，也不能简单断言梯度增大几倍，AdamW 的参数更新就一定增大几倍。

<table fit-page-width="true" header-row="true">
<tr>
<td>
项目
</td>
<td>
含义
</td>
</tr>
<tr>
<td>
输入
</td>
<td>
反向传播后各参数的 <code>.grad</code>，以及阈值 c
</td>
</tr>
<tr>
<td>
处理对象
</td>
<td>
所有传入参数中非空的梯度；概念上把这些元素合成一个向量
</td>
</tr>
<tr>
<td>
输出
</td>
<td>
原位修改后的梯度；函数另返回裁剪前的总范数
</td>
</tr>
<tr>
<td>
执行位置
</td>
<td>
<code>loss.backward()</code> 之后，<code>optimizer.step()</code> 之前
</td>
</tr>
<tr>
<td>
此时参数是否改变
</td>
<td>
裁剪这一步只修改梯度，参数稍后由 AdamW 更新
</td>
</tr>
</table>

把所有梯度元素记为 g₁ 到 gₙ，先计算 L2 范数，也就是整个向量的长度：

$$
G=\sqrt{\sum_{i=1}^{N}g_i^2}
$$

给定阈值 c，规则是：

$$
g_i'=\begin{cases}
g_i,&G\leq c,\\
\dfrac{c}{G}g_i,&G>c.
\end{cases}
$$

<b>所有梯度使用同一个缩放比例。</b> 因而缩小时保留整体方向和各分量的比例；这是全局范数裁剪，不是分别把每个元素截断到某个区间，也不是把每个参数矩阵各自归一化到 1。

用两个参数的梯度 g=\[3,4\]，阈值 c=1 手算：

$$
G=\sqrt{3^2+4^2}=5,\qquad \frac{c}{G}=0.2
$$

$$
g'=[0.6,0.8],\qquad \sqrt{0.6^2+0.8^2}=1
$$

如果 g=\[0.3,0.4\]，其长度只有 0.5，没有超过 1，就保持原样。裁剪不会把较小的梯度强行放大。

PyTorch 2.2.2 的实际缩放系数还包含数值保护项：

$$
\alpha=\min\left(1,\frac{c}{G+10^{-6}}\right),\qquad g'=\alpha g
$$

因此实际得到的 0.6、0.8 会略小于手算值；保护项也避免 G=0 时除零。<code>clip_grad_norm_</code> 尾部下划线表示原位修改，返回的范数是裁剪前的 G；例子中返回约 5。[PyTorch 源码](https://github.com/pytorch/pytorch/blob/v2.2.2/torch/nn/utils/clip_grad.py)

AdamW 随后使用裁剪后的梯度递推历史状态：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t',\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)(g_t')^2
$$

原有 m/v 不清空，之后照常进行偏差修正和参数更新。阈值 1 限制的是本次梯度的整体长度，不是 loss、参数大小或最终参数更新量的长度。

## 十五、学习率调度：预热、余弦衰减、下限保持

<b>问题与目的。</b> 训练初期可能不稳定，适合先以较小的学习率起步；逐渐提高到设定上限后，再随训练进度降低更新幅度。AdamW 根据每个参数的 m/v 自适应处理梯度，学习率 η 再控制更新尺度。两者同时起作用。

例如暂时只看梯度更新项，假设 m̂/(√v̂+ε)=0.5：η=0.0006 时，这一项为 0.0003；η=0.00006 时，这一项为 0.00003。其他量相同时，学习率降低十倍，这部分更新幅度也降低十倍。完整 AdamW 公式中的权重衰减项也含学习率。

<b>调度器的输入</b>是当前更新序号、预热步数、衰减终点、最大与最小学习率；<b>输出</b>是本次 <code>optimizer.step()</code> 使用的一个学习率数值。这种调度由预先设定的进度规则计算，不根据当前 loss 自动选择阶段。

<image source="notion-file-block://ad69c648-3750-4818-81f3-cb7cd34d7aa1/0c654614-2168-41dc-be94-5d01f9e9ed2c?space_id=6ac5907c-dabf-4363-a84a-c2c8afc00634&name=gpt2-learning-rate-three-phases.png" />

学习率三段规则：线性预热、余弦衰减、下限保持。

图中阶段宽度仅为示意；为说明第三段，画出了衰减结束后继续训练的部分。

<table fit-page-width="true" header-row="true">
<tr>
<td>
阶段
</td>
<td>
计算规则
</td>
<td>
目的
</td>
</tr>
<tr>
<td>
① 线性预热
</td>
<td>
从较小的学习率逐渐升至最大值
</td>
<td>
减轻训练初期过大更新的风险
</td>
</tr>
<tr>
<td>
② 余弦衰减
</td>
<td>
从最大值平滑降至最小值
</td>
<td>
随进度逐渐缩小更新幅度
</td>
</tr>
<tr>
<td>
③ 下限保持
</td>
<td>
到达下限后，若继续训练就保持该值
</td>
<td>
用设定的小学习率继续更新
</td>
</tr>
</table>

<b>“预热结束”是第一、二段之间的交界点，不是额外的一段。</b> 第三段保持的是学习率；前向、反向、梯度裁剪、m/v 与参数更新仍然继续。

<b>第一段怎样计算。</b> 沿用作者示例，最大学习率为 0.0006，前 W=10 次更新预热。代码更新序号 s 从 0 开始，第 s+1 次更新使用：

$$
\eta(s)=\eta_{\max}\frac{s+1}{W},\qquad 0\leq s<W
$$

<table fit-page-width="true" header-row="true">
<tr>
<td>
代码序号 s
</td>
<td>
第几次更新
</td>
<td>
学习率
</td>
</tr>
<tr>
<td>
0
</td>
<td>
第 1 次
</td>
<td>
0.000060
</td>
</tr>
<tr>
<td>
1
</td>
<td>
第 2 次
</td>
<td>
0.000120
</td>
</tr>
<tr>
<td>
4
</td>
<td>
第 5 次
</td>
<td>
0.000300
</td>
</tr>
<tr>
<td>
9
</td>
<td>
第 10 次
</td>
<td>
0.000600
</td>
</tr>
</table>

使用 s+1，第一次更新就能得到一个正的学习率。这里 s 是代码从 0 开始的序号；前面 Adam 偏差修正中的 t 从 1 开始计更新次数，二者计数起点不同。

<b>这里的预热与性能测量预热要区分。</b> 学习率预热改变每次更新的学习率；之前 benchmark 的预热，是先运行几步让设备、缓存、编译等准备充分，再统计耗时。二者都可以实际更新参数，但目的不同。本项目已有 <code>--warmup-steps</code> 属于性能测量预热，不能直接当成已经实现的学习率预热。

<b>第二段怎样计算。</b> 取最小学习率为最大值的十分之一：

$$
\eta_{\max}=0.0006,\qquad \eta_{\min}=0.00006
$$

令 D 为衰减终点的代码序号，衰减进度是：

$$
r=\frac{s-W}{D-W},\qquad W\leq s\leq D
$$

所以 r=0 表示刚开始衰减，r=0.5 表示衰减走了一半，r=1 表示到达衰减终点。它只描述第二段，不是整个训练的完成比例。

$$
\eta(r)=\eta_{\min}+\frac{1+\cos(\pi r)}{2}(\eta_{\max}-\eta_{\min})
$$

<table fit-page-width="true" header-row="true">
<tr>
<td>
衰减进度 r
</td>
<td>
cos(πr)
</td>
<td>
系数 (1+cos(πr))/2
</td>
<td>
学习率
</td>
</tr>
<tr>
<td>
0
</td>
<td>
1
</td>
<td>
1
</td>
<td>
0.000600
</td>
</tr>
<tr>
<td>
0.5
</td>
<td>
0
</td>
<td>
0.5
</td>
<td>
0.000330
</td>
</tr>
<tr>
<td>
1
</td>
<td>
−1
</td>
<td>
0
</td>
<td>
0.000060
</td>
</tr>
</table>

<image source="notion-file-block://334c3d2f-d27a-4d48-b5bc-b1e8d8994222/3b103310-1887-42c6-8c80-e1ec0a3c412b?space_id=6ac5907c-dabf-4363-a84a-c2c8afc00634&name=gpt2-cosine-learning-rate.png" />

余弦公式的曲线：开始、中点和结束的学习率。

余弦系数从 1 降到 0，因此学习率从最大值降到最小值。两端下降较慢，中间下降较快。公式中的 πr 使用弧度，r 从 0 到 1 正好覆盖半个余弦周期。

<b>第三段怎样计算。</b> 到达衰减终点后，如果继续训练：

$$
\eta(s)=\eta_{\min}=0.00006,\qquad s\geq D
$$

保持学习率不意味着保持参数，也不意味着每步参数变化一样大：梯度和 m/v 仍会变化。训练也可以在衰减阶段末尾结束，这时不会实际运行额外的下限保持阶段。

<b>原课示例的边界。</b> 作者 <code>90e5d15</code> 使用 W=10、D=50，循环为 <code>range(50)</code>，实际序号是 0—49：s=9 的预热已到最大值，s=10 的余弦起点仍是最大值；s=49 的学习率约为 0.0000608323，接近下限。若计算 s=50，余弦公式恰好返回下限；s>50 则由保持分支返回下限。因此这个 50 步演示没有额外跑第三段，图中的完整三段用于说明调度规则。[作者实现](https://github.com/karpathy/build-nanogpt/commit/90e5d15)

## 十六、把裁剪、调度和 AdamW 串回一次训练更新

下面是对照作者实现的衔接示意，不是声称当前旧训练入口已经加入裁剪和学习率调度。第十三节保留了现有项目循环；本次只扩充笔记和配图。

```txt
当前参数 + 当前 batch
        ↓ 前向与 loss
        ↓ backward：生成当前梯度
        ↓ 梯度裁剪：必要时按同一比例缩小各梯度
        ↓ 按当前 step 计算并设置学习率
        ↓ AdamW：使用裁剪后的梯度更新 m/v
                 偏差修正 + 权重衰减 + 参数更新
        ↓ 下一批前向使用更新后的参数
```

学习率的计算可以放在更新前的其他位置，关键是 <code>optimizer.step()</code> 使用到本次的新值；梯度裁剪必须等梯度产生后、参数更新前执行。

```py
# 优化器在训练循环外创建一次，历史 m/v 随训练保留。
optimizer.zero_grad(set_to_none=True)
logits, loss = forward_and_loss(model, x, y, device, precision)
loss.backward()

# 修改 .grad，返回裁剪前的整体梯度长度。
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# get_lr 按上节的三段规则返回本次学习率。
lr = get_lr(step)
for group in optimizer.param_groups:
    group["lr"] = lr

# 使用当前梯度、既有历史状态和本次学习率更新参数。
optimizer.step()
```

<table fit-page-width="true" header-row="true">
<tr>
<td>
操作
</td>
<td>
修改什么
</td>
<td>
保留什么
</td>
</tr>
<tr>
<td>
<code>zero_grad</code>
</td>
<td>
清理之前累积的梯度
</td>
<td>
参数、优化器 m/v
</td>
</tr>
<tr>
<td>
梯度裁剪
</td>
<td>
当前 <code>.grad</code>
</td>
<td>
当前参数、既有 m/v
</td>
</tr>
<tr>
<td>
设置学习率
</td>
<td>
优化器参数组中的 <code>lr</code>
</td>
<td>
参数、梯度、m/v
</td>
</tr>
<tr>
<td>
<code>optimizer.step()</code>
</td>
<td>
m/v、更新次数和参数
</td>
<td>
同一个优化器对象与参数对象
</td>
</tr>
</table>

这里修改已有优化器的配置即可；每步重新创建 AdamW 会丢掉之前的历史状态。

## 十七、梯度累积：多次反向，一次参数更新

### 1. 当前问题与这一步的目的

前面每次处理一个 batch，完成一次反向后就更新参数。如果希望一次更新参考更多样本，直接加大 batch 会增加中间激活的显存占用，可能超出设备容量。

<b>梯度累积（Gradient Accumulation）的做法：把这一次更新要用的数据分成多个微批次，依次前向、反向，把梯度累加起来，最后只更新一次参数。</b> 四个微批次内部都使用同一套参数，包括同一份 Embedding；直到最后的 <code>optimizer.step()</code> 才改变参数。

原视频约 2:35—2:46 讨论这个问题，单设备例子为 <code>B=16、T=1024、K=32</code>，每次更新使用 524,288 个 Token 预测位置。[作者加入梯度累积的历史提交](https://github.com/karpathy/build-nanogpt/commit/01be6b3)与此前核对的自动字幕相互对应。下面缩小为四轮，以便逐项计算；矩阵数值是教学例子。

<table fit-page-width="true" header-row="true">
<tr>
<td>
名称
</td>
<td>
本例
</td>
<td>
含义
</td>
</tr>
<tr>
<td>
微批次大小 B
</td>
<td>
2
</td>
<td>
一次前向输入 2 条序列
</td>
</tr>
<tr>
<td>
序列长度 T
</td>
<td>
32
</td>
<td>
每条序列提供 32 个有效预测位置
</td>
</tr>
<tr>
<td>
累积次数 K
</td>
<td>
4
</td>
<td>
连续处理 4 个微批次后更新一次
</td>
</tr>
<tr>
<td>
有效 batch 大小
</td>
<td>
B × K = 8 条序列
</td>
<td>
一次更新合计参考的数据量
</td>
</tr>
<tr>
<td>
每次更新的有效 Token 数
</td>
<td>
B × T × K = 256
</td>
<td>
本例参与平均交叉熵的总位置数
</td>
</tr>
</table>

这里沿 batch 维拆成四组，每条序列的上下文长度仍为 32。把一条长序列切成短段会改变可见上下文，不能直接当成同一件事。本节先限定单设备，不引入后续 DDP。

### 2. 全流程图：先看控制流程，再看四轮矩阵

<image source="notion-file-block://08f161e8-92b2-473f-afe6-ac98f6e18930/da2fe7b0-fdb2-4b6a-b115-1ff501760858?space_id=6ac5907c-dabf-4363-a84a-c2c8afc00634&name=gpt2-gradient-accumulation-flow.png" />

原尺寸 PNG（项目内 <code>docs/diagrams/gpt2-gradient-accumulation-flow.png</code>） · 可缩放 SVG（项目内 <code>docs/diagrams/gpt2-gradient-accumulation-flow.svg</code>） · 配图阅读说明（项目内 <code>docs/diagrams/gpt2-gradient-accumulation-flow.md</code>）

图的上半部展开每个微批次的输入、loss 和反向；中间逐轮展示同一个参数矩阵的 <code>.grad</code>；下半部展示累计完成后的裁剪、学习率设置和 AdamW 更新。

### 3. 每一轮输入什么、计算什么、留下什么

<table fit-page-width="true" header-row="true">
<tr>
<td>
环节
</td>
<td>
输入
</td>
<td>
计算与判断
</td>
<td>
输出及目的
</td>
</tr>
<tr>
<td>
外层开始
</td>
<td>
上次更新后的模型和优化器
</td>
<td>
清 <code>.grad</code> 一次
</td>
<td>
开始一份新的累计梯度，保留参数与 m/v
</td>
</tr>
<tr>
<td>
取第 i 个微批次
</td>
<td>
loader 的下一个位置
</td>
<td>
读取新 x/y，并移动到设备
</td>
<td>
两个形状为 <code>[2,32]</code> 的 ID 张量；y 是右移一位的监督目标
</td>
</tr>
<tr>
<td>
前向
</td>
<td>
当前固定参数 θ、x
</td>
<td>
Embedding → Blocks → LN\_f → lm\_head
</td>
<td>
logits，形状 <code>[2,32,V]</code>；V 是输出词表大小
</td>
</tr>
<tr>
<td>
当前批损失
</td>
<td>
logits、y
</td>
<td>
计算 64 个有效位置的平均交叉熵
</td>
<td>
标量 Lᵢ，表示这个微批次的平均损失
</td>
</tr>
<tr>
<td>
跨批缩放
</td>
<td>
Lᵢ、K=4
</td>
<td>
Lᵢ ÷ 4
</td>
<td>
让本批对最终平均损失贡献四分之一
</td>
</tr>
<tr>
<td>
反向与累积
</td>
<td>
Lᵢ/4 对应的计算图、已有 <code>.grad</code>
</td>
<td>
链式法则求导，再逐元素相加
</td>
<td>
每个参数的累计梯度；参数本身不变
</td>
</tr>
<tr>
<td>
判断是否满 K 次
</td>
<td>
当前微批次计数
</td>
<td>
未满则读取新批次；已满则退出内层
</td>
<td>
控制何时统一裁剪与更新
</td>
</tr>
<tr>
<td>
外层更新
</td>
<td>
累计梯度、参数、m/v
</td>
<td>
裁剪 → 设置学习率 → AdamW
</td>
<td>
新参数和新 m/v，供下一次外层更新使用
</td>
</tr>
</table>

<b>累计保存在各个 </b><b><code>Parameter.grad</code></b><b> 中。</b> 如果 W 的形状是 <code>[2,2]</code>，<code>W.grad</code> 的形状也是 <code>[2,2]</code>。它不是保存四张梯度矩阵的列表，而是一张不断更新的累加结果。实际 GPT 的每个可训练参数都有自己的梯度张量。

<code>zero_grad(set_to_none=True)</code> 把 <code>.grad</code> 置为 <code>None</code>；第一次反向建立梯度，后面的反向继续相加。公式中的零矩阵表示数学起点，不表示程序一定先分配零矩阵。这个累加行为由 PyTorch 的 <code>backward()</code> 完成。[PyTorch 2.2 backward 文档](https://docs.pytorch.org/docs/2.2/generated/torch.Tensor.backward.html)

### 4. 为什么 loss 要除以 K：两层平均

用 u 表示序列内的预测位置。先对一个微批次内部的 64 个位置求平均：

$$
L_i=\frac{1}{BT}\sum_{b=1}^{B}\sum_{u=1}^{T}\ell_{i,b,u}
=\frac{1}{64}\sum_{b,u}\ell_{i,b,u}.
$$

四个微批次都有 64 个有效位置，所以整个有效 batch 的平均损失为：

$$
L_{\mathrm{effective}}=\frac{L_1+L_2+L_3+L_4}{4}
=\frac{1}{256}\sum_{i=1}^{4}\sum_{b,u}\ell_{i,b,u}.
$$

令 gᵢ 表示 Lᵢ 对参数 W 的梯度。在同一份 W 下，求导可以分配到求和的各项：

$$
G=\frac{\partial L_{\mathrm{effective}}}{\partial W}
=\frac14\sum_{i=1}^{4}\frac{\partial L_i}{\partial W}
=\frac{g_1+g_2+g_3+g_4}{4}.
$$

所以每轮调用 <code>(loss / 4).backward()</code>，本轮贡献是 gᵢ/4，PyTorch 再把它加到旧 <code>.grad</code> 上。无需先保存四个完整计算图再统一求导。

$$
\Delta_i=\frac{\partial(L_i/4)}{\partial W}=\frac{g_i}{4},
\qquad G_0=0,\qquad G_i=G_{i-1}+\Delta_i.
$$

在一个微批次内部，同一参数来自各预测位置、各使用路径的梯度也按链式法则汇总。例如共享的 Token Embedding/lm\_head 参数会收到两条使用路径的贡献。实际 autograd 批量计算，不必显式保存 64 张单位置梯度矩阵。

如果没有除以 K，最终得到的是四个微批次梯度之和，本例比平均梯度大四倍。对 AdamW，不能进一步断言参数更新恰好大四倍，因为 m/v、ε 和梯度裁剪都会影响最终步幅。这里除以 K 的依据是匹配<b>平均损失的梯度</b>。[PyTorch 梯度累积示例](https://docs.pytorch.org/docs/2.2/notes/amp_examples.html#gradient-accumulation)

### 5. 四轮实际怎样累加：与图中数字一致

观察同一张 W 的梯度。下面用 <code>[[第一行], [第二行]]</code> 表示 2×2 矩阵；W 在四轮之间始终不变。

<table fit-page-width="true" header-row="true">
<tr>
<td>
微批次
</td>
<td>
未经跨批缩放的 gᵢ
</td>
<td>
本轮贡献 Δᵢ = gᵢ/4
</td>
<td>
加入后保存的 W.grad
</td>
</tr>
<tr>
<td>
1
</td>
<td>
<code>[[2,-2],[4,0]]</code>
</td>
<td>
<code>[[0.5,-0.5],[1,0]]</code>
</td>
<td>
<code>[[0.5,-0.5],[1,0]]</code>
</td>
</tr>
<tr>
<td>
2
</td>
<td>
<code>[[4,2],[0,4]]</code>
</td>
<td>
<code>[[1,0.5],[0,1]]</code>
</td>
<td>
<code>[[1.5,0],[1,1]]</code>
</td>
</tr>
<tr>
<td>
3
</td>
<td>
<code>[[6,-2],[4,4]]</code>
</td>
<td>
<code>[[1.5,-0.5],[1,1]]</code>
</td>
<td>
<code>[[3,-0.5],[2,2]]</code>
</td>
</tr>
<tr>
<td>
4
</td>
<td>
<code>[[8,2],[0,8]]</code>
</td>
<td>
<code>[[2,0.5],[0,2]]</code>
</td>
<td>
<code>[[5,0],[2,4]]</code>
</td>
</tr>
</table>

例如第二轮，读取第一轮的累计值，逐元素相加：

$$
G_2=
\begin{bmatrix}0.5&-0.5\\1&0\end{bmatrix}
+\begin{bmatrix}1&0.5\\0&1\end{bmatrix}
=\begin{bmatrix}1.5&0\\1&1\end{bmatrix}.
$$

只观察左上角，四轮就是 <code>0 + 0.5 + 1 + 1.5 + 2 = 5</code>，等于原始梯度 <code>(2+4+6+8)/4</code>。其他位置按同样的规则相加，互不串位；梯度矩阵的形状不会改变。

### 6. 接回项目函数：两层循环及每一步的位置

下面是<b>接线示意</b>：复用项目已有 <code>forward_and_loss</code> 的调用方式，但旧训练入口目前仍是每批一次更新。示例假设 model/optimizer/loader 已准备好、模型处于训练模式、<code>get_lr</code> 使用前面介绍的三段规则，并且每个累积窗口恰好有 K 个等大有效微批次。

```py
K = 4
# optimizer 在两层循环外创建一次，m/v 与更新次数持续保留。

for step in range(max_updates):  # 外层：每轮只做一次参数更新
    optimizer.zero_grad(set_to_none=True)  # 每个累积窗口开始时清一次
    loss_for_log = torch.zeros((), device=device)

    for micro_step in range(K):  # 内层：取 K 个新微批次
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)

        # 参数没有更新；每批建立自己的前向计算图。
        # 已有函数内部按 precision 控制 autocast，并求平均交叉熵。
        logits, loss = forward_and_loss(model, x, y, device, precision)
        scaled_loss = loss / K

        # 仅记录数值，防止日志变量连住之前微批次的计算图。
        loss_for_log += scaled_loss.detach()

        # 原梯度 + 本批梯度/K；只累加 .grad，不改变参数或 m/v。
        scaled_loss.backward()
        del logits, loss, scaled_loss  # 不保留本批输出的额外引用

    # 满 K 次之后，对所有参数的最终累计梯度统一裁剪一次。
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # step 是参数更新编号，不是 micro_step；调度一次。
    lr = get_lr(step)
    for group in optimizer.param_groups:
        group["lr"] = lr

    # 读取累计并裁剪后的 .grad，更新 m/v、偏差修正计数和参数一次。
    optimizer.step()
    # 下一轮用新参数；只有回到外层开头才清 .grad。
```

<code>loss_for_log</code> 累计的是脱离计算图后的损失数值，仅供显示；真正驱动训练的是每次 <code>scaled_loss.backward()</code> 写入的 <code>.grad</code>。手动 <code>del</code> 是释放额外引用，梯度累积不依靠它实现；普通反向已经释放本次求导保存的中间激活，无需 <code>retain_graph=True</code>。

在这一轮有效 batch 中：取数、前向、loss 和 backward 各 4 次；清梯度、统一裁剪、学习率设置、AdamW step 各 1 次。第一次 AdamW step 会为需要更新的参数建立状态，随后保留历史；本轮 <code>.grad is None</code> 的参数会被 AdamW 跳过。

### 7. 哪些状态会变，哪些状态要保留

<table fit-page-width="true" header-row="true">
<tr>
<td>
对象
</td>
<td>
四个微批次之间
</td>
<td>
外层更新后
</td>
</tr>
<tr>
<td>
模型参数 θ，包括 Embedding
</td>
<td>
不变，下一微批次继续使用同一份数值
</td>
<td>
AdamW 改变数值，供下一窗口使用
</td>
</tr>
<tr>
<td>
各参数的 <code>.grad</code>
</td>
<td>
不清空，逐次累加
</td>
<td>
step 不自动清除；下一窗口开始再清
</td>
</tr>
<tr>
<td>
优化器 m/v
</td>
<td>
不更新
</td>
<td>
根据最终梯度递推一次，继续保留
</td>
</tr>
<tr>
<td>
AdamW 偏差修正计数 t
</td>
<td>
不增加
</td>
<td>
对参与本次更新的参数增加一次
</td>
</tr>
<tr>
<td>
前向图与中间激活
</td>
<td>
每批重新产生，普通反向后释放所保存的激活
</td>
<td>
不需要保存四批的完整计算图
</td>
</tr>
<tr>
<td>
学习率调度编号 step
</td>
<td>
四批属于同一个外层编号
</td>
<td>
下一次参数更新才进入下一个编号
</td>
</tr>
</table>

前面的 m/v 是<b>跨参数更新的历史统计</b>；这里 <code>.grad</code> 是<b>一次更新内的微批次贡献之和</b>，二者存储位置、生命周期和计算规则不同。AdamW 的 v 接收最终梯度的平方，不是逐微批次梯度平方的平均。

如果在每个微批次里调用 <code>zero_grad()</code>，前面几批的贡献会丢失；如果每批调用 <code>optimizer.step()</code>，参数与 m/v 会中途变化，这就成了四次小 batch 更新。两者都不等于本例的一次有效大 batch 更新。

<b>裁剪也必须放在累计之后。</b> 先分别裁剪四份梯度再相加，通常不等于先求和再裁剪。整体裁剪的长度从所有参数的梯度计算，不能只看图中那一张 W；各参数共同使用一个缩放系数。fused AdamW 改变底层执行方式，不改变这里“一次更新”的时机。

### 8. 节省了什么，以及等价成立的条件

主要节省的是<b>中间激活的峰值显存</b>：处理完当前微批次并反向后，就可以释放它为反向保存的激活，再处理下一批。参数、累计梯度与 AdamW 状态仍需存储，总显存不会严格降为四分之一。总共仍需完成四份数据的前向和反向，也不保证比一次可容纳的大 batch 更快。

<code>loss / K</code> 要求各微批次参与平均的有效位置数相同。若存在不同数量的 padding、忽略标签或最后一个短批，令第 i 批有效位置数为 nᵢ，N 为本窗口总数，应按有效位置数加权：

$$
L_{\mathrm{effective}}=\sum_i\frac{n_i}{N}L_i,
\qquad N=\sum_i n_i.
$$

另一种实现是累计逐位置损失和的梯度，最后在裁剪前统一除以 N。最后不足 K 批的窗口也要用实际总数，不能仍机械地除以设定 K。

可沿 batch 维独立计算、保持同一参数、损失定义及随机选择一致时，梯度累积与整批平均目标在数学上对应。实际运行的 Dropout 随机数、浮点求和顺序和计算内核可能造成数值差异；含跨样本统计的计算还需单独分析。本节沿用 FP32 原参数及已有 BF16 autocast 方案，loss 除以 K 是平均化，不是 FP16 GradScaler 的数值放大。

图中数值复算（项目内 <code>artifacts/gradient-accumulation-diagram-check.json</code>）使用 PyTorch 2.2.2、CPU/float64、2×2 参数和二次损失：一次处理 8 条样本与分四次每次 2 条样本，累计梯度、统一裁剪后的梯度、一次 AdamW 后的参数最大差均为 0；四次反向之间参数和优化器状态不变，更新次数为 1；清梯度后参数与 m/v 保留。这只验证教学例子，不是完整 GPT 训练或 CUDA 性能结论。

## 十八、复习时最容易混淆的对象

| 容易混淆的说法 | 应该怎样理解 |
| --- | --- |
| AdamW 修正随机初始化的模型参数 | 修正的是 m/v 从 0 开始的统计偏差 |
| 所有参数使用同一个 m/v | 同一算法；每个元素拥有各自的历史记录 |
| 更新的是 Q/K/V | 更新 W\_Q/W\_K/W\_V 等参数；Q/K/V 之后重新计算 |
| 归一化后梯度结果是 1 | 归一化的是历史梯度系数，总和为 1 |
| m̂\_t 就是当前梯度 | 它是归一化的历史加权平均 |
| E\[g\] 是本次梯度 | 它是可能梯度按概率加权的平均 |
| (1−βᵗ)μ 是梯度期望 | 它是未修正历史记录 m\_t 的期望 |
| 优化器必须知道 μ 才能修正 | μ 用于理论分析；实际程序无需输入 μ |
| v 是方差或二阶导数 | v 记录梯度平方的指数移动平均 |
| 用 m̂/v̂ 覆盖 m/v，再递推 | 原始记录继续递推，修正量用于当前更新 |
| 每步清梯度会丢掉 m/v | zero\_grad 不清优化器历史状态 |
| 使用 AdamW 就保证每步 loss 下降 | batch 波动、步幅、历史方向等都会影响结果 |

复习题：用 β=0.9、g₁=2、g₂=4，算出 m₁、m₂、m̂₂；说明新系数为何加起来是 1。再说出“每次梯度固定 0.2”和“梯度为 0.1/0.3 各半”在 E\[g²\] 上的区别。

<toggle>
展开核对答案

m₁=0.2；m₂=0.9×0.2+0.1×4=0.58；m̂₂=0.58/0.19≈3.052632。新系数为 9/19 和 10/19，总和为 1。

固定 0.2 时，E\[g²\]=0.04。0.1/0.3 各半时，E\[g\]=0.2，但 E\[g²\]=0.05。两种情形的均值相同，平方的均值不同。
</toggle>

梯度累积复习题：B=2、T=32、K=4 时，一次更新用了多少个有效位置？四次 backward 会改变 Embedding 吗？每次都清梯度会留下哪一批的贡献？AdamW 的偏差修正次数增加几次？

<toggle>
展开梯度累积答案

一次更新使用 256 个有效位置；四次 backward 只累计梯度，Embedding 等参数直到最后一次 optimizer.step 才改变。若每个微批次开始都清梯度，最后只留下第四批的缩放梯度。参与更新的参数，其 AdamW 更新计数增加一次；m/v 也只递推一次。
</toggle>

## 十九、来源、复算与当前学习边界

- 课程位置：[Karpathy 原视频约 2:14:55 起](https://www.youtube.com/watch?v=l8pRSuU81PU&t=8095s)；此前定位使用[非官方自动字幕副本](https://github.com/nolimitkun/ai-kol-wiki/blob/main/sources/karpathy/20240609-l8pRSuU81PU/transcript.md)。
- 作者代码：[105f117，AdamW 参数与梯度裁剪](https://github.com/karpathy/build-nanogpt/commit/105f117)、[90e5d15，学习率预热与余弦调度](https://github.com/karpathy/build-nanogpt/commit/90e5d15)。已核对视频相应自动字幕；没有把本地改编当成课程原始证据。
- 原理：[Adam](https://arxiv.org/abs/1412.6980)、[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)。
- 版本实现：[PyTorch 2.2.2 AdamW](https://github.com/pytorch/pytorch/blob/v2.2.2/torch/optim/adamw.py)、[梯度裁剪](https://github.com/pytorch/pytorch/blob/v2.2.2/torch/nn/utils/clip_grad.py)。
- 项目对应：优化器创建（项目内 <code>train_compile.py</code>）、反向与更新（项目内 <code>train_mixed_precision.py</code>）。
- 本篇数值复算（项目内 <code>artifacts/adamw-bias-correction-check.json</code>）：两个标量更新与 PyTorch 对照；恒定梯度三步；随机例子分别枚举 2、4、8 条序列核验一阶/二阶矩期望；核验归一化加权平均。都是教学数值检查，未重跑完整 GPT 训练。
- 补充数值复算（项目内 <code>artifacts/adamw-clipping-schedule-check.json</code>）：裁剪阈值与方向、实际 PyTorch 数值、预热/余弦/保持关键点和原课循环边界。两张曲线图及重绘脚本在 <code>docs/diagrams/</code>。
- 梯度累积：[原视频约 2:35—2:46](https://www.youtube.com/watch?v=l8pRSuU81PU&t=9313s)、[作者历史提交 01be6b3](https://github.com/karpathy/build-nanogpt/commit/01be6b3)、[PyTorch 2.2 backward](https://docs.pytorch.org/docs/2.2/generated/torch.Tensor.backward.html)、[AMP 文档中的梯度累积](https://docs.pytorch.org/docs/2.2/notes/amp_examples.html#gradient-accumulation)。视频段落沿用此前自动字幕定位，没有声称直接观看。
- 梯度累积配套：全流程图与阅读说明（项目内 <code>docs/diagrams/gpt2-gradient-accumulation-flow.md</code>）、重绘和数值验证脚本（项目内 <code>docs/diagrams/render_gradient_accumulation_flow.py</code>）、四轮矩阵与一次 AdamW 复算（项目内 <code>artifacts/gradient-accumulation-diagram-check.json</code>）。PNG/SVG 已保存在项目，本篇嵌入 PNG。

本篇现已覆盖 AdamW 名称与更新机制、初始化偏差修正、梯度期望、梯度裁剪、三段学习率调度，以及梯度累积的公式、状态、矩阵例子和执行顺序。权重衰减分组、fused AdamW 与 optimizer.step 已在讨论中涉及；本次续写重点是梯度累积。掌握程度保留“理解中”，旧训练入口尚未整合裁剪、调度和梯度累积，完整工程实践仍待补；不把笔记扩充记成整个 2.3 已完成。完整段落计数仍为单元 2 的 1/5、阶段二 5/32。
