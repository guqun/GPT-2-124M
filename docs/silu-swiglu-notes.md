# SiLU 与 SwiGLU：双分支门控如何加工特征

记录日期：2026-11-11。本篇为现代模型结构学习的独立补充，说明 SiLU 的数学计算、SwiGLU 为什么有两条分支，以及它在 Transformer Block 中如何接收和输出数据。数值为教学选值，不是模型实测权重。

## 1. 先定位：SwiGLU 在 Block 的什么位置

注意力从上下文汇总信息；MLP（也称前馈网络 FFN）随后分别加工每个 Token 的特征。一个 Pre-Norm Block 的主要关系是：

~~~text
Block 输入 X
  → 第一次归一化 → 注意力 → 输出投影 O
  → 第一次残差 R = X + O
  → 第二次归一化 Z = Norm₂(R)
  → MLP：Fout = MLP(Z)               ← SwiGLU 位于这里
  → 第二次残差 Y = R + Fout
  → 下一个 Block
~~~

第二次归一化在每个 Block 内部、MLP 之前；所有 Block 结束后还有最终归一化，二者不是同一个环节。这里残差加回的是 R，不是归一化后的 Z，也不是再次加 X。

假设 Z 形状为 [B,T,D]=[2,5,16]，MLP 分别加工 10 条 16 维向量，所有位置使用同一套 MLP 参数。这里不跨 Token 汇总，但输入已经包含前面注意力读到的上下文。

## 2. SiLU 是什么：用输入自身决定缩放系数

SiLU 全称 Sigmoid Linear Unit，公式为：

$$
\sigma(x)=\frac1{1+e^{-x}},\qquad
\operatorname{SiLU}(x)=x\sigma(x)=\frac{x}{1+e^{-x}}
$$

先算 Sigmoid 系数，再乘原数值。标准 SiLU 不含可训练参数；向量中各元素独立计算，形状保持不变。

| x | Sigmoid(x)，近似值 | SiLU(x)，近似值 |
| --- | --- | --- |
| -3 | 0.047426 | -0.142278 |
| -1 | 0.268941 | -0.268941 |
| 0 | 0.5 | 0 |
| 1 | 0.731059 | 0.731059 |
| 2 | 0.880797 | 1.761594 |
| 3 | 0.952574 | 2.857722 |

较大的正数近似保留；很负的数被压到接近 0；零附近平滑变化。介于 0 和 1 的是 Sigmoid 系数，SiLU 输出本身可以为负或大于 1。SiLU 不是 softmax，不跨维求和，不保证输出和为 1，也不是归一化层。

例如向量 [-1,0,2] 经过 SiLU，得到约 [-0.268941,0,1.761594]。这只是逐元素函数计算，不涉及新的矩阵乘法。

## 3. 为什么引入 SiLU

连续两次线性变换可以合并：

$$
(zW_1)W_2=z(W_1W_2)
$$

加入非线性激活后，特征变换会随输入数值变化，增加可表达的关系。例如 SiLU(2) 不等于 2×SiLU(1)。

相比 ReLU 将所有负输入直接置零，SiLU 保留经过压缩的负值，并在零附近平滑变化。其负半轴导数不整段恒为零，但不能由此声称消除了梯度消失或在所有任务中一定更好。

在 SwiGLU 中，两条输入相关分支的相乘本身也具有非线性；不要把所有非线性能力都归因于 SiLU。

## 4. SwiGLU 为什么有两条分支

SwiGLU 是采用 Swish/SiLU 的门控线性单元变体。它的结构选择是：一路生成中间特征，另一路生成调节系数，随后逐元素相乘。

同一个完整输入 z 分别使用两套不同的权重：

$$
a=zW_g,\qquad v=zW_v
$$

$$
g=\operatorname{SiLU}(a),\qquad
u=g\odot v,\qquad
F_{\mathrm{out}}=uW_d
$$

~~~text
                    完整输入 z
                    /       \
                  ×Wg       ×Wv
                   ↓         ↓
                   a         v：特征分支
                   ↓ SiLU    │
                   g         │
                   └── ⊙ ────┘
                       ↓
                       u
                       ↓ ×Wd
                     Fout
~~~

两路都接收完整 z，不是把 z 切成两半，也不是两批不同样本。Wg、Wv 不同，所以同一个输入可以生成不同结果。两条分支来自两个线性投影，不是 SiLU 函数自己产生两路。

对中间维度 j：

$$
u_j=
\underbrace{\operatorname{SiLU}(zW_g)_j}_{\text{当前输入产生的调节系数}}
\underbrace{(zW_v)_j}_{\text{当前输入产生的特征}}
$$

“特征”和“调节”是解释运算的角色，不是人为规定每维的语义；两套投影通过同一个最终损失一起训练。

门控不是 0/1 开关。g 可以为负、为零或大于 1，因此能抑制、放大或改变另一分支特征的符号。

这里小写 v 是 MLP 分支的输出，与注意力中的 V、KV Cache 没有直接的数据复用关系。

## 5. 每一步的输入、参数与输出

采用总览图的教学配置 D=16、中间宽度 F=48，省略偏置，符合所讨论 Llama 风格 MLP 的写法。48 只是教学设定，不是所有 Llama 模型的固定宽度。

| 步骤 | 输入 | 参数 | 计算 | 输出 |
| --- | --- | --- | --- | --- |
| 门控分支投影 | Z [2,5,16] | Wg [16,48] | a=ZWg | a [2,5,48] |
| 特征分支投影 | 同一个 Z | Wv [16,48] | v=ZWv | v [2,5,48] |
| 激活 | a | 无 | g=SiLU(a) | g [2,5,48] |
| 门控相乘 | g、v | 无 | u=g⊙v | u [2,5,48] |
| 输出投影 | u | Wd [48,16] | Fout=uWd | Fout [2,5,16] |
| 第二次残差（MLP 之后） | R、Fout | 无新增参数 | Y=R+Fout | Y [2,5,16] |

⊙ 是对应元素相乘，不是矩阵乘法。扩展维度通过矩阵乘法产生新特征，不是补零或复制。输出投影把中间特征混合回 D 维，便于与同形状 R 相加。

本文使用行向量记法，W 的形状为 [输入维度,输出维度]；PyTorch Linear 的权重存储方向通常为 [输出维度,输入维度]，不要因存储转置把公式误认为另一种运算。

## 6. 一个完整数值例子：同一个 z 如何生成不同分支

为了手算，将维度进一步缩小到 D=F=2；本例不演示扩宽，只演示相同运算。参数全是教学选值。

$$
z=[1,2],\qquad
W_g=\begin{bmatrix}1&0\\0&1\end{bmatrix},\qquad
W_v=\begin{bmatrix}2&0\\0&3\end{bmatrix}
$$

第一步，两路投影：

$$
a=zW_g=[1,2],\qquad v=zW_v=[2,6]
$$

第二步，激活门控分支：

$$
g=\operatorname{SiLU}(a)
=\left[\frac1{1+e^{-1}},\frac2{1+e^{-2}}\right]
\approx[0.731059,1.761594]
$$

第三步，逐元素相乘：

$$
u=g\odot v
\approx[0.731059\times2,\;1.761594\times6]
\approx[1.462117,10.569565]
$$

第四步，取输出投影矩阵：

$$
W_d=\begin{bmatrix}1&1\\1&-1\end{bmatrix}
$$

$$
F_{\mathrm{out}}=uW_d
=[u_0+u_1,\;u_0-u_1]
\approx[12.031682,-9.107448]
$$

这就是 MLP 输出，尚未做残差相加。下一步使用此前保留的 R，计算 Y=R+Fout；不将已经归一化的 z 当作残差输入。以上结果按未舍入中间值计算，显示时四舍五入。

## 7. 与 GPT-2 的 GELU MLP 比较

GPT-2 风格：

$$
F_{\mathrm{out}}
=\operatorname{GELU}(zW_1+b_1)W_2+b_2
$$

Llama 风格 SwiGLU：

$$
F_{\mathrm{out}}
=\left[\operatorname{SiLU}(zW_g)\odot(zW_v)\right]W_d
$$

| 对比 | GPT-2 GELU MLP | 本文 SwiGLU MLP |
| --- | --- | --- |
| 扩展分支 | 一条 | 两条，使用同一个完整输入 |
| 参数矩阵 | W1、W2，共两套 | Wg、Wv、Wd，共三套 |
| 非线性加工 | GELU 作用于一路特征 | SiLU 激活门控分支，再与另一分支相乘 |
| 偏置 | 有 | 所讨论的 Llama 实现没有 |
| 输出维度 | 回到 D | 回到 D |
| 是否跨 Token 聚合 | 否 | 否 |

两路增加了不同投影特征之间的乘法交互，但也改变参数和计算开销。在相同中间宽度 F、不计偏置时，两层 MLP 有 2DF 个权重，SwiGLU 有 3DF 个；实际架构会选择合适的 F。总览图的 64 与 48 不应当作严格等参数的实验对照。

原 GLU 变体论文在其 Transformer 实验中观察到包括 SwiGLU 在内的部分门控变体改善质量；这支持该结构选择，但不意味着只凭公式就能证明它在所有任务中优于 GELU。本文没有做替换训练或性能测量。

## 8. 参数在哪里，学习发生在哪里

Wg、Wv、Wd 是可训练矩阵。同一次前向中，a、v、g、u、Fout 都是根据当前输入计算出的中间结果。SiLU 公式和逐元素乘法本身没有独立可训练参数。

反向传播时，损失通过输出投影、乘法、SiLU 和两路投影传回去，从而更新三套矩阵。门控系数 g 随输入变化，但不是额外维护的一张固定参数表。

## 9. 复习问答

- 两条分支是不是两个输入？不是，同一个完整 z 用两套不同权重投影。
- SiLU 和 SwiGLU 是同一个东西吗？不是。SiLU 是逐元素激活函数；SwiGLU 是包含两路投影、激活和门控相乘的结构。
- 门控 g 必须介于 0 和 1 吗？不必；只有 SiLU 内的 Sigmoid 系数介于 0 和 1。
- MLP 的 v 是注意力的 V 吗？不是，来自不同投影、处在不同环节。
- 门控乘法会改变维度吗？不会，两个同形状张量逐元素相乘。
- Fout 算完，这个 Block 就结束了吗？还要完成 Y=R+Fout。
- “MLP 输出是残差”意味着它是损失吗？不意味着；它是加到主路径表示上的分支输出，loss 在后续训练目标处计算。

## 10. 参考资料

- [PyTorch SiLU 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html)：函数定义及逐元素处理。
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)：SwiGLU 结构及论文实验依据。
- [原始 Llama 3 官方实现](https://github.com/meta-llama/llama3/blob/main/llama/model.py)：FeedForward 使用 w2(silu(w1(x)) * w3(x))；其中 w1 对应本文的门控投影 Wg，w3 对应特征投影 Wv，w2 对应输出投影 Wd。代码矩阵存储遵循框架约定。
- [OpenAI GPT-2 官方实现](https://github.com/openai/gpt-2/blob/master/src/model.py)：两层线性变换与 GELU。

本篇只记录已讨论的补充知识，掌握程度为理解中；模型替换和对照实验另行记录。

相关前置：[LayerNorm/RMSNorm 与注意力笔记](layernorm-rmsnorm-notes.md) · [GPT-2/Llama 3 全流程图](diagrams/gpt2-vs-llama3-forward-flow.svg)
