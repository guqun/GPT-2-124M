# FlashAttention：分块计算，累计输出

对应阶段二 · 单元 2 · 2.2 训练提速，承接 `torch.compile` 后的 FlashAttention 原理讨论。

![普通 Attention 与 FlashAttention 对照流程](gpt2-flash-attention-flow.png)

[打开可缩放 SVG](gpt2-flash-attention-flow.svg) · [PNG 图片](gpt2-flash-attention-flow.png)

## 阅读顺序

1. **顶部定位：** LN1 和 Q/K/V 投影之后，优化的是注意力的匹配、归一化和汇总 V；随后仍进行多头拼接、输出投影和残差相加。
2. **左侧普通实现：** 写入完整分数 S，读取 S 并生成权重 A，再读取 A 汇总 V。大中间矩阵引入显存占用与读写。
3. **右侧 FlashAttention：** 固定一块 Q，遍历 K/V 块，当前分数块随即参与累计；跨块保留统计量与输出累计值。图用这种循环便于讲解，具体内核可采用不同循环顺序、并行与调度。
4. **数字例子：** 分母累计为 `0 → 3 → 10`，分子累计为 `0 → 50 → 300`，最终 `300 / 10 = 30`。普通完整权重计算同样得到 30。
5. **底部补充：** 实际实现加入最大值修正保证数值稳定；反向传播根据已保存的数据与统计量局部重算权重。

## 数字例子的含义

这是自拟教学例子，只看一个查询、V 的一个维度，四个上下文位置均可见，Dropout 为 0。

- `s_j = q · k_j / sqrt(d)` 为缩放后的分数；为便于计算，假设 `exp(s) = [1, 2, 3, 4]`。
- `V` 的这一维为 `[10, 20, 30, 40]`。
- 普通权重为 `[0.1, 0.2, 0.3, 0.4]`，输出为 `1 + 4 + 9 + 16 = 30`。
- 简化累计使用 `L = Σ exp(s_j)`、`U = Σ exp(s_j) v_j`，最终 `o = U / L`。
- 实际 V 有 d 维，U 就是 d 维向量，所有维度共用当前查询的同一个分母 L。处理多个查询时，各查询各自保留状态。

主例省略最大值修正，用于说明“分子分母可分块相加”，不是完整内核伪代码。稳定形式维护 `(m, L, U)`，新块到达时：

```text
m_new = max(m, 新块中允许位置的最大分数)
alpha = exp(m - m_new)
L_new = alpha * L + sum(exp(s_j - m_new))
U_new = alpha * U + sum(exp(s_j - m_new) * v_j)
```

上面求和只遍历新块；最终 `o = U_new / L_new`。因果遮罩禁止的位置贡献为零。实现还需正确处理一个块全部被遮挡的情况；图没有展开此分支。

## 适用范围

- HBM 和片上存储都是 GPU 内部的存储层级。图中往返读写不是 CPU 与 GPU 之间的逐步传输。
- 普通矩阵乘法也会分块；FlashAttention 将注意力的多个操作按块组织与融合，避免把完整 S/A 矩阵存入 HBM。原始算法可能读写局部输出及统计量；图没有声称所有输入都只读一次或整个计算只有一次显存访问。
- 数学目标为相同的密集注意力。主要计算量仍随 T² 增长，减少的是大中间结果的存储和显存读写；浮点执行顺序不同允许小的数值差异。
- 本图为算法教学图，没有表示本机已经运行 CUDA FlashAttention，也没有提供性能加速倍数。
- 本图未修改模型或训练入口。当前显式注意力可对照 [gpt2_model.py](../../gpt2_model.py) 的 `CausalSelfAttention.forward`。

## 来源与复算

- [Karpathy 原视频约 1:59:46—2:06:54](https://www.youtube.com/watch?v=l8pRSuU81PU&t=7186s)：本次通过[非官方自动字幕副本](https://github.com/nolimitkun/ai-kol-wiki/blob/main/sources/karpathy/20240609-l8pRSuU81PU/transcript.md)定位，再交叉核对作者历史代码；未直接观看视频画面。
- [作者历史提交 7ee630c](https://github.com/karpathy/build-nanogpt/commit/7ee630c)：用 `scaled_dot_product_attention` 替换显式注意力计算。
- [FlashAttention 论文 §3.1](https://arxiv.org/html/2205.14135v2#S3.SS1)：分块、稳定在线归一化、融合与反向重算。主图的数值例子为独立教学推导。
- [PyTorch 2.2 SDPA 文档](https://docs.pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html)：接口会根据输入和后端选择实现，不能仅凭接口调用成功认定使用了 FlashAttention。

生成脚本会独立复算普通完整权重、简化累计、带最大值修正的累计，确认输出均为 30；无需训练依赖。

```bash
/Users/guqun/micrograd/venv/bin/python docs/diagrams/render_flash_attention_flow.py --png
```

不加 `--png` 时只生成 SVG；PNG 导出沿用本项目的 macOS AppKit 方式。
