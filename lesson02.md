# 阶段二 · 单元 2 · 2.1 · 知识块 2：把模型搭起来，并验证前向计算

本次目标：让自己实现的 GPT-2 使用已有的预训练参数，对相同 Token ID 算出与参考模型接近的 logits。先验证模型实现，之后再学习训练循环。

课程来源：[Karpathy — Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)、[build-nanogpt](https://github.com/karpathy/build-nanogpt)。本项目是中文教学实现，不是视频代码的逐字抄录。

## 你这次需要做的三件事

### 1. 先运行对照实验

先按 README 完成运行准备。在 VS Code 中打开本仓库和 `compare_forward.py`，选择仓库内的解释器 `.venv/bin/python`，点击“运行 Python 文件”。也可以在仓库根目录执行：

```bash
.venv/bin/python compare_forward.py
```

本次直接使用准备好的 Token ID，离线加载本机已有的 GPT-2 权重。运行后看三个结果：

- 第一组输入的 `embedding → block_01…block_12 → final_norm → logits` 逐层比较是否全部 PASS。
- 三组输入各自的 logits 误差是否通过比较；最高分 ID 用于辅助观察，完整 logits 的比较才是主要判据。
- 改变最后一个输入 Token 后，较早位置是否保持不变。这是在验证因果遮罩。

程序自动保存 `artifacts/lesson02-report.json`；每次运行会更新这份实验报告。两边都运行在 CPU / FP32，关闭 Dropout，并关闭本次计算的梯度记录。程序不训练、不采样生成，也不修改下载的预训练文件。

### 2. 只跟踪一条主数据流

结合 [代码调用与计算逻辑对照图](docs/diagrams/gpt2-forward-call-map.md) 阅读：它区分创建模型与触发计算，并逐层展示调用、返回、两次残差及 Tensor 形状。

打开 `gpt2_model.py`，先看 `GPT.forward`，再看 `Block.forward`。本轮先掌握模块边界和输入输出，`load_hf_weights`、初始化和底层矩阵计算可以在对应知识块继续分析。

| 位置 | 本例输出形状 | 为什么这样 |
|---|---|---|
| 输入 ID | `[2,4]` | 2 段序列，各 4 个 Token |
| Token 向量 | `[2,4,768]` | 每个 ID 查出一个 768 维向量 |
| 位置向量 | `[4,768]` | 位置 0～3 查出 4 个向量，沿 batch 广播 |
| 两者相加 | `[2,4,768]` | 同维度相加，保持每个位置的向量宽度 |
| 每个 Block 输出 | `[2,4,768]` | 注意力和 MLP 更新内容，保持接口形状 |
| 最终 LayerNorm | `[2,4,768]` | 沿每个位置的特征维归一化 |
| lm_head | `[2,4,50257]` | 每个位置对所有词表 ID 打分 |

用三条式子连接旧知识：

```text
X0 = TokenEmbedding(ids) + PositionEmbedding(positions)
R  = X + Attention(LN1(X))
Y  = R + MLP(LN2(R))
```

后两条构成一个 Block。12 个 Block 的结构相同、参数各自独立；这一层的 Y 是下一层的 X。最后再做 `LN_f` 和 `lm_head`。

### 3. 做一个预测，再看运行结果

对第二组 ID，只改最后一个数字，前 3 个位置的 logits 会不会变化？先解释理由，再对照程序输出的“因果遮罩检查”。

这里的“前 3 个位置”指输出张量的时间维。每个位置仍然有完整的 50,257 个词表分数。

## 这次验证了什么

**课程内容 / 源码事实**：GPT-2 的前向组件、Pre-LN 两次残差、QKV、多头、GELU、共享 Embedding；加载 Hugging Face 的 Conv1D 投影权重时，需要转置为 Linear 的存储方向。

**独立教学安排**：额外记录每层输出；使用三组短输入，加入改变末尾 Token 的因果性检查。显式 Attention 便于看见 mask 和矩阵乘法；没有采用最终课程代码的 FlashAttention。保留配置中的 Dropout，评估时关闭。

**实际运行**：程序的报告记录环境、输入 ID、每层最大/平均绝对误差、权重加载数量和因果性结果。单个元素的判据为 `abs(自己 - 参考) <= atol + rtol * abs(参考)`，本实验两项容差均为 `1e-4`。

测试通过说明这些案例的前向计算与参考实现相符；本块还没有验证训练梯度、优化器更新、padding 或 KV cache。训练和生成仍按 2.1 后续知识块推进。

## 本轮自检

1. 12 个 Block 都输出 768 维，为什么经过每一层仍然有意义？
2. `[2,4,50257]` 中的 4 是什么？它代表未来 4 个新 Token 吗？
3. 为什么同一份权重、同一输入，要先关闭 Dropout 再比较？
4. 如果 embedding 已一致，但 block_01 首次不一致，应该优先检查哪里？

本轮完成标准：能解释形状变化、两次残差的输入，以及对齐实验的目的。暂时无需逐行记住 Python 写法。
