# GPT-2：代码调用与计算逻辑对照

![GPT-2 前向计算：代码调用与计算逻辑对照](gpt2-forward-call-map.png)

本图对应阶段二 · 单元 2 · 2.1 · 知识块 2 的独立 GPT-2 实现，依据 2026-10-16 的本地代码绘制。尺寸采用 GPT-2 124M：12 个 Block、768 维特征、12 个注意力头、每头 64 维、50,257 个词表项。当前对照实验使用已加载的预训练参数、CPU / FP32 和 eval 模式。

## 按这个顺序读图

1. **准备阶段**：`GPT(config)` 调用构造过程，建立各层；`load_hf_weights(...)` 写入已有参数；`eval()` 切换评估模式。
2. **计算入口**：`own(input_ids, trace=own_trace)` 经 `nn.Module` 的调用机制进入 `GPT.forward()`。
3. **整体流程**：位置编号与向量查表 → Embedding 相加 → 12 个 Block → 最终 LayerNorm → lm_head → 返回 logits。
4. **Block 内部**：先 LN1、Attention、第一次残差，再 LN2、MLP、第二次残差。Attention 和 MLP 都返回到 Block；两次残差分别加回 `x` 和 `r`。
5. **子模块展开**：Attention 展示 QKV、多头、遮罩、softmax、V 汇总和输出投影；MLP 展示扩展、GELU、压回和 Dropout。
6. **返回外层**：一个 Block 的结果返回到 GPT，成为下一个 Block 的输入；最终 logits 返回给对照程序。

## 对应源码

| 在图里查什么 | 对应文件与方法 |
|---|---|
| 构建、加载、调用自己的模型 | [compare_forward.py](../../compare_forward.py) 的 `main()` |
| 完整前向计算 | [gpt2_model.py](../../gpt2_model.py) 的 `GPT.forward()` |
| 每个 Block 的六步 | 同文件的 `Block.forward()` |
| 多头注意力 | 同文件的 `CausalSelfAttention.forward()` |
| 逐位置 MLP | 同文件的 `MLP.forward()` |
| 复制和转置预训练权重 | 同文件的 `GPT.load_hf_weights()` |

图中 Block 的嵌套表达式按执行顺序展开为六步，`attn_input`、`attn_update`、`mlp_input`、`mlp_update` 是阅读用的临时名称，已在源代码注释中对应。实际可执行表达式仍为：

```python
r = x + self.attn(self.ln_1(x))
return r + self.mlp(self.ln_2(r))
```

`self.transformer.h` 是一个 `nn.ModuleList`，保存 12 个独立 Block 对象；`h[0]` 才是其中第一个 Block。`start=1` 只决定记录名称的编号，不改变从第一个 Block 开始的执行顺序。

位置编号 `positions = [0, 1, ..., T-1]` 是确定生成的；`wpe(positions)` 才是根据编号查出本次加载的预训练位置向量。图中所有 `[B,T,768]` 均表示“B 条序列、每条 T 个位置、每个位置 768 维”。

`trace` 记录中间 Tensor，供与 Hugging Face 比较；它不改变主计算顺序。本图覆盖前向计算，损失、反向传播、优化器更新和采样生成属于后续流程。GELU 使用 `approximate="tanh"`；图中 Dropout 在本次 eval 模式下原样通过。

<details>
<summary>版式与维护规则：对照秩序</summary>

空间依靠稳定的对齐关系传达意义。同一横行的两侧相互解释，嵌套区域表达包含关系，返回线表达结果的去向。细致的边距与网格让复杂结构可以沿一个方向逐步阅读。

颜色承担明确的职责：蓝色指向计算与调用，紫色标记局部变换，琥珀色强调旁路相加，绿色表示返回的结果。大面积白底与轻浅底色保留对比，逐项校准的线条和色彩避免视觉噪声。

中文说明与等宽代码各有清晰的字体角色。标题建立层级，表达式保留准确字符，形状标签作为紧邻计算的辅助信息。精心控制字号与行距，让读者放大时仍能辨认每一个调用。

箭头的方向和终点需要反复核对，特别是局部调用结束后的返回关系。更新时优先维护内容的准确性、完整的留白和细致的对齐，不通过增加装饰补充信息；图、说明与实际实现保持一致。

</details>
