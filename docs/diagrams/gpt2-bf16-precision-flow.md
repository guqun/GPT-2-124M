# BF16：前向、梯度与参数更新的精度变化

对应阶段二 · 单元 2 · 2.2 · 知识块 9。方案是 **FP32 参数 + BF16 autocast**，源码为 [train_mixed_precision.py](../../train_mixed_precision.py)，讲义为 [lesson09.md](../../lesson09.md)。

![一次训练更新中的精度变化](gpt2-bf16-precision-flow.png)

[打开可缩放的 SVG](gpt2-bf16-precision-flow.svg) · [PNG 图片](gpt2-bf16-precision-flow.png)

## 阅读顺序

从左至右看三个阶段，每一列从上往下读。左列的 loss 连到中列的反向入口；中列的 `W.grad` 连到右列的 AdamW。顶部蓝色虚线表示原始参数供前向取值，底部灰色虚线表示进入下一批。

1. **前向：** x/y 是 int64 ID；查 FP32 Embedding 与位置表。对选择 BF16 的 Linear/矩阵乘法，浮点输入和权重按需转换成 BF16，输出也为 BF16。模型中的 LN、残差等按其输入与运算规则选择精度；本例主要残差流为 FP32。最终 logits 为 BF16，交叉熵在 autocast 内转 FP32，得到 FP32 loss。
2. **反向：** 在 autocast 外调用 `loss.backward()`。梯度沿前向计算图回传，中间同时存在 BF16、FP32 路径。浮点类型转换也有反向规则；经过权重转换节点时，将梯度转换回原参数的 FP32 类型。共享参数的多条使用路径最终累积到同一个 `.grad`。
3. **更新：** AdamW 使用 FP32 的原参数、参数梯度和 m/v，更新 m/v 并把新数值写回原始参数。下一批继续保留这些状态，先清空旧梯度，再进行新的前向。BF16 计算表示在需要时从更新后的原参数生成。

## 图中的“精度”具体指什么

- 色块标签表示 **Tensor 的存储 dtype**。右侧原参数与左侧 `W32` 是同一参数的不同引用位置；没有新增一套独立学习的 BF16 参数。
- 矩阵乘法卡片内的 **FP32 累加**描述硬件内部计算，不代表最终输出 Tensor 一定是 FP32。A100 Tensor Core 的 BF16 模式支持 BF16 输入、FP32 累加；具体后端、内核及相关设置可能影响归约过程，不能将所有设备笼统视为同一种内部实现。
- BF16 转 FP32 能精确表示原来的有限 BF16 数值，但不会恢复已经丢失的低位。新增的小数低位补零，随后 FP32 运算与参数更新可以产生新的低位信息。
- `backward()` 放在 autocast 外，不意味着整个反向统一使用 FP32。图中原参数 `.grad` 的 FP32，与中间梯度的混合类型是两件事。
- 本课使用 AdamW。图中 η 是学习率，λ 是权重衰减，m̂/v̂ 是偏差修正后的统计，ε 是分母中的小常数。

本图为精度主线示意，没有逐节点展开 12 个 Block、注意力中的全部算子或 AdamW 内部公式。没有声称 GPU 已实跑；现有报告来自本机 CPU。

## 核对依据

- [本机完整模型 dtype 报告](../../artifacts/lesson09-cpu-dtypes.json)：Embedding/LN/Block 输出为 FP32；QKV、MLP 线性层与 lm_head 输出为 BF16；loss 为 FP32。
- [本机 3 次训练更新报告](../../artifacts/lesson09-cpu-bf16-smoke.json)：参数、参数梯度、AdamW m/v 为 FP32，logits 为 BF16。
- [PyTorch 2.2.2 autocast 说明](https://github.com/pytorch/pytorch/blob/v2.2.2/torch/amp/autocast_mode.py)：作用范围和反向执行约定。
- [PyTorch 自动转换实现](https://github.com/pytorch/pytorch/blob/v2.2.2/aten/src/ATen/autocast_mode.cpp)及[类型转换的反向规则](https://github.com/pytorch/pytorch/blob/v2.2.2/tools/autograd/derivatives.yaml)：计算表示与原参数的连接。
- [NVIDIA A100 精度说明](https://developer.nvidia.com/blog/accelerating-tensorflow-on-a100-gpus/)：Tensor Core 的 BF16 输入和 FP32 累加。

## 重新生成

在项目目录中运行：

```bash
/Users/guqun/micrograd/venv/bin/python docs/diagrams/render_bf16_precision_flow.py --png
```

脚本读取已有报告核对节点类型，生成 SVG，并通过 macOS AppKit 导出 PNG；不启动训练。不加 `--png` 时只生成 SVG。
