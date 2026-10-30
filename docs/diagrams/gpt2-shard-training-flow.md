# 分片训练：代码全流程图

[PNG 原图](gpt2-shard-training-flow.png) · [可缩放 SVG](gpt2-shard-training-flow.svg) · [配套讲义](../../lesson12-shard-training.md)

![分片读取、微批次累积、参数更新、验证与生成](gpt2-shard-training-flow.png)

## 按什么顺序读

1. 顶部 `main()`：选择分片目录，创建独立的 train / val 加载器，随机初始化一次模型与 AdamW，再进入主循环。
2. 左侧先初始验证，再清梯度。内层循环每次读一个微批次，搬运 x/y，前向求平均 loss，执行 `(loss/K).backward()`。重复 K 次后才裁剪、设置学习率、调用一次 `optimizer.step()`。
3. 右侧 A 展开 `next_batch()`：当前分片在 CPU；取 B×T+1 个 ID，错开一位形成 x/y，推进下一次游标并判断换片，最后返回本次切出的 x/y。预载下一片不会改写已切出的 batch。
4. 更新完成后分别判断验证与生成。右侧 B 只重置验证加载器，评估后恢复模式；右侧 C 使用固定提示词和当前参数，不调用数据加载器，逐 Token 生成，恢复模式后返回主循环。
5. 下一次更新先清梯度，再从训练游标继续。底部记录第 2 次更新跨两个文件、观察后第 3 次仍从 `000001:6` 继续。

虚线是同步函数的调用 / 返回，不代表并发。左侧初始验证也调用 B，只省略重复的跨图连线。B 内部读取的是 val 对象，使用与 A 相同的类方法。

## 代码定位

| 图中环节 | 源码 |
|---|---|
| 主入口 | [train_shards.py L237](../../train_shards.py#L237) |
| 创建两个加载器 | [train_shards.py L91](../../train_shards.py#L91) |
| 更新 / 微批次两层循环 | [train_shards.py L145](../../train_shards.py#L145) |
| 读取、错位、推进、换片 | [load_shards.py L84](../../load_shards.py#L84) |
| 梯度累积 | [train_shards.py L192](../../train_shards.py#L192) |
| 参数更新 | [train_shards.py L205](../../train_shards.py#L205) |
| 独立验证循环 | [train_shards.py L108](../../train_shards.py#L108) |
| 生成循环 | [train_generation.py L46](../../train_generation.py#L46) |
| 汇总并保存报告 | [train_shards.py L349](../../train_shards.py#L349) |

行号由绘图脚本从源码提取；M/T/S/G 标记对应源码注释。A/B/C 是图中三个展开区域，不是额外的函数。

## 本图的具体计算

- 每个微批次 `x/y=[2,3]`：两行序列，每行三个预测位置，共 6 对输入 / 目标。每个位置用因果上下文预测对应 y；不是只有每行最后一格有 loss。
- 为获得这 6 对监督信号，需要连续 7 个 ID：前 6 个组成 x，后 6 个组成 y。读 7 个 ID 不等于有 7 个预测位置。
- `loss` 已是 6 个位置的平均损失；每次更新要求 12 个位置，因此 `K=12/(2×3×1)=2`。两个微批次都用同一组参数 θ，梯度最终是 `(g₁+g₂)/2`。
- `.grad` 由 `backward()` 累加；代码的 `loss_accum += scaled_loss.detach()` 只是记录 loss 指标，不能替代梯度累积。
- 当前分片长度 16，从 p=0 取完后下一起点为 6，`6+6+1≤16`，留在当前片；从 p=6 取完后下一起点为 12，`12+6+1>16`，预载下一片。末尾不足一批的部分不跨文件拼接。
- 长度 8 的第二片只够从 p=0 取一次，随后绕回首片。一次参数更新的两个微批次可以跨文件，但单个微批次始终来自一个文件。换片不清梯度、不重建优化器、不重置调度。
- 本次验证 / 生成都在更新 2、3 次后触发；两项判断各自独立。验证数据是 `val_000000.npy` 的起点 0、6；生成从固定 8 个提示词 ID 开始，4 条序列各新增 24 个 ID。
- 验证只测量指标，本入口不会根据验证 loss 自动改学习率或早停。生成文字只供观察，不是下一批训练样本。JSON 是实验报告，不是可恢复的权重检查点。

## 范围与复现

数字来自 [CPU tiny 报告](../../artifacts/lesson12-shard-training-cpu.json)：3 次更新、36 个训练预测位置。模型为随机初始化的 2 层、64 维教学配置；默认不加 `--model-size tiny` 时入口创建完整 GPT-2。图中两句 demo 不代表 FineWeb 语料，短跑不衡量语言质量。

多卡只作为旁注：每进程的起点加 `rank×B×T`，全局游标步长为 `B×T×R`；每次更新覆盖 `K×B×T×R` 个位置。前 K−1 次 `no_sync()` 同时包住前向与反向，最后反向同步参数梯度；验证中的 all-reduce 只汇总指标。主图数值是 R=1，不能当作 CUDA/NCCL 实测。

原课来源和本地改编边界保留在[讲义第 0 节](../../lesson12-shard-training.md)。本次只绘图，不重跑训练，不修改旧训练代码。

重建（项目已配置的 Python 环境）：

```bash
python docs/diagrams/render_shard_training_flow.py --png
```

SVG / Markdown 只需要标准库及项目绘图组件，PNG 导出复用 macOS AppKit。脚本核对源码定位、分片边界、真实读取顺序、验证 / 生成触发、观察前后的游标与生成长度；结果保存在 [制图核对记录](../../artifacts/lesson12-shard-training-diagram-check.json)。
