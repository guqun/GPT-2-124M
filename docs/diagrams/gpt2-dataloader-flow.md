# 顺序数据加载器：代码与流程对照

学习位置：**阶段二 · 单元 2「复现 GPT-2」· 2.1 · 知识块 5**。

![顺序数据加载器：代码与流程对照](gpt2-dataloader-flow.png)

[打开可放大的 SVG 原图](gpt2-dataloader-flow.svg)。源代码：[train_dataloader.py](../../train_dataloader.py)，完整讲义：[lesson05.md](../../lesson05.md)。

## 阅读顺序

1. **左列 M1—M3：每次启动只准备一次。** 全文编码；构造加载器并保存位置；未开启预览时，创建随机模型和优化器。`--preview-only` 调用同一个 `next_batch()` 三次，打印后直接退出。
2. **左列 M4：每个训练步调用一次 next_batch。** 蓝线进入右列 L1，绿虚线从 `return x, y` 返回左列，接着清梯度、前向、loss、反传与更新。
3. **右列 L1—L4：本块新增逻辑。** 取 129 个 ID → 得到各含 128 个 ID 的 x/y → 前进 128 → 检查下一批是否装得下。
4. **区分“本次数据”和“下次位置”。** L2 先构造本次 x/y；L4 重置的是下次读取位置；本次返回的仍然是已构造的数据。下一次调用才会从头读取。
5. **更新后回到 M4，不回到 M2/M3。** 模型、优化器、加载器都继续沿用。达到训练步数才保存报告。

图中蓝色为 main 主流程，绿色为加载器，虚线为返回；跨线处白色间隙表示没有连接。图展开正常训练路径，参数、文件或有限性检查失败时会退出。

## 图与源码定位

图中 M/L 编号来自源码注释；第几行由绘图脚本自动提取。源码增删后可重新生成；阅读时优先搜索稳定编号。

| 编号 | 作用 | 源码 |
| --- | --- | --- |
| `M1` | 选项、检查、读取全文与编码 | [第 71 行](../../train_dataloader.py#L71) |
| `M2` | 只创建一次加载器，保存整段 tokens 和位置 | [第 109 行](../../train_dataloader.py#L109) |
| `M3` | 只创建一次随机模型和 AdamW | [第 130 行](../../train_dataloader.py#L130) |
| `M4` | 逐步获取下一批并训练，更新后回到循环入口 | [第 139 行](../../train_dataloader.py#L139) |
| `L1` | 从当前位置取 B*T+1 个 ID | [第 55 行](../../train_dataloader.py#L55) |
| `L2` | 错开一位并排成 x/y [B,T] | [第 58 行](../../train_dataloader.py#L58) |
| `L3` | 下次位置前进 B*T | [第 61 行](../../train_dataloader.py#L61) |
| `L4` | 判断下一批是否装得下；不足时重置下次位置 | [第 63 行](../../train_dataloader.py#L63) |
| — | 加载器构造函数 | [第 42 行](../../train_dataloader.py#L42) |
| — | 只预览前三批的分支 | [第 121 行](../../train_dataloader.py#L121) |
| — | 训练循环调用 next_batch | [第 142 行](../../train_dataloader.py#L142) |
| — | 清空梯度与前向 | [第 143 行](../../train_dataloader.py#L143) |
| — | 全位置 loss 与有限性检查 | [第 145 行](../../train_dataloader.py#L145) |
| — | 反传和更新 | [第 148 行](../../train_dataloader.py#L148) |
| — | 记录本步更新前的 loss | [第 152 行](../../train_dataloader.py#L152) |
| — | 返回已经构造的 x/y | [第 67 行](../../train_dataloader.py#L67) |
| — | 循环后保存 JSON 报告 | [第 159 行](../../train_dataloader.py#L159) |

## 结合本次讨论看图

- **位置与数量：** 位置 0～127 包含两端，共 128 个 Token；x 为位置 0～127，y 为位置 1～128，各排成 `[4,32]`。这些数字是语料位置，不是 ID 数值。
- **每个位置都预测：** `model(x)` 得到 `[4,32,50257]`。每条序列独立做因果注意力，128 个位置都参与交叉熵；`next_token_loss` 已有对应标签，不再右移一次，也不先取 argmax。
- **固定可见范围：** 每条序列内的位置使用 32×32 的因果遮罩。四条序列之间、前后 batch 之间不延续隐藏状态。详见 [Attention 代码](../../gpt2_model.py#L138) 和 [前向调用图](gpt2-forward-call-map.md)。
- **每轮重复：** 只重置读取位置，不重置模型或优化器。无打乱、随机偏移；重新走到同一位置时 x/y 相同，模型参数可以不同。
- **两个循环边界：** 当前数据可取 2640 个完整 batch 才回绕；默认外层只有 50 步，只处理 6400 个监督位置，并没有读完整轮。数据末尾不足一批的部分跳过。
- **日志与更新：** `.step()` 后记录的 `loss.item()` 仍来自本轮更新前的前向，参数更新不会让它自动重新计算。

数据量与首轮起点来自 [50 步归档报告](../../artifacts/lesson05-dataloader-report.json)；这里整理已有代码和讨论，未改变训练行为。上一块的 `train_step.py` 和 `lesson04.md` 继续作为固定批次实验的配套。

## 重新生成

```bash
.venv/bin/python docs/diagrams/render_dataloader_flow.py
```

更新 SVG 与定位说明。macOS 加 `--png` 同时生成 PNG，使用系统 AppKit；缓存与临时 Swift 文件均保存在当前项目的 `.cache/` 中。

```bash
.venv/bin/python docs/diagrams/render_dataloader_flow.py --png
```

脚本只读取源码与已有报告，不导入训练脚本、不启动训练。绘制依据：2026-10-18 的本地代码。
