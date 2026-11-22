# 阶段二 · 单元 3 · 3.9：全局与滑动窗口注意力的代码对照

本块承接已经学过的窗口遮罩、KV 缓存与完整 Block 数值图。目标是看清：**同样的输入、参数和 Q/K/V，怎样仅通过可见范围的变化，得到不同的注意力权重和输出。**

新入口：[compare_attention_windows.py](compare_attention_windows.py)。新计算模块：[sliding_window_attention.py](sliding_window_attention.py)。原有模型、训练、生成和课程文件均保留。

## 1. 本次来源、输入与范围

- 原理参考：[Mistral 7B §2](https://arxiv.org/html/2310.06825v1#S2)，以及已学的 [Notion 窗口笔记](https://app.notion.com/p/41eeaea3141481c7b0d1f3258ed57b75)。这是单元 3 的论文/实现学习，不是 Karpathy GPT-2 视频后续片段。
- 窗口约定沿用本项目：W 包含当前位置，允许 `0 <= i-j < W`。不同资料/接口对窗口边界的命名可能不同，以明确的不等式为准。
- 默认输入是人工生成的隐藏向量 U，形状 `[1,5,16]`，代表 Block 归一化之后的输入。A—E 是位置名称，不是五个实际字符的分词结果。
- 参数来自固定种子 42 下新建的 `ModernAttention`（沿用其子层默认初始化）；没有加载预训练权重，也没有使用 `ModernLM` 的整模型初始化流程。
- 4 个 Q 头、2 个 KV 头、每头 4 维；沿用 QK-Norm → RoPE → GQA → 拼头 → 输出投影的顺序。输出是注意力分支 O，不是词表 logits，未接残差/MLP/loss。
- 本节运行只做前向观察。新注意力函数支持反向，测试检查梯度，但本次没有优化器训练或模型质量实验。
- 本节先对照完整序列遮罩，不接滚动缓存；缓存淘汰与绝对位置的实现留作下一小块。
- 为打印完整权重 A，本实现仍构造 T×T 分数。遮住窗口外位置并不自动带来稀疏计算加速，不能根据本程序声称速度或峰值内存提升。

## 2. 运行

在项目根目录使用当前项目解释器：

```bash
.venv/bin/python compare_attention_windows.py
```

先观察默认 W=3；再改窗口做对照：

```bash
.venv/bin/python compare_attention_windows.py --window 1
.venv/bin/python compare_attention_windows.py --window 5
```

W=1 时每个位置只能读自己；W>=5 时本例与全局因果注意力完全一致。保存自己的新报告：

```bash
.venv/bin/python compare_attention_windows.py --report .cache/sliding-window/my-report.json
```

报告已存在会拒绝覆盖，默认不写文件。助手的 CPU/FP32 运行记录位于 [实跑报告](artifacts/sliding-window-attention-report.json)。

## 3. 从 main() 按执行顺序读

### M1：准备一份输入与一套参数

`main()` 读取 `--window`，调用 `run(window)`。`run()` 新建一个注意力层，生成 U，并保存参数快照用于检查。

两种模式没有分别创建或训练两套模型。只有一个 layer，后面共用其参数，避免把初始化差异误当作窗口效果。

### M2 → P1：Q/K/V 只计算一次

`project_once(layer,u)` 依次调用已有的三路线性投影，拆头，对 Q/K 做头内归一化和 RoPE，V 保持投影后的结果。

| 数据 | 形状 | 含义 |
| --- | --- | --- |
| U | `[1,5,16]` | 1 条序列、5 个位置、每位置 16 维 |
| Q | `[1,4,5,4]` | 4 个 Q 头，每头 4 维 |
| K / V | `[1,2,5,4]` | 2 组 KV，每组 4 维 |

接下来只改函数参数：

```python
global_trace = attention_with_window(q, k, v)
local_trace = attention_with_window(q, k, v, window)
```

两次接收的是同一份 Q/K/V。滑动窗口不会移动 WQ/WK/WV，也不会重新生成一套投影参数。

### A1：两边先算出相同的原始分数

4 个 Q 头按两组组织：Q0/Q1 读 KV0，Q2/Q3 读 KV1。对各自对应的 K 计算点积并除以 sqrt(4)。得到 `[1,4,5,5]` 的 scores。

最后两轴分别是查询位置和被读取位置。此时两边 scores 数值完全相同。

### A2：真正的差异在这里

```python
distance = positions[:, None] - positions[None, :]
allowed = distance >= 0
if window is not None:
    allowed = allowed & (distance < window)
```

`distance[i,j]` 是查询位置 i 与 K/V 位置 j 的距离。全局模式只挡住未来；窗口模式再挡住太早的位置。W=3：

| 当前 Q | 全局可读 | 窗口可读 |
| --- | --- | --- |
| A | A | A |
| B | AB | AB |
| C | ABC | ABC |
| D | ABCD | BCD |
| E | ABCDE | CDE |

### A3：先遮罩，再重新计算 softmax

```python
weights = torch.softmax(
    scores.masked_fill(~allowed, float('-inf')), dim=-1
)
```

被屏蔽的位置变为 0 权重；剩余位置的权重重新归一化。不是先得到全局权重后简单把旧位置设为 0。

本次随机教学输入的第 0 个 Q 头在 D 位置，打印到四位小数：

```text
位置       A       B       C       D       E
全局     0.1327  0.1509  0.0398  0.6766  0.0000
窗口     0.0000  0.1740  0.0458  0.7802  0.0000
```

D 对自身的原始分数没有变化，但屏蔽 A 之后，D 的归一化权重上升。这正是 softmax 分母变化的结果。

### A4 → P2：A 读取 V，再拼头与输出投影

```text
A：[1,4,5,5]
    × 各 Q 头所属组的 V
H：[1,4,5,4]
    ↓ transpose + reshape
拼接：[1,5,16]
    ↓ 同一套输出投影
O：[1,5,16]
```

V 相同，A 不同，因此 H 和 O 可以不同。前三个位置的可见范围相同，输出相同；本例从 D 开始出现差异。实跑 O 最大绝对差约为 0.236368，仅说明当前这套教学数值下的前向差异。

### M3/M4：核验并输出

1. 全局分支与旧 `ModernAttention.forward()` 的结果对照。
2. W=序列长度时与全局分支完全一致。
3. 参数快照未变化。
4. 打印每个位置可读的 K/V、第 0 个 Q 头的 A/H，以及 O 的差值。

程序最后另跑之前学过的单头人工算例：q_D=1，K_A..D=[0,0,0,ln2]，V_A..D=[10,2,4,6]。全局 D 输出 5.6，W=3 时 4.5。它直接指定中间特征，和上面的随机投影实验是两组不同例子，不能混用数值。

## 4. 验证结果与当前学习状态

新增 5 项行为测试：全局/大窗口对齐旧内核（MHA/MQA/GQA）、逐头逐行切片参考的前向与梯度、窗口外/未来位置的直接影响及梯度为零、单 Token/短序列/W=1/非法窗口，以及真实投影链路的反向和已学算例。

```bash
.venv/bin/python -m unittest discover -s tests -p test_sliding_window_attention.py -v
```

助手已在项目环境完成默认 CPU/FP32 实跑及上述5项检查。2026-11-22归档确认：用户已亲自运行默认对照并提供成功输出，O最大差0.236368，检查全部通过，反馈“大概了解了”。窗口对照记录为用户实践完成、理解中；整节3.9尚未验收。没有进行模型训练或GPU性能实验。

后续按总计划已学习[基础线性注意力原理与三种机制数值对照](docs/linear-attention-learning-notes.md)，下一步是独立线性代码实践。滚动缓存保留为工程待办：窗口裁剪后不能直接把缓存长度当成绝对Token位置，仍需跟踪已经处理的位置数并核对缓存/完整窗口重算输出。
