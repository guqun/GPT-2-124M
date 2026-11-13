# 第23块：把现代模型结构接成完整训练与生成流程

当前位置：阶段二 → 单元3 → 3.7 整合实践。前课 QK-Norm 对照已经由用户运行通过；本课继续连接完整模型、训练、检查点与生成，不覆盖旧代码。

## 1. 本次解决什么问题

之前为隔离变量，先建立 GPT，再逐项替换 Norm、位置方案、MLP、注意力。这种写法方便对照，但完整结构散在多个转换函数里。本课新增直接构造的 `ModernLM`：从 Token Embedding 到两个 Block、最终 RMSNorm、lm_head 都有明确入口。

目标：能沿代码追踪数据，分清参数、激活值和缓存；能训练、保存、重新加载同一个模型，并使用 KV Cache 生成。默认从全新随机参数开始，不导入官方权重，也不把两个模型的权重合并。每个矩阵直接初始化，不再先平均旧 MHA 权重得到 GQA。

**这是具有 Qwen3 风格计算结构的教学模型。** GPT-2 分词器、共享输出权重、theta=10000、eps=1e-5、相邻配对 RoPE、小模型尺寸均保留本地设置；不是官方 Qwen3 检查点兼容复现。当前全部 Dropout=0、单进程 FP32、等长无 padding、固定上下文，无 YaRN/滑动窗口/MoE。本次不完成 E1—E5 汇总与三家配置比较，也不据此结算整个3.7。

## 2. 先看代码流程图

![现代模型整合全流程](diagrams/modern-integration-flow.png)

[放大 SVG](diagrams/modern-integration-flow.svg)。图中 T1—T6、M1—M3、A1—A4、B1、F1、G1 与代码注释对应，标明输入输出、计算式和设备。读取/分词在 CPU；默认模型运算也在 CPU，指定设备后模型与 batch 才移动到对应设备。图是确定的计算流程和形状，不代表实测权重值。

## 3. 运行方法

在项目目录下先跑3次更新，检查整个链路：

```bash
/Users/guqun/micrograd/venv/bin/python train_modern.py --steps 3 --output .cache/modern-integrated/my-smoke
```

再做默认100次更新的本机短训练：

```bash
/Users/guqun/micrograd/venv/bin/python train_modern.py
```

默认输出 `.cache/modern-integrated/default/`；如已有文件，必须用 `--output` 指定一个新目录。既有模型与报告不会被覆盖。助手验证使用单独的 `.cache/runtime-check/modern-integrated-verified/`，不会占用你的默认目录。

训练脚本最后自动保存、重载、生成并检查缓存；也可单独重载自己的默认检查点：

```bash
/Users/guqun/micrograd/venv/bin/python generate_modern.py
```

如前面使用了3步检查目录，则显式指定：

```bash
/Users/guqun/micrograd/venv/bin/python generate_modern.py --checkpoint .cache/modern-integrated/my-smoke/checkpoint.pt
```

设备默认 CPU，可加 `--device mps` 或 `--device cuda`（须设备/解释器支持）。本次验证只报告 CPU 结果。生成固定16个新Token，使用贪心，无提前EOS停止；这不是对话模型，小语料短跑可能只生成换行、标点或重复词。

## 4. 先读哪些代码

| 文件 / 函数 | 作用、输入、输出 |
| --- | --- |
| [配置](../configs/modern-integrated-local.json) | 模型结构、训练预算、数据位置和生成提示词；不存参数值 |
| [train_modern.py](../train_modern.py) `load_config → run` | 读配置，准备数据，构造模型/AdamW，组织更新、验证、保存和生成 |
| [modern_model.py](../modern_model.py) `ModernLM.forward` | ID `[B,T]` → Embedding → Blocks → 最终Norm → logits `[B,T,V]` |
| `ModernBlock.forward` | 输入/输出 `[B,T,D]`；两次独立Norm、注意力/MLP以及两次残差 |
| `ModernAttention.forward` | QKV投影 → QK-Norm → RoPE → 按需接缓存 → GQA → 拼接/输出投影 |
| `ModernMLP.forward` | 两路扩展 → SiLU门控乘法 → 缩回D；不包含残差 |
| [architecture_training.py](../architecture_training.py) `train_update` | 复用前课的取批、loss/K反传、裁剪、AdamW和清梯度，不修改旧文件 |
| `evaluate` | 固定验证窗口，只观察loss，恢复原模式 |
| `load_checkpoint` | 新建同配置模型，严格加载本课自存权重；用于生成，不自动续训 |
| `generate → generation_check` | 逐Token生成；对照缓存与完整重算 |

读取主线是 `run → model(x) → ModernLM.forward → ModernBlock.forward`。`model(x)` 由 PyTorch 的 `nn.Module.__call__` 调用 `forward`，不需要手动在主循环写 `model.forward(x)`。

## 5. 模型与数据配置

| 项目 | 默认值与含义 |
| --- | --- |
| V | 50257，GPT-2 Token ID 的取值范围 |
| D / L | 128维、2个Block |
| Hq / Hkv / d | 4个Q头、2个KV头、每头32维 |
| MLP中间宽度 | 341，接近8D/3；三矩阵无bias |
| Norm | Block和最终Norm沿128维；QK-Norm沿32维；各实例独立gamma |
| 位置 | 无学习式位置表；每层Q/K相邻维配对RoPE；容量256 |
| 共享 | 输入Embedding与lm_head共享同一个Parameter；配置也支持不共享 |
| 数据 | 本地Tiny Shakespeare，GPT-2分词；Token位置前90%训练、后10%验证 |
| B / T / K | 2条序列、每条128个预测位置、累积4个微批次 |
| 预算 | 每次更新1024个预测位置；默认100次=102400个预测位置 |

为复用前课数据准备接口，JSON 保留 `model` 小节，同时 `modern_model` 明确现代结构；加载时核对共同维度是否一致。修改对应维度必须同步两处。参数初始值由 seed=42 的随机初始化产生；相同 seed 不意味着与旧“先构建后替换”的模型参数相同，因为分配顺序和初始化方式变了。

## 6. Block 内部与反向传播

设入口 `X[B,T,D]`：

```text
U = RMSNorm1(X)
Q/K/V = 对 U 的三套独立线性投影，再拆头
Qn = RMSNormQ(Q)；Kn = RMSNormK(K)
Qp/Kp = RoPE(Qn/Kn, 当前位置)
A = softmax(Qp Kpᵀ / sqrt(d) + causal_mask)
H = A V
O = 输出投影(拼接H)
R = X + O
Z = RMSNorm2(R)
F = down(SiLU(gate(Z)) * up(Z))
Y = R + F
```

GQA 按组广播K/V，每个Q仍有自己的A和H。V只在投影后进入AV，不经过QK-Norm或RoPE。两次残差分别加回X、R，不是同一个向量加两次。模型最后的RMSNorm在全部Block之外。

本项目负责上述顺序、张量形状、因果mask和缓存边界；`nn.Linear`、Embedding查表、matmul、softmax、SiLU和自动求导由PyTorch执行。RMSNorm、RoPE和分组注意力复用前课数学原语，不复制或改写旧课文件。

训练对所有有效位置计算交叉熵平均，每个微批的 `loss/4` 反传累积到参数的 `.grad`；这四轮里参数不变。累积完才裁剪并 `optimizer.step()`。新的Q/K gamma与其他参数一起被优化；一维gamma不做权重衰减，共享E在优化器中仅计一次。

## 7. 训练与缓存的边界

`model(x)` 返回所有位置的logits，训练不传缓存。`model(x, past, use_cache=True, last_only=True)` 返回末位置logits与每层的新缓存，要求 `eval()` 且关闭梯度。

缓存是当前参数下的激活值，不属于 `state_dict`。参数一旦更新，旧缓存必须丢弃。本入口每次生成请求从空缓存开始；不在训练微批次间复用缓存。调用者不能把其他模型或之前参数版本的缓存传进来——形状检查无法发现这种语义错误。

默认提示词编码为P个ID。prefill处理P个位置，选出第一个新ID；后面只输入新ID，同时读每层历史K/V。生成N个ID结束，缓存长度为P+N−1，因为最后选出的ID尚未再次前向。

## 8. 保存了什么、验证什么

- `checkpoint.pt`：模型配置、参数、AdamW状态、更新次数、训练规格、数据游标。只加载自己可信的本课文件。
- `report.json`：数据/代码指纹、训练和验证loss、gamma实际变化、重载检查、生成ID与缓存字节。
- `metrics.jsonl`：逐次更新记录。

本课只实现检查点重载生成，**未实现断点续训协议**。虽然保存AdamW与游标，不能把这些字段存在就当成实现了完整恢复。

专项检查包括：与前课逐项替换模型在完全相同权重下的前向和全部参数梯度对齐；因果性；多种分块缓存；微批累积与整批梯度等价；gamma更新及共享权重去重；共享/不共享检查点重载；错误缓存/容量/模式边界。

本轮直接初始化后训练的数值不能与前课报告简单相减就当成单因素消融结论。前课对齐测试会严格复制全部权重，随后比较前向和梯度；实际本课训练则从新的随机起点开始，二者目的分开。

## 9. 来源与范围

- [Raschka：Qwen3 from scratch](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3)：完整模型与缓存版本的学习参照，属于原书附加材料，不是Karpathy视频新增章节。
- [Transformers v4.51.3 Qwen3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py)：核对Q/K头内Norm→RoPE→缓存、SwiGLU与残差组织顺序。采用固定版本，不把主分支新功能混进来。
- 这里复用本地FP32原语、GPT-2分词器、相邻配对RoPE与教学尺寸；不直接兼容官方参数排列和全部配置，不声称是官方Qwen3复现或质量评测。

## 10. 本次助手验证（2026-11-13）

默认CPU/FP32、4线程、100次更新已完整运行，处理102400个训练预测位置。参数6,793,856；验证loss从10.762307到6.934774，四组Q/K gamma均更新。保存重载的提示词logits完全一致；训练后缓存/重算逐轮最大差9.54e-7，贪心ID一致。提示词8个Token，生成16个后缓存长23；每层K/V各[1,2,23,32]，共23552字节。生成结果为连续换行，不代表质量合格。

5项新整合检查与6项前课QK-Norm回归通过；72个旧Python文件哈希不变。完整实跑报告见[报告](../artifacts/modern-integration-report.json)，验证摘要见[记录](../artifacts/modern-integration-verification.json)。以上是助手验证，用户本课运行与理解反馈尚待补充。
