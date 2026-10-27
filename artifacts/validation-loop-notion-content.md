学习日期：<mention-date start="2026-10-26"/>。所属：阶段二 → 单元 2「预训练工程：复现 GPT-2」→ 2.4「多卡与数据」→ 知识块 13。

**本块已学习，用户反馈“大概了解了”，并亲自完成完整 GPT-2 的 CPU 短跑。** 掌握程度记录为“理解中”；本次运行验证了流程，不等于完成正式预训练或整个 2.4。

项目：[guqun/GPT-2-124M](https://github.com/guqun/GPT-2-124M)。配套：课程讲义（项目内：`lesson13.md`）、训练入口（项目内：`train_validation.py`）、用户实跑报告（项目内：`artifacts/lesson13-user-gpt2-cpu-report.json`）。

## 1. 为什么训练过程中还要验证

训练 loss 回答：当前模型在参与参数更新的数据上预测得怎么样？验证 loss 回答：当前参数在留出的、没有用于这次训练更新的数据上预测得怎么样？验证仍然做前向和交叉熵计算，但不做反向或参数更新。

验证指标有两种不同的“反馈”方式：

- **对参数的直接反馈**：训练 loss 经 `backward()` 形成梯度，再由 `optimizer.step()` 更新参数。验证 loss 不走这条链路。
- **对训练决策的反馈**：人或另写的控制逻辑可以据此选择 checkpoint、提前停止、调整训练策略。这些功能需要显式实现，不会因为算出了 val_loss 自动发生。

本节 `train_validation.py` 只打印并保存验证记录。学习率仍由更新序号决定，没有把 val_loss 传给 AdamW 或学习率调度器，也没有自动早停、最佳 checkpoint 选择或模型保存。

## 2. 模型参数从哪里来

**当前入口从新初始化的参数开始训练。** 之前做前向对照时加载过 GPT-2 预训练权重，现在学习的是随机参数起点的预训练工程。

<table header-row="true">
	<tr>
		<td>内容</td>
		<td>当前是否使用</td>
		<td>用途</td>
	</tr>
	<tr>
		<td>`config.json`</td>
		<td>使用</td>
		<td>确定层数、隐藏维度、头数、词表大小等模型结构</td>
	</tr>
	<tr>
		<td>`tokenizer.json`</td>
		<td>使用</td>
		<td>将文本转换为 Token ID</td>
	</tr>
	<tr>
		<td>`model.safetensors`</td>
		<td>不使用</td>
		<td>它保存已有模型的训练后参数；本入口不读取这些权重</td>
	</tr>
</table>

主入口 M2 的关键代码：

```python
raw_model = GPT(config, tie_embeddings=False, gpt2_init=False).float()
tie_weights_like_course(raw_model)
initialize_like_course(raw_model)
```

`gpt2_init=False` 把课程专用初始化留给后面的显式调用，并不是加载预训练参数。Token/Position Embedding 和线性层权重按规则随机初始化；线性层偏置为 0；LayerNorm 缩放为 1、偏移为 0。`lm_head` 与 Token Embedding 共享权重。

初始化在训练开始前执行一次。下一批沿用上一次更新后的参数，验证也使用调用当时的同一份参数。`eval()` 不会复制模型、重置权重或加载另一套模型。

## 3. 全流程和代码位置

<image source="file-upload://405eaea3-1414-81e0-a26d-00b228161c74" />

可缩放 SVG（项目内：`docs/diagrams/gpt2-validation-loop-flow.svg`） · 流程图阅读说明与源码定位（项目内：`docs/diagrams/gpt2-validation-loop-flow.md`）

**图中底部数值来自协作验证用的 tiny 模型报告；本次用户运行的是完整 GPT-2，数值见第 9 节。两者控制流相同，网络规模不同，loss 不需要一致。**

阅读顺序：顶部 `main()` 建立模型、优化器和两套 loader → 左侧 `train_with_validation()` 控制何时更新/验证 → 紫色 A/B 调用进入右侧 `evaluate_validation()` → 恢复原模式、保存记录 → 返回原调用点继续训练。

<table header-row="true">
	<tr>
		<td>函数</td>
		<td>输入</td>
		<td>输出与目的</td>
	</tr>
	<tr>
		<td>`make_train_val_loaders()`</td>
		<td>完整 Token 序列、B/T、rank/R</td>
		<td>两套独立 loader 和切分位置</td>
	</tr>
	<tr>
		<td>`forward_and_loss()`</td>
		<td>当前模型、x/y、设备和精度</td>
		<td>logits 与本批平均交叉熵</td>
	</tr>
	<tr>
		<td>`evaluate_validation()`</td>
		<td>当前模型、验证 loader、运行信息、验证批次数</td>
		<td>全局平均验证 loss、取数位置及数量</td>
	</tr>
	<tr>
		<td>`record_validation()`</td>
		<td>已完成的参数更新次数</td>
		<td>调用验证、加标签、加入记录列表并打印</td>
	</tr>
	<tr>
		<td>`train_with_validation()`</td>
		<td>模型、优化器、两套 loader 与训练设置</td>
		<td>training / validation 两组记录</td>
	</tr>
</table>

`GPT.forward()` 本身只返回 logits；交叉熵由 `forward_and_loss()` 在模型调用之后计算。

## 4. 训练数据和验证数据怎样隔离

本地语料编码后共有 338,025 个 Token 位置，先按位置切分，再分别构造 x/y：

```python
split = len(token_ids) * 9 // 10
train_loader = ResettableDataLoaderLite(token_ids[:split], ...)
val_loader = ResettableDataLoaderLite(token_ids[split:], ...)
```

上面的 `...` 只省略 B/T/rank/R 等构造参数；完整代码在训练入口。

- 训练范围：`[0, 304222)`。
- 验证范围：`[304222, 338025)`。
- 两边各自生成右移一位的目标 y；训练最后的目标不会跨入验证范围。
- 隔离的是原始数据位置，同一个 Token ID 可以在两边出现。

两套 loader 分别保存游标。`val_loader.reset()` 只让验证重新读取同一段位置，不会重置训练游标。本次每次验证的相对子集起点都是 `[0, 64]`；训练游标在三次更新后是 512、1024、1536。

这是 Tiny Shakespeare 的位置切分，不是文档级划分或文本去重。原视频使用 FineWeb-Edu 的 train/val 磁盘分片；本地示例没有代替尚待补做的 FineWeb 实验。

## 5. V1—V6：验证函数内部做什么

<table header-row="true">
	<tr>
		<td>标记</td>
		<td>操作</td>
		<td>为什么这样做</td>
	</tr>
	<tr>
		<td>V1</td>
		<td>保存 `was_training`，调用 `model.eval()`</td>
		<td>切换模块行为，例如关闭 Dropout</td>
	</tr>
	<tr>
		<td>V2</td>
		<td>重置验证游标，令 `val_loss_accum=0`</td>
		<td>每次用固定验证片段比较参数变化</td>
	</tr>
	<tr>
		<td>V3</td>
		<td>进入 `torch.no_grad()`，循环取验证批次</td>
		<td>前向不记录反向计算图，节省不需要的反向资源</td>
	</tr>
	<tr>
		<td>V4</td>
		<td>`forward_and_loss()`，累计 `loss / validation_batches`</td>
		<td>取得本卡各批次的平均指标</td>
	</tr>
	<tr>
		<td>V5</td>
		<td>多进程时 `all_reduce(SUM)`，再除以 R</td>
		<td>得到全局平均验证 loss</td>
	</tr>
	<tr>
		<td>V6</td>
		<td>`finally: model.train(was_training)`</td>
		<td>在返回调用者前恢复原模式；异常时也恢复</td>
	</tr>
</table>

`eval()` 与 `no_grad()` 不是同一件事。前者改变模块的训练/评估行为，后者控制自动求导是否记录计算图，都不会把参数的 `requires_grad` 改为 False。当前配置的 Dropout 为 0，但仍保留正确的模式切换边界。

验证内部没有 `backward()`、`zero_grad()` 或 `optimizer.step()`：不生成新参数梯度，不清除已有 `.grad`，不推进 AdamW 的 m/v/step。下一次训练开头才清梯度。try 内发生异常时先恢复模式再抛出，不能理解为报错后仍按正常流程继续训练。

## 6. 一批数据怎样变成一个 loss

```python
logits = model(x)
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    y.reshape(-1),
)
```

本例每个进程的 B=2、T=32、词表大小 V=50257：

```text
x / y：[2, 32] 个 Token ID
    ↓ 前向：Embedding → Blocks → 最终 LN → lm_head
logits：[2, 32, 50257]
    ↓ 展平前两个维度
分数：[64, 50257]；目标：[64]
    ↓ 对每个位置的正确目标算交叉熵，再求平均
loss：一个标量，代表本批 64 个预测位置
```

每个位置仍只能读取当前和过去的上下文。验证使用真实文本中的右移目标评分，没有通过采样生成 Token 来代替正确目标。

## 7. 为什么还要在多批次和多卡之间平均

设一次验证每个 rank 读 M 批，每批有相同的有效位置数。每批 loss 已是批内均值，本卡再计算：

```python
val_loss_accum += loss / validation_batches
```

例如 M=2，本卡结果就是 `(loss_1 + loss_2) / 2`。多进程时所有 rank 都调用：

```python
dist.all_reduce(val_loss_accum, op=dist.ReduceOp.SUM)
val_loss_accum /= runtime.world_size
```

全局均值为：

$$
L_{val}=\frac{1}{RM}\sum_{r=0}^{R-1}\sum_{m=1}^{M}L_{r,m}
$$

这里通信的是 **loss 指标标量**，不是梯度或模型参数。不能只让 rank 0 参与通信；仅打印限制在 rank 0。本次 R=1，所以跳过跨进程通信。

均值再平均成立的前提是每批、每 rank 有效预测位置数相同。以后若有不等批次、padding 或忽略位置，应累计损失之和与有效位置数，再相除，不能机械平均各个均值。

## 8. 三种循环和验证触发条件

<table header-row="true">
	<tr>
		<td>名称</td>
		<td>本次值</td>
		<td>它到底数什么</td>
	</tr>
	<tr>
		<td>`steps`</td>
		<td>3</td>
		<td>参数总共更新几次</td>
	</tr>
	<tr>
		<td>`accum_steps` / K</td>
		<td>8</td>
		<td>每次更新前累计几个训练微批次的梯度</td>
	</tr>
	<tr>
		<td>`val-batches` / M</td>
		<td>2</td>
		<td>一次验证每个 rank 读取几批</td>
	</tr>
	<tr>
		<td>`val-every`</td>
		<td>2</td>
		<td>每更新几次参数验证一次</td>
	</tr>
</table>

训练每个微批次先计算 `loss/K` 再反向，8 次反向累计 `.grad` 后，统一裁剪、设置学习率、`optimizer.step()` 更新一次。等量数据下，每次更新的有效训练量为：

$$
B\times T\times K\times R=2\times32\times8\times1=512
$$

验证初始参数后进入训练；每次更新结束再判断：

```python
if (step + 1) % validation_every == 0 or step + 1 == steps:
    record_validation(step + 1)
```

因此本次顺序是：验证 0 → 训练 1 → 训练 2 → 验证 2 → 训练 3 → 验证 3 → 保存指标报告。第 1 次不满足间隔且不是最后一次，所以不验证；第 3 次虽然不满足间隔，但属于最后一次更新，因此验证。

## 9. 用户亲自运行的完整 GPT-2 结果

本次实际命令没有传 `--model-size tiny`，所以默认创建完整 GPT-2：12 个 Block、隐藏维度 768、12 个注意力头，共 **124,439,808 个独立参数**。CPU / FP32，随机初始化，未加载预训练权重。流程图中的 tiny 模型是 2 层、64 维、3,324,736 个参数；两者 loss 不应要求相同。

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/train_validation.py
```

<table header-row="true">
	<tr>
		<td>顺序</td>
		<td>动作</td>
		<td>loss</td>
		<td>学习率</td>
	</tr>
	<tr>
		<td>1</td>
		<td>验证：更新 0 次</td>
		<td>11.036055</td>
		<td>不更新参数</td>
	</tr>
	<tr>
		<td>2</td>
		<td>训练：更新 1 次</td>
		<td>10.958789</td>
		<td>0.0006</td>
	</tr>
	<tr>
		<td>3</td>
		<td>训练：更新 2 次</td>
		<td>9.674578</td>
		<td>0.0006</td>
	</tr>
	<tr>
		<td>4</td>
		<td>验证：更新 2 次</td>
		<td>9.143974</td>
		<td>不更新参数</td>
	</tr>
	<tr>
		<td>5</td>
		<td>训练：更新 3 次</td>
		<td>8.898489</td>
		<td>0.00033</td>
	</tr>
	<tr>
		<td>6</td>
		<td>验证：更新 3 次</td>
		<td>9.071917</td>
		<td>不更新参数</td>
	</tr>
</table>

结果与脚本设置一致：每次更新处理 512 个训练位置，共处理 1,536 个；每次验证读 `2 × 2 × 32 = 128` 个位置，来自同一段固定验证数据。学习率是预先配置的预热/余弦计划，不是验证下降触发的调整。

**如何理解下降：** 相同验证片段上的 loss 从 11.036055 降到 9.071917，说明这三次更新改善了该片段上的预测表现。只有三次更新、128 个验证位置，不能据此认定训练充分、泛化良好或已经复现官方 GPT-2 的能力。

训练 loss 与验证 loss 使用不同数据；训练 loss 来自本次更新前的前向，随后验证使用更新后的参数。它们不是同一批数据在同一时刻的两种计算，不要求数值相等。

开头的 `NotOpenSSLWarning` 是本机 urllib3 与 SSL 库的兼容性提示。本次本地训练已经完成并生成报告；这条警告没有阻止此次运行。当前 JSON 保存的是指标和配置，`checkpoint_saved=false`，没有保存训练后的模型权重；重新启动脚本会重新初始化，不会接着这三次更新继续训练。

运行报告已从会被下次默认运行覆盖的 `.cache/runtime-check/lesson13-report.json` 原样归档到 lesson13-user-gpt2-cpu-report.json（项目内：`artifacts/lesson13-user-gpt2-cpu-report.json`）。其 SHA-256 为 `cec78dea85508cce1d231c5c9696fed5215e0a21e4f98c510fc355e991213296`。

## 10. 工程核对与学习进度

此前协作验证已有 5 项行为测试：数据及错位目标隔离；验证均值与联合 batch 一致并保持参数/已有梯度/优化器/随机状态；异常模式恢复；非零 Dropout 下插入验证不改变训练轨迹；真实双进程 CPU/Gloo 与单进程联合 batch 对照。float64 小例子的验证 loss 误差约 `8.88e-16`，后续更新的参数误差约 `1.87e-16`，见 数值报告（项目内：`artifacts/lesson13-validation-equivalence.json`）。这些是已归档的检查，本次整理笔记没有重跑。

本次新增的学习证据是：**用户亲自运行完整 GPT-2 的 CPU 训练/验证入口，并反馈“大概了解了”。** 这与助手之前运行 tiny、CPU/Gloo 数值对照分别记录；不再把本块用户单进程实践写成未运行。

下一步按原课接“训练中定期生成文本”，观察 loss 指标之外的输出，再进入 2.5 HellaSwag 与正式训练。FineWeb-Edu 独立分片代码/实跑、用户 DDP 多进程实践、CUDA/NCCL 与 2.2 性能对照继续待补。2.3 的优化与累积已经在这次单进程入口实际使用，独立配置/多卡与 GPU 对照仍分开保留。

知识块 13 记为“已学习、初步理解；完整 GPT-2 单进程 CPU 实践完成”。2.4 整节没有完成，完整段落计数仍为单元 2 的 **1/5**、阶段二 **5/32**；自检用于复习，不另设完成门槛。

## 11. 来源与改编边界

- 课程主线：[Karpathy《Let's reproduce GPT-2 (124M)》约 3:23—3:25](https://www.youtube.com/watch?v=l8pRSuU81PU&t=12185s)。此前教学已核对非官方自动字幕和作者代码，未直接观看视频画面。
- 作者实现：[21d3d32 的验证循环](https://github.com/karpathy/build-nanogpt/blob/21d3d32/train_gpt2.py)。视频此处采用 FineWeb-Edu、较大的验证间隔和批次数；本地使用已有 Tiny Shakespeare、初始/每 2 次/最终验证，以短跑观察控制流。最后一次额外验证是本地教学补充。
- PyTorch：[eval 与关闭梯度计算的区别](https://docs.pytorch.org/docs/2.2/notes/autograd.html#locally-disabling-gradient-computation)。本地环境和执行行为以本项目源码、测试及报告为准。
- 衔接笔记：<mention-page url="https://app.notion.com/p/404eaea31414818b9df0c8ab083cdbe0"/>、<mention-page url="https://app.notion.com/p/400eaea3141481a39db3f37c52896a35"/>、<mention-page url="https://app.notion.com/p/402eaea31414814c80bad0fa8f02bdd7"/>。

本地代码与报告用于解释实现和记录实验，不反过来充当课程安排的独立来源。本次仅整理已学习内容与进度，没有新增训练或下载数据。
