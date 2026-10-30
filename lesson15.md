# 2.5 · 知识块 15：把 HellaSwag 评测接入训练

新入口：[train_hellaswag.py](train_hellaswag.py)。单题处理和评测循环：[hellaswag_eval.py](hellaswag_eval.py)。之前的 train_shards.py、验证、生成、加载器及所有旧 Python 文件保留原样。

## 1. 本次解决什么问题

前面已理解候选评分。现在把它放进真实控制流程：**使用当前参数，逐题评分，统计准确率，恢复训练，再更新参数**。

输入有两路：训练仍读 train/val 的 npy 分片；四选一评测独立读取带 `ctx`、四个 `endings` 和 `label` 的 JSONL。评测题不会送进训练分片，也没有反向传播。

### 来源和改编边界

- 原视频：[约 3:35:15 的独立评测](https://www.youtube.com/watch?v=l8pRSuU81PU&t=12915s)、[约 3:38:16—3:41:18 的训练接入与多卡计数](https://www.youtube.com/watch?v=l8pRSuU81PU&t=13096s)。核对了非官方自动字幕副本与作者历史源码，没有直接观看视频画面。
- 历史代码：[8018ed2 / train_gpt2.py](https://github.com/karpathy/build-nanogpt/blob/8018ed2c82c78a0600b2f7c66b1e606a93c60584/train_gpt2.py)、[hellaswag.py](https://github.com/karpathy/build-nanogpt/blob/8018ed2c82c78a0600b2f7c66b1e606a93c60584/hellaswag.py)。选择平均候选交叉熵最小的一行，按题号取模分卡，最终 SUM 正确数和题数。
- 原课以 250 次迭代为间隔；本地短跑改成初始，以及**完成第 2、3 次参数更新后**评测，按“已完成更新次数”记日志。最后一次必评；可用 `--hellaswag-every` 调整。
- 本地单题编码继续使用已缓存的 GPT-2 tokenizer，分别编码上下文和前置空格的候选。原脚本使用 tiktoken；本次未宣称对任意文本两个 tokenizer 都已逐项核验。
- 本地直接调用 `raw_model` 做评测，避免各卡题数不等时 DDP 在前向同步 buffer；每个进程都有训练同步后的参数。结束时才合并计数。

| 设置 | 本次行为 |
|---|---|
| 参数起点 | 每次启动随机初始化；不承接上一脚本，不加载官方预训练权重 |
| 默认模型 | 完整 GPT-2 124M；建议先用 `--model-size tiny` 看流程 |
| tiny 配置 | 2 层、64 维、2 头，词表 50257，位置上限保留 1024；与上一入口 tiny 的 128 位置不同，因此随机初始化短跑数值也不同 |
| 训练数据 | 默认两句教学文本的 train/val 分片；不是 FineWeb；已有分片用 `--data-dir` |
| 默认评测题 | **真实 HellaSwag 验证集前 8 题**的本地预览，启动不联网 |
| 执行 | eager；CPU/FP32 已检查；训练与评测可按原入口选择 BF16；生成保留 FP32 |
| 输出 | JSON 观察报告；不保存模型检查点、不自动早停、不按准确率改学习率 |

## 2. 先运行，再看调用链

先用 CPU 小模型跑三次更新：

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/train_hellaswag.py --device cpu --model-size tiny
```

若要使用原来的完整 GPT-2 模型，去掉 `--model-size tiny`。8 道题也各自包含四条完整候选，因此完整模型的评测比短序列训练步更慢。

报告默认写到 `/Users/guqun/micrograd/gpt2_pretrain_lab/.cache/runtime-check/lesson15-report.json`。

```text
main
  M1 复用两个独立分片加载器
  M2 准备分词器和配置，H1/H2 读取并检查所有评测题，再随机初始化模型和 AdamW
  M3 train_with_hellaswag
       初始验证 → 初始四选一评测
       每次更新：K 个微批次 → backward → 裁剪 → 设置 lr → AdamW.step
       到验证间隔：evaluate_shards → 返回
       到评测间隔：record_hellaswag → evaluate_hellaswag → 返回
       到生成间隔：sample_current_model → 返回
       继续下一次参数更新
  M4 保存训练、验证、生成、评测及每卡记录
```

建议阅读顺序：`train_hellaswag.py` 的 M2 和 `record_hellaswag()` → T7 的触发条件 → `hellaswag_eval.py` 的 H1—H4 → 回来看 M4 的报告。

## 3. H1/H2：一道题如何成为模型输入

真实文件每行一题。例如下面仅展示结构，文本是教学示意：

```json
{"ctx":"A person opens a book. They", "endings":["begin reading.","eat the table.","turn into a tree.","fly into space."], "label":0}
```

`label` 可以是整数或字符串形式的 0～3；它只用于判对错，不输入模型。无标签 test 集不能直接计算准确率。

每个候选前加入一个空格再编码，按原课组合 `context_ids + ending_ids`。四行右补齐到最长长度，形成 `[4,T]`。`mask` 同形状，只有候选结尾为 1。**补齐 ID 0 与 mask 的 0 是不同概念**：真实候选中即使出现 ID 0，也照常计分。

默认最多取文件前 8 题；`--hellaswag-limit 0` 取全文件。上限在分卡之前生效，是全局题数。空文件、缺少四个选项、非法标签、空候选、越界 ID 或超出位置上限都会报错，不静默截断或跳过题目。

## 4. H3：代码与前面讨论的公式对应

模型得到 `logits [4,T,V]`，然后：

1. `logits[:, :-1]` 对应 `tokens[:, 1:]`；每个位置预测下一个 Token。
2. `cross_entropy(..., reduction='none')` 保留每个位置的损失。
3. `mask[:, 1:]` 跟随目标对齐，排除上下文和右侧补齐。
4. 每行损失加权求和，除以该行 mask 中 1 的数量，得到四个平均损失。
5. `argmin()` 返回选项编号，再与 `label` 比较；并列时沿用原课选择第一个。

例如输入 `[10,20,31,32,33,0]`，前两项是上下文，最后一项是补齐。目标成为 `[20,31,32,33,0]`，评分 mask 为 `[0,1,1,1,0]`，分母为 3。

不是先生成一个 Token 再决定下一步的输入。完整候选已给定，因果遮罩保证每个目标仅使用此前的上下文/候选前缀，因此一次前向即可得到所有需要的分数。

`score_candidates()` 不接收 label。它输出每行损失之和、候选 Token 数、平均损失及最小值编号；公式与判分逻辑分开。

## 5. H4/T7：多进程与回到训练

每个 rank 只处理 `question.index % world_size == rank` 的题目。一张卡负责该题的全部四个候选，不把四个选项拆到不同卡。

评测使用 `eval()` 加 `no_grad()`；结束后用 `finally` 恢复之前模式。它不清梯度，不修改参数/AdamW，也不推进 train/val 分片游标。训练仍由 loss/K 的梯度驱动。

汇总只传 `[答对题数, 处理题数]` 两个整数。各卡做 SUM 后再相除，不直接平均每卡准确率；某张卡没有分到题，也要参加汇总。使用 raw_model 避免不等次数的 DDP 前向通信，之后训练继续用原 DDP 包装对象。

本机两个 CPU 进程示例（没有使用两张 GPU）：

```bash
GLOO_SOCKET_IFNAME=lo0 /Users/guqun/micrograd/venv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29519 /Users/guqun/micrograd/gpt2_pretrain_lab/train_hellaswag.py --device cpu --model-size tiny --batch-size 1 --hellaswag-limit 3
```

`lo0` 是本机 macOS 的回环接口；其他系统应使用其对应接口或让 Gloo 自动选择。B=1、T=3、R=2，total-tokens=12，因此 K=2；评测 3 题分为 2+1。

## 6. 下载结果与完整数据怎么接入

GitHub 原课程 URL 本次返回 404，HF rows 接口首次连接重置；随后通过作者数据集的 HF first-rows 接口成功取得前 8 题。保存文件与来源：

- [真实验证题预览](artifacts/lesson15-hellaswag-preview.jsonl)
- [下载 URL、日期与 SHA-256](artifacts/lesson15-hellaswag-preview-source.json)
- [作者数据集页面](https://huggingface.co/datasets/Rowan/hellaswag)
- [原始 JSONL 数据目录](https://github.com/rowanz/hellaswag/tree/master/data)

数据下载已完成到“8 题预览”，尚未下载或运行完整验证集。预览文件的校验值在加载时核对。

如果手动下载完整 `hellaswag_val.jsonl` 到项目 `.cache/hellaswag/`，用：

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/train_hellaswag.py --device cpu --model-size tiny --hellaswag-file /Users/guqun/micrograd/gpt2_pretrain_lab/.cache/hellaswag/hellaswag_val.jsonl --hellaswag-limit 0
```

输入要求是 JSONL；从 HF 下载的 parquet 不能直接改扩展名当作 JSONL。默认正上限 8 表示仅预览前 8 题，完整评测务必设置 0。外部文件的实际来源由提供者确认，程序不会根据文件名宣称是完整官方基准。

网络失败期间准备的[自编教学题](artifacts/lesson15-teaching-questions.jsonl)另存保留；它不是默认数据，使用时日志会明确显示“非 HellaSwag”。没有在真实下载失败时静默切换题集的逻辑。

## 7. 检查结果与学习边界

已完成数值和状态检查：手算概率对应平均交叉熵；实际模型整批评分等于逐个前缀评分；右补齐不影响真实候选得分；真实 ID 0 保留；非法输入拒绝；模式、参数、梯度和随机数状态保持；非零 Dropout 下新增入口与旧入口的训练/验证/生成及 AdamW 历史完全一致。

真实 CPU/Gloo 检查包含 3 题分成 2+1，以及 1 题分成 1+0；汇总后能继续 DDP 反向，更新与单进程联合 batch 对照一致。此前分片训练的四项检查也通过。

[tiny CPU 三次更新报告](artifacts/lesson15-hellaswag-cpu.json)中，初始、第 2 次及第 3 次更新后分别为 1/8、3/8、3/8。只训练两句文本、只有八道题，不能把这个变化解释为可靠的常识能力提升，更不能与原课完整基准成绩比较。

报告中 `hellaswag` 保存来源和全局配置，`ranks[*].hellaswag` 保存每轮正确数、总题数、各卡题号、前四题的得分明细及评测前后游标。完整模型本节用户实践、完整验证集和 CUDA/NCCL 仍待继续；本轮不结算 2.5，也不更新 Notion。

[双进程 CPU 入口报告](artifacts/lesson15-hellaswag-ddp-cpu.json)也已跑通三次参数更新，题目按 2+1 分配；与单进程相同三题的判分一致。训练/验证 loss 最大差为 9.54e-07。[历史源码对照与旧文件校验](artifacts/lesson15-alignment.json)记录八题预测均与作者原函数一致，31 个既有 Python 文件保持原样。
