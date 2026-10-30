# GPT-2 预训练工程学习

[中文](README.md) | [English](README.en.md)

课程主线：[Karpathy — Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)。配套参考：[build-nanogpt](https://github.com/karpathy/build-nanogpt)。

**当前：阶段二 → 单元 2 → 2.5，知识块 15：HellaSwag 评测接入训练（2026-10-30）。** 评测原理、代码接入与 CPU/GPU 分工已覆盖，用户初步理解；代码与流程图已完成，助手 CPU/Gloo 验证与用户亲自实践分别记录。下一步学习正式训练配置、训练量及结果与基线对照，随后接原课 llm.c 收尾。2.5 尚未完成；Notion 总计划已同步，完整段落计数保持单元 2 的 1/5、阶段二 5/32。

新增 [train_hellaswag.py](train_hellaswag.py) / [hellaswag_eval.py](hellaswag_eval.py) / [第 15 块讲义](lesson15.md)。保留所有旧 Python 文件；初始状态和默认更新 2、3 次后用当前参数评测，恢复模式继续训练。重试下载成功，默认使用本地保存的 **HellaSwag 验证集前 8 道真实题目**，不是完整基准。CPU tiny 短跑和评分/状态隔离检查通过；完整验证集及 CUDA/NCCL 仍待补。

先运行小模型观察流程：

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/train_hellaswag.py --device cpu --model-size tiny
```

真实题预览已随项目保存，启动无需联网。完整 JSONL 用 `--hellaswag-file` 指定，并以 `--hellaswag-limit 0` 选择全文件；默认正上限为 8。来源校验、各卡统计、候选得分和模式恢复见讲义。

新增 [HellaSwag CPU / GPU 流程图 PNG](docs/diagrams/gpt2-hellaswag-cpu-gpu-flow.png)（[SVG](docs/diagrams/gpt2-hellaswag-cpu-gpu-flow.svg) / [代码定位说明](docs/diagrams/gpt2-hellaswag-cpu-gpu-flow.md)）：按源码行号标出数据准备、设备搬运、GPU 评分、CPU 判对、计数归约及返回训练；单独区分 CUDA 分工示意和已有 CPU/Gloo 结果。

此前新增 [train_shards.py](train_shards.py) / [实践（三）讲义](lesson12-shard-training.md)，复用独立的 train/val 加载器与已学训练主干，保留所有旧代码。默认教学数据按两句话分别分配 train/val；支持指定已有分片目录。tiny CPU 单进程与真实双进程 Gloo 三次更新已跑通，13 项相关检查通过；29 个此前 Python 文件校验值保持不变。真实 FineWeb 和 CUDA/NCCL 未实跑，不能把本次流程检查记为用户完成正式训练。Notion 总计划已同步至 10 月 30 日；2.4 未整体完成，计数保持 1/5、5/32。

2026-10-30 已将原数据笔记补全为[从原始文本、分片到模型训练](docs/fineweb-edu-sharding-notes.md)（[同一篇 Notion 笔记](https://app.notion.com/p/404eaea31414818b9df0c8ab083cdbe0)）。第十三节串联正文/EOT、npy、CPU/设备、x/y、六个预测位置、微批次累积、跨片更新及验证/生成返回；本地内嵌完整代码流程图。笔记归档与总计划结算分开，不改变完成计数。

前两块的 [prepare_fineweb.py](prepare_fineweb.py) / [数据准备讲义](lesson12-data-preparation.md)与 [load_shards.py](load_shards.py) / [读取讲义](lesson12-shard-batches.md)保留原样。实际 FineWeb 20 条预览此前因连接重置未下载，本次不重试下载。

此前知识块 14 的[复习笔记](docs/periodic-generation-notes.md)（[Notion](https://app.notion.com/p/406eaea31414814ea785fee733047ffc)）与[完整模型用户报告](artifacts/lesson14-user-gpt2-cpu-report.json)保留。用户 DDP 多进程、CUDA/NCCL 和 2.2 工程仍待补；2.3 优化与累积已在用户单进程入口运行，独立配置与多卡/GPU 对照另记。

每次一个知识块：先说明解决什么问题、为什么学、要学会什么，再结合例子、输入输出、公式与形状，最后落到代码和实验；已学基础简要带过，自检供复习。这里的脚本是按课程整理的独立教学实现；记录中的数值来自实际文件，不把手算值当作模型输出。

[学习计划核对与执行修正](docs/learning-plan-review.md)：对照 Notion 的整体目标、原始资料和视频顺序，明确已学基础、改编边界与验收标准。进度见 [Notion 总计划](https://app.notion.com/p/340eaea31414819197bde537d339d0fc)；[2.1 收尾回顾](docs/section-2.1-review.md)分别记录内容覆盖、工程验证与独立掌握。

[梯度累积全流程图](docs/diagrams/gpt2-gradient-accumulation-flow.md)：展开微批次内的 loss 缩放与反向、四轮 2×2 梯度矩阵累计，以及累计完成后的统一裁剪和 AdamW 更新。[PNG](docs/diagrams/gpt2-gradient-accumulation-flow.png) / [SVG](docs/diagrams/gpt2-gradient-accumulation-flow.svg) 已保存；教学例已复算，训练入口和学习完成计数不变。

[DDP / NCCL / All-Reduce 双卡流程图](docs/diagrams/gpt2-ddp-allreduce-flow.md)：两条 GPU 泳道展开本地梯度、分块求和、交换汇总结果、求平均及各自更新，标明 CPU 调度与 GPU 计算的位置。[PNG](docs/diagrams/gpt2-ddp-allreduce-flow.png) / [SVG](docs/diagrams/gpt2-ddp-allreduce-flow.svg) 为 2.4 的原理讨论配图，数字是教学例子。

[把 DDP 接到 lm_head 的具体计算](docs/diagrams/gpt2-ddp-lm-head-example.md)：用两条样本展开线性层、softmax、交叉熵、两份完整权重梯度、跨卡归约和 AdamW，持续追踪同一个 `W[0,0]`。[PNG](docs/diagrams/gpt2-ddp-lm-head-example.png) / [SVG](docs/diagrams/gpt2-ddp-lm-head-example.svg) 已保存；固定特征的 2×2 输出头例子已用 PyTorch 对照整批计算，属于 CPU 数值验证。

[DDP、平均损失与梯度同步笔记](docs/ddp-batch-gradient-notes.md) · [Notion](https://app.notion.com/p/402eaea31414814c80bad0fa8f02bdd7)：归档各 GPU 的数据归属、分块通信、单卡整批矩阵求导与双卡平均的对应，重点回答“一个参数为什么只有一个梯度”“为什么需要 batch loss”“平均之后如何沿计算图求导”。含既有配图、数值例子和折叠复习答案；2026-10-23 归档后另行同步了 Notion 总计划中的学习位置，未改变训练行为或完成计数；最新进度见下文。

[FineWeb-Edu、分片与文档边界笔记](docs/fineweb-edu-sharding-notes.md) · [Notion](https://app.notion.com/p/404eaea31414818b9df0c8ab083cdbe0)：知识块 12 的学习归档，包含大规模数据流程图、Tiny Shakespeare 对照、EOT 的作用与局限、文档级遮罩的取舍，以及不足一个 batch 的 shard 尾部如何处理。按原课方案继续；已开始[独立数据准备实践](lesson12-data-preparation.md)，分片准备、加载和训练串联已完成教学数据 CPU 检查，真实 FineWeb 和用户新入口实践待继续。

## 新机器上的运行准备

以下命令在仓库根目录执行。已验证的环境为 macOS / Python 3.9.6，依赖版本固定在 `requirements.txt`。本机已经完成这一步时，直接使用下面的运行命令即可。

```bash
/usr/bin/python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/hf download openai-community/gpt2 config.json model.safetensors tokenizer.json \
  --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --local-dir .cache/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e
```

权重约 548 MB，分词器约 1.4 MB。实验脚本优先读取仓库内的固定版本缓存；也可通过 `--model-dir` 指定目录。下载只需准备时执行，实验本身离线读取文件。

```bash
.venv/bin/python inspect_checkpoint.py --json .cache/runtime-check/lesson01-report.json
.venv/bin/python compare_forward.py --report .cache/runtime-check/lesson02-report.json
.venv/bin/python generate_text.py --prompt "Hello, I'm a language model," --check-reference --report .cache/runtime-check/lesson03-greedy.json
```

`.venv/` 和 `.cache/` 已被 Git 忽略。上述命令将本机运行报告保存到缓存目录。

## Python 知识补充

[Python 迁移教材](python_foundations/README.md) 面向熟悉 Java、C++、Dart 的资深程序员，共 17 个单元，重点讲解 Python 语义差异，以及阅读大模型代码所需的 Tensor、autograd 和 `nn.Module` 知识。

- [Markdown 教材](python_foundations/textbook.md)：可直接在 GitHub 阅读。
- [HTML 教材](python_foundations/textbook.html)：克隆仓库后用浏览器打开，包含目录筛选、代码复制和折叠实验。
- [可运行示例](python_foundations/examples/)、[配置接口练习](python_foundations/exercises/config_report.py) 与 [参考实现](python_foundations/answers/config_report.py)。

这部分作为知识补充，不新增主线进度。主课程已覆盖阶段二 · 单元 2 · 2.2 的理论内容，接续讨论了 2.3 优化器与梯度累积、2.4 DDP、FineWeb-Edu 分片与验证集；验证与定期生成均已学习并完成用户单进程 CPU 实践；分片与训练/验证/生成已串联，当前接续 2.5 HellaSwag 评测的新入口。

## 2026-10-27 归档：定期生成已学习，完整 GPT-2 CPU 实践完成

以下保留归档时状态；2026-10-28 已先接续数据准备实践，见顶部与下方新入口。

阶段一的基础课程已经学完。阶段二单元 1 的 **1.1—1.4 已完成**，单元 2 的 **2.1 已完成**。2.2 理论已覆盖，部分工程与 CUDA 对照待补；2.3 优化与累积已在用户单进程运行，多卡/GPU 实践另记；2.4 原理与数据串联已覆盖，用户完整 GPT-2 CPU 验证/生成短跑完成，教学分片的新入口有助手 CPU/Gloo 验证，真实 FineWeb、用户新入口/多进程及 CUDA/NCCL 待补。**当前 2.5 的 HellaSwag 原理、训练接入和 CPU/GPU 分工已覆盖，下一步正式训练配置与结果对照；2.5 未完成。** 单元 3—7 尚未开始。完整段落计数为单元 2 **1/5**、阶段二 **5/32**；详见[学习计划与证据](docs/learning-plan-review.md)。

2026-10-23 已将梯度累积加入原 AdamW 笔记，并接续 2.4 的 DDP 与平均损失讨论。当时归档为答疑笔记，并按用户要求同步了 Notion 总计划；后续数据与验证内容尚未覆盖，多进程运行待实践。该历史记录不用于判断今天的位置。

2026-10-24 接续 [知识块 11](lesson11.md)：新增 `train_ddp.py`，CPU/Gloo 双进程取数、累积与同步已运行，并和单进程联合 batch 核对梯度、参数及 AdamW 状态。新入口接通分组、裁剪、调度和累积；前十块入口保留。当时开始代码走读，完整段落计数仍为 1/5、5/32，CUDA/NCCL、FineWeb-Edu 与验证集列为后续。

2026-10-25 已讲解 [知识块 12：FineWeb-Edu 与数据分片](docs/fineweb-edu-sharding-notes.md)的核心内容并归档答疑，掌握程度“理解中”。用户选择按原课继续：EOT 加普通因果注意力，shard 尾部不足时跳过；文档级遮罩与跨片补齐不加入必学实现。Notion 总计划与本地进度已同步；下一步训练 Token 总量、步数和 warmup 的简短换算，再学验证集与 validation loss。独立分片代码和实际运行待补，整个 2.4 未完成，1/5、5/32 不变。

已学习配置与权重检查、独立模型与前向对照，以及 `forward` 的逐层调用。复习见 [阶段笔记：检查点、独立实现与前向对照](docs/lesson01-02-notes.md)，包含本次追问、调用图和已有实验结果；掌握程度保留“理解中”。

文本生成、单批次过拟合、数据加载、权重共享和初始化已逐块讲解并核对实验。2.2 接续讨论设备、计时、TF32/BF16、compile/kernel、FlashAttention 和词表对齐；本机 CPU 编译对照与 MPS 普通训练已有报告，SDPA/对齐独立入口与 CUDA 逐项实验仍待补。Notion 总计划 10 月 25 日同步到知识块 12，10 月 26 日同步到验证循环已学习及用户完整 GPT-2 CPU 实跑完成；已学习、协作验证、亲自运行与独立掌握分别记录。

[AdamW、初始化偏差修正与梯度期望](docs/adamw-bias-correction-notes.md) · [Notion](https://app.notion.com/p/400eaea3141481a39db3f37c52896a35)：保留完整推导、手算、梯度裁剪与三段学习率。第十七节续入梯度累积：loss/K、四轮 .grad 矩阵相加、两层循环、统一裁剪与一次 AdamW 更新，并嵌入全流程 PNG，项目另有 SVG。教学数值已复算；优化主干已接通，用户本次完整 GPT-2 单进程实验实际使用了这些步骤，独立配置/多卡与 GPU 对照另记；掌握程度“理解中”，2.3 尚未结算为完成。

[GPU 训练提速（一）](docs/gpu-training-notes.md) · [Notion](https://app.notion.com/p/3feeaea3141481af8e6fd509bee27fc7)：设备、计时、TF32/BF16 与前向到参数更新的精度变化。

新增 [训练提速（二）笔记](docs/gpu-training-optimization-notes.md) · [Notion](https://app.notion.com/p/3ffeaea3141481f9bd20ca4745fa8b80)：compile/kernel、FlashAttention、词表 50257→50304，以及额外 logits 对 loss/梯度的影响与遮罩方案。含流程图、输入输出、公式、项目对应、实测表和复习自检。遮罩属于教学补充，尚未集成完整训练。

接续的 [FlashAttention 流程图](docs/diagrams/gpt2-flash-attention-flow.md)已保存为 [PNG](docs/diagrams/gpt2-flash-attention-flow.png) 和 [SVG](docs/diagrams/gpt2-flash-attention-flow.svg)：对比完整中间矩阵与片上分块累计，用两块数据得到 `300 / 10 = 30`，并说明稳定在线归一化和反向重算。这是原理讨论配图，尚未作为 CUDA 性能实验结果。

## 2.4 实践（三）：分片接入完整训练循环

新增 [train_shards.py](train_shards.py) / [lesson12-shard-training.md](lesson12-shard-training.md)。阅读 M1—M4（准备和调用）、T1—T6（训练/观察）、V1—V4（多分片验证）。分片切换只影响取数；模型、AdamW 状态及更新计数继续沿用。单个 batch 不跨文件，但同一次更新累计的多个微批次可以跨文件。

配套[完整代码流程图 PNG](docs/diagrams/gpt2-shard-training-flow.png)（[可缩放 SVG](docs/diagrams/gpt2-shard-training-flow.svg) / [读图说明](docs/diagrams/gpt2-shard-training-flow.md)）：展开分片读取、微批次累积、一次 AdamW 更新、独立验证和生成的调用 / 返回，底部对照三次更新的真实文件与游标。

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/train_shards.py --device cpu --model-size tiny
```

默认 B2/T3/R1/K2，三次更新共 36 个训练预测位置，初始及更新 2/3 次后验证，更新 2/3 次后生成。见 [CPU 报告](artifacts/lesson12-shard-training-cpu.json)、[双进程 CPU/Gloo 报告](artifacts/lesson12-shard-training-ddp-cpu.json)及[来源/核对记录](artifacts/lesson12-shard-training-alignment.json)。默认数据是两句教学文本，不用于语言质量评估；`--data-dir` 可改用已有 GPT-2 uint16 分片。

## 2.4 实践（二）：分片文件 → CPU Tensor → x/y

新增 [load_shards.py](load_shards.py) / [lesson12-shard-batches.md](lesson12-shard-batches.md)，保留旧准备和训练代码。按 S1—S7 阅读：加载当前文件、筛选 train/val、取 B×T+1、错位并 reshape、前进 B×T、检查下一批、换片/回绕。默认只观察数据，不创建模型。

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/load_shards.py
```

本节把两句教学文本重复 5 组，另写 16/16/8 三片；train 只读后两片。B=2、T=3 的四次起点为文件 000001 的 0、6，文件 000002 的 0，再回 000001 的 0。[CPU 报告](artifacts/lesson12-shard-batches-cpu.json)与[来源/改编记录](artifacts/lesson12-shard-loader-alignment.json)已归档。5 项检查含精确边界、split/reset 隔离及双 rank 取数对照；不是实际 DDP/GPU 或 FineWeb 训练。本块开始学习，理解状态待讨论，完成计数不变。

## 2.4 补充实践：正文 → Token ID → 分片文件

先读 [lesson12-data-preparation.md](lesson12-data-preparation.md)，再看 [prepare_fineweb.py](prepare_fineweb.py) 的 P1—P5：读取正文、前置 EOT、检查 ID 范围、填缓冲区、写满换片和保存尾片。这个阶段不构造模型，也不更新参数。

配套[代码流转图](docs/diagrams/gpt2-data-preparation-flow.md)（[PNG](docs/diagrams/gpt2-data-preparation-flow.png) / [SVG](docs/diagrams/gpt2-data-preparation-flow.svg)）展开主入口、惰性编码、外层文档循环与内层搬运循环、完整片与尾片保存，并以实际 demo 的 8 个 ID 对照每轮 take / used / offset。行号从源码提取，数值从已归档报告核对。

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/prepare_fineweb.py
```

默认使用两句教学文本，每次写入新的缓存目录。加 `--source fineweb-preview` 可请求官方 20 条真实记录，默认每片 2048 Token；本机此下载尚未成功，不以 demo 冒充真实 FineWeb 结果。新环境依赖见 [requirements-data.txt](requirements-data.txt)，[demo 报告](artifacts/lesson12-data-preparation-demo.json)已归档。

## 阶段二 · 单元 2 · 2.4 · 知识块 14：训练中定期生成文本

新入口 [train_generation.py](train_generation.py) 与 [lesson14.md](lesson14.md) 将生成观察接入训练；上一节文件保留原样。先读 G1—G6，再读主循环 T4：固定提示词 → 当前参数前向 → top-k 抽样 → 追加 ID → 恢复训练。生成没有 backward、参数更新，也不把生成文字放回训练数据。

配套[代码流转图](docs/diagrams/gpt2-periodic-generation-flow.md)（[PNG](docs/diagrams/gpt2-periodic-generation-flow.png) / [SVG](docs/diagrams/gpt2-periodic-generation-flow.svg)）标出主循环的触发条件、生成内部的形状变化、独立 RNG、模式恢复和返回位置；数值、ID 与代码行号由绘图脚本读取核对。

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/train_generation.py --device cpu --model-size tiny
```

默认在更新 2、3 次后各生成 4 条，每条 24 个新 Token。[CPU tiny 报告](artifacts/lesson14-cpu-report.json)与第 13 块同配置的训练/验证记录一致；4 项行为检查含非零 Dropout 下的训练轨迹保持。文字仍不连贯，这只是短跑流程验证，不代表完成预训练。[来源与改编](artifacts/lesson14-course-alignment.json)对应视频 3:25—3:28 与作者历史提交；独立 RNG、每次从同一提示词开始及当时的 compile 问题见讲义。默认报告单独写入 `.cache/runtime-check/lesson14-report.json`，不保存 checkpoint。

2026-10-27 用户已运行默认完整 GPT-2：训练与验证记录和上一块完整模型报告完全一致；更新 2、3 次后各生成 4×24 个新 Token，分别以换行和逗号为主。该结果符合本节流程观察预期，不代表文本质量合格。详见[归档笔记](docs/periodic-generation-notes.md)（[Notion](https://app.notion.com/p/406eaea31414814ea785fee733047ffc)）及[完整模型用户报告](artifacts/lesson14-user-gpt2-cpu-report.json)。现已初步理解，下一步 2.5；图中 tiny 示例保留并明确区分。

## 阶段二 · 单元 2 · 2.4 · 知识块 13：验证循环

2026-10-26 新增 [train_validation.py](train_validation.py) / [lesson13.md](lesson13.md)，保留旧训练入口。先看 `evaluate_validation()`（V1—V6），再看 `train_with_validation()`（T1—T3）：单独取验证数据、eval/no_grad、等量批次及跨卡均值、恢复模式、继续训练。本机可复用已有环境：

配套[代码流转图](docs/diagrams/gpt2-validation-loop-flow.md)（[PNG](docs/diagrams/gpt2-validation-loop-flow.png) / [SVG](docs/diagrams/gpt2-validation-loop-flow.svg)）将主循环、验证函数、返回位置和实际运行顺序对应起来；源码行号与数值由绘图脚本读取并核对。

```bash
/Users/guqun/micrograd/venv/bin/python train_validation.py --device cpu --model-size tiny
```

默认在更新 0、2、3 次时验证，每次读取 2 批/进程；tiny 保留 GPT-2 词表并缩小网络，省略该选项则创建完整 124M。用户 2026-10-26 亲自运行默认完整模型，CPU/FP32、3 次更新，val loss 为 **11.036055 → 9.143974 → 9.071917**，报告已另存 [lesson13-user-gpt2-cpu-report.json](artifacts/lesson13-user-gpt2-cpu-report.json)。本块记为已学习、初步理解、用户单进程实践完成。完整[验证循环笔记](docs/validation-loop-notes.md)（[Notion](https://app.notion.com/p/405eaea3141481c98d06ce42da3d9212)）包含参数起点、模式切换、代码图与结果解读；图中 tiny 数值与本次完整 GPT-2 分开标注。原课 FineWeb 与本地位置切分的差异见讲义。既有 5 项行为检查和 CPU/Gloo 协作验证保留；本次短跑不代表正式预训练，FineWeb/用户多进程/CUDA 实践待继续。

## 阶段二 · 单元 2 · 2.4 · 知识块 11：DDP、数据分配与梯度累积

打开 [train_ddp.py](train_ddp.py)，配合 [lesson11.md](lesson11.md)，先从 `main()` 看启动进程与数据预览，再看 D1—D4 的游标、T1—T5 的累积同步和更新。仍使用 Tiny Shakespeare 与随机初值；默认完整 124M，`--model-size tiny` 是流程验证用的两层小模型。

```bash
.venv/bin/python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29511 \
  train_ddp.py --device cpu --preview-only
```

移除 `--preview-only` 并添加 `--model-size tiny` 即可运行双进程短训练。默认每进程 B=2/T=32，R=2 时累积 K=4，每次更新处理 512 个位置，共更新 3 次。新入口用公开 `no_sync()` 包住前向和反向，按全局块起点一起回绕，并接上此前已讨论的 AdamW 参数分组、裁剪与学习率调度。

6 项行为测试通过，包含真实 CPU/Gloo 双进程与单进程全局 batch 的梯度及更新对照；[小模型双进程报告](artifacts/lesson11-ddp-cpu-report.json)、[完整 124M 单进程短跑](artifacts/lesson11-gpt2-single-process-smoke.json)和[数值对照](artifacts/lesson11-ddp-equivalence.json)已归档。这些结果验证训练流程，CUDA/NCCL 与性能仍待运行；本块未加入 FineWeb-Edu 或验证集。

## 阶段二 · 单元 2 · 2.2 · 知识块 10：torch.compile

新配套：[train_compile.py](train_compile.py) / [lesson10.md](lesson10.md)。对应[视频约 1:47:34—1:59:46](https://www.youtube.com/watch?v=l8pRSuU81PU&t=6454s)及作者 [fb8bd6e 历史提交](https://github.com/karpathy/build-nanogpt/commit/fb8bd6e)。在模型放到设备后、训练循环前包装一次；复用上一节 BF16/FP32 循环。新增 eager/compile 对照开关，报告分别列出首次更新与排除预热后的计时。

直接运行时，自动选择 CUDA，否则使用 CPU；支持 BF16 的 CUDA 默认 BF16，其余默认 FP32。MPS 不参与本课版本的自动编译对照。启动时打印实际设备和精度；默认 B=4/T=32，下面使用更小的 B=1/T=8 做短跑。

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/train_compile.py --device cpu --precision fp32 --execution compile --batch-size 1 --seq-len 8 --threads 1 --warmup-steps 3 --steps 5
```

将 `--execution compile` 换成 `--execution eager`，其余条件保持一致，即可对照。当前 CPU 实验用于验证编译、反向及参数更新；CUDA 性能对照待补。此版本只编译模型，外部交叉熵与 AdamW 沿用普通执行。为兼容 PyTorch 2.2.2 图捕获，模型中的 `enumerate(..., start=1)` 等价改写为 `enumerate(..., 1)`。

31 项测试通过；完整 124M 单线程 CPU 的 [eager](artifacts/lesson10-cpu-eager.json) 与 [compile](artifacts/lesson10-cpu-compile.json) 各 8 次更新已完成。排除预热后约 849 / 846 ms 每步，本次未显示明显提速。当前 Intel Mac 的 4 线程编译反向曾停在矩阵乘法同步等待，因此本节默认单线程；细节与首次编译开销见讲义。

Mac GPU 普通训练也已验证：使用 `--device mps --execution eager --precision fp32`，完整 124M 的三次小批次更新成功，见 [MPS 报告](artifacts/lesson10-mps-eager-smoke.json)。`eager` 只是关闭编译，计算仍在 GPU 上；当前课程版本的 MPS 普通训练与 CUDA 编译提速实验分别记录。

## 阶段二 · 单元 2 · 2.2 · 知识块 9：BF16 自动混合精度

新配套：[train_mixed_precision.py](train_mixed_precision.py) / [lesson09.md](lesson09.md)。对应[视频约 1:39:28—1:47:34](https://www.youtube.com/watch?v=l8pRSuU81PU&t=5968s)及作者 [177e4cd 历史提交](https://github.com/karpathy/build-nanogpt/commit/177e4cd5b4cc05df4bb637ed1a9e55911d6f1e2c)：前向与交叉熵进入 `autocast`，部分运算和输出使用 BF16，参数仍为 FP32，反向与更新在区域外执行。

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/train_mixed_precision.py --device cpu --probe-only --batch-size 1 --seq-len 8
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/train_mixed_precision.py --device cpu --precision bf16 --batch-size 1 --seq-len 8 --warmup-steps 1 --steps 2
```

第一条对同一组参数/输入比较 FP32 和 BF16，不更新；第二条运行完整训练链路。29 项测试通过；完整 124M 的 [dtype 对照](artifacts/lesson09-cpu-dtypes.json)与 [3 次 CPU 更新](artifacts/lesson09-cpu-bf16-smoke.json)已归档，参数、参数梯度、loss 和 AdamW 的 m/v 保持 FP32，logits 为 BF16。CPU 验证使用 B=1/T=8；GPU 相同条件的性能对照命令见讲义，尚未实跑，不能把本机结果当作 GPU 提速证据。

## 阶段二 · 单元 2 · 2.2 · 知识块 8：GPU 执行、训练计时与 TF32

新配套：[benchmark_training.py](benchmark_training.py) / [lesson08.md](lesson08.md)。对应[视频约 1:22—1:39](https://www.youtube.com/watch?v=l8pRSuU81PU&t=4938s)及作者 [5265b20 历史提交](https://github.com/karpathy/build-nanogpt/commit/5265b2042aaaa84943313002671b8b3ca7749b06)：先识别硬件与精度，再正确计时，用训练 Token/s 衡量速度，并准备 CUDA TF32 对照。

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/benchmark_training.py --inspect-only
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/benchmark_training.py --device cpu
```

默认完整 124M、B=4/T=32、2 步预热 + 5 步测量。本块当时的 25 项测试通过，[本机 CPU 报告](artifacts/lesson08-cpu-baseline.json)已归档。当时工具隔离环境未检测到 CUDA/MPS；2026-10-20 已在普通终端环境验证 MPS 普通训练，见知识块 10。CUDA 的 highest/high 对照命令和记录规范见讲义，TF32 提速实验尚待实跑。用户反馈初步理解后，继续上方知识块 9 的 BF16。

## 阶段二 · 单元 2 · 2.1 · 知识块 7：参数初始化

新增 [train_initialization.py](train_initialization.py) 与 [lesson07.md](lesson07.md)，沿用上一块的共享关系、顺序加载器和随机训练起点。对应[视频约 1:13:03—1:22:11](https://www.youtube.com/watch?v=l8pRSuU81PU&t=4383s)及作者 [9ac321e 历史提交](https://github.com/karpathy/build-nanogpt/commit/9ac321e299f06c0414bd577fd2776f4562283a8a)：普通权重标准差 0.02，Linear 偏置为零，LayerNorm 保留 gamma=1/beta=0；两类残差输出投影标准差为 `0.02/sqrt(2L)`。初始化只在训练开始前执行。

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/train_initialization.py --inspect-only
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/train_initialization.py
```

其他机器使用项目 `.venv/bin/python`。第一条显示初值统计与视频的随机累加实验；第二条训练 50 步。本块当时的 22 项行为测试通过，完整模型的 50 步记录在[训练报告](artifacts/lesson07-initialization-report.json)，历史小模型数值对照在[源码对齐报告](artifacts/lesson07-course-alignment.json)。用户已反馈初步理解，并确认所属的 2.1 整节学习完成；自检保留作复习。

## 阶段二 · 单元 2 · 2.1 · 知识块 6：权重共享

新文件 [train_weight_tying.py](train_weight_tying.py) 与 [lesson06.md](lesson06.md)。对照[视频约 1:05:55—1:13:03](https://www.youtube.com/watch?v=l8pRSuU81PU&t=3955s)及作者 [ec39245 历史提交](https://github.com/karpathy/build-nanogpt/commit/ec39245425b2c82a38c0787f27e5f9d7082c973b)，保留原输出表，让 `wte.weight` 引用 `lm_head.weight`。重点是同一参数的两种用途、参数数量减少，以及两条梯度路径共同更新一次。

```bash
.venv/bin/python train_weight_tying.py --inspect-only
.venv/bin/python train_weight_tying.py
```

当前 Mac 已授权复用原解释器，可将 `.venv/bin/python` 替换为 `/Users/guqun/micrograd/venv/bin/python`。第一条只看共享关系与参数量；第二条从随机初值开始，用上一块的顺序数据执行 50 次更新。全模型参数从 163,037,184 降为 124,439,808，减少 38,597,376。19 项行为测试通过，完整 50 步已执行，见[实跑报告](artifacts/lesson06-weight-tying-report.json)。本块不加载预训练权重，专用初始化仍留给下一块。

## 阶段二 · 单元 2 · 2.1 · 知识块 5：顺序数据加载器

打开新文件 [train_dataloader.py](train_dataloader.py)，配合 [顺序数据加载器学习单](lesson05.md)。沿用上一块的训练主干，重点看 `DataLoaderLite` 保存 `current_position`，每次取 `B*T+1` 个 ID、构造 x/y、前进 `B*T`、不足下一批时回到开头。加载器在循环外创建，`next_batch()` 在循环内调用。

配套[代码流程图与源码定位表](docs/diagrams/gpt2-dataloader-flow.md)已保存到项目，含 [PNG](docs/diagrams/gpt2-dataloader-flow.png) 和 [SVG](docs/diagrams/gpt2-dataloader-flow.svg)：按 M1—M4 / L1—L4 对照执行顺序、张量形状、回绕与返回。

```bash
.venv/bin/python train_dataloader.py --preview-only
.venv/bin/python train_dataloader.py
```

第一条命令只展示前三批；第二条用完整 Tiny Shakespeare 编码后的 338,025 个 Token，按 B=4/T=32 顺序更新 50 步。每步数据不同，loss 可以波动，不再以同批次 loss 逼近零为目标。参数仍从随机初值开始，暂不共享权重、不使用专用初始化；旧脚本没有保存检查点，本脚本也不承接上次训练后的权重。

本块对照[视频约 1:01:52—1:05:55](https://www.youtube.com/watch?v=l8pRSuU81PU&t=3712s)及作者 [631f7d6 历史代码](https://github.com/karpathy/build-nanogpt/blob/631f7d6f080effc4342b5e2f2f41879642c76fd1/train_gpt2.py)。17 项测试通过，50 步实跑归档在[训练报告](artifacts/lesson05-dataloader-report.json)，与历史代码的取批行为对照在[加载器对齐报告](artifacts/lesson05-loader-alignment.json)。原来的 `train_step.py` 和 `lesson04.md` 继续保留。

## 阶段二 · 单元 2 · 2.1 · 知识块 4：Tiny Shakespeare 单批次过拟合

打开 [train_step.py](train_step.py)，配合 [单批次过拟合学习单](lesson04.md)，先看训练数据与预期，再从 `main()` 依次读：原文编码 → 错位得到 x/y → 随机参数模型 → 全位置前向 → 交叉熵 → 反传与更新 → 同一批重复 50 次。M / D / T 编号与学习单流程图对应。

```bash
.venv/bin/python train_step.py
```

默认使用项目内的 [Tiny Shakespeare 原文](data/tiny_shakespeare/input.txt) 前 1000 字符；编码为 285 个 Token，取前 129 个构造 `x/y=[4,32]`，共 128 个监督位置。CPU / FP32、seed=42、AdamW lr=3e-4，固定这一批更新 50 次。原视频此刻尚未共享输入/输出表，也未引入专用初始化，因此临时有 163,037,184 个参数；共享后回到 124,439,808。

实测同批次 loss 从 `10.927578` 降到 `0.002746`，给定真实前缀的下一 Token 预测正确数从 `0/128` 到 `128/128`。这验证固定小批次能被学好，不衡量自由生成或泛化。默认报告在 `.cache/runtime-check/lesson04-report.json`，本次结果另存为[实跑归档](artifacts/lesson04-overfit-report.json)。

参数起点与数据/循环设置由作者 [ce84eb3](https://github.com/karpathy/build-nanogpt/commit/ce84eb34ea813ec463a2587ddd8147f158df4c82) 和 [7822fce](https://github.com/karpathy/build-nanogpt/commit/7822fce8ebf46ba3907ee8f7cf0356f8ec1c63fa) 核对。视频位置、数据版本，以及本地 tokenizer、设备和 seed 的区别见[学习单第 0 节](lesson04.md#0-课程依据为什么这里切换为随机参数)。

## 阶段二 · 单元 2 · 2.1 · 知识块 3：使用自己的模型生成文本

按 [文本生成学习单](lesson03.md) 从上一块的 `[B,T,V]` 接起：取最后位置 → 选一个 ID → 追加到输入 → 再次前向。

配合 [代码与完整流程对照图](docs/diagrams/gpt2-generation-flow.md) 阅读：图中 `M/G/S` 编号与源码注释一致，并标出张量形状、分支、循环返回和停止条件。

打开 [generate_text.py](generate_text.py) 并点击“运行 Python 文件”，默认执行贪心生成。先读 `generate()` 中标为 ①～⑤ 的循环，再读 `select_next_token()` 的贪心和采样分支。程序逐步打印长度、形状与选中的 Token。

```bash
.venv/bin/python generate_text.py --check-reference
.venv/bin/python generate_text.py --strategy sample --temperature 0.8 --top-k 50 --seed 42
```

已实跑课堂固定提示词 `Hello, I'm a language model,` 的 20 个新 Token：贪心 ID 与 Hugging Face 参考结果一致；采样实验与 5 个行为测试通过。无参数运行使用 `main()` 中 `--prompt` 的当前默认值；复现图中 8 个 Token 的示例时，按学习单显式传入课堂提示词。脚本只处理单条输入，使用 CPU / FP32，每步重算完整前缀。

## 阶段二 · 单元 2 · 2.1 · 知识块 2：独立模型与前向对照

先看 [代码调用与计算逻辑对照图](docs/diagrams/gpt2-forward-call-map.md)：图中对应 `GPT.forward → Block.forward → Attention / MLP.forward` 的调用、计算与返回，并标注输入输出形状。

![GPT-2 代码调用与计算逻辑对照图](docs/diagrams/gpt2-forward-call-map.png)

先按 [本次学习操作单](lesson02.md) 运行一个实验，再跟踪主数据流：

- `gpt2_model.py`：独立 GPT-2 实现；先看 `GPT.forward` 和 `Block.forward` 的输入输出。
- `compare_forward.py`：加载同一份本地权重，与 Hugging Face 逐层比较，再检查因果遮罩。
- `artifacts/lesson02-report.json`：运行后保存本次实际比较结果。

```bash
.venv/bin/python compare_forward.py
```

本块验证前向计算；后续的生成、单批次训练、数据加载、共享和初始化均已逐块学习。2.1 已完成，随后接续 2.2 提速与 2.3 优化器学习。

## 阶段二 · 单元 2 · 2.1 · 知识块 1：配置与权重

文件 `inspect_checkpoint.py` 对应视频开头检查 GPT-2 参数的内容。读取本机已缓存的预训练配置和 safetensors 权重，使用 Hugging Face 的现有模型类检查名称、形状、参数值和权重共享。本块还没有开始自己实现 GPT-2。

- 模型代码：规定 LN、Attention、残差、MLP 等按什么顺序运算。
- 配置：规定多少层、几维、几个头等，也包含激活函数、归一化 epsilon 等设置。
- 权重：保存训练得到的具体数值，包括矩阵、偏置、LayerNorm 的缩放和平移参数。
- Tokenizer 文件：规定文本与 ID 的对应方式，和神经网络参数一起保持匹配。

这次读取的 `model.safetensors` 是发布的预训练权重文件。用于恢复训练的 checkpoint 还可以包含优化器状态、步数、随机数状态等；这些内容不一定存在于发布权重中。

## 在 VS Code 运行

在 VS Code 中打开本仓库文件夹。本机复用已有课程环境，请通过 **Python: Select Interpreter** 选择 `/Users/guqun/micrograd/venv/bin/python`。其他机器按前面的准备步骤创建项目环境后，选择仓库内的解释器：

```text
.venv/bin/python
```

打开 `inspect_checkpoint.py`，使用“运行 Python 文件”。项目设置已给出 `.venv/bin/python` 作为默认解释器。

在仓库根目录运行的等价命令：

```bash
.venv/bin/python inspect_checkpoint.py
```

显示全部参数：追加 `--all`。观察另一个 ID：追加 `--token-id 17250`。脚本从文件自身与用户目录定位路径，不依赖终端当前目录。

当前验证环境为 Python 3.9.6、PyTorch 2.2.2、Transformers 4.57.6。脚本离线读取已存在的 safetensors，不调用在线下载，也不修改模型权重。若模型放在其他目录，可用 `--model-dir` 指定包含两个文件的目录。此次环境在导入库时可能打印 urllib3 / LibreSSL 兼容提示；本脚本的离线读取已经实际验证。

## 第一块该怎么看

先观察五项配置：`vocab_size=50257`、`n_positions=1024`、`n_embd=768`、`n_layer=12`、`n_head=12`。因此单头维度为 64。

然后找以下三行：

```text
transformer.wte.weight       [50257, 768]   Token 向量表
transformer.wpe.weight       [1024, 768]    位置向量表
transformer.h.0.ln_1.weight  [768]          第一个 Block 的 LN1 gamma
```

`h.0` 是从 0 开始编号的第一个 Block；`.weight` 只是参数名称的一部分，不保证都是二维矩阵。LayerNorm 的 weight 就是一维的 gamma。

`E[220]` 是表中第 220 行的 768 个已训练数值。输入 ID 220 时才查出这一行作为该位置的 Token 向量。训练好的 Q/K/V 权重存在文件中；当前输入对应的 Q/K/V 则需要前向计算后才产生。

最后查看参数总数与共享关系。独立参数总数为 124,439,808，E 与 lm_head 引用同一参数。不要把 `state_dict()` 所有名称下的元素直接相加来统计独立参数，否则共享表会重复计算。

Hugging Face GPT-2 的合并 QKV 权重是 `[768, 2304]`，使用 Conv1D 包装实现线性投影；PyTorch Linear 的存储方向相反。后续写自己的模型并加载权重时，再具体处理转置。本块先辨认它对应哪个环节。

## 检查范围

`artifacts/lesson01-report.json` 与 `artifacts/lesson01-output.txt` 保存实际读取结果：核心配置、参数名称与形状、一行向量的前八个数、参数数量、权重共享以及加载完整性。

本块完成的是“现成权重能在参考模型中正确加载并检查”；自己的模型实现、logits 对齐与训练验证在后续知识块进行。

## 两个自检问题

1. 只有配置文件，能否知道某个 Token 向量的具体 768 个数？为什么？
2. 把 `n_embd` 从 768 改成 512，原来的 `[50257,768]` 向量表能否原样放进去？

这一块讨论清楚后，再开始根据配置搭建自己的模型结构。
