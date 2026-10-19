# GPT-2 预训练工程学习

[中文](README.md) | [English](README.en.md)

课程主线：[Karpathy — Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)。配套参考：[build-nanogpt](https://github.com/karpathy/build-nanogpt)。

**本项目所属：阶段二「现代大模型：从分词到部署」→ 单元 2「预训练工程：复现 GPT-2」。当前位置：第 2.2 节 → 知识块 9「BF16 自动混合精度」，已讨论并反馈初步理解。** 知识块按本项目连续编号，8 是 2.2 的第一块；用户已对设备、计时、TF32 和 BF16 反馈初步理解，GPU 提速仍待实测。2.1 的七块已覆盖并回顾，用户于 2026-10-19 确认整节学习完成。

每次一个知识块：目的 → 输入输出 → 公式与形状 → 源码 → 实际运行 → 自检。这里的脚本是按课程整理的独立教学实现；记录中的数值来自实际文件，不把手算值当作模型输出。

[学习计划核对与执行修正](docs/learning-plan-review.md)：对照 Notion 的整体目标、原始资料和视频顺序，明确已学基础、改编边界与验收标准。进度见 [Notion 总计划](https://app.notion.com/p/340eaea31414819197bde537d339d0fc)；[2.1 收尾回顾](docs/section-2.1-review.md)分别记录内容覆盖、工程验证与独立掌握。

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

这部分作为知识补充，不新增主线进度。主课程当前进入阶段二 · 单元 2 · 2.2 · 知识块 9。

## 当前进度：阶段二 · 单元 2 · 2.2 · 知识块 9 初步理解

阶段一的基础课程已经学完。阶段二内部：单元 1「分词器」的 1.1—1.4 已学完；单元 2 的 **2.1 已完成**（2026-10-19 用户确认），现在学习 2.2；单元 3—7 尚未开始。学习完成进度为单元 2 **1/5**、阶段二 **5/32**。自检题保留作后续复习。依据、补充内容与掌握程度的区别见[阶段二已学内容核对](docs/learning-plan-review.md)。

已学习配置与权重检查、独立模型与前向对照，以及 `forward` 的逐层调用。复习见 [阶段笔记：检查点、独立实现与前向对照](docs/lesson01-02-notes.md)，包含本次追问、调用图和已有实验结果；掌握程度保留“理解中”。

文本生成、单批次过拟合、数据加载、权重共享和初始化已逐块讲解并核对实验。2026-10-19 用户对初始化反馈“大概清楚了”，并确认 2.1 整节学习完成；知识块 5—7 保留初步理解的记录，自检题用于复习。2.2 的知识块 8 已讨论设备、预热、吞吐率与 TF32，用户反馈初步理解；知识块 9 的 BF16、转换开销、反向与 FP32 更新也已讨论，用户反馈“大概理解”。下一块为 torch.compile，尚未开始。每块使用新文件，之前的代码与讲义保留。Notion 最近已同步的里程碑是 2.1 完成；本次细分知识块的位置先记录在本地。

新增 [GPU 训练提速专题笔记](docs/gpu-training-notes.md) · [Notion 归档](https://app.notion.com/p/3feeaea3141481af8e6fd509bee27fc7)：汇总设备/MPS、计时与吞吐、TF32、BF16 前向到更新的精度变化，包含讨论中的数值例子、完整流程图、代码定位及复习自检。新笔记已归档；Notion 总计划本轮未修改。

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

默认完整 124M、B=4/T=32、2 步预热 + 5 步测量。本块当时的 25 项测试通过，[本机 CPU 报告](artifacts/lesson08-cpu-baseline.json)已归档。CUDA/MPS 在当前环境不可用；GPU 上的 highest/high 对照命令和记录规范见讲义，GPU 提速实验尚待实跑。用户反馈初步理解后，继续上方知识块 9 的 BF16。

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

本块验证前向计算；后续的生成、单批次训练、数据加载、共享和初始化均已逐块学习。2.1 已完成，当前进入 2.2 提速。

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
