# GPT-2 预训练工程学习

[中文](README.md) | [English](README.en.md)

课程主线：[Karpathy — Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)。配套参考：[build-nanogpt](https://github.com/karpathy/build-nanogpt)。

每次一个知识块：目的 → 输入输出 → 公式与形状 → 源码 → 实际运行 → 自检。这里的脚本是按课程整理的独立教学实现；记录中的数值来自实际文件，不把手算值当作模型输出。

## Python 知识补充

[Python 迁移教材](python_foundations/README.md) 面向熟悉 Java、C++、Dart 的资深程序员，共 17 个单元，重点讲解 Python 语义差异，以及阅读大模型代码所需的 Tensor、autograd 和 `nn.Module` 知识。

- [Markdown 教材](python_foundations/textbook.md)：可直接在 GitHub 阅读。
- [HTML 教材](python_foundations/textbook.html)：克隆仓库后用浏览器打开，包含目录筛选、代码复制和折叠实验。
- [可运行示例](python_foundations/examples/)、[配置接口练习](python_foundations/exercises/config_report.py) 与 [参考实现](python_foundations/answers/config_report.py)。

这部分作为知识补充。主课程现接续配置与权重检查，进入下方第二块的前向模型对照。

## 当前进度：2.1 前两个知识块已归档

已学习配置与权重检查、独立模型与前向对照，以及 `forward` 的逐层调用。复习见 [阶段笔记：检查点、独立实现与前向对照](docs/lesson01-02-notes.md)，包含本次追问、调用图和已有实验结果；掌握程度保留“理解中”。

下一知识块是使用自己的模型生成文本。2.1 整节仍在进行中，生成、训练循环、单批次过拟合、数据加载与初始化专题尚未完成。

## 2.1 第二块，独立模型与前向对照

先看 [代码调用与计算逻辑对照图](docs/diagrams/gpt2-forward-call-map.md)：图中对应 `GPT.forward → Block.forward → Attention / MLP.forward` 的调用、计算与返回，并标注输入输出形状。

![GPT-2 代码调用与计算逻辑对照图](docs/diagrams/gpt2-forward-call-map.png)

先按 [本次学习操作单](lesson02.md) 运行一个实验，再跟踪主数据流：

- `gpt2_model.py`：独立 GPT-2 实现；先看 `GPT.forward` 和 `Block.forward` 的输入输出。
- `compare_forward.py`：加载同一份本地权重，与 Hugging Face 逐层比较，再检查因果遮罩。
- `artifacts/lesson02-report.json`：运行后保存本次实际比较结果。

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/compare_forward.py
```

本块先验证前向计算。2.1 中的生成、训练循环、单批次过拟合和初始化专题仍在后续学习，不据此把整节标记为完成。

## 2.1 第一块，配置与权重

文件 `inspect_checkpoint.py` 对应视频开头检查 GPT-2 参数的内容。读取本机已缓存的预训练配置和 safetensors 权重，使用 Hugging Face 的现有模型类检查名称、形状、参数值和权重共享。本块还没有开始自己实现 GPT-2。

- 模型代码：规定 LN、Attention、残差、MLP 等按什么顺序运算。
- 配置：规定多少层、几维、几个头等，也包含激活函数、归一化 epsilon 等设置。
- 权重：保存训练得到的具体数值，包括矩阵、偏置、LayerNorm 的缩放和平移参数。
- Tokenizer 文件：规定文本与 ID 的对应方式，和神经网络参数一起保持匹配。

这次读取的 `model.safetensors` 是发布的预训练权重文件。用于恢复训练的 checkpoint 还可以包含优化器状态、步数、随机数状态等；这些内容不一定存在于发布权重中。

## 在 VS Code 运行

打开 `/Users/guqun/micrograd` 项目，选择解释器：

```text
/Users/guqun/micrograd/venv/bin/python
```

打开 `/Users/guqun/micrograd/gpt2_pretrain_lab/inspect_checkpoint.py`，使用“运行 Python 文件”；也可以在“运行和调试”中选择 **GPT-2 2.1：检查配置与权重**，按 F5。

直接运行的等价命令：

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/inspect_checkpoint.py
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
