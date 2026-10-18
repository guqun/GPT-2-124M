# 2.1 第三块：使用自己的 GPT-2 生成文本

衔接 [Notion 学习计划](https://app.notion.com/p/340eaea31414819197bde537d339d0fc) 中的下一知识块：提示词编码 → 前向 → 取最后位置 logits → 贪心 / 采样 → 追加新 ID → 循环与停止。前两个知识块的复习见 [阶段笔记](docs/lesson01-02-notes.md)。

本次目标：能解释每次前向如何生成一个新 Token，并读懂自己的生成循环。2026-10-18 已完成本轮讲解与试跑，用户反馈“大概了解”；记录为初步理解，独立自检待完成。

先打开 [代码与完整流程对照图](docs/diagrams/gpt2-generation-flow.md)，配合 `generate_text.py` 阅读。图与注释共用 `M`（启动与收尾）、`G`（生成循环）、`S`（选择 Token）三组编号，可在源码中直接搜索，例如 `[G3]`。

![文本生成：代码与完整流程对照](docs/diagrams/gpt2-generation-flow.png)

## 1. 从上一块的输出接起

上一块的接口是 `logits = model(input_ids)`：

```text
input_ids  [B, T]          当前已有的 Token ID
logits     [B, T, 50257]   每个输入位置对下一 Token 的词表打分
```

`T` 是当前输入长度。第一个位置利用第一个 Token 预测其后的 Token；最后一个位置利用整个输入前缀，预测接下来应追加的 Token。因此生成时取 `logits[:, -1, :]`，形状变成 `[B, 50257]`。

只取最后位置的输出，并没有丢掉模型对前文的计算。最后位置经过因果注意力，已经可以利用前面所有位置的信息。本实现每次仍将整个前缀送入模型。

本课固定 `B=1`。文中与流程图统一使用课堂示例 `Hello, I'm a language model,`，它实际编码为：

```text
[15496, 11, 314, 1101, 257, 3303, 2746, 11]
```

输入形状是 `[1,8]`。模型前向输出 `[1,8,50257]`，取最后位置后得到 `[1,50257]`，从这 50,257 个分数中选出一个 ID。

## 2. 先读懂贪心生成

打开 [generate_text.py](generate_text.py)，先读 `generate()` 内标为 ①～⑤ 的步骤，再看 `select_next_token()` 中的 `greedy` 分支。下面提取相同逻辑；假设模型、分词器和输入已按脚本的 `main()` 准备好：

```python
ids = input_ids.clone()
model.eval()
with torch.inference_mode():
    for _ in range(max_new_tokens):
        if ids.shape[1] >= model.config.n_positions:
            break
        logits = model(ids)                              # [1,T,V]
        last_logits = logits[:, -1, :]                   # [1,V]
        next_id = last_logits.argmax(dim=-1, keepdim=True)  # [1,1]
        ids = torch.cat((ids, next_id), dim=1)            # [1,T+1]
        if next_id.item() == tokenizer.eos_token_id:
            break
text = tokenizer.decode(ids[0].tolist(), clean_up_tokenization_spaces=False)
```

- `argmax` 返回最大分数所在的词表下标，这个下标就是 Token ID。softmax 保持大小顺序，因此贪心选择可以直接使用 logits。
- `keepdim=True` 保留长度为 1 的维度，让新 ID 的形状是 `[1,1]`，可以和 `[1,T]` 拼接。
- `dim=1` 表示沿序列长度方向追加；batch 数仍是 1。
- 下一轮把增长后的 `ids` 再送入同一个模型。模型参数不变，输入前缀变长；这就是自回归生成。
- `eval()` 关闭本模型的 Dropout，`inference_mode()` 关闭此次推理的自动求导记录，两者用途不同。

下面是本机实际运行的前三步，字符串引号保留了 Token 自带的前导空格：

| 步骤 | 当前输入 | 前向输出 | 选出的新 ID | 解码片段 | 追加后的输入 |
| --- | --- | --- | ---: | --- | --- |
| 1 | `[1,8]` | `[1,8,50257]` | 407 | `' not'` | `[1,9]` |
| 2 | `[1,9]` | `[1,9,50257]` | 257 | `' a'` | `[1,10]` |
| 3 | `[1,10]` | `[1,10,50257]` | 8300 | `' programming'` | `[1,11]` |

第三次前向的输入已经包含前两次选出的 `' not'` 和 `' a'`。这解释了为什么要逐轮计算：后续预测依赖前面实际选出的内容。

## 3. 在 VS Code 运行一次

打开 `generate_text.py`，使用已经选好的项目解释器，点击“运行 Python 文件”。无参数时使用 `main()` 中 `--prompt` 的当前默认值，进行贪心生成，最多新增 20 个 Token，并打印每一步的长度、输出形状和新 ID。下面的命令显式指定课堂提示词，以复现图中的 8 个 Token 示例。

在仓库根目录运行并增加参考验证：

```bash
.venv/bin/python generate_text.py --prompt "Hello, I'm a language model," --check-reference \
  --report .cache/runtime-check/lesson03-greedy.json
```

本机实际输出包含原提示词：

```text
Hello, I'm a language model, not a programming language. I'm a language model. I'm a language model. I'm a
```

停止原因是 `max_new_tokens`：新增了 20 个 Token，最终共 28 个。预算耗尽时可以停在句子中间。

每步选择局部最高分并不保证整段文字最好，也可能进入重复。首步原始分布中，`' not'` 的概率约 16.43%，`' and'` 约 15.16%，`' I'` 约 10.75%；贪心始终选择其中最高的 `' not'`。这些是本次实际模型输出。

## 4. 再把“取最大值”换成“按概率抽样”

读取 `select_next_token()` 的采样分支，依次做四件事：

1. 用正数 `temperature` 除 logits，调整分布的集中程度。
2. 保留分数最高的 `top_k` 个候选，其余分数设为负无穷；本脚本 `top_k=0` 表示使用全词表。
3. 对处理后的分数做 softmax，得到总和为 1 的概率。
4. 用 `torch.multinomial(..., num_samples=1)` 按概率抽出一个 ID，再沿用相同的追加与循环逻辑。

对最后位置的分数向量 $z$，温度为 $\tau>0$ 时，未过滤的概率为：

$$
p_i = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}
$$

小于 1 的温度让分布更集中，大于 1 的温度让分布更平缓。top-k 过滤后，softmax 会在保留的候选之间重新归一化。温度和候选范围改变的是选取规则，模型权重保持不变。

```bash
.venv/bin/python generate_text.py --prompt "Hello, I'm a language model," --strategy sample \
  --temperature 0.8 --top-k 50 --seed 42 \
  --report .cache/runtime-check/lesson03-sample.json
```

本机实际输出：

```text
Hello, I'm a language model, I'm a problem solver.

Yes, I know, like a lot of these people
```

这次第一步抽到了 `' I'`，它在调整后的分布中概率约 14.87%，并非最高概率候选。后续前缀随之改变，之后的每步分布也会改变。

观察第 5、6 步：`' sol'` 和 `'ver'` 两个 Token 合起来组成 `solver`。一个 Token 可以是单词的一部分、带空格的片段、标点或换行。GPT-2 使用字节级分词，单个 Token 也可能只包含一个字符的部分字节；最终文本应解码完整 ID 序列，不能普遍用单个 Token 的显示文字直接拼接。

本脚本使用独立随机数生成器；在相同环境、输入与参数下，固定 seed 便于复现实验。不同设备、库版本或采样实现之间不保证逐字一致。贪心分支不使用 temperature、top-k 和 seed 来作选择。

## 5. 什么情况下停止

| 停止原因 | 本课含义 |
| --- | --- |
| `max_new_tokens` | 已达到新增 Token 数量上限，提示词不计入该预算 |
| `eos` | 新生成的 ID 是 GPT-2 的结束符 `50256`，该 ID 已追加到结果中 |
| `context_limit` | 达到本教学实现设置的完整结果长度上限 `1024` |

GPT-2 的位置向量表支持 1024 个输入位置。本教学循环把最终结果也限制在 1024 个 Token 内，保证结果可以再次直接作为输入；这是本脚本的边界约定。输入已经长达 1024 时，本脚本会直接停止。没有进行滑动截断或位置编号重排。

本课每一步都重新计算完整前缀，暂不使用 KV cache，只处理单条非空输入。后续可在掌握循环后讨论缓存和批量生成。

## 6. 已验证的范围

2026-10-17，本机 Python 3.9.6 / PyTorch 2.2.2 / Transformers 4.57.6，CPU / FP32：

- 使用课堂固定提示词生成 20 个新 Token，独立模型的完整贪心 ID 序列与 Hugging Face `generate()` 一致；两边关闭 KV cache。
- 已实际运行 `temperature=0.8, top_k=50, seed=42` 的采样实验。
- 5 个行为测试通过，覆盖前缀回传、最后位置选择、EOS、长度预算、上下文上限、零预算、采样候选范围、温度、固定随机种子及非法输入。

本机完整记录保存在 `.cache/runtime-check/lesson03-greedy.json` 和 `lesson03-sample.json`。默认不指定 `--report` 时保存到 `lesson03-report.json`，后续运行会覆盖同一路径的报告。缓存目录不提交 Git。

```bash
.venv/bin/python -m unittest discover -s tests -v
```

参考对照验证的是这个案例的生成 ID 一致性；采样结果用于观察生成机制，未要求与另一个实现逐字一致。本课使用预训练参数，训练循环留在下一知识块。

## 7. 本轮自检

1. 提示词有 8 个 Token，连续新增 3 个 Token 且没有提前停止：三次前向的输入形状分别是什么？最终结果多长？
2. 为什么对 `[1,8,50257]` 的所有 8 个位置同时取 argmax，不能直接得到接下来的 8 个新 Token？
3. 第一轮生成 `' not'` 后，第二轮模型收到的输入是什么？哪一个变量变了，模型参数有没有变化？
4. `top_k=50`、`num_samples=1` 分别控制什么？采样为什么可能选到概率第二或第三高的 Token？

能独立解释这些问题，再记录本知识块的理解情况。2.1 整节仍在进行中。

## 分词器准备与来源

当前机器已经准备好分词器，可以直接运行。新机器先按 README 安装依赖与权重；已有权重但缺少分词器时，补充同版本的文件：

```bash
.venv/bin/hf download openai-community/gpt2 tokenizer.json \
  --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --local-dir .cache/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e
```

`tokenizer.json` 约 1.4 MB，与权重来自相同固定版本。生成脚本离线读取本地三个文件：`config.json`、`model.safetensors`、`tokenizer.json`。

课程的生成循环可对照 [Karpathy build-nanogpt 源码](https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py)；贪心与采样策略的参考说明见 [Hugging Face 生成策略文档](https://huggingface.co/docs/transformers/v4.57.1/en/generation_strategies)。本课增加逐步形状日志、明确停止原因和独立行为测试，并先用单条贪心生成建立直观理解。
