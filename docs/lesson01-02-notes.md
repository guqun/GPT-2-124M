# GPT-2 预训练工程（一）：检查点、独立实现与前向对照

> **本篇定位**：阶段二 · 单元 2 · 2.1 的前两个知识块。已经完成配置与权重检查、独立模型的前向对照，并通过源码讨论理解调用关系；2.1 整节仍在进行中，下一块是用自己的模型生成文本。
>
> **这次建立的关键联系**：配置规定结构，权重提供数值，代码规定运算顺序；调用模型对象后，数据才经过各层变成 logits。能够加载参数，并不自动证明自己的计算正确，所以还要与参考实现逐层对照。

归档日期：2026-10-16。掌握程度保留为“理解中”；学习讨论与实验通过，不等于已经独立重写或完成整节训练。

**承接**：[GPT 训练与推理全流程复习](https://app.notion.com/p/3f9eaea3141481d19044e32c68f97520)。上一篇保留训练、推理的两份交互动画；本篇把其中的前向路线落到真实 124M 模型代码。

**来源标记**：**「课程主线」**来自 Karpathy 的 [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) 与 [build-nanogpt](https://github.com/karpathy/build-nanogpt)；**「本地源码」**指本项目及当前安装的 PyTorch / Transformers；**「教学补充」**是为阅读代码增加的解释、hook 与多案例对照；**「实际记录」**来自已经保存的实验报告。本次归档核对报告和源文件，没有重新执行训练或生成。

## 一、先用一张图把调用与计算对上

![GPT-2：代码调用与计算逻辑对照](diagrams/gpt2-forward-call-map.png)

**读图顺序**：顶部先分清“创建并加载模型”和“触发计算”；中间左列看代码、右列看目的与输出；在 Block 中追两次调用及返回；最后看 Attention 和 MLP 的内部展开。

最容易混淆的关系是：**GPT 调用 Block；Block 分别调用 Attention 和 MLP；两个分支各自返回 Block，由 Block 完成残差相加。Attention 不负责调用 MLP。**

配套：[原图与维护说明](diagrams/gpt2-forward-call-map.md)。图中 Block 的临时变量用于展开嵌套表达式，已经对应到源码注释；可执行源码仍是两行残差表达式。

## 二、这次到底在做什么，对照的又是谁

**课程主线**：先搭出与 GPT-2 兼容的网络，载入公开的预训练参数，确认前向计算正确，再继续生成和训练。当前阶段复用了别人已经训练好的数值。

| 角色 | 本次具体对象 | 负责什么 |
|---|---|---|
| 模型架构 | GPT-2 124M | 规定 Embedding、12 个 Block、最终 LN 和输出层的组织方式 |
| 已训练参数 | 本地 GPT-2 checkpoint | 提供各张参数表中的具体数值 |
| 我们的实现 | `gpt2_model.py` 中的 `GPT` | 自己组织前向计算、分头、mask、残差及权重映射 |
| 对照实现 | Transformers 的 `GPT2LMHeadModel` | 用另一份现成实现计算同一输入，作为参考 |
| 数值工具 | PyTorch | 执行 Tensor 运算、标准层、设备管理及比较 |
| 实验程序 | `compare_forward.py` | 安排相同输入、收集中间结果、计算误差并保存报告 |

**“自己的计算”指自己写网络的组合与数据流。** 本项目仍然调用 `nn.Linear`、`nn.LayerNorm`、矩阵乘法和 softmax，没有重新实现矩阵运算库；自己的前向没有调用 Hugging Face 的 `GPT2Block`。

**预训练是什么意思？** GPT-2 的原始训练方先用大量文本训练“根据前缀预测下一 Token”的模型，得到参数。我们现在加载这些已经学到的参数，称为加载预训练模型。本课并未在本机完成那次大规模训练，也没有进行微调。

## 三、模型代码、配置、权重、Tokenizer 各保存什么

| 材料 | 保存的内容 | 不包含什么 |
|---|---|---|
| 模型 Python 类 | 构造哪些层，`forward` 按什么顺序调用 | 不包含这份 checkpoint 的全部训练后数值 |
| `config.json` | 层数、特征维度、头数、词表大小、激活与 Dropout 等规格 | 不保存每个权重的实际数值 |
| `model.safetensors` | 按名称组织的 Tensor：dtype、shape 和二进制数据 | 不保存可执行的 `forward` 函数 |
| `generation_config.json` | 生成时使用的默认选项，例如 BOS / EOS | 不决定 QKV 权重，也不会自行运行采样循环 |
| Tokenizer 文件 | 文本如何切分、词条与 ID 如何对应、特殊标记 | 不包含 Transformer 的参数 |

**只拿到权重文件还不够**：需要兼容的网络结构去使用它，还需要匹配的 Tokenizer 来解释输入输出 ID。只拿到配置也不够：可以建出同样大小的网络，却不知道预训练权重具体是多少。

### 1. 这份配置的核心字段

| 字段 | 当前值 | 具体作用 |
|---|---|---|
| `vocab_size` | 50257 | 输入词表行数，也是每个位置输出分数的数量 |
| `n_embd` | 768 | 每个位置的特征宽度、残差主干宽度 |
| `n_layer` | 12 | 顺序执行 12 个参数独立的 Block |
| `n_head` | 12 | 每层 12 个注意力头，每头 `768/12=64` 维 |
| `n_positions` | 1024 | 学习式位置表有 1024 行，对应位置 0～1023 |
| `n_ctx` | 1024 | 旧配置保留的上下文长度字段；当前安装版本的相关尺寸由 `n_positions` 及其别名控制 |
| `activation_function` | `gelu_new` | 当前实现使用与之对应的 tanh 近似 GELU |
| `layer_norm_epsilon` | `1e-5` | LayerNorm 分母中的稳定项 |
| `initializer_range` | 0.02 | 基础正态初始化的标准差；加载预训练参数后，初值被覆盖 |
| `embd_pdrop` | 0.1 | Embedding 相加后的训练态 Dropout |
| `attn_pdrop` | 0.1 | 注意力权重上的训练态 Dropout |
| `resid_pdrop` | 0.1 | Attention / MLP 分支输出上的训练态 Dropout |
| `bos_token_id` / `eos_token_id` | 50256 | GPT-2 配置里的开始 / 结束标记编号 |

**本地源码限定**：`n_ctx` 与 `n_positions` 的说明对应当前 Transformers 4.57.6。不要把“修改 `n_ctx` 就能扩展模型上下文”当成结论；位置表大小和模型训练方式仍需匹配。

`architectures` 和 `model_type` 帮助描述、识别架构；本次明确调用 `GPT2LMHeadModel` 类。`summary_*` 是其他头或历史任务相关设置，当前普通语言模型前向没有依赖这组摘要头字段。`task_specific_params` 中的生成选项属于任务配置，不是训练参数。

### 2. generation_config.json 为什么这么短

当前文件包含 `bos_token_id=50256`、`eos_token_id=50256`、`transformers_version` 和 `_from_model_config=true`。它记录了生成配置及其来源元数据；没写出的选项可能使用库的默认值，也可以由生成调用显式指定。

本次 `compare_forward.py` 没有调用生成接口，因此这个文件不参与“逐层 logits 对照”。它存在，不代表我们已经运行了文本生成。

### 3. REVISION 和缓存目录是什么

`REVISION` 固定了一个 Hugging Face 模型仓库快照版本，便于反复使用同一套文件。缓存的是模型配置、权重及已经下载的其他仓库文件，不是某次输入算出的 Q/K/V。

本机快照绝对路径：

```text
/Users/guqun/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e
```

`Path.home()` 取得用户主目录；连续 `/` 是 `pathlib.Path` 的路径拼接，不是数学除法。`local_files_only=True` 要求这次加载使用已有本地文件；缺失时不会自动在线补齐。

## 四、二进制权重如何保存，再变回模型参数

**格式事实**：safetensors 的开头是 8 字节小端整数 N，随后是 N 字节的 UTF-8 JSON 头，再后面是 Tensor 数据区。JSON 头保存每个 Tensor 的名称、dtype、shape 和数据区间。详见 [Safetensors 格式说明](https://github.com/safetensors/safetensors#format)。

```text
文件开头
├─ 8 字节：JSON 头的长度 N
├─ N 字节：名称、类型、形状、偏移等元信息
└─ Tensor 数据区：各张量的二进制数值
```

**本地文件实际检查**：当前文件的头长度为 `14283` 字节。此前追问的条目是原始文件中的 `wte.weight`：

```json
{
  "dtype": "F32",
  "shape": [50257, 768],
  "data_offsets": [287209472, 441598976]
}
```

`data_offsets` 是相对于 **Tensor 数据区起点** 的字节偏移，采用左闭右开区间。该张量占用：

```text
441598976 - 287209472 = 154389504 字节
50257 × 768 × 4       = 154389504 字节
```

F32 每个数占 4 字节，两种算法一致。若要计算文件中的绝对偏移，还要加上 `8 + N`。这些数是存储位置，不是 Token ID，也不是参数的数值。

加载库按元信息定位数据、还原 Tensor，再根据模型的命名规则放到相应参数中。原始文件的 `wte.weight` 与 HF 模型对象中的 `transformer.wte.weight` 名称可以经过加载适配；不能假定磁盘键名总与包装后的模型键名逐字相同。

**保存时的逆向关系**：收集待保存的张量，写出名称 / dtype / shape / offsets 头信息与连续数据区。共享权重需要保存工具正确处理别名；模型恢复时还要重建共享关系。普通发布权重与“可恢复训练的检查点”也不同：后者通常还需优化器、步数等训练状态。本轮只读取现成权重，没有进行保存后恢复训练的实验。

## 五、from_pretrained 返回的两个对象是什么

实际使用的写法：

```python
reference, loading_info = GPT2LMHeadModel.from_pretrained(
    str(model_dir),
    local_files_only=True,
    use_safetensors=True,
    attn_implementation="eager",
    output_loading_info=True,
)
```

- `GPT2LMHeadModel` 是 Transformers 中定义的模型类，本机定义位于 `venv/lib/python3.9/site-packages/transformers/models/gpt2/modeling_gpt2.py`。
- `from_pretrained(...)` 是类提供的加载入口：按配置构建模型，并载入预训练参数。
- `reference` 是已经创建并加载的模型对象，可以调用它做前向。
- `loading_info` 是加载诊断字典，包含缺失键、多余键、形状不匹配和错误消息。
- 因为指定了 `output_loading_info=True`，当前调用返回两个值供解包；不要把这理解成所有 `from_pretrained()` 调用都固定返回两个值。

**实际记录**：第一块的 `missing_keys`、`unexpected_keys`、`mismatched_keys`、`error_msgs` 都为空。它证明这次参数加载完整；前向是否对齐，要由第二块的实验继续检查。

## 六、nn、Module、GPT、own 分别是谁

```python
from torch import nn

class GPT(nn.Module):
    ...

own = GPT(config)
```

| 名称 | Python 中的角色 | 对熟悉 Java / C++ 的读者可以怎样理解 |
|---|---|---|
| `nn` | `torch.nn` 模块 | 提供神经网络工具的命名空间 |
| `nn.Module` | 基类 | 管理子模块、参数、设备与调用机制 |
| `GPT` | 本项目定义的子类 | `class GPT(nn.Module)` 表示继承 |
| `own` / `model` | 普通变量名，引用模型对象 | 变量名不是框架保留字，具体类型取决于创建了谁 |
| `self` | 当前对象 | 类似 `this`，但 Python 方法显式写出这个参数 |

`__init__` 中的 `super().__init__()` 初始化 Module 的管理机制；把层赋给 `self.xxx`，以及使用 `ModuleList` / `ModuleDict`，可让参数遍历、设备移动与模式切换递归找到子模块。

**Tensor 与 PyTorch**：Tensor 是 PyTorch 的核心多维数组类型，携带形状、数据类型、设备等信息。可训练权重通常由 `nn.Parameter` 表示，它是 Tensor 的子类；输入 ID、位置编号和注意力 mask 也都是 Tensor，但并非所有 Tensor 都是可训练参数。

## 七、创建模型时会不会执行 forward

下面是当前实验的关键次序，省略了文件检查和统计打印：

```python
own = GPT(config).cpu().float()  # 创建结构及初始参数，统一设备与精度
own.load_hf_weights(reference.state_dict())  # 覆盖为预训练数值
own.eval()                      # 切换评估行为，关闭 Dropout 随机性

with torch.inference_mode():    # 本次不记录求导过程
    own_trace = {}
    logits = own(input_ids, trace=own_trace)  # 这里才触发前向
```

**本地源码**：`own(input_ids, ...)` 使用继承自 `nn.Module` 的对象调用机制，最终进入这个 GPT 对象的 `forward(...)`。简化关系：

```text
调用对象 own(...)
→ nn.Module 的 __call__ 调度（包含 hook 等机制）
→ own.forward(...)
→ 返回 logits
```

`GPT(config)` 的构造过程会创建层并初始化参数，却没有输入文本可供执行完整前向。`.cpu()` 与 `.float()` 只设置模型设备和浮点精度，也不会触发文本计算。

正常使用 `own(...)`、`block(...)`、`self.mlp(...)`，让 Module 的调用机制管理 hook；直接调用 `.forward(...)` 会绕过该对象的调用包装。

**eval 与 inference_mode 的区别**：`eval()` 改变 Dropout 等模块的行为；`inference_mode()` 控制自动求导记录及相关开销。两者职责不同，`eval()` 本身不会关闭梯度记录，也不会删除模型参数。

## 八、GPT.forward 如何对应我们学过的逻辑

统一记号：B 是 batch 内序列条数，T 是本次输入序列长度，C 是每个位置的特征维数；本模型 `C=768`。`[B,T]` 中存整数 ID，Embedding 后才变成浮点特征。

| 逻辑环节 | 对应调用 | 输入 → 输出 |
|---|---|---|
| 产生位置编号 | `torch.arange(length, device=input_ids.device)` | 长度 T → `[T]` |
| Token 向量查表 | `self.transformer.wte(input_ids)` | `[B,T]` → `[B,T,768]` |
| 位置向量查表 | `self.transformer.wpe(positions)` | `[T]` → `[T,768]` |
| 两者相加 | `token_vectors + position_vectors` | 广播后得到 `[B,T,768]` |
| Embedding Dropout | `self.transformer.drop(x)` | 形状不变，本次 eval 下原样通过 |
| 顺序经过 12 层 | 循环内的 `x = block(x)` | 每层 `[B,T,768]` → `[B,T,768]` |
| 最终归一化 | `self.transformer.ln_f(x)` | `[B,T,768]` → 同形状 |
| 全词表打分 | `self.lm_head(x)` | `[B,T,768]` → `[B,T,50257]` |
| 返回计算结果 | `return logits` | 返回刚才的张量，不再做一次额外的 logit 计算 |

`wte` 沿用 GPT-2 的 Token embedding 命名，`wpe` 是 Position embedding。两张表的参数都已经由 checkpoint 载入，但索引是本次输入决定的。

### 1. position 到底是不是预训练好的

```python
positions = torch.arange(length, device=input_ids.device)
position_vectors = self.transformer.wpe(positions)
```

这两行产生的是不同东西：

- `positions` 是确定的整数编号，例如 `[0,1,2,3]`，每次按输入长度生成。
- `position_vectors` 是拿这些编号去 `wpe.weight` 中查出的预训练向量，形状 `[4,768]`。

`device` 表示 Tensor 所在的计算设备。本次是 CPU；写 `device=input_ids.device` 让新位置编号与输入放在同一设备，模型本身也要放在兼容设备上。device 不决定向量是否随机。

若从零训练，wte / wpe 起初是随机参数，随后会更新；当前实验先创建它们，再用 checkpoint 覆盖。后续换输入不会重新初始化权重。

### 2. h 为什么不是 block

本项目保留 HF 的命名，便于对齐参数键：

```python
"h": nn.ModuleList([Block(config) for _ in range(config.n_layer)])
```

`h` 只是保存所有 Block 的属性名，没有额外数学操作。`self.transformer.h` 是容器，`self.transformer.h[0]` 才是第一个 Block。每次 `Block(config)` 都创建新对象，因此 12 层结构相同、参数独立。

```python
for index, block in enumerate(self.transformer.h, start=1):
    x = block(x)
    if trace is not None:
        trace[f"block_{index:02d}"] = x
```

`enumerate` 是 Python 内置函数，每轮给出“编号、元素”。第一轮为 `(1, h[0])`，第二轮为 `(2, h[1])`；`start=1` 只改编号，不跳过第一个元素。`02d` 将整数显示为两位，例如 `block_01`。

`x = block(x)` 把当前 x 交给该对象的 forward，再让变量 x 引用新结果。不同层仍输出 768 维，但向量内容逐层改变。

## 九、GPT、Block、Attention、MLP 的 forward 如何相互调用

**函数名相同，不意味着执行同一个函数体。** 调用哪个对象，就进入那个对象对应的 forward；层与层之间靠返回的 Tensor 衔接。

```text
compare_forward.main()
└─ own(input_ids, trace=own_trace)
   └─ GPT.forward()
      ├─ wte(...) + wpe(...) → Dropout
      ├─ 第 1 个 block(x) → Block.forward(x)
      │  ├─ ln_1(x) → PyTorch LayerNorm.forward
      │  ├─ attn(...) → CausalSelfAttention.forward
      │  │  └─ 返回注意力分支更新量
      │  ├─ 加回原始 x，得到 r
      │  ├─ ln_2(r) → PyTorch LayerNorm.forward
      │  ├─ mlp(...) → MLP.forward
      │  │  └─ 返回 MLP 分支更新量
      │  └─ 加回 r；返回 GPT.forward
      ├─ 第 2～12 个 Block 重复相同结构
      ├─ ln_f(x)
      ├─ lm_head(x)
      └─ 返回 logits 给 main()
```

### 1. Block 中的两行，按执行顺序展开

实际源码：

```python
r = x + self.attn(self.ln_1(x))
return r + self.mlp(self.ln_2(r))
```

用于理解的等价展开：

```python
attn_input = self.ln_1(x)
attn_update = self.attn(attn_input)
r = x + attn_update

mlp_input = self.ln_2(r)
mlp_update = self.mlp(mlp_input)
y = r + mlp_update
return y
```

第一次加回 LN1 之前的 x，第二次加回 LN2 之前的 r。两个分支返回的是更新量，完整 Block 输出是 `y`；“残差连接”描述相加结构，不是把 MLP 算出的结果一概改名为“残值”。

### 2. Attention 返回之前已经做了什么

| 计算 | 形状与目的 |
|---|---|
| `self.c_attn(x)`，再 `split` | `[B,T,768] → [B,T,2304]`，拆成三份 Q/K/V，各 `[B,T,768]` |
| `view + transpose` | 每份变为 `[B,12,T,64]`；每个头都处理所有位置 |
| `q @ k.transpose(-2,-1) / sqrt(64)` | 得到 `[B,12,T,T]` 的位置匹配分数 |
| 因果 mask、softmax | 未来位置分数设为负无穷，读取权重为 0；每行沿上下文位置归一化 |
| `attn_dropout(weights) @ v` | 汇总内容，得到 `[B,12,T,64]` |
| 拼接各头 | `[B,12,T,64] → [B,T,768]` |
| `c_proj` 与 `resid_dropout` | 学习组合各头特征，返回 `[B,T,768]` 的更新量 |

Q/K/V 是当前输入经过参数投影后得到的激活值；`c_attn.weight` 才是参数。多头切分的是投影后的特征维，不是把输入序列分给不同头。Attention 的 softmax 是“对上下文位置分配读取权重”，不是“对整个词表选下一 Token”。

Attention 内部已做输出投影和分支 Dropout；**返回到 Block 后才执行第一次残差相加**。本次 eval 下 Dropout 原样通过。

### 3. MLP 的两个线性层各管什么

```python
expanded = self.c_fc(x)                         # 768 → 3072
activated = F.gelu(expanded, approximate="tanh") # 3072 → 3072
return self.dropout(self.c_proj(activated))     # 3072 → 768
```

输入来自 LN2。第一层把已有特征组合成更宽的中间表示；GELU 引入逐元素非线性；第二层组合这些中间特征，回到残差所需的 768 维。若没有非线性，两次仿射变换可合并成一次。

MLP 对各位置分别计算，所有位置复用这一层的同一套参数，不直接读取其他位置。返回 Block 后，再加回 r。

### 4. lm_head 的输出就是 logits

本项目将 `lm_head.weight` 与 `wte.weight` 指向同一个 Parameter E：

```python
self.lm_head.weight = self.transformer.wte.weight
```

最后的计算是 `logits = x @ E.T`，形状 `[B,T,50257]`。`logits[b,t,j]` 给词表 ID j 打分，表示读到位置 t 后对下一 Token 的预测分数。本函数还没有做词表 softmax、采样或把新 ID 接回输入。

T 是已有输入的位置数，不是一次生成了 T 个未来 Token。

## 十、load_hf_weights 如何把参考参数装进自己的模型

```python
def load_hf_weights(self, source: Mapping[str, torch.Tensor]) -> Dict[str, int]:
    ...
```

**输入**：`reference.state_dict()`，即名称到 Tensor 的映射。**副作用**：把相应数值写入 own 的参数。**返回值**：加载统计字典，含命名条目数量与转置数量；不是返回另一个模型。

类型注解中的 `Mapping[str, torch.Tensor]` 说明键为字符串、值为 Tensor；`Dict[str, int]` 说明返回的统计值为整数。注解本身不会完成加载或自动检查全部数据。

本实现按以下顺序工作：

1. 取得自己的 `state_dict`，过滤参考实现某些版本中的旧固定 mask 条目。
2. 核对缺失键与多余键，检查 Embedding / 输出表符合共享约束。
3. 对四类线性权重转置，并核对所有形状。
4. 在 `torch.no_grad()` 中通过 `copy_` 原地写入已有参数存储。
5. 返回统计，供实验报告记录。

### 为什么需要转置

HF GPT-2 的 Conv1D 包装在这里执行线性投影，保存 `W_hf[输入维,输出维]`，算 `x @ W_hf + b`。PyTorch Linear 保存 `W_linear[输出维,输入维]`，算 `x @ W_linear.T + b`。为了得到相同结果，要设置 `W_linear = W_hf.T`。

| 权重 | HF 形状 | 本项目 Linear 形状 |
|---|---|---|
| Attention QKV | `[768,2304]` | `[2304,768]` |
| Attention 输出投影 | `[768,768]` | `[768,768]`，仍须转置 |
| MLP 扩展层 | `[768,3072]` | `[3072,768]` |
| MLP 压回层 | `[3072,768]` | `[768,3072]` |

**尺寸相同不表示方向相同**，方阵也不能漏转。Embedding、LayerNorm 和偏置保持原方向。

默认 `state_dict()` 中的 Tensor 与模型参数共享底层存储，所以 `target[name].copy_(value)` 能写回模型。若只写 `target[name] = value`，只是重新绑定字典条目，不能替代这里的原地加载。

**实际记录**：复制 149 个命名状态条目，转置 `12×4=48` 个矩阵。149 不是独立参数对象数，更不是标量参数总数；共享表可能对应两个名称。

## 十一、capture_reference 为什么能看到中间结果

**教学补充**：只比较最终 logits，失败时很难定位。因此对照程序在相同观察点记录两条路径的 Tensor。

| 路径 | 怎样记录 | 得到什么 |
|---|---|---|
| 自己的模型 | 在 `GPT.forward` 中填写传入的 `trace` 字典 | `embedding`、12 个 Block、`final_norm`、`logits` |
| HF 参考模型 | `capture_reference()` 给模块注册 forward hook | 相同名称的观察结果 |

`capture_reference` 的输入是已 eval 的参考模型和 `[B,T]` 输入 ID；它注册 hook，执行一次参考前向，最后在 `finally` 中移除 hook，返回名称到 Tensor 的字典。

```python
def capture(module, inputs, output):
    trace[name] = output[0] if isinstance(output, tuple) else output
```

模块算完后，PyTorch 把模块、输入和输出交给回调。某些参考模块返回 tuple，第一项是 hidden states；其他层直接返回 Tensor。这里按真实类型选择要记录的值。回调隐式返回 `None`，因此保留原输出；hook 本身也可以通过返回值替换输出，但本实验不这样做。

`recorder(name)` 用闭包记住观察点名称；并不是每次 forward 都重新训练这一层。`capture_reference()` 只负责采集，误差比较发生在外层 `comparison()`。

### trace 是不是 GPT 的第二个返回值

不是。当前 `GPT.forward` 始终 `return logits`。`own_trace` 是调用方传入的可变字典，函数在同一个对象中写入条目，所以调用后外面也能读取。

```python
own_trace = {}
logits = own(input_ids, trace=own_trace)
# 返回值是 logits；中间结果已经写入 own_trace。
```

保存的是 Tensor 引用，后面 `x = block(x)` 绑定到新张量，不会自动修改此前记录的旧张量。本次在 inference_mode 中观察；训练时长期保留这类引用可能连计算图一起保留。

## 十二、阅读代码时用到的 Python / PyTorch 小工具

### 1. tuple 是什么，在哪里定义

tuple 是 Python 内置类型，中文叫“元组”，所以文件里不需要再写一个 tuple 类。它是有顺序、长度固定、槽位不能重新赋值的容器，元素可以是不同类型；元素若引用可变对象，对象自身仍可能被修改。

```python
pair = ("block_01", tensor)  # 两个不同类型的元素
one = (tensor,)              # 一项元组需要逗号
a, b = pair                 # 解包
```

`isinstance(output, tuple)` 才是在运行时判断类型。括号也可能只是分组，不能见到括号就认定是元组。`return a, b` 通常形成元组供调用方解包；当前自己的 GPT 返回的是一个 Tensor。

### 2. detach、cpu、tolist 分别做什么

```python
row = model.transformer.wte.weight[args.token_id, :8].detach().cpu().tolist()
```

| 操作 | 本句的含义 |
|---|---|
| `weight[args.token_id, :8]` | 取指定 ID 对应行的前 8 个标量 |
| `.detach()` | 取得与自动求导记录分离的 Tensor；不是独立数据拷贝，仍可共享存储 |
| `.cpu()` | 让结果位于 CPU；若本就在 CPU，通常不必搬运 |
| `.tolist()` | 转为普通 Python 列表，便于打印或 JSON 保存 |

`detach()` 不会搬到 CPU，`.cpu()` 也不是关闭求导；两者解决不同问题。需要独立数据副本时应明确使用 `clone()` 等复制操作。

### 3. 本项目用到的常见工具

| 工具 | 职责 |
|---|---|
| `nn.Embedding / Linear / LayerNorm / Dropout` | 标准神经网络层 |
| `nn.ModuleList / ModuleDict` | 保存并注册子模块，让框架递归管理 |
| `model.apply(fn)` | 递归对模块执行函数，本项目用于初始化 |
| `parameters() / named_parameters()` | 遍历参数，后者同时返回名称 |
| `state_dict()` | 提供参数与持久化 buffer 的命名映射 |
| `register_buffer(...)` | 注册随模型管理的非参数 Tensor；本例 mask 使用 `persistent=False`，不进入 state_dict |
| `Tensor.numel()` | 数标量元素个数，不是数层数 |
| `Tensor.t()` / `copy_()` | 二维转置 / 原地写入数值 |
| `view / transpose / contiguous` | 调整形状或维度顺序，必要时整理存储 |
| `F.softmax / F.gelu` | 执行对应数学变换 |
| `register_forward_hook` | 模块计算结束后观察输入输出 |
| `torch.allclose` | 按绝对与相对容差逐元素比较 |

Python 系统学习仍放在 [Python 迁移教材](../python_foundations/README.md)，本篇仅收录影响模型阅读的语法点。

## 十三、已有实验究竟验证到了哪里

**实际记录**：来自 `artifacts/lesson01-report.json` 与 `artifacts/lesson02-report.json`。本次读取并核对了这些报告；没有把讨论中的手算值当作运行值。

运行环境：Python 3.9.6、PyTorch 2.2.2、Transformers 4.57.6；CPU / FP32。两边使用相同预训练参数、相同输入 ID，均处于 eval；HF 使用 eager Attention、关闭 KV cache。对照在 inference_mode 中执行。

| 案例 | 输入形状 | logits 形状 | logits 最大绝对误差 | 逐层结果 |
|---|---|---|---|---|
| 两段各四个 Token | `[2,4]` | `[2,4,50257]` | 0 | 15/15 通过 |
| 单个 Token | `[1,1]` | `[1,1,50257]` | `3.814697265625e-5` | 15/15 通过 |
| 同一 ID 出现在五个位置 | `[1,5]` | `[1,5,50257]` | 0 | 15/15 通过 |

每个案例的 15 个观察点是 Embedding、12 个 Block、最终 LN 和 logits，共 45 次逐层比较。实际通过规则要求形状相同、数值有限，且每个元素满足：

```text
abs(自己 - 参考) <= 1e-4 + 1e-4 × abs(参考)
```

这不是“最大误差必须小于 1e-4”。例如单 Token 案例某些 Block 的最大绝对误差大于 1e-4，但相应元素仍可满足相对容差。最高分 ID 相同只是辅助观察，不能替代完整 logits 比较。

**因果性检查**：只改变每段最后一个输入 ID，比较前面三个位置的全部词表分数，最大变化为 0。最后位置的分数允许变化。这与“每一层只读取当前位置及之前位置”一致。

| 参数组 | 独立标量数量 |
|---|---:|
| Token Embedding | 38,597,376 |
| Position Embedding | 786,432 |
| 12 个 Block | 85,054,464 |
| 最终 LayerNorm | 1,536 |
| 合计 | **124,439,808** |

Embedding 与 lm_head 是同一个 Parameter，统计一次；首两个 Block 的 QKV 权重对象独立，源码的逐次构造也表明每个 Block 各自创建参数。报告记录 `all_passed=true`、`training_ran=false`、`generation_ran=false`。

**验证边界**：这些结果支持当前短输入案例的前向实现与 HF 一致，尚未验证训练梯度、优化器更新、padding、KV cache 或多轮生成，更不能据此宣布“已复现完整预训练”。

## 十四、以后复习时从哪些文件进入

项目根目录：`/Users/guqun/micrograd/gpt2_pretrain_lab`。

| 文件 | 阅读目的 |
|---|---|
| `inspect_checkpoint.py` | 看配置、参数名称、形状、数值和共享关系 |
| `gpt2_model.py` | 先 GPT.forward，再 Block.forward，最后进入两个分支 |
| `compare_forward.py` | 看两边如何加载、观察和比较 |
| `lesson02.md` | 本次操作顺序与完成标准 |
| `docs/diagrams/gpt2-forward-call-map.png` | 快速恢复调用与逻辑的对应关系 |
| `artifacts/lesson01-report.json` | 已有权重检查记录 |
| `artifacts/lesson02-report.json` | 已有前向误差及因果性记录 |
| `python_foundations/` | 面向 Java / C++ / Dart 程序员的 Python 知识补充 |

在 VS Code 中选择解释器 `/Users/guqun/micrograd/venv/bin/python`，再运行文件。另一个 Python 安装即使版本更高，也可能没有当前课程的 torch 等依赖。

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/inspect_checkpoint.py
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/compare_forward.py
```

第一条读取配置与权重；第二条运行前向对照并更新报告。二者都使用已有本地 checkpoint。

## 十五、自检与后续学习

1. `config.json`、权重文件、模型类分别决定什么？
2. `GPT(config)` 和 `own(input_ids)` 的作用有什么不同？
3. `nn`、`nn.Module`、`GPT`、`own` 分别是模块、类还是对象引用？
4. 位置编号 `[0,1,2,3]` 与位置向量 `[4,768]` 各从哪里来？
5. `enumerate(h, start=1)` 会跳过 `h[0]` 吗？
6. `self.attn(...)` 返回之后，哪个函数继续执行？MLP 由谁调用？
7. 两次残差分别加回哪个张量？
8. `[2,4,50257]` 中的 4 为什么不是未来四个新 Token？
9. 为什么 HF 的 `[768,768]` 输出投影也要转置？
10. `capture_reference` 为什么需要判断 output 是否为 tuple？
11. `eval()` 能否替代 `inference_mode()`？`detach()` 能否替代 `.cpu()`？
12. 当前对照通过，为什么仍不能把整节 2.1 标为完成？

<details>
<summary>简要核对答案</summary>

1. 配置决定规格，权重给出数值，模型类规定运算和连接。
2. 前者构造并初始化；后者经 Module 调用机制执行前向。
3. nn 是模块，Module 是基类，GPT 是子类，own 引用 GPT 实例。
4. 编号由 arange 按长度生成，向量从已经载入参数的 wpe 表查得。
5. 不会。编号从 1 开始，元素仍从 h[0] 开始。
6. 返回 Block.forward；Block 完成第一次相加后再调用 MLP。
7. 第一次加回 LN1 前的 x；第二次加回 LN2 前的 r。
8. 它是四个已有输入位置，每个位置预测自己的下一 Token。
9. 存储方向不同；方阵尺寸一样也不能省略转置。
10. 参考模块的返回结构可能不同，元组第一项才是需要比较的 hidden states。
11. 都不能替代；前一组分别控制模块行为与求导记录，后一组分别控制求导关系与设备。
12. 只核对了加载和前向；生成、训练循环、单批次过拟合、数据加载和初始化专题还在后面。

</details>

**下一知识块**：让自己的 GPT-2 生成文本。提示词编码 → 前向 logits → 取最后位置 → 贪心或采样选一个 ID → 追加到输入 → 再次前向 → 满足结束条件后解码。之后再进入训练循环、单批次过拟合、简易数据加载器及初始化专题。

**进度口径**：2.1 前两个知识块已学习并归档，2.1 整段仍在进行中；单元 2 完整段数仍为 0/5，阶段二已完成 4/32 段。Python 补充不另计主课程段数。
