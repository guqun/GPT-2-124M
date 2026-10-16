## 13 · 实战：按对象与契约读懂检查脚本

**迁移目标：**把已有工程阅读能力直接用于 [inspect_checkpoint.py](../inspect_checkpoint.py)：追踪对象类型、状态来源、返回契约和副作用。此处核对真实源码，不重新讲模型前向原理。

### 对象流先于逐行翻译

| 对象 | 类型/角色 | 谁创建它 | 它不包含什么 |
|---|---|---|---|
| `args` | argparse Namespace | CLI 解析 | 模型参数 |
| `model_dir` | Path | 参数转换与路径解析 | 目录内文件内容 |
| `config` | dict | JSON 解析 | 训练后的完整权重 |
| `model` | GPT2LMHeadModel 实例 | 类方法工厂 | Tokenizer 的完整预处理流程 |
| `loading_info` | dict | 模型加载器 | 模型的计算图和模块结构 |
| `state` | 名称到 Tensor 的映射 | `state_dict()` | 独立冻结的参数副本 |
| `report` | 由程序组织的 dict | 统计与采样 | 可恢复完整训练状态的 checkpoint |

### 核心调用：由库契约决定返回形状

```python
model, loading_info = GPT2LMHeadModel.from_pretrained(
    str(model_dir), local_files_only=True, use_safetensors=True,
    output_loading_info=True,
)
```

按层次拆：

1. `GPT2LMHeadModel` 是类；`from_pretrained` 是可经类访问的加载工厂。
2. `str(model_dir)` 在调用前求值；其余三个是命名实参。
3. 前两个开关限制文件来源与权重格式，第三个要求返回诊断信息。
4. 调用结果含模型和诊断两项；左侧只是 Python 拆包。

因此 `output_loading_info=True` 是 Transformers API 行为，`model, loading_info = ...` 是语言行为。类型提示和赋值变量的数量都不会自行改变加载函数的返回值。

### 紧凑表达式需要还原的执行顺序

```python
config = json.loads(config_path.read_text(encoding="utf-8"))
has_problem = any(loading_info.get(key) for key in
                 ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
```

第一行先 I/O，再解析；第二行创建惰性检查过程，任意列表非空即可短路。这些是加载结构匹配检查，不是模型质量评价。

```python
row = model.transformer.wte.weight[args.token_id, :8].detach().cpu().tolist()
```

属性链定位参数 → 基本索引取片段 → detach 梯度关系 → 放到 CPU → 转普通列表。`detach()` 自身不复制存储；最后 `.tolist()` 得到的才是用于输出的普通数值容器。这条链在查看少量参数时合理，不应把模型主计算搬到列表后期待 autograd 继续跟踪。

### 参数枚举与快照语义

```python
state = model.state_dict()
total = sum(p.numel() for p in model.parameters())
tied = model.transformer.wte.weight is model.lm_head.weight
```

- `parameters()` 枚举注册参数，默认会对共享参数去重；参数是否冻结还要看 `requires_grad`。
- `state_dict()` 还可包含持久 buffer；不能假定其所有条目都是独立可训练参数。
- 返回状态映射不意味着底层 Tensor 存储已复制；保存“最佳模型”的内存快照需另行复制或立即序列化。
- `is` 检查这里的 E 与输出层是否引用同一 Parameter，比数值相等更强。

脚本把报告交给 `json.dumps`，保存的是统计资料。完整参数的保存与恢复由另一组 API 负责，见第 15 单元。

### 接口和副作用审查

这份脚本会读取本地配置和权重、创建模型、打印报告；启用 `--json` 时会写报告。它调用 `eval()`，但没有用输入执行前向，也没有 `backward()`、优化器更新或文本生成。

`REVISION` 的全大写只是一种常量命名惯例，解释器不会强制禁止重新赋值。固定缓存版本是复现实验选择，不是 Python 的特殊语法。

<details><summary>展开核对：当前真实源文件</summary>

<!-- SOURCE: ../inspect_checkpoint.py -->

</details>

### 代码审查题

把 `state = model.state_dict()` 存到变量后继续训练，是否拿到了训练前的稳定快照？`model.eval()` 为什么不能替代一次 `model(input_ids)`？

<details><summary>核对答案</summary>

没有保证快照独立，后续更新可能通过共享存储反映在 `state` 中。`eval` 修改模式；`model(input_ids)` 才进入具体前向。下一单元把 Python 对象、Tensor 存储和 autograd 三种关系分别验证。

</details>
