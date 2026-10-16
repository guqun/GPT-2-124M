# 从 Java / C++ / Dart 迁移到 Python

面向资深程序员的大模型代码阅读与实作指南

**读者前提：你熟悉 Java、C++、Dart，具备工程和算法经验，正在补 Python。**本版默认你理解变量、循环、函数、类、异常和基本数据结构；篇幅集中在 Python 的具体语法、运行语义与工程边界。遇到熟悉概念，用对照表定位差异，再用小实验验证。

目标是回到 GPT-2 复现课程 2.1：读懂 `inspect_checkpoint.py`，能修改检查逻辑，并正确理解 PyTorch 模型的对象、张量和调用关系。主线仍暂停在“配置与权重检查”，这份材料是语言迁移补充。

## 先校准几种容易迁移错的经验

| 已有经验 | Python 中需要重新确认的边界 |
|---|---|
| C++ 对象赋值可能有值拷贝语义 | 名称赋值通常只建立绑定，不调用隐含的对象复制 |
| Java 对象的 `==` 判断身份 | Python 的 `==` 调用相等性语义；身份判断用 `is` |
| Java / Dart 的 `String.length` | Python `len(str)` 数码点；UTF-16 码元和 UTF-8 字节要另算 |
| 静态类型与泛型检查 | Python 注解通常交给检查工具；解释器不会据此自动验证参数 |
| Dart 的命名参数、C++ 的默认参数 | Python 有 `/`、`*`、`*args`、`**kwargs`，默认对象在函数定义时创建 |
| Java 注解、C++ attribute | Python 装饰器会实际调用代码，可能替换函数或类 |
| 容器中的对象可被应用层访问 | PyTorch 普通 `list` 里的层不一定注册到 `nn.Module` |
| “复制了变量”或“拿到了 state” | Python 别名、Tensor 共享存储、autograd 连接是三个不同维度 |

对照只用来定位差异，不把三门语言当成一种语义。各章会说明哪些能类比、哪些要以 Python 实际行为为准。

## 两条阅读路线

**快速迁移路线：** [01 对象绑定](#unit-01) → [07 参数绑定](#unit-07) → [09 模块](#unit-09) → [10 对象协议](#unit-10) → [13 真实脚本](#unit-13) → [14 Tensor](#unit-14) → [15 模型生命周期](#unit-15)。其余章节按读代码时遇到的问题查阅。

**系统核对路线：**按 00—16 顺序。章节标记为“速查”“重点”“实战”，无需把每章投入相同时间。先做章末语义判断，能准确解释结果就跳过展开实验；不确定时运行后再回看规则。

| 阅读级别 | 单元 | 重点 |
|---|---|---|
| 速查 | 00、02、11 | 环境、文本、I/O 的 Python 写法与边界 |
| 重点 | 01、03—10、12 | 绑定、可变性、协议、作用域、参数和动态类型 |
| 实战 | 13—16 | 真实 GPT-2 检查脚本、PyTorch 与配置检查器 |

## 教材的三层边界

- **语言本身：**名称绑定、函数调用、索引、迭代、上下文管理协议。
- **标准库：**`pathlib`、`json`、`argparse`、`dataclasses` 等具体实现。
- **框架约定：**Transformers 的加载参数，PyTorch 的注册、梯度与存储行为。

例如 `model(...)` 使用 Python 的可调用对象协议；如何注册层、进入 `forward` 和执行钩子，则由 `nn.Module` 实现。知道语法之后，仍需查看对象类型与 API 契约。

## 材料与验证

本教材是独立编写的迁移材料，不是 Karpathy 视频逐字稿。第 13 单元对照当前真实文件，其余例子以小数据验证语义，不替代完整 GPT-2 实现。

- [网页阅读版](textbook.html) / [Markdown 版](textbook.md)：同步生成，示例默认折叠。
- [运行与文件说明](README.md)：已有项目环境，16 个可运行程序。
- [综合练习](exercises/config_report.py) / [参考答案](answers/config_report.py)：用配置检查任务综合语法。
- [验证记录](verification.json)：保留实跑输出和源文件校验值。

已有环境为 Python 3.9.6；涉及版本差异会明确标出。教材不要求为了语法学习升级环境或重新下载模型。


---

<a id="unit-00"></a>

## 00 · 执行环境：解释器、依赖与启动方式

**迁移目标：**建立“哪个解释器执行了哪个入口”的确定性。你已有终端与 IDE 使用经验，本章只保留 Python 环境定位。

### 一张定位表

| 需要知道什么 | 直接观察 | 避免的误判 |
|---|---|---|
| 当前解释器 | `sys.executable` | 系统 `python`、Python 3.12 和项目 venv 是同一个环境 |
| 是否在 venv | `sys.prefix != sys.base_prefix` | 终端目录在项目里就自动使用项目环境 |
| 包能否被找到 | `importlib.util.find_spec("torch")` | 找到包就代表它及原生依赖一定能导入成功 |
| 脚本资源目录 | `Path(__file__).resolve().parent` | 当前工作目录就是脚本目录 |
| 包安装归属 | `指定解释器 -m pip ...` | 裸 `pip` 一定对应正在运行的 Python |

你之前的 `No module named 'torch'` 发生在切换解释器之后，应首先检查环境绑定，而不是修改模型代码。本课程当前解释器：`/Users/guqun/micrograd/venv/bin/python`。

### 两种入口都要能识别

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/inspect_checkpoint.py
```

这是按文件运行；`python -m package.module` 则按可导入模块运行，能建立包上下文。选择取决于项目组织，不是哪个写法更高级。相对导入和入口保护见第 09 单元。

<details><summary>运行环境定位实验</summary>

**可运行例子：** [examples/00_environment.py](examples/00_environment.py)


```python
"""00：解释器、环境与模块定位；诊断而非安装脚本。"""
import importlib.util
from pathlib import Path
import sys

print("解释器：", sys.executable)
print("版本：", sys.version.split()[0])
print("当前是虚拟环境：", sys.prefix != sys.base_prefix)
print("脚本目录：", Path(__file__).resolve().parent)
# 查找模块的入口，不导入 torch，不下载权重。
spec = importlib.util.find_spec("torch")
print("torch 可定位：", spec is not None)
```

**运行方式：**

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/python_foundations/examples/00_environment.py
```

**本机实际输出：**

```text
解释器： /Users/guqun/micrograd/venv/bin/python
版本： 3.9.6
当前是虚拟环境： True
脚本目录： /Users/guqun/micrograd/gpt2_pretrain_lab/python_foundations/examples
torch 可定位： True
```


</details>

<details><summary>VS Code 解释器定位速查</summary>

命令面板选择 **Python: Select Interpreter**，指定 `/Users/guqun/micrograd/venv/bin/python`。实际执行后以 `sys.executable` 为准。需要核对安装位置时运行这个解释器的 `-m pip show torch`；本机依赖已经具备，无需再安装。

</details>

### 语义自检

IDE 能补全 `torch`，运行却找不到它，优先核对哪两个环境？为什么“find_spec 成功”和“import 成功”不是同一件事？

<details><summary>核对答案</summary>

先核对 IDE 分析/运行配置与实际 `sys.executable`。定位模块只说明找到了导入入口；执行模块代码、解析传递依赖或加载原生扩展仍可能失败。

</details>


---

<a id="unit-01"></a>

## 01 · 名称绑定、对象身份与可变性

**迁移目标：**精准判断赋值和传参是否复制、是否修改调用者状态。先分“名字重新绑定”与“对象被修改”，后面再区分 Tensor 存储与梯度图。

### 传参共享对象，局部重绑定不回写调用者

```python
def update(values):
    values.append(30)
    values = [999]
    return values

ids = [10, 20]
other = update(ids)
```

结果：`ids == [10,20,30]`，`other == [999]`。调用时形参与实参名字可访问同一个列表；`append` 修改那个对象；后面的赋值只改变局部名字 `values`。

可以借用 Java “对象引用值传递”的直觉理解这个例子，但 Python 对整数、函数、类等同样使用对象与名字的模型，也没有这里可对应的 C++ `T&` 形参声明。称为“按对象共享传递”比笼统地说“全部按引用传递”更不易误读。

### 三种写法，三种契约

| 写法 | 判断/操作 | 不能顺手推断什么 |
|---|---|---|
| `b = a` | 新绑定到已有对象 | 不意味着新建对象或复制数据 |
| `a == b` | 相等性比较，可由类型定义 | 不保证两个引用相同；Tensor 甚至返回逐元素结果 |
| `a is b` | 对象身份相同 | 不用来比较字符串、整数的内容 |

Java 对象的 `==` 与 Python 的 `is` 更接近。C++ 的运算符重载和 Dart 的 `operator ==` 更能提醒你：相等性要看对象类型的实现。不要用整数驻留或字符串复用的偶然结果推断 `is` 的内容比较规则。

### 数值语义也有几个差别

- Python `int` 是任意精度整数，受资源限制；普通整数运算没有固定 32/64 位溢出回绕。
- `/` 是普通除法；`//` 向负无穷取整：`-7 // 2 == -4`。别套用 Java/C++ 整数除法向零截断的结果。
- `%` 与这个取整规则配套，`-7 % 2 == 1`；乘方用 `**`，`^` 是异或。
- `bool` 是 `int` 的子类，`isinstance(True, int)` 成立；严格配置尺寸校验需要排除布尔值。
- `x += y` 由类型决定：整数重新绑定，列表可能原地修改，Tensor 还涉及 autograd 约束。

<details><summary>实验：共享修改、重新绑定与整数边界</summary>

**可运行例子：** [examples/01_values.py](examples/01_values.py)


```python
"""01：对象绑定、共享修改与重新绑定。"""

def mutate_then_rebind(values):
    values.append(30)  # 修改调用者也能访问的同一个对象。
    values = [999]    # 只改变本函数局部名字的绑定。
    return values

original = [10, 20]
returned = mutate_then_rebind(original)
print("调用者的列表：", original)
print("函数返回的新列表：", returned)
assert original == [10, 20, 30] and returned == [999]

alias = original
same_value = original.copy()
print("alias is original：", alias is original)
print("copy == original / copy is original：", same_value == original, same_value is original)

integer = 1
old_integer = integer
integer += 1
print("整数重新绑定：", old_integer, integer)
print("普通除法 / 向下整除 / 余数：", 7 / 2, -7 // 2, -7 % 2)
print("整数不固定为 64 位：", 2 ** 70)
print("bool 是 int 的子类：", isinstance(True, int))
```

**运行方式：**

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/python_foundations/examples/01_values.py
```

**本机实际输出：**

```text
调用者的列表： [10, 20, 30]
函数返回的新列表： [999]
alias is original： True
copy == original / copy is original： True False
整数重新绑定： 1 2
普通除法 / 向下整除 / 余数： 3.5 -4 1
整数不固定为 64 位： 1180591620717411303424
bool 是 int 的子类： True
```


</details>

### 代码审查题

`cache = original` 可以当备份吗？如何解释 `wte.weight is lm_head.weight` 与“两个权重当前数值一样”的差别？

<details><summary>核对答案</summary>

前者只是别名。备份要选择符合对象类型的复制策略，嵌套容器和 Tensor 不能仅凭一个等号判断。共享权重检查比较参数对象身份；两个独立参数即使数值一致，也不会因此在后续更新中自动同步。

</details>


---

<a id="unit-02"></a>

## 02 · 文本边界：str、bytes 与格式化

**迁移目标：**保留你已经理解的 Unicode/Tokenizer 概念，校准 Python 对字符串长度、索引和字节的实际定义。

### 同一个字符，三个不同计数

| 文本 | Python `len(str)`：码点 | Java / Dart 字符串长度：UTF-16 码元 | UTF-8 字节数 |
|---|---|---|---|
| `你` | 1 | 1 | 3 |
| `😊` | 1 | 2 | 4 |
| `e` 加组合重音 `\u0301` | 2 | 2 | 3 |

Java 的字符串索引和 Dart `length` 使用 UTF-16 码元；Python 的 `str` 按码点访问。这些都不是用户感知字素簇的通用计数方式。[Java String 说明](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html)、[Dart length 定义](https://api.dart.dev/dart-core/String/length.html)。

Python 没有独立的 `char` 类型：`text[0]` 仍是长度 1 的 `str`。`bytes[0]` 却是 0—255 的整数；`bytes[:1]` 仍是 `bytes`。这直接对应此前字节级 BPE 的输入处理。

### 只记几条 Python 特有的 API 习惯

| 写法 | 契约 |
|---|---|
| `text.encode("utf-8")` / `data.decode("utf-8")` | `str ↔ bytes`，不是 Tokenizer 的词表编码 |
| `"a  b".split()` | 合并连续空白、丢掉两端空白后的分段 |
| `"a  b".split(" ")` | 按指定分隔符切，保留中间产生的空字符串 |
| `"a.b".split(".")` | 分隔符是字面字符串；不像 Java `String.split` 默认接收正则 |
| `" ".join(pieces)` | 连接器是接收者，元素需要是字符串 |
| `f"{loss:.3f}"` | 在执行该表达式时格式化，不改变原数值 |
| `f"{value!r}"` | 使用 `repr` 表示；便于观察转义和边界 |

不要把 `str(bytes_object)` 当解码，它得到的是对象的显示表示。原始字符串 `r"\d+"` 减少 Python 对反斜杠的处理，但正则引擎随后仍会解释自己的语法。

<details><summary>实验：三种长度、字节索引与 split</summary>

**可运行例子：** [examples/02_strings.py](examples/02_strings.py)


```python
"""02：码点、UTF-16 码元、UTF-8 字节的计数对照。"""
for text in ("你", "😊", "e\u0301"):
    data = text.encode("utf-8")
    utf16_units = len(text.encode("utf-16-le")) // 2
    print(repr(text), "码点", len(text), "UTF-16 码元", utf16_units, "UTF-8 字节", len(data))
    assert data.decode("utf-8") == text

raw = "你".encode("utf-8")
print("bytes 索引与切片类型：", type(raw[0]).__name__, type(raw[:1]).__name__)
print("字节数值：", list(raw))
print("空白拆分：", "a  b".split())
print("指定普通空格：", "a  b".split(" "))
print("字面分隔符不是正则：", "a.b".split("."))
loss = 2.34567
print(f"显示 {loss:.3f}；原值 {loss!r}；参数量 {50257 * 768:,}")
```

**运行方式：**

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/python_foundations/examples/02_strings.py
```

**本机实际输出：**

```text
'你' 码点 1 UTF-16 码元 1 UTF-8 字节 3
'😊' 码点 1 UTF-16 码元 2 UTF-8 字节 4
'é' 码点 2 UTF-16 码元 2 UTF-8 字节 3
bytes 索引与切片类型： int bytes
字节数值： [228, 189, 160]
空白拆分： ['a', 'b']
指定普通空格： ['a', '', 'b']
字面分隔符不是正则： ['a', 'b']
显示 2.346；原值 2.34567；参数量 38,597,376
```


</details>

### 语义自检

为什么“取字符串的前 8 项”和“取 UTF-8 的前 8 字节”可能无法互换？`repr(text)`、UTF-8 编码与 Token ID 序列分别解决什么问题？

<details><summary>核对答案</summary>

前者按码点，后者可能截断一个多字节字符，得到不能独立解码的片段。`repr` 用于观察对象；UTF-8 是字符到字节的编码；Token ID 还要经过具体 Tokenizer 的切分与词表映射。

</details>


---

<a id="unit-03"></a>

## 03 · 容器、切片与复制边界

**迁移目标：**熟悉 Python 序列语法，并能区分“外层复制”“内部共享”“深复制”。后面 Tensor 的切片会再引入存储视图语义。

### 索引语法快速映射

| Python | 需要识别的结果 |
|---|---|
| `ids[-1]` | 最后一个元素 |
| `ids[start:stop:step]` | 左闭右开；步长可为负 |
| `x, y = ids[:-1], ids[1:]` | 两份列表切片，准备右移监督 |
| `first, *middle, last = ids` | 星号吸收中间项，`middle` 是列表 |
| `(768,)` | 单项元组；关键是逗号 |
| `list.sort()` / `sorted(values)` | 前者原地改并返回 `None`；后者产生新列表 |

列表切片创建新的外层列表，元素仍是原对象的引用。与 C++ `std::vector` 的按值复制经验对照时，要看 Python 容器里的元素对象是否继续共享。

### 两个高频反例

```python
rows = [[1, 2], [3, 4]]
backup = rows[:]
backup[0].append(9)  # rows[0] 也改变

matrix = [[0] * 2] * 3
matrix[0][0] = 7     # 三行都看到 7
```

第一例只复制外层。第二例把同一个行对象的引用重复了三次；正确的独立构造是 `[[0] * 2 for _ in range(3)]`。

标准库 `copy.copy` 与切片一样只做浅复制；`copy.deepcopy` 递归复制适用对象，并处理共享/循环引用。资源句柄、框架对象以及 Tensor 应按各自契约决定是否及怎样复制。[copy 官方说明](https://docs.python.org/3.9/library/copy.html)。

### 不可变 tuple 不等于深不可变

`record = ("tokens", [1,2])` 不允许重新赋值 `record[1]`，但 `record[1].append(3)` 合法。`tuple` 固定的是容器槽位，不能类比为包含对象全都被深度冻结。

`append(value)` 把一个对象加入列表；`extend(values)` 消费一个可迭代对象。二者返回 `None`，不提供 fluent 风格链式调用。`ids = ids.append(1)` 会丢失原列表绑定。

<details><summary>实验：切片、浅复制、深复制与共享行</summary>

**可运行例子：** [examples/03_sequences.py](examples/03_sequences.py)


```python
"""03：列表切片复制外层；tuple 不保证内部对象不可变。"""
import copy

ids = [10, 20, 30, 40]
x, y = ids[:-1], ids[1:]
print("右移监督：", list(zip(x, y)))
print("负索引与步长：", ids[-1], ids[::2], ids[::-1])
first, *middle, last = ids
print("星号拆包：", first, middle, last)

rows = [[1, 2], [3, 4]]
shallow = rows[:]
deep = copy.deepcopy(rows)
shallow[0].append(9)
shallow.append([5, 6])
print("原始 / 浅复制 / 深复制：", rows, shallow, deep)
assert len(rows) == 2 and rows[0] == [1, 2, 9] and deep[0] == [1, 2]

wrong = [[0] * 2] * 3
wrong[0][0] = 7
right = [[0] * 2 for _ in range(3)]
right[0][0] = 7
print("复用同一行：", wrong)
print("独立创建每行：", right)

record = ("tokens", [1, 2])
record[1].append(3)
print("tuple 内的列表仍可变：", record)
values = [3, 1, 2]
returned = values.sort()
print("原地排序结果 / 返回值：", values, returned)
```

**运行方式：**

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/python_foundations/examples/03_sequences.py
```

**本机实际输出：**

```text
右移监督： [(10, 20), (20, 30), (30, 40)]
负索引与步长： 40 [10, 30] [40, 30, 20, 10]
星号拆包： 10 [20, 30] 40
原始 / 浅复制 / 深复制： [[1, 2, 9], [3, 4]] [[1, 2, 9], [3, 4], [5, 6]] [[1, 2], [3, 4]]
复用同一行： [[7, 0], [7, 0], [7, 0]]
独立创建每行： [[7, 0], [0, 0], [0, 0]]
tuple 内的列表仍可变： ('tokens', [1, 2, 3])
原地排序结果 / 返回值： [1, 2, 3] None
```


</details>

### 代码审查题

为每个 batch 保存 `history.append(batch[:])`，是否已取得稳定快照？tuple 包装一个列表后，能直接作为字典键吗？

<details><summary>核对答案</summary>

取决于内部元素：嵌套可变对象仍共享，Tensor 还要核对存储和梯度关系。包含列表的 tuple 不能哈希，因此不能直接作字典键；tuple 的每个元素都满足可哈希要求才行。

</details>


---

<a id="unit-04"></a>

## 04 · dict / set：缺失、顺序与哈希契约

**迁移目标：**读词表与配置时，区分缺键、空值、覆盖与去重；审查代码不能只看容器名字像哪一种 Map。

### 常用行为对照

| 表达式 | 精确语义 |
|---|---|
| `mapping[key]` | 不存在则 `KeyError`，不会插入默认值 |
| `mapping.get(key, fallback)` | 缺失才返回默认；存在但为 `None` 时保留 `None` |
| `key in mapping` | 检查键，不检查值 |
| `mapping.keys()` / `.items()` | 动态视图，后续改动可以反映出来 |
| `{**base, **overrides}` | 新外层字典；后面的同名键覆盖前面的，仍是浅组合 |
| `set()` / `{}` | 空集合 / 空字典 |

Python 3.7 起 dict 插入顺序是语言保证；覆盖已有键不把键移到末尾；删除后重新插入则是新的插入位置。它不是按键排序。set 不提供你可以依赖的插入次序，BPE 的合并优先级应来自 rank，不能来自集合遍历位置。

### 缺失与 None 要明确区分

```python
sentinel = object()
value = config.get("checkpoint", sentinel)
if value is sentinel:
    print("根本没提供这个字段")
elif value is None:
    print("提供了字段，并明确使用空值")
```

如果只 `.get(key)`，两种情况都会得到 `None`。`setdefault` 则会在缺键时插入值；并且传入的默认表达式会在调用前求值，不是惰性工厂。

### 键的相等与哈希会影响数据

`{1: "a", True: "b", 1.0: "c"}` 最终只有一个键，因为它们相等且哈希一致。不要把配置中的布尔值直接放进要求严格整数的 ID/维度路径。

像 `("a","b")` 这样的字符串元组可以作为 BPE pair 的键；列表不能哈希。自定义类若重定义 `__eq__`，也需要理解其 `__hash__` 契约，不能在作为键后再改变影响相等/哈希的状态。

<details><summary>实验：动态视图、哨兵与数值键合并</summary>

**可运行例子：** [examples/04_mappings.py](examples/04_mappings.py)


```python
"""04：字典顺序、缺失语义与哈希键。"""
config = {"width": 768, "dropout": None}
print("get 无法区分两种情况：", config.get("dropout"), config.get("missing"))
sentinel = object()
print("sentinel 检查缺键：", config.get("missing", sentinel) is sentinel)
print("键存在但值为空：", "dropout" in config)

keys = config.keys()
config["heads"] = 12
print("keys 是动态视图：", list(keys))
config["width"] = 16
print("覆盖值不改变既有顺序：", list(config.items()))
print("仅构造新字典：", {**config, "heads": 2})

pairs = {("a", "b"), ("a", "b"), ("b", "c")}
print("去重的符号对：", sorted(pairs))
print("相等的数值键合并：", len({1: "整数", True: "布尔", 1.0: "浮点"}))
assert len({1, True, 1.0}) == 1
try:
    hash(("tokens", [1, 2]))
except TypeError as error:
    print("含列表的 tuple 不可哈希：", type(error).__name__)
```

**运行方式：**

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/python_foundations/examples/04_mappings.py
```

**本机实际输出：**

```text
get 无法区分两种情况： None None
sentinel 检查缺键： True
键存在但值为空： True
keys 是动态视图： ['width', 'dropout', 'heads']
覆盖值不改变既有顺序： [('width', 16), ('dropout', None), ('heads', 12)]
仅构造新字典： {'width': 16, 'dropout': None, 'heads': 2}
去重的符号对： [('a', 'b'), ('b', 'c')]
相等的数值键合并： 1
含列表的 tuple 不可哈希： TypeError
```


</details>

### 代码审查题

`decoder = {v:k for k,v in encoder.items()}` 在什么情况下会丢失映射？为什么 `config.get("width", 768)` 不一定返回 768 或一个合法整数？

<details><summary>核对答案</summary>

如果多个键对应同一个值，反向构造时会覆盖。`get` 只处理缺键，不做 schema 校验；已有值可能是 `None`、字符串、负数或布尔值。

</details>


---

<a id="unit-05"></a>

## 05 · 真值、短路与作用域差异

**迁移目标：**熟悉 Python 的控制流语义差异。`if/for` 的概念无需重学，但真值转换和作用域不能机械沿用 Java/C++/Dart 的习惯。

### 真值不是“只允许 bool”

`None`、0、空字符串和空容器按假处理；非空 `[False]` 按真处理。`and/or` 短路，而且**返回被选中的操作数，不必是布尔值**。

```python
token_id = 0
wrong = token_id or 220
right = 220 if token_id is None else token_id
```

`wrong` 为 220，`right` 为 0。`or` 不是 Dart `??` 那样只针对空值的回退操作。有效 ID 0、合法的空列表、`dropout=0.0` 都可能被错误覆盖。

`0 <= token_id < vocab_size` 是链式比较；中间表达式只求值一次。`&`、`|` 是另一组可被类型定义的运算，不是通用的 `and/or` 替代品；张量逻辑到第 14 单元单独处理。

### 没有一般的 if / for 块作用域

```python
for i in range(3):
    pass
print(i)  # 2
```

只要循环曾执行，绑定就留在所在作用域；空循环不会替你初始化名字。Python 3 的列表推导式则有自己的循环变量作用域，`[j for j in ...]` 不把 `j` 泄漏出去。

### 局部绑定由整个函数体决定

```python
scale = 10

def broken():
    print(scale)
    scale = 2
```

调用时抛出 `UnboundLocalError`：函数中存在 `scale` 的赋值，因而它在整个函数中都被视为局部名字，第一行读取时尚未绑定。不会因为读取写在赋值前，就先读外层同名变量。

普通函数嵌套可用 LEGB 理解：Local、Enclosing function、Global(module)、Builtins。`global` 声明模块级绑定，`nonlocal` 声明外层函数的已有绑定；它们影响重新绑定的位置，不是修改列表内容的前提。类体等场景有额外规则，LEGB 不应当作全部 Python 名称解析的完整形式化定义。[执行模型](https://docs.python.org/3.9/reference/executionmodel.html)。

<details><summary>实验：ID 0、真值和局部绑定</summary>

**可运行例子：** [examples/05_conditions.py](examples/05_conditions.py)


```python
"""05：真值规则、短路返回值与函数作用域。"""
user_id = 0
print("误用 or 提供默认值：", user_id or 220)
print("只为 None 提供默认值：", 220 if user_id is None else user_id)
print("空列表 / 非空列表 [False]：", bool([]), bool([False]))
print("and/or 返回操作数：", [] and "unused", "ready" or "fallback")

for i in range(3):
    pass
print("for 不新建块作用域：", i)
_ = [j for j in range(3)]
print("推导式的 j 未泄漏：", "j" not in globals())

scale = 10

def broken_scope():
    print(scale)
    scale = 2  # 这个绑定使 scale 在整个函数中被判定为局部名字。

try:
    broken_scope()
except UnboundLocalError as error:
    print("先读后赋的局部变量：", type(error).__name__)

assert (user_id or 220) == 220
assert (220 if user_id is None else user_id) == 0
```

**运行方式：**

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/python_foundations/examples/05_conditions.py
```

**本机实际输出：**

```text
误用 or 提供默认值： 220
只为 None 提供默认值： 0
空列表 / 非空列表 [False]： False True
and/or 返回操作数： [] ready
for 不新建块作用域： 2
推导式的 j 未泄漏： True
先读后赋的局部变量： UnboundLocalError
```


</details>

### 代码审查题

`def add(x): cache.append(x)` 需要 `global cache` 吗？改成 `cache = cache + [x]` 呢？

<details><summary>核对答案</summary>

前者只读取外层绑定并修改对象，不因这一操作需要 `global`。后者在函数内赋值，名字被判定为局部，右侧先读时会失败；如果确实要改模块级绑定，应显式声明，但通常更清楚的设计是传入状态并返回结果。

</details>


---

<a id="unit-06"></a>

## 06 · 迭代协议、生成器与消费状态

**迁移目标：**把 Python 的循环看成协议调用，识别数据是可重复遍历的容器，还是已经消费到某个位置的迭代器。

### Iterable 和 Iterator 不应混用

`for x in source` 大致取得 `iter(source)`，然后反复 `next`，遇到 `StopIteration` 正常结束。可迭代对象能产生迭代器；迭代器本身保存遍历状态。

| 对象 | 重复读取时要注意 |
|---|---|
| `list`、`tuple`、`range` | 通常每次 `iter` 都能从头产生新迭代器 |
| `iter(list)` | 已经是有状态的迭代器，用完就结束 |
| 生成器表达式、含 `yield` 的函数返回值 | 惰性、通常只能消费一次 |
| `zip`、`enumerate`、`map`、`filter` | Python 3 中返回迭代器，不是现成列表 |

默认 `zip` 以最短输入结束。如果这会静默丢掉标签，应显式检查长度或选择符合数据源约束的策略。Python 3.10 才加入 `zip(..., strict=True)`；本项目 3.9 不直接使用它。

### `yield` 把函数变成可恢复的生产过程

```python
def batches(sequence, size):
    for offset in range(0, len(sequence), size):
        yield sequence[offset:offset + size]
```

调用 `batches(...)` 得到生成器，通常此时还没执行函数体；消费时执行到 `yield` 返回一批并保存现场，下次接着走。它与 Dart `sync*` 有相近的惰性生产直觉，但具体对象仍要区分“Iterable 工厂”与“这一个已生成的迭代器”。`yield` 本身不是 `await`。

### 训练循环中的直接后果

如果在 epoch 外只创建一次 `samples = (read(x) for x in paths)`，第一轮遍历耗尽后，第二轮可能没有数据。解决方式可以是每个 epoch 重新创建生成器，或者由数据集对象的 `__iter__` 提供新迭代器；不是一律把全量数据转成列表。

`any`/`all` 会短路，可能只消费迭代器的一部分；`sum` 通常消费到结束。打印调试时 `list(iterator)` 也会改变后续读取状态。

<details><summary>实验：zip 耗尽、any 短路与 yield 分批</summary>

**可运行例子：** [examples/06_loops.py](examples/06_loops.py)


```python
"""06：可迭代对象和有消费状态的迭代器。"""
ids = [10, 20, 30]
iterator = iter(ids)
print("第一次 next：", next(iterator))
print("消费剩余：", list(iterator))
print("已耗尽：", list(iterator))
print("原列表可重新迭代：", list(ids))

pairs = zip(ids, [20, 30])
print("zip 按最短输入结束：", list(pairs))
print("zip 也只消费一次：", list(pairs))
print("enumerate 起始编号：", list(enumerate(ids, start=1)))

checks = iter([False, True, False])
print("any 短路：", any(checks))
print("any 没消费到最后：", list(checks))
assert list(iterator) == []

def batches(sequence, size):
    if size <= 0:
        raise ValueError("size 必须为正")
    for offset in range(0, len(sequence), size):
        yield sequence[offset:offset + size]

print("yield 分批：", list(batches(ids, 2)))
```

**运行方式：**

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/python_foundations/examples/06_loops.py
```

**本机实际输出：**

```text
第一次 next： 10
消费剩余： [20, 30]
已耗尽： []
原列表可重新迭代： [10, 20, 30]
zip 按最短输入结束： [(10, 20), (20, 30)]
zip 也只消费一次： []
enumerate 起始编号： [(1, 10), (2, 20), (3, 30)]
any 短路： True
any 没消费到最后： [False]
yield 分批： [[10, 20], [30]]
```


</details>

### 代码审查题

`print(list(model.parameters()))` 后再调用一次 `model.parameters()` 能否重新遍历？如果先保存 `params=model.parameters()`，连续消费 `params` 两次呢？

<details><summary>核对答案</summary>

再次调用该方法可以取得新的遍历过程；同一个已经耗尽的 `params` 迭代器则不会自动复位。区别在于“重新调用工厂”和“重复使用同一个消费对象”。

</details>


---

<a id="unit-07"></a>

## 07 · 参数绑定、默认值与返回协议

**迁移目标：**读函数签名时准确还原参数绑定，不把 Python 当成“省略类型的 Java 方法”。这直接对应当前的 `from_pretrained(...)` 调用。

### 一次读全五种形参位置

```python
def inspect(config, /, token_id=0, *extra, output_info=False, **options):
    ...
```

| 部分 | 契约 |
|---|---|
| `/` 之前的 `config` | 只能按位置传入 |
| `token_id=0` | 既可按位置，也可按名称，有默认值 |
| `*extra` | 收集额外位置实参，结果为 tuple |
| `output_info=False` | 只能用名称传入；也可用单独 `*` 引入这一段 |
| `**options` | 收集未被前面形参接走的关键字实参，结果为 dict |

这是签名识读示意，不要求把所有函数都写得如此通用。Dart 的命名参数经验可以迁移，但要按 Python 的 `/` 与 `*` 分界重新读；实参重复绑定同一个参数会报 `TypeError`。

调用侧 `f(*values, **options)` 是展开，定义侧 `*args/**kwargs` 是收集。关键字名字属于 API 契约，传给库的名称必须匹配，而不是由调用者随便选。

### 默认值在执行 def 时求值

```python
def collect(token_id, history=[]):
    history.append(token_id)
    return history
```

这里的默认列表在该次函数定义时创建，省略 `history` 的后续调用会复用它。字符串、整数等不可变默认值通常不产生这个共享修改问题，但其表达式也同样在定义时求值。

`history=None` 后在函数内新建列表是常见修正。如果 `None` 本身也是有意义的实参，则使用独立 sentinel。时间戳、随机数、配置对象也要检查“默认表达式何时执行”，不能只盯着列表。

### 返回两个结果不是声明两个返回槽

`return model, info` 返回一个两项 tuple；`model, info = result` 消费并拆包该对象。数量不匹配会报错。调用左侧写两个名字，不会改变函数的返回契约。

你当前的 `output_loading_info=True` 决定库返回两项；Python 只按这个实际返回值拆包。若函数根据标志返回不同结构，调用方必须跟标志一致，类型提示也应表达这种差别。

### 函数对象与重定义

`key=rank_of` 传的是可调用对象，`key=rank_of()` 则先执行它。重复 `def f(...)` 通常会重新绑定名字、覆盖前一个定义，并不自动建立 Java/C++ 式重载集合。`typing.overload` 描述的是静态类型接口，不会自动完成运行时重载分派。

<details><summary>实验：位置专用参数、默认值时机与返回值拆包</summary>

**可运行例子：** [examples/07_functions.py](examples/07_functions.py)


```python
"""07：参数绑定、定义时求值、可变默认值与拆包。"""

def inspect_config(config, /, token_id=0, *, output_info=False):
    report = {"valid": 0 <= token_id < config["vocab_size"]}
    info = {"missing_keys": []}
    return (report, info) if output_info else report

config = {"vocab_size": 50257}
options = {"token_id": 220, "output_info": True}
report, info = inspect_config(config, **options)
print("返回值拆包：", report, info)
try:
    inspect_config(config=config)
except TypeError as error:
    print("位置专用参数拒绝关键字：", type(error).__name__)

DEFAULT_WIDTH = 16

def width(value=DEFAULT_WIDTH):
    return value

DEFAULT_WIDTH = 32
print("默认值在定义时取得：", width())

def bad_collect(token_id, values=[]):
    values.append(token_id)
    return values

first = bad_collect(1)
second = bad_collect(2)
print("可变默认参数跨调用共享：", first, second, first is second)

def collect(token_id, values=None):
    if values is None:
        values = []
    values.append(token_id)
    return values

print("每次独立创建：", collect(1), collect(2))
assert width() == 16 and first is second
```

**运行方式：**

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/python_foundations/examples/07_functions.py
```

**本机实际输出：**

```text
返回值拆包： {'valid': True} {'missing_keys': []}
位置专用参数拒绝关键字： TypeError
默认值在定义时取得： 16
可变默认参数跨调用共享： [1, 2] [1, 2] True
每次独立创建： [1] [2]
```


</details>

### 代码审查题

`def load(options={}): ...` 与 `def load(options=None): options=options or {}` 各有什么问题？在默认参数中写 `dtype=DEFAULT_DTYPE` 后，再改全局 `DEFAULT_DTYPE`，默认值会跟着变吗？

<details><summary>核对答案</summary>

前者共享可变默认对象。后者虽然避开了默认对象共享，却会替换调用者明确传入的空字典；是否可接受要看 API 是否要保留原对象。通常使用 `if options is None:`。全局名字之后重新绑定，不会替换已保存的默认值；如果默认对象自身可变，它的内部修改则仍可见。

</details>


---

<a id="unit-08"></a>

## 08 · 推导式、闭包与装饰器

**迁移目标：**区分“表达式更短”与“执行时机改变”。重点看捕获了什么对象、什么时候调用、什么时候消费结果。

### 三种表达式的执行模型

| 写法 | 返回什么 | 何时计算元素 |
|---|---|---|
| `[f(x) for x in xs if keep(x)]` | list | 立即遍历并求值 |
| `{k: f(v) for k,v in items}` | dict | 立即遍历，同键后写覆盖 |
| `(f(x) for x in xs)` | generator | 元素表达式通常在消费时求值 |

推导式适合表达数据转换；包含多层副作用、异常恢复或复杂状态时，普通循环往往更清楚。生成器表达式最左侧的可迭代表达式在创建时求值，别把“惰性”理解为整个表达式的任何部分都不执行。

### 闭包读取绑定，不自动保存每轮值

```python
late = [lambda: i for i in range(3)]
snapshots = [lambda i=i: i for i in range(3)]
```

随后分别调用，得到 `[2,2,2]` 和 `[0,1,2]`。第一组闭包读取同一个外层绑定，调用时循环已经结束。第二组把当时的 `i` 作为默认实参保存。

用 C++ lambda 经验类比时，要主动询问“我需要按值快照还是共享绑定”，但 Python 没有同样的捕获列表语法。`i=i` 保存的是当时对象的引用；如果该对象是可变列表，这并不是深复制。

### 装饰器会真正执行，不只是元数据标记

```python
@decorate
def forward(x):
    return x
```

对单个简单函数，可以理解为先定义原函数，再执行 `forward = decorate(forward)`。装饰器在定义被执行时应用，返回值重新绑定到原名字。它可能返回包装函数、登记信息或改变行为。

`@torch.no_grad()` 中的括号先构造/取得装饰器，然后应用于函数。包装函数通常用 `*args, **kwargs` 转发参数，用 `functools.wraps` 保留名称、docstring 等元数据；这不等于静态语言中只被编译器读取的注解。

### 接回 BPE 与模型代码

`min(pairs, key=lambda p: ranks.get(p, inf))` 用辅助值比较，返回原始候选对。模型代码中的 `@classmethod`、`@property`、`@dataclass` 则是不同装饰器的具体契约，不是可以互换的语法标签。

<details><summary>实验：晚绑定、默认参数快照与装饰时机</summary>

**可运行例子：** [examples/08_compact_syntax.py](examples/08_compact_syntax.py)


```python
"""08：推导式、闭包晚绑定与装饰器的实际执行。"""
from functools import wraps

ranks = {("a", "b"): 0, ("b", "c"): 2}
pairs = [("b", "c"), ("c", "a"), ("a", "b")]
print("按 key 选原对象：", min(pairs, key=lambda pair: ranks.get(pair, float("inf"))))

late = [lambda: i for i in range(3)]
snapshots = [lambda i=i: i for i in range(3)]
print("闭包读取同一个绑定：", [fn() for fn in late])
print("默认参数保存当时的整数：", [fn() for fn in snapshots])
assert [fn() for fn in late] == [2, 2, 2]
assert [fn() for fn in snapshots] == [0, 1, 2]

def trace(function):
    print("装饰发生：", function.__name__)
    @wraps(function)
    def wrapper(*args, **kwargs):
        print("调用发生：", function.__name__)
        return function(*args, **kwargs)
    return wrapper

@trace
def head_dim(width, heads):
    return width // heads

print("尚未显式调用，装饰已完成")
print("计算结果：", head_dim(768, heads=12))
print("wraps 保留函数名：", head_dim.__name__)
```

**运行方式：**

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/python_foundations/examples/08_compact_syntax.py
```

**本机实际输出：**

```text
按 key 选原对象： ('a', 'b')
闭包读取同一个绑定： [2, 2, 2]
默认参数保存当时的整数： [0, 1, 2]
装饰发生： head_dim
尚未显式调用，装饰已完成
调用发生： head_dim
计算结果： 64
wraps 保留函数名： head_dim
```


</details>

### 代码审查题

在循环里创建多个 logging hook，全部最后打印同一个层编号，首先检查什么？为什么给 wrapper 加 `@wraps` 不等于让它自动做类型检查？

<details><summary>核对答案</summary>

检查闭包是否共享循环变量绑定，必要时用工厂函数或定义时参数绑定保存期望值。`wraps` 主要保留包装关系和元数据，不替你校验参数类型、返回类型或张量形状。

</details>


---

<a id="unit-09"></a>

## 09 · import、模块缓存与包上下文

**迁移目标：**把模块导入当作运行时动作来审查，而不是 C++ 预处理 include 或单纯的编译期名称声明。

### 首次导入通常会执行模块顶层代码

Python 导入系统查找模块、建立模块对象、执行其顶层代码，并在 `sys.modules` 中缓存。后续普通导入通常复用同一模块对象。CPython 可以编译和缓存字节码，但这不消除模块初始化的运行时副作用。

因此，把训练、下载或文件覆盖放在顶层，会使“别人 import 我”也触发这些动作。`if __name__ == "__main__": main()` 将脚本入口与被导入时的定义分开，作用只覆盖它保护的部分。

### import 两种形式的绑定差别

```python
import module
from module import factory
```

第一种取得模块对象，之后 `module.factory` 会再次读取其当前属性；第二种把当时取到的对象绑定到本地名字 `factory`。如果以后模块把属性重绑定为另一对象，本地名字不会自动跟随改变。导入语句不是“永久建立一个别名路径”。

<details><summary>实验：模块缓存与 from-import 的既有绑定</summary>

**可运行例子：** [examples/09_modules.py](examples/09_modules.py)


```python
"""09：模块缓存与 from-import 的名称绑定。"""
import importlib
from pathlib import Path
import sys
from mini_package import helpers
from mini_package.helpers import make_xy


def main():
    again = importlib.import_module("mini_package.helpers")
    print("两次导入取得同一模块：", helpers is again)
    print("sys.modules 中的对象：", sys.modules[helpers.__name__] is helpers)
    original = helpers.make_xy
    try:
        helpers.make_xy = lambda ids: (["replaced"], [])
        print("模块属性的新绑定：", helpers.make_xy([10, 20, 30]))
        print("from-import 的既有绑定：", make_xy([10, 20, 30]))
    finally:
        helpers.make_xy = original
    print("脚本名与模块名：", Path(__file__).name, __name__)


if __name__ == "__main__":
    main()
```

**运行方式：**

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/python_foundations/examples/09_modules.py
```

**本机实际输出：**

```text
两次导入取得同一模块： True
sys.modules 中的对象： True
模块属性的新绑定： (['replaced'], [])
from-import 的既有绑定： ([10, 20], [20, 30])
脚本名与模块名： 09_modules.py __main__
```


</details>

### 相对导入依赖包身份

本例目录：`examples/mini_package/{__init__.py, helpers.py, __main__.py}`。`from .helpers import make_xy` 里的点表示包内定位，不是任意脚本当前目录的文本替换。

```bash
cd /Users/guqun/micrograd/gpt2_pretrain_lab/python_foundations/examples
/Users/guqun/micrograd/venv/bin/python -m mini_package
```

**本机实际输出：**

```text
包入口： __main__
x、y： ([10, 20], [20, 30])
```


你以前直接运行包内 `demo.py` 得到 `attempted relative import with no known parent package`，应检查启动方式与包上下文；`No module named 'regex'` 则要检查依赖是否在当前解释器搜索路径内。两种错误的修正位置不同。

### 项目工程边界

直接运行文件时，脚本所在目录通常加入 `sys.path`；`-m` 运行方式的搜索起点有所不同。应用目录里的 `torch.py`、`json.py` 可能遮住真正的库，诊断时查看模块的 `__file__`。

包根下的 `__init__.py` 可以重新导出公开接口，也可能触发额外导入。阅读 `from transformers import GPT2LMHeadModel` 时，最终类的定义未必就在包根文件里。课程代码用固定目录和显式入口，比靠临时修改 `sys.path` 猜测导入结果更容易核对。

### 代码审查题

为什么改了一个模块全局常量，另一个文件 `from module import CONSTANT` 得到的名字可能还是旧值？Python 相对导入与当前工作目录是否是同一个概念？

<details><summary>核对答案</summary>

导入时形成的是本地绑定；原模块属性后续重绑定不自动重写该绑定。相对导入依据包上下文解析；工作目录只影响部分搜索路径与相对文件地址。

</details>


---

<a id="unit-10"></a>

## 10 · 对象协议、绑定方法与 dataclass

**迁移目标：**利用已有 OOP 经验快速读 Python 类，重点掌握动态成员、显式 self、工厂与协议，而不是重学封装和继承的概念。

### 与熟悉的类语法对照

| Python | 迁移时需要知道 |
|---|---|
| `self` | 显式写在实例方法签名里；通过实例调用时自动绑定 |
| `__init__` | 初始化实例，通常返回 `None`；分配/构造过程另有 `__new__` |
| `@classmethod` / `cls` | 绑定调用的类，适合多态工厂 |
| `@staticmethod` | 不自动传实例或类；只是放在类命名空间的函数 |
| `@property` | 属性读取可触发方法执行，不必是直接访问存储字段 |
| `super()` | 按 MRO 找下一站；单继承时常可按父类理解 |
| `_name` / `__name` | 前者是内部使用约定，后者有名字改写；不等同于 Java private |

`obj.method(x)` 对普通实例方法可按 `Type.method(obj,x)` 理解，self 类似显式写出的接收者。类实例通常可以动态增加属性，但有 `__slots__`、描述符和框架钩子等例外。

### 类属性与实例属性别混淆

```python
class Cache:
    entries = []
```

这张列表在类上共享，两个实例都可能通过属性查找到它。执行 `self.entries = []` 创建实例自己的绑定，和 `self.entries.append(...)` 修改已有共享列表不是一回事。无意共享训练历史、缓存、子层列表是常见错误。

### `model(...)` 属于可调用对象协议

`__call__` 使实例可调用，`__getitem__` 支持索引，`__len__` 支持 `len`。这些“特殊方法”让框架对象接入 Python 语法。

PyTorch 已在 `nn.Module` 提供调用流程，进入子类 `forward` 并处理相关钩子。正常使用 `model(inputs)`；直接调用 `model.forward(inputs)` 会绕过 Module 的调用入口。不要为了模仿普通函数而覆盖掉这层机制。

`from_pretrained` 是类方法工厂，和 `__init__`、`forward` 分别处于加载入口、实例初始化与输入计算三个层面。

### dataclass 减少样板，不提供完整 schema

`@dataclass` 可以生成初始化、显示与相等性等方法；`field(default_factory=list)` 为每个实例新建列表。`frozen=True` 阻止常规字段重新赋值，但不深度冻结字段指向的列表，也不自动验证类型和维度。

它可以承担配置对象的角色，但不能直接等同于“编译器保证字段类型且整个对象图不可变”。

<details><summary>实验：多态工厂、属性协议与 frozen 的边界</summary>

**可运行例子：** [examples/10_objects.py](examples/10_objects.py)


```python
"""10：绑定方法、工厂、多态构造与 dataclass 的边界。"""
from dataclasses import dataclass, field, FrozenInstanceError
from typing import List


@dataclass(frozen=True)
class ModelConfig:
    width: int = 16
    notes: List[str] = field(default_factory=list)


class TokenTable:
    def __init__(self, rows):
        self.rows = rows

    def lookup(self, token_id):
        return self.rows[token_id]

    def __getitem__(self, token_id):
        return self.lookup(token_id)

    def __len__(self):
        return len(self.rows)

    @property
    def width(self):
        return len(self.rows[0]) if self.rows else 0

    @classmethod
    def zeros(cls, count, width):
        return cls([[0.0] * width for _ in range(count)])

    @staticmethod
    def valid_id(token_id, count):
        return 0 <= token_id < count


class NamedTable(TokenTable):
    pass


table = NamedTable.zeros(3, 2)
print("classmethod 沿用调用的子类：", type(table).__name__)
print("绑定方法 / 显式传实例：", table.lookup(1), TokenTable.lookup(table, 1))
print("协议访问 / property / staticmethod：", table[1], table.width, table.valid_id(2, len(table)))
a, b = ModelConfig(), ModelConfig()
a.notes.append("可以修改内部列表")
print("独立的 default_factory：", a.notes, b.notes)
try:
    a.width = 32
except FrozenInstanceError as error:
    print("frozen 阻止字段重新赋值：", type(error).__name__)
assert a.notes is not b.notes
```

**运行方式：**

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/python_foundations/examples/10_objects.py
```

**本机实际输出：**

```text
classmethod 沿用调用的子类： NamedTable
绑定方法 / 显式传实例： [0.0, 0.0] [0.0, 0.0]
协议访问 / property / staticmethod： [0.0, 0.0] 2 True
独立的 default_factory： ['可以修改内部列表'] []
frozen 阻止字段重新赋值： FrozenInstanceError
```


</details>

### 代码审查题

`@classmethod def create(cls): return Base()` 会保留子类构造行为吗？配置设为 `frozen=True` 后，能否无条件假设适合做哈希键？

<details><summary>核对答案</summary>

写死 `Base()` 不会；合适情况下应使用 `cls(...)`，并保证子类构造契约兼容。frozen dataclass 可能生成哈希，但若字段包含列表等不可哈希对象，实际计算哈希仍会失败。冻结字段赋值不等于所有字段可哈希。

</details>


---

<a id="unit-11"></a>

## 11 · with、路径与序列化边界

**迁移目标：**将已有资源管理经验迁移到上下文管理协议，同时区分“地址”“文件内容”“解析后的对象”。

### with 是可扩展的进入/退出协议

`with resource as x:` 通过上下文管理器的 `__enter__` / `__exit__` 管理作用域；`x` 接收 `__enter__` 的返回结果，不一定就是原管理器对象。正常离开或异常离开都会执行退出逻辑；`__exit__` 还可以选择抑制异常。

可以类比 C++ RAII 和 Java try-with-resources 的确定性清理目的，但 Python 的 `with` 不要求对象此时销毁，也不只用于文件资源。`torch.no_grad()` 临时改变梯度记录状态，就是相同语法下的另一种协议实现。不要依赖 `__del__` 的调用时机来代替确定性清理。

### Path 与 JSON 的最小速查

| 写法 | 实际动作 |
|---|---|
| `Path.home() / "dir" / name` | 创建路径对象；`/` 被路径类型定义为拼接 |
| `Path(__file__).resolve().parent` | 定位脚本资源目录，不是 `cwd()` |
| `.read_text(encoding="utf-8")` | 磁盘数据 → str |
| `json.loads(text)` / `json.load(file)` | 文本 / 文件流 → Python 对象 |
| `json.dumps(obj)` / `json.dump(obj,file)` | Python 对象 → 文本 / 文件流 |
| `.read_bytes()` | 原始 bytes，不尝试文本解码 |

JSON 的 `true/false/null` 变为 Python `True/False/None`。解析只保证语法合法，不保证字段存在、维度合法或数据结构满足模型要求。

### 格式与所有权要分开

`config.json` 给结构设置，权重文件给张量数值。Path 不下载数据，JSON 解析器不理解模型结构，safetensors 也不替你保存 Python 类的完整实现。

`data_offsets=[start,end]` 使用左闭右开区间；在 safetensors 中相对于数据区起点，不是整个文件第 0 字节。读格式库的契约比按熟悉的文件格式类比更可靠。

<details><summary>实验：JSON 往返与异常发生后的上下文退出</summary>

**可运行例子：** [examples/11_files.py](examples/11_files.py)


```python
"""11：路径对象、数据转换与上下文的异常退出。"""
from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def session(label):
    print("进入上下文：", label)
    try:
        yield {"label": label}
    finally:
        print("退出上下文：", label)


with TemporaryDirectory() as directory:
    path = Path(directory) / "config.json"
    config = {"width": 16, "checkpoint": None, "training": False}
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    print("JSON 往返：", loaded == config)
    print("JSON 与 Python 的空值写法：", json.dumps(None), repr(None))
    binary = Path(directory) / "data.bin"
    binary.write_bytes(bytes([10, 20, 30, 40]))
    print("偏移 [1,3) 的字节：", list(binary.read_bytes()[1:3]))

try:
    with session("toy") as context:
        print("拿到资源：", context)
        raise ValueError("模拟块内失败")
except ValueError:
    print("外层捕获；清理已经执行")
```

**运行方式：**

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/python_foundations/examples/11_files.py
```

**本机实际输出：**

```text
JSON 往返： True
JSON 与 Python 的空值写法： null None
偏移 [1,3) 的字节： [20, 30]
进入上下文： toy
拿到资源： {'label': 'toy'}
退出上下文： toy
外层捕获；清理已经执行
```


</details>

### 代码审查题

为什么 `with torch.no_grad(): ...` 结束后可以恢复先前状态，而不需要销毁模型？把报告 dict 写成 JSON，等不等于保存了可直接恢复的模型？

<details><summary>核对答案</summary>

上下文管理器控制的是临时状态的进入和恢复，与模型生命周期无关。报告是你主动组织的可序列化信息；模型还需要结构、参数、配置，以及恢复训练时可能需要的优化器和随机状态等。

</details>


---

<a id="unit-12"></a>

## 12 · 类型提示、异常链与输入契约

**迁移目标：**保留静态语言中重视接口契约的习惯，同时明确 Python 解释器、类型检查器与运行时校验的职责。

### 类型提示不会自动执法

```python
def identity(value: int) -> int:
    return value

result = identity("text")
```

这段代码在解释器中可以运行并返回字符串；静态检查工具可以指出它违反注解。`list[int]`、`Optional[int]`、`TypedDict` 等也不会自动变成序列化框架或 schema 验证器。

`Optional[int]` 表示 `int` 或 `None`，不表示参数可以省略；能否省略由默认值决定。`Any` 表达类型检查中的动态逃生口，`object` 则仍要求使用对任意对象都合法的操作，二者不是同一含义。

本项目 Python 3.9 支持 `list[int]`，但常用的 `int | None` 联合类型运行时写法从 3.10 开始；这里使用 `typing.Optional`。类型注解通常也不检查 Tensor 的 shape、dtype 或 device，这些仍需库契约与必要的显式检查。

### 错误类型与调用边界

Python 没有 Java checked exception 那样由调用签名强制处理的机制。异常沿调用栈传播，`except` 按类匹配。`raise NewError(...) from error` 保留因果链；`from None` 会抑制原异常上下文的展示，但不适合作为默认隐藏原因的做法。

`except Exception` 和裸 `except:` 不一样：后者连 `KeyboardInterrupt`、`SystemExit` 等也会捕获。只有在确有统一处理策略时才使用很宽的捕获范围。

`assert` 表达内部不变量，可在优化模式下被移除；外部配置校验要使用显式判断与异常。这与你审查边界输入时区分“契约校验”与“调试断言”的习惯一致。

### argparse 只是 CLI 到对象的适配层

`add_argument("--token-id", type=int, default=220)` 把终端字符串转成整数，`--token-id` 对应 `args.token_id`；`action="store_true"` 提供布尔开关。`type=int` 只负责转换，不知道词表上界，仍需检查 `0 <= token_id < vocab_size`。

<details><summary>实验：类型提示不拦截、异常链与合法/非法 ID</summary>

**可运行例子：** [examples/12_errors_cli.py](examples/12_errors_cli.py)


```python
"""12：注解不做运行时校验；CLI 的边界显式验证。"""
import argparse
from typing import Optional


def annotated(value: int) -> int:
    return value


def validate_id(token_id: int, vocab_size: int) -> int:
    if type(token_id) is not int or not 0 <= token_id < vocab_size:
        raise ValueError("token_id 必须是 0 到 vocab_size - 1 之间的整数")
    return token_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-id", type=int, default=220)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    try:
        token_id = validate_id(args.token_id, 50257)
    except ValueError as error:
        parser.error(str(error))
    print("已检查 ID：", token_id)
    print("是否显示全部：", args.all)
    result = annotated("仍然可以传入字符串")
    print("注解没有拦截：", type(result).__name__)
    maybe_id: Optional[int] = None
    print("Optional 的运行时值：", maybe_id)
    try:
        try:
            int("abc")
        except ValueError as error:
            raise RuntimeError("解析配置失败") from error
    except RuntimeError as error:
        print("保留异常链：", type(error).__name__, "<-", type(error.__cause__).__name__)
    assert validate_id(0, 50257) == 0


if __name__ == "__main__":
    main()
```

**运行方式：**

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/python_foundations/examples/12_errors_cli.py
```

**本机实际输出：**

```text
已检查 ID： 220
是否显示全部： False
注解没有拦截： str
Optional 的运行时值： None
保留异常链： RuntimeError <- ValueError
```


可再用 `--token-id 50256 --all` 运行；用 `--token-id -1` 验证拒绝路径。验证程序已覆盖这两种情况。

</details>

### 代码审查题

`def f(x: Optional[int])` 可以无参调用吗？一个 JSON 配置通过类型检查，是否就能保证运行时 `width % heads == 0`？

<details><summary>核对答案</summary>

不能，除非另有默认值。类型检查不能替代外部 JSON 校验，更不会仅凭两个 `int` 的注解证明整除约束；应在系统边界进行运行时检查。

</details>


---

<a id="unit-13"></a>

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


```python
"""2.1 第一块：实际读取 GPT-2 的配置和预训练参数。

课程对应：Karpathy《Let's reproduce GPT-2 (124M)》开头的权重检查。
本文件是独立教学脚本：打印实际数据，不训练、不生成，也不自动下载模型。
在 VS Code 中选择项目 venv，打开此文件后运行，或使用对应的 F5 配置。
"""

import argparse
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent.parent
# 固定本机已经缓存的版本；避免之后 main 更新，课堂使用的参数也跟着改变。
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
DEFAULT_MODEL_DIR = (
    Path.home() / ".cache/huggingface/hub/models--gpt2/snapshots" / REVISION
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--token-id", type=int, default=220,
                        help="查看哪一个 Token 的向量；默认 220 对应空格")
    parser.add_argument("--all", action="store_true", help="列出全部独立参数")
    parser.add_argument("--json", type=Path, help="额外保存本次实际读取的检查报告")
    args = parser.parse_args()

    # 第一步：直接读 JSON。它给出结构尺寸和计算设置，没有训练好的向量数值。
    model_dir = args.model_dir.expanduser().resolve()
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        parser.error("目录中需要 config.json 和 model.safetensors；可用 --model-dir 指定。")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {"model_type": "gpt2", "vocab_size": 50257, "n_embd": 768,
                "n_layer": 12, "n_head": 12, "n_positions": 1024}
    if any(config.get(key) != value for key, value in expected.items()):
        parser.error("本课固定检查 GPT-2 124M；该目录中的配置与本课模型不匹配。")
    if not 0 <= args.token_id < config["vocab_size"]:
        parser.error("--token-id 必须在 0 到 vocab_size - 1 之间。")

    try:
        import torch
        from transformers import GPT2LMHeadModel
    except ImportError as error:
        raise SystemExit(
            f"无法导入课程依赖：{error}\n"
            f"请在 VS Code 选择解释器：{PROJECT_DIR / 'venv/bin/python'}"
        ) from error

    # 第二步：库中的 GPT2LMHeadModel 类定义计算流程；配置决定各层尺寸；
    # from_pretrained 将保存的权重填进这个结构。只读本地文件，使用 CPU。
    model, loading_info = GPT2LMHeadModel.from_pretrained(
        str(model_dir), local_files_only=True, use_safetensors=True,
        output_loading_info=True,
    )
    model.eval()
    if any(loading_info.get(key) for key in
           ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"权重加载需要核对：{loading_info}")

    # 第三步：state_dict 是「名称 → 张量」的映射。名称帮助定位模块；
    # shape 说明存了多少行、多少列；张量中的数才是训练得到的参数值。
    state = model.state_dict()
    selected = {
        "transformer.wte.weight": "Token 向量表 E：按 Token ID 查行",
        "transformer.wpe.weight": "位置向量表 P：按位置编号查行",
        "transformer.h.0.ln_1.weight": "第 1 个 Block 的 LN1 缩放系数 gamma",
        "transformer.h.0.ln_1.bias": "第 1 个 Block 的 LN1 平移系数 beta",
        "transformer.h.0.attn.c_attn.weight": "Q/K/V 的合并投影权重",
        "transformer.h.0.attn.c_attn.bias": "Q/K/V 的合并投影偏置",
        "transformer.h.0.attn.c_proj.weight": "注意力输出投影权重",
        "transformer.h.0.mlp.c_fc.weight": "MLP 扩展层：768 → 3072",
        "transformer.h.0.mlp.c_proj.weight": "MLP 压回层：3072 → 768",
        "transformer.ln_f.weight": "所有 Block 之后的最终 LN 缩放系数",
        "lm_head.weight": "输出词表打分；与 Token 向量表共享",
    }

    print("2.1 第一块：配置、权重、模型代码怎样配合")
    print(f"实际读取位置：{model_dir}\n")
    for key, meaning in [
        ("vocab_size", "词表行数"), ("n_positions", "位置表行数"),
        ("n_embd", "每个位置的向量维度"), ("n_layer", "Block 个数"),
        ("n_head", "注意力头数"),
    ]:
        print(f"{key:16s} = {config[key]:>6}  {meaning}")
    print(f"每头维度 = {config['n_embd']} / {config['n_head']}"
          f" = {config['n_embd'] // config['n_head']}\n")

    print("实际参数名称 → 形状 → 所属环节")
    for name, meaning in selected.items():
        print(f"{name:41s} {str(list(state[name].shape)):18s} {meaning}")
    print("\n注意：h.0 是第 1 个 Block。")
    print("Hugging Face GPT-2 的这些投影使用 Conv1D 包装，权重按 [输入维, 输出维] 存；")
    print("改用 PyTorch Linear 时存储方向相反，加载时需转置。这个细节后续再实现。")

    # 第四步：只取一行的前 8 个真实数。它不是输入文本，而是参数表的一小部分。
    row = model.transformer.wte.weight[args.token_id, :8].detach().cpu().tolist()
    print(f"\nE[{args.token_id}] 的前 8 个参数值（完整一行有 768 个）：")
    print("[" + ", ".join(f"{x:.7f}" for x in row) + "]")

    # 第五步：model.parameters() 默认对共享参数去重。
    # 直接累加 state_dict 的所有条目会重复计算 E / lm_head 这一共享表。
    tied = model.transformer.wte.weight is model.lm_head.weight
    groups = {
        "Token Embedding": model.transformer.wte.weight.numel(),
        "Position Embedding": model.transformer.wpe.weight.numel(),
        "12 个 Block": sum(p.numel() for p in model.transformer.h.parameters()),
        "最终 LayerNorm": sum(p.numel() for p in model.transformer.ln_f.parameters()),
    }
    total = sum(p.numel() for p in model.parameters())
    assert sum(groups.values()) == total
    assert tied
    print("\n独立可训练参数的个数：")
    for label, count in groups.items():
        print(f"{label}: {count:,}")
    print(f"合计：{total:,}，约 {total / 1_000_000:.2f}M")
    print(f"E 与 lm_head 是否为同一个参数对象：{tied}")
    print("权重加载检查：没有缺失项、多余项或形状冲突。")
    print("本次仅检查配置和权重；尚未执行生成、训练或与自写模型对齐。")

    parameters = [
        {"name": name, "shape": list(p.shape), "numel": p.numel(), "dtype": str(p.dtype)}
        for name, p in model.named_parameters()
    ]
    if args.all:
        print("\n全部独立参数（共享项只列一次）：")
        for item in parameters:
            print(item["name"], item["shape"])
    if args.json:
        report = {
            "source": {"model_dir": str(model_dir), "snapshot_revision": model_dir.name,
                       "config_file": str(config_path), "weights_file": str(weights_path)},
            "environment": {"python": sys.version.split()[0], "torch": torch.__version__},
            "config": config, "loading_info": loading_info,
            "selected_shapes": {name: list(state[name].shape) for name in selected},
            "embedding_sample": {"token_id": args.token_id, "dimensions": [0, 8], "values": row},
            "parameter_groups": groups, "total_parameters": total,
            "embedding_and_head_same_parameter": tied, "parameters": parameters,
            "forward_ran": False, "training_ran": False,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"检查报告：{args.json.resolve()}")


if __name__ == "__main__":
    main()
```


</details>

### 代码审查题

把 `state = model.state_dict()` 存到变量后继续训练，是否拿到了训练前的稳定快照？`model.eval()` 为什么不能替代一次 `model(input_ids)`？

<details><summary>核对答案</summary>

没有保证快照独立，后续更新可能通过共享存储反映在 `state` 中。`eval` 修改模式；`model(input_ids)` 才进入具体前向。下一单元把 Python 对象、Tensor 存储和 autograd 三种关系分别验证。

</details>


---

<a id="unit-14"></a>

## 14 · Tensor：形状、存储别名与 autograd

**迁移目标：**用你对数组、视图和内存的经验理解 Tensor，同时补上 autograd 这个独立维度。关键是分别询问对象身份、底层存储与梯度路径。

### 三种“相同”，互不等价

| 表达式 | Python 对象 | 底层数据 | 梯度关系 |
|---|---|---|---|
| `y = x` | 同一个对象 | 共享 | 原样 |
| `y = x.detach()` | 新 Tensor 对象 | 共享 | 切断该结果的 autograd 连接 |
| `y = x.clone()` | 新 Tensor 对象 | 复制 | 可以保留到 x 的梯度路径 |
| `y = x.detach().clone()` | 新 Tensor 对象 | 复制 | 不再通过该结果跟踪到 x |

所以“detach 后修改不会影响原参数”是错误推断。“clone 一定关闭梯度”同样错误。示例使用同起点张量的 `data_ptr` 辅助核对共享；一般带偏移视图不应只靠起始地址判断所有别名关系。

### 索引既影响 shape，也可能影响所有权

设 `X.shape=[B,T,C]=[2,3,4]`：

| 表达式 | shape | 读取语义 |
|---|---|---|
| `X[:, -1, :]` | `[2,4]` | 整数索引移除位置轴，基本索引通常为视图 |
| `X[:, -1:, :]` | `[2,1,4]` | 切片保留轴，基本索引通常为视图 |
| `X[[0,1]]` | `[2,3,4]` | 这里是高级索引，读取产生副本 |

Python list 切片新建外层；Tensor 基本切片通常共享数据，规则不同。高级索引“读取产生副本”也不能推导出 `X[mask] = value` 不会修改 X：后者执行的是索引赋值。

### 算子与广播快速核对

手工 `W.shape=[4,2]` 时，`X @ W + bias` 输出 `[2,3,2]`，偏置 `[2]` 从尾部轴广播。`nn.Linear(4,2)` 保存的权重则为 `[2,4]`，内部按转置后的方向参与计算。

`*` 是逐元素乘，`@` 是矩阵乘法；`==` 返回逐元素布尔 Tensor。多个元素的 Tensor 不能直接作为 Python `if` 条件，需明确使用 `.any()`、`.all()` 等归约。逐元素逻辑常写 `(x > 0) & (x < 1)`，不要把 Python `and` 当作按元素运算。

### reshape 与 transpose 不是一个动作

`transpose`/`permute` 重排轴，经常产生非连续视图。`view` 需要布局兼容；`reshape` 可在必要时复制，不能据方法名判断结果必然共享。`contiguous()` 也不是无条件复制：输入已经连续时可以复用。

学习多头注意力代码时，同时写出语义轴 `[B,T,H,Dh]` 与实际内存布局；元素数匹配只证明能排出某种形状，不证明 Token、头和维度的对应关系正确。

<details><summary>实验：视图修改、布局与 detach/clone</summary>

**可运行例子：** [examples/14_tensors.py](examples/14_tensors.py)


```python
"""14：形状、基本索引的视图、复制与 autograd 边界。"""
import torch

x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
bias = torch.tensor([0.5, -0.5])
print("X / W / Y：", tuple(x.shape), tuple(weight.shape), tuple((x @ weight + bias).shape))
print("首位置投影结果：", (x @ weight + bias)[0, 0].tolist())
print("整数索引消轴 / 切片留轴：", tuple(x[:, -1, :].shape), tuple(x[:, -1:, :].shape))

base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
view = base[:, 1:]
snapshot = base.clone()
view[0, 0] = -10
print("修改视图后原值改变：", base.tolist())
print("clone 的独立快照：", snapshot.tolist())
assert base[0, 1].item() == -10 and snapshot[0, 1].item() == 1

transposed = x.transpose(1, 2)
print("转置 shape / 连续性：", tuple(transposed.shape), transposed.is_contiguous())
print("连续布局后 view：", tuple(transposed.contiguous().view(2, 12).shape))
try:
    bool(torch.tensor([1, 2]))
except RuntimeError as error:
    print("多元素张量真值不明确：", type(error).__name__)

leaf = torch.tensor([2.0], requires_grad=True)
detached = leaf.detach()
cloned = leaf.clone()
frozen = leaf.detach().clone()
print("detach：共享存储 / 需要梯度：", detached.data_ptr() == leaf.data_ptr(), detached.requires_grad)
print("clone：独立存储 / 有梯度路径：", cloned.data_ptr() != leaf.data_ptr(), cloned.grad_fn is not None)
print("detach+clone：独立存储 / 需要梯度：", frozen.data_ptr() != leaf.data_ptr(), frozen.requires_grad)
assert detached.data_ptr() == leaf.data_ptr() and cloned.grad_fn is not None
```

**运行方式：**

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/python_foundations/examples/14_tensors.py
```

**本机实际输出：**

```text
X / W / Y： (2, 3, 4) (4, 2) (2, 3, 2)
首位置投影结果： [2.5, 3.5]
整数索引消轴 / 切片留轴： (2, 4) (2, 1, 4)
修改视图后原值改变： [[0.0, -10.0, 2.0], [3.0, 4.0, 5.0]]
clone 的独立快照： [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
转置 shape / 连续性： (2, 4, 3) False
连续布局后 view： (2, 12)
多元素张量真值不明确： RuntimeError
detach：共享存储 / 需要梯度： True False
clone：独立存储 / 有梯度路径： True True
detach+clone：独立存储 / 需要梯度： True False
```


</details>

### 原地更新是另一个契约

PyTorch 常用末尾下划线表示原地操作，如 `add_`、`copy_`。在梯度跟踪过程中改动被后续反向依赖的数据，可能被版本检查拒绝。受控加载参数常在 `torch.no_grad()` 中执行 `copy_`，这仍是在修改原参数，不是创建新对象。

Tensor 视图规则可核对 [PyTorch 官方视图说明](https://docs.pytorch.org/docs/2.14/tensor_view.html)；本教材的具体例子在本机 PyTorch 2.2.2 上运行验证，升级文档不替代本地行为核对。

### 代码审查题

`best = model.wte.weight.detach()` 能作为独立的最佳权重备份吗？`weights == other_weights` 能直接拿去 `if` 吗？

<details><summary>核对答案</summary>

前者仍共享存储，通常需要 `detach().clone()` 或序列化快照。后者为逐元素结果，需明确比较语义：如 `torch.equal` 做严格整体相等，或 `torch.allclose` 做浮点容差比较，不能让多元素结果隐式转 bool。

</details>


---

<a id="unit-15"></a>

## 15 · nn.Module：注册、执行、更新与保存

**迁移目标：**理解框架如何建立对象图并发现参数。你已学过训练数学，本单元关注 Python 代码中的机制和生命周期边界。

### 注册不是“对象存在”就自动发生

```python
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.visible = nn.Linear(4, 4)
        self.hidden = [nn.Linear(4, 4)]
        self.blocks = nn.ModuleList([nn.Linear(4, 4)])
```

`self.visible` 与 `self.blocks` 及其中层会被 Module 管理；`self.hidden` 是普通列表，列表内部层不会仅凭这个容器赋值被自动注册。因此它们可能不出现在 `parameters()`、`state_dict()` 和设备迁移的递归操作中。

这不是 Python “看不见对象”，而是框架注册协议没有沿任意对象图递归。需要管理层列表时使用 `ModuleList`，参数列表用 `ParameterList`；非参数但需要随模型迁移/保存的状态可用适当的 buffer 注册方式。

### 三个入口对应三个阶段

| 入口 | 生命周期职责 |
|---|---|
| `__init__` | 建立可复用层与参数，先初始化 Module 基础设施 |
| `forward` | 描述输入如何使用当前状态计算结果 |
| `model(inputs)` | 经过 Module 调用入口进入 forward，包含相关 hook 机制 |

在 `forward` 临时新建 Linear 会产生新的随机参数，不能当成已有层持续训练。循环和条件可以参与构造执行路径，但不能因此忽略参数注册与复用。

### 状态变化逐项核对

| 操作 | 改变什么 | 未完成什么 |
|---|---|---|
| `zero_grad()` | 清理已累积梯度 | 不重新初始化权重 |
| `model(x)` | 产生本次前向结果及可能的图 | 不根据损失更新参数 |
| `loss.backward()` | 沿图累积梯度 | 不等同于优化器 step |
| `optimizer.step()` | 按梯度及优化器状态更新参数 | 不自动清掉下一轮要累积的梯度 |
| `eval()` | 修改模式标志与相关层行为 | 不关闭 autograd，不执行前向 |
| `no_grad()` | 临时关闭梯度记录 | 不自动让 Dropout 进入评估模式 |

### 小模型的输入输出契约

配套例子用 `Embedding(6,4)` 与 `Linear(4,6)`：输入 ID `[1,3]` → hidden `[1,3,4]` → logits `[1,3,6]` → 标量交叉熵。`F.cross_entropy` 接受 logits，不先做 softmax。参数总数为 54，这里两层不共享权重。

这是独立的接口实验，没有 Attention 或位置向量，不是缩小版完整 GPT-2。固定 seed 用于可复查本次实验，不声称跨所有平台/版本能逐位复现任意训练。

<details><summary>实验：一次更新、模块注册与保存后重载</summary>

**可运行例子：** [examples/15_tiny_model.py](examples/15_tiny_model.py)


```python
"""15：模块注册、调用、梯度和序列化的语义边界。

它只有 Embedding 与 Linear，没有 Attention，也没有位置向量，不能称为 GPT。
不读取、不下载、不修改真实 GPT-2 权重。
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file


class TinyNextToken(nn.Module):
    def __init__(self, vocab_size=6, width=4):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, width)
        self.lm_head = nn.Linear(width, vocab_size)

    def forward(self, input_ids):
        hidden = self.wte(input_ids)
        logits = self.lm_head(hidden)
        return logits


torch.manual_seed(7)
model = TinyNextToken()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
targets = torch.tensor([[2, 3, 4]], dtype=torch.long)

model.train()
optimizer.zero_grad()
logits = model(input_ids)
loss = F.cross_entropy(logits.reshape(-1, 6), targets.reshape(-1))
state_view = model.state_dict()  # 映射是新的，但值的底层存储并非独立快照。
before = model.wte.weight.detach().clone()
loss.backward()
print("输入 ID 形状：", tuple(input_ids.shape))
print("输出 logits 形状：", tuple(logits.shape))
print("总参数量：", sum(parameter.numel() for parameter in model.parameters()))
print("损失形状（标量）：", tuple(loss.shape))
print("产生了 Embedding 梯度：", model.wte.weight.grad is not None)
optimizer.step()
print("更新后权重改变：", not torch.equal(before, model.wte.weight))
print("state_dict 与参数共享存储：", state_view["wte.weight"].data_ptr() == model.wte.weight.data_ptr())

model.eval()
with torch.no_grad():
    scores = model(input_ids)
print("eval 后 training 标记：", model.training)
print("no_grad 中输出需要梯度：", scores.requires_grad)

with TemporaryDirectory() as temp_dir:
    path = Path(temp_dir) / "tiny.safetensors"
    save_file(model.state_dict(), str(path))
    restored = TinyNextToken()
    restored.load_state_dict(load_file(str(path)))
    restored.eval()
    with torch.no_grad():
        reloaded_scores = restored(input_ids)
    print("保存并重载后输出相同：", torch.equal(scores, reloaded_scores))

class RegistryProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.plain_list = [nn.Linear(2, 2)]
        self.module_list = nn.ModuleList([nn.Linear(2, 2)])


probe = RegistryProbe()
print("自动注册的参数名：", [name for name, _ in probe.named_parameters()])
assert sum(p.numel() for p in probe.parameters()) == 6
```

**运行方式：**

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/python_foundations/examples/15_tiny_model.py
```

**本机实际输出：**

```text
输入 ID 形状： (1, 3)
输出 logits 形状： (1, 3, 6)
总参数量： 54
损失形状（标量）： ()
产生了 Embedding 梯度： True
更新后权重改变： True
state_dict 与参数共享存储： True
eval 后 training 标记： False
no_grad 中输出需要梯度： False
保存并重载后输出相同： True
自动注册的参数名： ['module_list.0.weight', 'module_list.0.bias']
```


</details>

### 状态映射不是独立快照，也不是完整程序

`state_dict()` 的值通常继续共享模型参数/buffer 的存储；它并不自动深复制模型状态。`safetensors.save_file` 序列化张量数值，不保存全部 Python 类代码和优化器上下文。恢复先建立兼容结构，再加载张量并核对键、shape 和输出。

本例写入临时目录、重载并对齐输出。真实续训还要根据目标保存优化器状态、步数、随机状态等；只读 GPT-2 预训练权重和恢复一次中断训练，契约不同。

### 代码审查题

一个模型能 forward，是否证明所有层都能被 optimizer 更新？为什么 `best_state=model.state_dict()` 后继续训练，可能导致“最佳权重”跟着变化？

<details><summary>核对答案</summary>

不能，forward 可以直接调用普通容器中的层，但优化器可能没拿到它们。状态映射中的张量可共享底层存储，后续原地更新会反映过去；需要独立复制或在目标时间点实际写出快照。

</details>


---

<a id="unit-16"></a>

## 16 · 综合实战：实现并审查配置接口

**迁移目标：**用一个熟悉的工程任务检验 Python 语义。功能规模保持小，重点审查输入契约、对象所有权、异常和返回形状。

### 任务与接口约束

读取 [教学配置](fixtures/demo_gpt2_config.json)，生成 Token/Position Embedding 的形状与参数量，以及每头维度。实现接口 `inspect_config(config, *, output_info=False)`：

1. 不修改传入字典；检查五个必需字段存在。
2. 维度必须为正整数，明确拒绝 Python 的 `bool`。
3. `n_embd % n_head == 0`，避免静默向下取整。
4. 默认返回报告 dict；开关开启时返回 `(report, info)`。
5. 每次调用创建自己的报告与信息，不能共享可变默认状态。
6. CLI 支持选择配置及可选 JSON 输出；输入问题应给出明确失败原因。

`argparse` 入口还应检查 JSON 最外层是对象。这个练习验证 Python API 的组织方式，不计算完整模型参数，也不读取预训练数值。

### 固定验收数据


```json
{
  "model_type": "gpt2",
  "vocab_size": 50257,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_positions": 1024
}
```


| 输出 | 期望 |
|---|---|
| Token 表 shape / 参数数 | `[50257,768]` / `38,597,376` |
| Position 表 shape / 参数数 | `[1024,768]` / `786,432` |
| head_dim | `64` |
| 缺字段、`n_head=13`、`n_embd=True` | 拒绝而非猜测修正 |

从 [练习起点](exercises/config_report.py) 开始；模板保留 `NotImplementedError`，不算已实现程序。代码量不大，建议先写完再和参考实现比较，不必追求逐行相同。

### 先审查这段有问题的实现

```python
def inspect(config, info={}):
    width = config.get("n_embd") or 768
    heads = config.get("n_head") or 12
    info["head_dim"] = width // heads
    return info
```

它有哪些隐藏契约？如果调用者有意传 0、配置缺键、上次返回的 info 被保留、宽度不能整除头数，各自发生什么？

<details><summary>审查结论</summary>

默认字典跨调用共享；`or` 把 0 与缺失混在一起；缺少严格类型与正数验证；不能整除时 `//` 隐藏问题；返回同一个可变对象导致调用之间互相影响。应先验证输入，再创建独立输出。不能靠类型注解替代这些验证。

</details>

<details><summary>参考实现与实际输出</summary>

**可运行例子：** [answers/config_report.py](answers/config_report.py)


```python
"""综合练习参考答案：读取配置并检查形状；不加载真实模型。"""

import argparse
import json
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "fixtures" / "demo_gpt2_config.json"


def inspect_config(config, *, output_info=False):
    required = ("vocab_size", "n_embd", "n_head", "n_layer", "n_positions")
    missing_keys = [key for key in required if key not in config]
    if missing_keys:
        raise ValueError("缺少字段：" + ", ".join(missing_keys))
    for key in required:
        value = config[key]
        if type(value) is not int or value <= 0:
            raise ValueError(key + " 必须是正整数")
    width = config["n_embd"]
    heads = config["n_head"]
    if width % heads != 0:
        raise ValueError("n_embd 必须能被 n_head 整除")

    vocab_size = config["vocab_size"]
    positions = config["n_positions"]
    report = {
        "token_embedding_shape": [vocab_size, width],
        "position_embedding_shape": [positions, width],
        "head_dim": width // heads,
        "token_embedding_parameters": vocab_size * width,
        "position_embedding_parameters": positions * width,
    }
    info = {"missing_keys": [], "checked_fields": list(required)}
    if output_info:
        return report, info
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("配置最外层必须是 JSON 对象")
        report, info = inspect_config(config, output_info=True)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    text = json.dumps({"report": report, "info": info}, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print("报告已保存：", args.output)


if __name__ == "__main__":
    main()
```

**运行方式：**

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/python_foundations/answers/config_report.py
```

**本机实际输出：**

```text
{
  "report": {
    "token_embedding_shape": [
      50257,
      768
    ],
    "position_embedding_shape": [
      1024,
      768
    ],
    "head_dim": 64,
    "token_embedding_parameters": 38597376,
    "position_embedding_parameters": 786432
  },
  "info": {
    "missing_keys": [],
    "checked_fields": [
      "vocab_size",
      "n_embd",
      "n_head",
      "n_layer",
      "n_positions"
    ]
  }
}
```


答案保留普通 dict，便于对照当前检查脚本。进一步可以用 dataclass 表达结构，但仍应保留外部输入校验；不能假定 dataclass 自动完成它。

</details>

### 回到 2.1 的准备度检查

能够解释以下行为，就可以回到主线边读边补剩余语法，不必先掌握整个 Python 生态：

- 调用、关键字实参、返回拆包与可变默认值的契约。
- 别名、浅/深复制、Tensor 视图与梯度连接的区别。
- 类方法工厂、Module 调用入口和注册参数的边界。
- 相对导入与解释器依赖问题各从哪里排查。
- `inspect_checkpoint.py` 哪些操作读数据、创建对象、检查状态或写文件。

后续课程遇到 multiprocessing、异步 I/O、打包发布或更复杂的类型设计时再补；这些暂不阻塞 GPT-2 复现。


---

## 速查 · 语言迁移中的十个审查问题

| 看到的写法 | 立即核对 |
|---|---|
| `b = a` | 重新绑定还是复制？后续修改是否共享？ |
| `a == b` / `a is b` | 比较值还是身份？返回 bool 还是 Tensor？ |
| `value or fallback` | 合法的 0、False、空容器是否被覆盖？ |
| `def f(x=[]):` | 默认对象是否跨调用复用？ |
| `lambda: i` | 读共享绑定还是期望每轮快照？ |
| `for x in source` | source 能否重新迭代，是否已被调试输出消费？ |
| `from m import x` | 本地绑定是否会随 m.x 的重绑定自动变化？ |
| `@decorator` | 何时执行、返回什么、是否替换被装饰对象？ |
| `self.layers=[...]` | Module 是否注册到了容器内部的层？ |
| `snapshot = tensor.detach()` | 切断梯度是否被误当成复制数据？ |

## 版本边界与官方核对

教材实际运行环境为 **Python 3.9.6、PyTorch 2.2.2、safetensors 0.7.0**。语言迁移中的行为实验以这个环境的输出为准；官方文档与程序角色不同，前者说明契约，后者核对本机例子。

| 特性 | 本教材处理 |
|---|---|
| dict 保留插入次序 | Python 3.7 起的语言保证 |
| 位置专用形参 `/` | Python 3.8 起支持，本机可用 |
| `list[int]` | Python 3.9 可用 |
| `int \| None`、`match`、`zip(strict=True)` | 常用写法从 3.10 引入，本机示例不直接使用 |

需要深入核对时使用官方资料：

- [Python 执行模型](https://docs.python.org/3.9/reference/executionmodel.html)：名称绑定、作用域和异常传播。
- [Python 函数与控制流](https://docs.python.org/3.9/tutorial/controlflow.html)：参数位置、默认值与调用规则。
- [复制协议](https://docs.python.org/3.9/library/copy.html)：浅复制、深复制与自定义对象。
- [Java String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html) / [Dart String.length](https://api.dart.dev/dart-core/String/length.html)：跨语言长度计数的核对。
- [PyTorch Tensor 视图](https://docs.pytorch.org/docs/2.14/tensor_view.html)：存储共享与视图行为，结合本机版本实测阅读。

## 验证范围

16 个可运行教学程序、包入口、CLI 合法/非法参数、配置边界与报告读写均已执行。核心易错语义在相应例子中包含断言。练习 TODO 只检查语法，真实 GPT-2 源码只做阅读核对；本次不重新加载大型权重、不修改课程模型代码。

[verification.json](verification.json) 记录实际输出与环境。正文中的输出框显示 stdout，完整记录另保留 stderr；本机小模型实验可能显示已有的 urllib3/LibreSSL 兼容性警告，成功退出与重载断言分别核对，不把警告当作模型计算结果。

材料已调整为 **Java/C++/Dart 资深开发者的 Python 迁移版**。阅读级别是建议投入方式，页面滚动进度也只是阅读位置；它们不自动改变 Notion 学习进度。
