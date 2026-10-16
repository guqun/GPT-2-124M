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

<!-- EXAMPLE: examples/02_strings.py -->

</details>

### 语义自检

为什么“取字符串的前 8 项”和“取 UTF-8 的前 8 字节”可能无法互换？`repr(text)`、UTF-8 编码与 Token ID 序列分别解决什么问题？

<details><summary>核对答案</summary>

前者按码点，后者可能截断一个多字节字符，得到不能独立解码的片段。`repr` 用于观察对象；UTF-8 是字符到字节的编码；Token ID 还要经过具体 Tokenizer 的切分与词表映射。

</details>
