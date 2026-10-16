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

<!-- EXAMPLE: examples/07_functions.py -->

</details>

### 代码审查题

`def load(options={}): ...` 与 `def load(options=None): options=options or {}` 各有什么问题？在默认参数中写 `dtype=DEFAULT_DTYPE` 后，再改全局 `DEFAULT_DTYPE`，默认值会跟着变吗？

<details><summary>核对答案</summary>

前者共享可变默认对象。后者虽然避开了默认对象共享，却会替换调用者明确传入的空字典；是否可接受要看 API 是否要保留原对象。通常使用 `if options is None:`。全局名字之后重新绑定，不会替换已保存的默认值；如果默认对象自身可变，它的内部修改则仍可见。

</details>
