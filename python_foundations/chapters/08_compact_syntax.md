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

<!-- EXAMPLE: examples/08_compact_syntax.py -->

</details>

### 代码审查题

在循环里创建多个 logging hook，全部最后打印同一个层编号，首先检查什么？为什么给 wrapper 加 `@wraps` 不等于让它自动做类型检查？

<details><summary>核对答案</summary>

检查闭包是否共享循环变量绑定，必要时用工厂函数或定义时参数绑定保存期望值。`wraps` 主要保留包装关系和元数据，不替你校验参数类型、返回类型或张量形状。

</details>
