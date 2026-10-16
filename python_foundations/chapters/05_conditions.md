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

<!-- EXAMPLE: examples/05_conditions.py -->

</details>

### 代码审查题

`def add(x): cache.append(x)` 需要 `global cache` 吗？改成 `cache = cache + [x]` 呢？

<details><summary>核对答案</summary>

前者只读取外层绑定并修改对象，不因这一操作需要 `global`。后者在函数内赋值，名字被判定为局部，右侧先读时会失败；如果确实要改模块级绑定，应显式声明，但通常更清楚的设计是传入状态并返回结果。

</details>
