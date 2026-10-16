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

<!-- EXAMPLE: examples/03_sequences.py -->

</details>

### 代码审查题

为每个 batch 保存 `history.append(batch[:])`，是否已取得稳定快照？tuple 包装一个列表后，能直接作为字典键吗？

<details><summary>核对答案</summary>

取决于内部元素：嵌套可变对象仍共享，Tensor 还要核对存储和梯度关系。包含列表的 tuple 不能哈希，因此不能直接作字典键；tuple 的每个元素都满足可哈希要求才行。

</details>
