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

<!-- EXAMPLE: examples/06_loops.py -->

</details>

### 代码审查题

`print(list(model.parameters()))` 后再调用一次 `model.parameters()` 能否重新遍历？如果先保存 `params=model.parameters()`，连续消费 `params` 两次呢？

<details><summary>核对答案</summary>

再次调用该方法可以取得新的遍历过程；同一个已经耗尽的 `params` 迭代器则不会自动复位。区别在于“重新调用工厂”和“重复使用同一个消费对象”。

</details>
