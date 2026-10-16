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

<!-- EXAMPLE: examples/04_mappings.py -->

</details>

### 代码审查题

`decoder = {v:k for k,v in encoder.items()}` 在什么情况下会丢失映射？为什么 `config.get("width", 768)` 不一定返回 768 或一个合法整数？

<details><summary>核对答案</summary>

如果多个键对应同一个值，反向构造时会覆盖。`get` 只处理缺键，不做 schema 校验；已有值可能是 `None`、字符串、负数或布尔值。

</details>
