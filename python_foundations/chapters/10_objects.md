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

<!-- EXAMPLE: examples/10_objects.py -->

</details>

### 代码审查题

`@classmethod def create(cls): return Base()` 会保留子类构造行为吗？配置设为 `frozen=True` 后，能否无条件假设适合做哈希键？

<details><summary>核对答案</summary>

写死 `Base()` 不会；合适情况下应使用 `cls(...)`，并保证子类构造契约兼容。frozen dataclass 可能生成哈希，但若字段包含列表等不可哈希对象，实际计算哈希仍会失败。冻结字段赋值不等于所有字段可哈希。

</details>
