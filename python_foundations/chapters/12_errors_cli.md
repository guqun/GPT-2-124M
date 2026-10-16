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

<!-- EXAMPLE: examples/12_errors_cli.py -->

可再用 `--token-id 50256 --all` 运行；用 `--token-id -1` 验证拒绝路径。验证程序已覆盖这两种情况。

</details>

### 代码审查题

`def f(x: Optional[int])` 可以无参调用吗？一个 JSON 配置通过类型检查，是否就能保证运行时 `width % heads == 0`？

<details><summary>核对答案</summary>

不能，除非另有默认值。类型检查不能替代外部 JSON 校验，更不会仅凭两个 `int` 的注解证明整除约束；应在系统边界进行运行时检查。

</details>
