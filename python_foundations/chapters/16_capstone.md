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

<!-- SOURCE: fixtures/demo_gpt2_config.json -->

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

<!-- EXAMPLE: answers/config_report.py -->

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
