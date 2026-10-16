## 速查 · 语言迁移中的十个审查问题

| 看到的写法 | 立即核对 |
|---|---|
| `b = a` | 重新绑定还是复制？后续修改是否共享？ |
| `a == b` / `a is b` | 比较值还是身份？返回 bool 还是 Tensor？ |
| `value or fallback` | 合法的 0、False、空容器是否被覆盖？ |
| `def f(x=[]):` | 默认对象是否跨调用复用？ |
| `lambda: i` | 读共享绑定还是期望每轮快照？ |
| `for x in source` | source 能否重新迭代，是否已被调试输出消费？ |
| `from m import x` | 本地绑定是否会随 m.x 的重绑定自动变化？ |
| `@decorator` | 何时执行、返回什么、是否替换被装饰对象？ |
| `self.layers=[...]` | Module 是否注册到了容器内部的层？ |
| `snapshot = tensor.detach()` | 切断梯度是否被误当成复制数据？ |

## 版本边界与官方核对

教材实际运行环境为 **Python 3.9.6、PyTorch 2.2.2、safetensors 0.7.0**。语言迁移中的行为实验以这个环境的输出为准；官方文档与程序角色不同，前者说明契约，后者核对本机例子。

| 特性 | 本教材处理 |
|---|---|
| dict 保留插入次序 | Python 3.7 起的语言保证 |
| 位置专用形参 `/` | Python 3.8 起支持，本机可用 |
| `list[int]` | Python 3.9 可用 |
| `int \| None`、`match`、`zip(strict=True)` | 常用写法从 3.10 引入，本机示例不直接使用 |

需要深入核对时使用官方资料：

- [Python 执行模型](https://docs.python.org/3.9/reference/executionmodel.html)：名称绑定、作用域和异常传播。
- [Python 函数与控制流](https://docs.python.org/3.9/tutorial/controlflow.html)：参数位置、默认值与调用规则。
- [复制协议](https://docs.python.org/3.9/library/copy.html)：浅复制、深复制与自定义对象。
- [Java String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html) / [Dart String.length](https://api.dart.dev/dart-core/String/length.html)：跨语言长度计数的核对。
- [PyTorch Tensor 视图](https://docs.pytorch.org/docs/2.14/tensor_view.html)：存储共享与视图行为，结合本机版本实测阅读。

## 验证范围

16 个可运行教学程序、包入口、CLI 合法/非法参数、配置边界与报告读写均已执行。核心易错语义在相应例子中包含断言。练习 TODO 只检查语法，真实 GPT-2 源码只做阅读核对；本次不重新加载大型权重、不修改课程模型代码。

[verification.json](verification.json) 记录实际输出与环境。正文中的输出框显示 stdout，完整记录另保留 stderr；本机小模型实验可能显示已有的 urllib3/LibreSSL 兼容性警告，成功退出与重载断言分别核对，不把警告当作模型计算结果。

材料已调整为 **Java/C++/Dart 资深开发者的 Python 迁移版**。阅读级别是建议投入方式，页面滚动进度也只是阅读位置；它们不自动改变 Notion 学习进度。
