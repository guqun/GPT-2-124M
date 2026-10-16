## 14 · Tensor：形状、存储别名与 autograd

**迁移目标：**用你对数组、视图和内存的经验理解 Tensor，同时补上 autograd 这个独立维度。关键是分别询问对象身份、底层存储与梯度路径。

### 三种“相同”，互不等价

| 表达式 | Python 对象 | 底层数据 | 梯度关系 |
|---|---|---|---|
| `y = x` | 同一个对象 | 共享 | 原样 |
| `y = x.detach()` | 新 Tensor 对象 | 共享 | 切断该结果的 autograd 连接 |
| `y = x.clone()` | 新 Tensor 对象 | 复制 | 可以保留到 x 的梯度路径 |
| `y = x.detach().clone()` | 新 Tensor 对象 | 复制 | 不再通过该结果跟踪到 x |

所以“detach 后修改不会影响原参数”是错误推断。“clone 一定关闭梯度”同样错误。示例使用同起点张量的 `data_ptr` 辅助核对共享；一般带偏移视图不应只靠起始地址判断所有别名关系。

### 索引既影响 shape，也可能影响所有权

设 `X.shape=[B,T,C]=[2,3,4]`：

| 表达式 | shape | 读取语义 |
|---|---|---|
| `X[:, -1, :]` | `[2,4]` | 整数索引移除位置轴，基本索引通常为视图 |
| `X[:, -1:, :]` | `[2,1,4]` | 切片保留轴，基本索引通常为视图 |
| `X[[0,1]]` | `[2,3,4]` | 这里是高级索引，读取产生副本 |

Python list 切片新建外层；Tensor 基本切片通常共享数据，规则不同。高级索引“读取产生副本”也不能推导出 `X[mask] = value` 不会修改 X：后者执行的是索引赋值。

### 算子与广播快速核对

手工 `W.shape=[4,2]` 时，`X @ W + bias` 输出 `[2,3,2]`，偏置 `[2]` 从尾部轴广播。`nn.Linear(4,2)` 保存的权重则为 `[2,4]`，内部按转置后的方向参与计算。

`*` 是逐元素乘，`@` 是矩阵乘法；`==` 返回逐元素布尔 Tensor。多个元素的 Tensor 不能直接作为 Python `if` 条件，需明确使用 `.any()`、`.all()` 等归约。逐元素逻辑常写 `(x > 0) & (x < 1)`，不要把 Python `and` 当作按元素运算。

### reshape 与 transpose 不是一个动作

`transpose`/`permute` 重排轴，经常产生非连续视图。`view` 需要布局兼容；`reshape` 可在必要时复制，不能据方法名判断结果必然共享。`contiguous()` 也不是无条件复制：输入已经连续时可以复用。

学习多头注意力代码时，同时写出语义轴 `[B,T,H,Dh]` 与实际内存布局；元素数匹配只证明能排出某种形状，不证明 Token、头和维度的对应关系正确。

<details><summary>实验：视图修改、布局与 detach/clone</summary>

<!-- EXAMPLE: examples/14_tensors.py -->

</details>

### 原地更新是另一个契约

PyTorch 常用末尾下划线表示原地操作，如 `add_`、`copy_`。在梯度跟踪过程中改动被后续反向依赖的数据，可能被版本检查拒绝。受控加载参数常在 `torch.no_grad()` 中执行 `copy_`，这仍是在修改原参数，不是创建新对象。

Tensor 视图规则可核对 [PyTorch 官方视图说明](https://docs.pytorch.org/docs/2.14/tensor_view.html)；本教材的具体例子在本机 PyTorch 2.2.2 上运行验证，升级文档不替代本地行为核对。

### 代码审查题

`best = model.wte.weight.detach()` 能作为独立的最佳权重备份吗？`weights == other_weights` 能直接拿去 `if` 吗？

<details><summary>核对答案</summary>

前者仍共享存储，通常需要 `detach().clone()` 或序列化快照。后者为逐元素结果，需明确比较语义：如 `torch.equal` 做严格整体相等，或 `torch.allclose` 做浮点容差比较，不能让多元素结果隐式转 bool。

</details>
