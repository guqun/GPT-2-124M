## 15 · nn.Module：注册、执行、更新与保存

**迁移目标：**理解框架如何建立对象图并发现参数。你已学过训练数学，本单元关注 Python 代码中的机制和生命周期边界。

### 注册不是“对象存在”就自动发生

```python
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.visible = nn.Linear(4, 4)
        self.hidden = [nn.Linear(4, 4)]
        self.blocks = nn.ModuleList([nn.Linear(4, 4)])
```

`self.visible` 与 `self.blocks` 及其中层会被 Module 管理；`self.hidden` 是普通列表，列表内部层不会仅凭这个容器赋值被自动注册。因此它们可能不出现在 `parameters()`、`state_dict()` 和设备迁移的递归操作中。

这不是 Python “看不见对象”，而是框架注册协议没有沿任意对象图递归。需要管理层列表时使用 `ModuleList`，参数列表用 `ParameterList`；非参数但需要随模型迁移/保存的状态可用适当的 buffer 注册方式。

### 三个入口对应三个阶段

| 入口 | 生命周期职责 |
|---|---|
| `__init__` | 建立可复用层与参数，先初始化 Module 基础设施 |
| `forward` | 描述输入如何使用当前状态计算结果 |
| `model(inputs)` | 经过 Module 调用入口进入 forward，包含相关 hook 机制 |

在 `forward` 临时新建 Linear 会产生新的随机参数，不能当成已有层持续训练。循环和条件可以参与构造执行路径，但不能因此忽略参数注册与复用。

### 状态变化逐项核对

| 操作 | 改变什么 | 未完成什么 |
|---|---|---|
| `zero_grad()` | 清理已累积梯度 | 不重新初始化权重 |
| `model(x)` | 产生本次前向结果及可能的图 | 不根据损失更新参数 |
| `loss.backward()` | 沿图累积梯度 | 不等同于优化器 step |
| `optimizer.step()` | 按梯度及优化器状态更新参数 | 不自动清掉下一轮要累积的梯度 |
| `eval()` | 修改模式标志与相关层行为 | 不关闭 autograd，不执行前向 |
| `no_grad()` | 临时关闭梯度记录 | 不自动让 Dropout 进入评估模式 |

### 小模型的输入输出契约

配套例子用 `Embedding(6,4)` 与 `Linear(4,6)`：输入 ID `[1,3]` → hidden `[1,3,4]` → logits `[1,3,6]` → 标量交叉熵。`F.cross_entropy` 接受 logits，不先做 softmax。参数总数为 54，这里两层不共享权重。

这是独立的接口实验，没有 Attention 或位置向量，不是缩小版完整 GPT-2。固定 seed 用于可复查本次实验，不声称跨所有平台/版本能逐位复现任意训练。

<details><summary>实验：一次更新、模块注册与保存后重载</summary>

<!-- EXAMPLE: examples/15_tiny_model.py -->

</details>

### 状态映射不是独立快照，也不是完整程序

`state_dict()` 的值通常继续共享模型参数/buffer 的存储；它并不自动深复制模型状态。`safetensors.save_file` 序列化张量数值，不保存全部 Python 类代码和优化器上下文。恢复先建立兼容结构，再加载张量并核对键、shape 和输出。

本例写入临时目录、重载并对齐输出。真实续训还要根据目标保存优化器状态、步数、随机状态等；只读 GPT-2 预训练权重和恢复一次中断训练，契约不同。

### 代码审查题

一个模型能 forward，是否证明所有层都能被 optimizer 更新？为什么 `best_state=model.state_dict()` 后继续训练，可能导致“最佳权重”跟着变化？

<details><summary>核对答案</summary>

不能，forward 可以直接调用普通容器中的层，但优化器可能没拿到它们。状态映射中的张量可共享底层存储，后续原地更新会反映过去；需要独立复制或在目标时间点实际写出快照。

</details>
