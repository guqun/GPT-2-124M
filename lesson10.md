# 知识块 10：把 torch.compile 接到我们的 GPT 训练代码

当前位置：**阶段二 → 单元 2「预训练工程：复现 GPT-2」→ 2.2 训练提速 → 知识块 10**。

承接上一块 BF16：网络结构、数据、损失和更新方法沿用已有实现。本块学习怎样改变模型的执行方式，以及怎样判断它是否正确、是否值得启用。

参数起点仍是随机初始化，语料仍为固定的 Tiny Shakespeare。读取本地 GPT-2 配置与 tokenizer，不读取预训练权重。

## 1. 来源与本地改编

- [原视频约 1:47:34—1:59:46](https://www.youtube.com/watch?v=l8pRSuU81PU&t=6454s)：在 BF16 后引入编译，解释 Python 调度、kernel 融合和中间数据读写。
- [作者历史提交 fb8bd6e：add torch compile](https://github.com/karpathy/build-nanogpt/commit/fb8bd6e)：只增加一行 `model = torch.compile(model)`，位置在 `.to(device)` 后、优化器创建前。
- 本次定位使用[非官方自动字幕副本](https://github.com/nolimitkun/ai-kol-wiki/blob/main/sources/karpathy/20240609-l8pRSuU81PU/transcript.md)并交叉核对作者提交；未直接观看视频画面。
- API 对照本项目使用的 [PyTorch 2.2 torch.compile](https://docs.pytorch.org/docs/2.2/generated/torch.compile.html) 与 [编译器说明](https://docs.pytorch.org/docs/2.2/torch.compiler.html)。

本地新增 [train_compile.py](train_compile.py)，上一块 [train_mixed_precision.py](train_mixed_precision.py) 保留。原课 GPU 使用更大的 B/T；本机 CPU 小批次运行只验证行为。暂不引入后面的 FlashAttention、词表大小对齐等改动。

**编译范围有一处本地差异：**作者模型接收 `x,y` 并返回 `logits,loss`；我们的 `GPT.forward(x)` 只返回 logits，因此这一版编译模型前向及其对应的反向，外部交叉熵仍普通执行。数据加载、计时、AdamW 也没有自动被编译。

## 2. 先找到唯一的新入口

主程序按 `[C1]—[C4]` 标注。准备工作完成后，核心是：

```python
raw_model.to(device)
raw_model.train()

# prepare_execution(raw_model, "compile") 内部就是这一行：
model = torch.compile(raw_model)

optimizer = torch.optim.AdamW(raw_model.parameters(), lr=3e-4)
```

| 名称 | 含义 | 谁提供 |
| --- | --- | --- |
| `raw_model` | 我们的 GPT 实例，持有 FP32 参数 | 本项目定义网络，PyTorch 管理参数 |
| `torch.compile` | 捕获、编译可支持的运算图 | PyTorch |
| `model` | 返回的编译包装对象，之后用它调用前向 | PyTorch |
| `optimizer` | 持有原参数引用，按梯度更新它们 | PyTorch AdamW |

`raw_model` 和包装后的 `model` 访问的是同一批 Parameter，原有 `wte.weight` 与 `lm_head.weight` 共享关系也保留。编译没有创建一份独立训练的模型。

保留 `raw_model` 只是方便访问原模块；也便于以后用 `raw_model.state_dict()` 保存原参数名称。本课只保存实验报告，不保存检查点。

## 3. 训练循环怎么接

新脚本将 `model` 传入上一块的 `train_amp_steps()`。实际调用关系为：

```text
train_compile.main()
  ├─ 创建 raw_model，初始化参数，放到 device
  ├─ prepare_execution(raw_model, execution)
  │    ├─ eager   → 直接返回 raw_model
  │    └─ compile → 返回 torch.compile(raw_model)
  └─ train_amp_steps(model, ...)
       └─ 每个 batch
            ├─ forward_and_loss(model, x, y, ...)
            │    └─ autocast 区域
            │         ├─ logits = model(x)       ← 进入编译后的模型
            │         └─ loss = cross_entropy(...) ← 外部普通执行
            ├─ loss.backward()                  ← 包含模型的编译反向
            └─ optimizer.step()                 ← 更新同一批 FP32 参数
```

把计时与日志略去，已有循环等价于：

```python
for step in range(total_steps):
    x, y = loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

    loss.backward()
    optimizer.step()
```

这是 BF16 路径的阅读摘录；完整脚本也支持 FP32。`x/y` 仍是 `[B,T]` 的整数 ID；`model(x)` 仍输出 `[B,T,V]` 的 logits；loss 仍是标量；参数梯度仍累积到原 FP32 参数。

**我们没有手写更快的矩阵乘法。** 我们定义模型与调用边界，PyTorch 编译器负责分析和生成执行代码。

## 4. 到底什么时候编译

1. **执行 `torch.compile(raw_model)`：**建立编译入口，尚未完成具体输入对应的编译。
2. **第一次调用 `model(x)`：**根据实际输入、模型和执行条件捕获运算，生成并使用优化代码。
3. **第一次 `loss.backward()`：**训练路径也会涉及反向图的编译准备，因此首个完整更新尤其慢。
4. **后续 batch：**条件符合已有缓存时，复用编译结果，再处理新的输入与当前参数。

缓存保存的是执行代码，不是上一个 batch 的 logits。正常的 `optimizer.step()` 改变参数数值，下一步会读取这些新数值，通常不因此重新编译。输入形状、dtype、设备或其他受检查的条件变化时，则可能出现新的编译。

因此只在循环外包装一次。固定 B/T 有助于对照实验，改变条件时要重新观察是否发生编译。

## 5. 为什么仍保留 BF16 与原来的训练方法

BF16 控制适用运算使用的精度；`torch.compile` 优化运算的组织与执行。两者可以一起使用。仍应在 autocast 外执行反向和 AdamW；本块不额外加入 GradScaler。

编译可能融合运算、减少中间内存读写和调度开销；不是所有操作都能融合。浮点运算顺序改变也可能造成小的数值差异，所以对照使用容差，不能要求逐 bit 一样。

默认后端是 Inductor。CPU 和 CUDA 的代码生成方式不同；不能根据 CPU 小实验推断 A100 或其他 GPU 的提速倍数。

## 6. 我们项目实际遇到的 graph break

PyTorch 2.2.2 捕获原模型时，不支持这一处关键字写法：

```python
enumerate(self.transformer.h, start=1)
```

已在 [gpt2_model.py](gpt2_model.py) 改为含义相同的位置参数：

```python
enumerate(self.transformer.h, 1)
```

`1` 仍表示从 1 编号，网络计算和 trace 的名称不变。默认 `fullgraph=False` 可以在断点前后分段编译，所以“程序能跑”不一定说明整个前向完整捕获。

测试另外使用 `fullgraph=True` 和计数后端检查：当前固定形状的前向可以整体捕获，连续改变数据和参数数值不会每步重新捕获。这个计数后端仅用于诊断；真正的编译训练对照使用 Inductor。

需要排查时，在运行命令前设置：

```bash
TORCH_LOGS=graph_breaks,recompiles .venv/bin/python train_compile.py --device cpu --precision fp32 --batch-size 1 --seq-len 8
```

日志在编译中有断点或重新编译时帮助定位；打印、`.item()`、调试 hook 等应尽量留在编译模型外。

## 7. 当前 Mac 怎么运行

现在也可以直接运行脚本，不传任何参数。默认使用 `--device auto --precision auto`，启动时显示实际选定的设备和精度：

| 检测情况 | 本节默认设备 | 默认精度 |
| --- | --- | --- |
| CUDA 可用，且支持 BF16 | CUDA | BF16 |
| CUDA 可用，不支持 BF16 | CUDA | FP32 |
| 没有 CUDA，包括检测到 MPS 的 Mac | CPU | FP32 |

两种 execution 的自动规则相同，避免只切换 eager/compile 就连设备、精度也一起改变。显式指定 `--precision bf16` 时保留并验证这个选择；显式请求不支持的编译设备时提示可运行的选项。

用户普通终端曾自动选择到 MPS，而先前脚本默认 BF16，导致在参数检查时退出。已修正这组默认值；工具运行环境未检测到 MPS，不应据此推断普通终端也检测不到。这里沿用课程 PyTorch 2.2.2 的 CPU/CUDA 编译范围。日志中的 urllib3/LibreSSL 警告与此次设备/精度配置错误分别处理，本次修复不修改依赖版本。

不传参数时采用完整 124M、B=4/T=32、单线程、3 次预热加 5 次测量：

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/train_compile.py
```

使用之前安装过依赖的解释器。下面使用完整 124M、CPU、FP32、B=1/T=8、单线程，便于先检查调用链；首次运行会生成编译产物，耗时较长。

本机 Intel Mac / PyTorch 2.2.2 曾在完整模型的 4 线程编译反向中长时间不返回，调用栈停在矩阵乘法的 OpenMP 同步等待处。该次运行已停止，不计为成功实验；改成单线程后完成下述对照。本脚本因此默认 `--threads 1`，两边保持一致。这是 CPU 线程设置，不是在限制 GPU 只能使用一个线程。

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/train_compile.py \
  --device cpu --precision fp32 --execution compile \
  --batch-size 1 --seq-len 8 --threads 1 --warmup-steps 3 --steps 5
```

再运行一遍，把 `--execution compile` 改成 `--execution eager`，其他条件不变。每次进程重新设相同 seed、创建同样初值、从语料起点开始；预热也会更新参数与推进 batch。

默认报告保存在 `.cache/runtime-check/lesson10-compile-fp32.json` 和对应的 eager 文件；编译产物在 `.cache/torch-compile/`，不作为模型参数提交。

报告分别记录：

- `first_update_ms`：第一个完整更新的耗时，包含训练与当次编译/缓存加载，不能当作纯编译时间。
- `summary.mean_step_ms`：排除预热后的每步平均耗时。
- `summary.tokens_per_second`：测量步总 Token 数除以测量总时间。
- `same_parameter_objects`：包装前后是否引用同一批参数。

默认预热 3 步不是通用保证。若测量期仍发生重新编译，应查原因、调整预热或固定形状后重测。编译可能不适合很短的运行；评估总耗时还要考虑首次开销。

### Mac 的 GPU 仍可以进行普通训练

当前机器在普通终端环境中 `MPS available=True`，已实际跑通 GPU 前向、反向与 AdamW 更新。[MPS 是 PyTorch 在 macOS 上使用 GPU 的后端](https://docs.pytorch.org/docs/2.2/notes/mps.html)。先前工具隔离环境的 MPS 探测结果为 False，不能据此认定这台机器没有可用 GPU。

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/train_compile.py \
  --device mps --execution eager --precision fp32 \
  --batch-size 1 --seq-len 8 --warmup-steps 1 --steps 2
```

`--device mps` 决定在 Mac GPU 上执行；`--execution eager` 表示不启用 `torch.compile`，并不表示 CPU。当前课程的 PyTorch 2.2.2 Inductor 设备代码生成注册只有 CPU/CUDA，所以本节仍用 CPU 跟读编译，CUDA 编译提速另作对照；不将普通 MPS 训练当作编译实验。

完整 124M、FP32、B=1/T=8 的三次 MPS 更新已完成，参数及梯度保持 FP32，共享关系保留。原始记录见 [MPS 普通训练短跑报告](artifacts/lesson10-mps-eager-smoke.json)。这是可用性验证，未与 CPU 做同一测量窗口的完整性能对照。

## 8. CUDA 上如何承接原课实验

在有兼容 CUDA 环境和 BF16 GPU 的机器上，保持参数相同，分别运行：

```bash
.venv/bin/python train_compile.py --device cuda --precision bf16 --execution eager \
  --batch-size 4 --seq-len 32 --warmup-steps 5 --steps 20
.venv/bin/python train_compile.py --device cuda --precision bf16 --execution compile \
  --batch-size 4 --seq-len 32 --warmup-steps 5 --steps 20
```

这是先跑通的较小配置；再按显存条件，将两边一起改成原课 B=16/T=1024。比较时设备、精度、B/T、语料、seed、优化器、预热和测量步数都要一致。B/T 太小时可能主要测到其他开销，不能期待固定倍数。

本机没有 CUDA；上述 GPU 性能对照待补。用户已反馈理解编译与 kernel 的基本概念，本块代码开始走读；2.2 仍在学习，完成计数不变。

## 9. 本次实际核验

2026-10-20，31 项测试通过，其中新增检查真实 Inductor 的 FP32/BF16 训练、原参数引用与共享、更新后重新读取参数，以及固定形状下三次更新只捕获一次完整前向。

完整 124M 的实际对照使用 Python 3.9.6 / PyTorch 2.2.2、Intel Mac CPU、FP32、1 线程、B=1/T=8、seed=42、3 次预热 + 5 次测量。两次运行的配置、语料、起点、精度和优化器参数核对一致。

| 本次读数 | 普通执行 eager | 编译执行 compile |
| --- | --- | --- |
| 首个完整更新 | 1,527.60 ms | 30,972.57 ms |
| 排除预热后平均每步 | 848.85 ms | 846.40 ms |
| 测量期吞吐 | 9.42 Token/s | 9.45 Token/s |
| 更新次数 / 测量 Token 数 | 8 / 40 | 8 / 40 |

本次后续速度基本相同，不能把小幅差异解释为可靠提速；短运行的总耗时还明显受首次编译影响。开启 graph break/recompile 日志时，本次完整模型运行未出现相应事件。两边 8 次更新前 loss 最大绝对差为 `0.0000314713`；参数对象与权重共享关系保留。

原始记录：[普通执行报告](artifacts/lesson10-cpu-eager.json)、[编译执行报告](artifacts/lesson10-cpu-compile.json)。这些是本机短跑的行为证据，不代表原课 GPU 的性能或长期训练质量。

修正设备/精度的自动选择后，6 项 compile 相关检查通过，包含模拟 MPS 可用时的默认配置回归。无参数命令也实际完成完整 124M、CPU/FP32、B=4/T=32 的 8 次更新；首个完整更新约 37.52 秒。该次报告位于本地 `.cache/runtime-check/lesson10-compile-fp32.json`；它与上表 B=1/T=8 的对照不是同一批次配置。
