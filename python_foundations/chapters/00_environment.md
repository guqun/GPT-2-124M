## 00 · 执行环境：解释器、依赖与启动方式

**迁移目标：**建立“哪个解释器执行了哪个入口”的确定性。你已有终端与 IDE 使用经验，本章只保留 Python 环境定位。

### 一张定位表

| 需要知道什么 | 直接观察 | 避免的误判 |
|---|---|---|
| 当前解释器 | `sys.executable` | 系统 `python`、Python 3.12 和项目 venv 是同一个环境 |
| 是否在 venv | `sys.prefix != sys.base_prefix` | 终端目录在项目里就自动使用项目环境 |
| 包能否被找到 | `importlib.util.find_spec("torch")` | 找到包就代表它及原生依赖一定能导入成功 |
| 脚本资源目录 | `Path(__file__).resolve().parent` | 当前工作目录就是脚本目录 |
| 包安装归属 | `指定解释器 -m pip ...` | 裸 `pip` 一定对应正在运行的 Python |

你之前的 `No module named 'torch'` 发生在切换解释器之后，应首先检查环境绑定，而不是修改模型代码。本课程当前解释器：`/Users/guqun/micrograd/venv/bin/python`。

### 两种入口都要能识别

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/inspect_checkpoint.py
```

这是按文件运行；`python -m package.module` 则按可导入模块运行，能建立包上下文。选择取决于项目组织，不是哪个写法更高级。相对导入和入口保护见第 09 单元。

<details><summary>运行环境定位实验</summary>

<!-- EXAMPLE: examples/00_environment.py -->

</details>

<details><summary>VS Code 解释器定位速查</summary>

命令面板选择 **Python: Select Interpreter**，指定 `/Users/guqun/micrograd/venv/bin/python`。实际执行后以 `sys.executable` 为准。需要核对安装位置时运行这个解释器的 `-m pip show torch`；本机依赖已经具备，无需再安装。

</details>

### 语义自检

IDE 能补全 `torch`，运行却找不到它，优先核对哪两个环境？为什么“find_spec 成功”和“import 成功”不是同一件事？

<details><summary>核对答案</summary>

先核对 IDE 分析/运行配置与实际 `sys.executable`。定位模块只说明找到了导入入口；执行模块代码、解析传递依赖或加载原生扩展仍可能失败。

</details>
