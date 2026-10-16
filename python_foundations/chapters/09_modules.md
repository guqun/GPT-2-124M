## 09 · import、模块缓存与包上下文

**迁移目标：**把模块导入当作运行时动作来审查，而不是 C++ 预处理 include 或单纯的编译期名称声明。

### 首次导入通常会执行模块顶层代码

Python 导入系统查找模块、建立模块对象、执行其顶层代码，并在 `sys.modules` 中缓存。后续普通导入通常复用同一模块对象。CPython 可以编译和缓存字节码，但这不消除模块初始化的运行时副作用。

因此，把训练、下载或文件覆盖放在顶层，会使“别人 import 我”也触发这些动作。`if __name__ == "__main__": main()` 将脚本入口与被导入时的定义分开，作用只覆盖它保护的部分。

### import 两种形式的绑定差别

```python
import module
from module import factory
```

第一种取得模块对象，之后 `module.factory` 会再次读取其当前属性；第二种把当时取到的对象绑定到本地名字 `factory`。如果以后模块把属性重绑定为另一对象，本地名字不会自动跟随改变。导入语句不是“永久建立一个别名路径”。

<details><summary>实验：模块缓存与 from-import 的既有绑定</summary>

<!-- EXAMPLE: examples/09_modules.py -->

</details>

### 相对导入依赖包身份

本例目录：`examples/mini_package/{__init__.py, helpers.py, __main__.py}`。`from .helpers import make_xy` 里的点表示包内定位，不是任意脚本当前目录的文本替换。

```bash
cd /Users/guqun/micrograd/gpt2_pretrain_lab/python_foundations/examples
/Users/guqun/micrograd/venv/bin/python -m mini_package
```

<!-- OUTPUT: python -m mini_package -->

你以前直接运行包内 `demo.py` 得到 `attempted relative import with no known parent package`，应检查启动方式与包上下文；`No module named 'regex'` 则要检查依赖是否在当前解释器搜索路径内。两种错误的修正位置不同。

### 项目工程边界

直接运行文件时，脚本所在目录通常加入 `sys.path`；`-m` 运行方式的搜索起点有所不同。应用目录里的 `torch.py`、`json.py` 可能遮住真正的库，诊断时查看模块的 `__file__`。

包根下的 `__init__.py` 可以重新导出公开接口，也可能触发额外导入。阅读 `from transformers import GPT2LMHeadModel` 时，最终类的定义未必就在包根文件里。课程代码用固定目录和显式入口，比靠临时修改 `sys.path` 猜测导入结果更容易核对。

### 代码审查题

为什么改了一个模块全局常量，另一个文件 `from module import CONSTANT` 得到的名字可能还是旧值？Python 相对导入与当前工作目录是否是同一个概念？

<details><summary>核对答案</summary>

导入时形成的是本地绑定；原模块属性后续重绑定不自动重写该绑定。相对导入依据包上下文解析；工作目录只影响部分搜索路径与相对文件地址。

</details>
