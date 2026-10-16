## 11 · with、路径与序列化边界

**迁移目标：**将已有资源管理经验迁移到上下文管理协议，同时区分“地址”“文件内容”“解析后的对象”。

### with 是可扩展的进入/退出协议

`with resource as x:` 通过上下文管理器的 `__enter__` / `__exit__` 管理作用域；`x` 接收 `__enter__` 的返回结果，不一定就是原管理器对象。正常离开或异常离开都会执行退出逻辑；`__exit__` 还可以选择抑制异常。

可以类比 C++ RAII 和 Java try-with-resources 的确定性清理目的，但 Python 的 `with` 不要求对象此时销毁，也不只用于文件资源。`torch.no_grad()` 临时改变梯度记录状态，就是相同语法下的另一种协议实现。不要依赖 `__del__` 的调用时机来代替确定性清理。

### Path 与 JSON 的最小速查

| 写法 | 实际动作 |
|---|---|
| `Path.home() / "dir" / name` | 创建路径对象；`/` 被路径类型定义为拼接 |
| `Path(__file__).resolve().parent` | 定位脚本资源目录，不是 `cwd()` |
| `.read_text(encoding="utf-8")` | 磁盘数据 → str |
| `json.loads(text)` / `json.load(file)` | 文本 / 文件流 → Python 对象 |
| `json.dumps(obj)` / `json.dump(obj,file)` | Python 对象 → 文本 / 文件流 |
| `.read_bytes()` | 原始 bytes，不尝试文本解码 |

JSON 的 `true/false/null` 变为 Python `True/False/None`。解析只保证语法合法，不保证字段存在、维度合法或数据结构满足模型要求。

### 格式与所有权要分开

`config.json` 给结构设置，权重文件给张量数值。Path 不下载数据，JSON 解析器不理解模型结构，safetensors 也不替你保存 Python 类的完整实现。

`data_offsets=[start,end]` 使用左闭右开区间；在 safetensors 中相对于数据区起点，不是整个文件第 0 字节。读格式库的契约比按熟悉的文件格式类比更可靠。

<details><summary>实验：JSON 往返与异常发生后的上下文退出</summary>

<!-- EXAMPLE: examples/11_files.py -->

</details>

### 代码审查题

为什么 `with torch.no_grad(): ...` 结束后可以恢复先前状态，而不需要销毁模型？把报告 dict 写成 JSON，等不等于保存了可直接恢复的模型？

<details><summary>核对答案</summary>

上下文管理器控制的是临时状态的进入和恢复，与模型生命周期无关。报告是你主动组织的可序列化信息；模型还需要结构、参数、配置，以及恢复训练时可能需要的优化器和随机状态等。

</details>
