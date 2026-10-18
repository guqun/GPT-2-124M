# Python 迁移教材：面向 Java / C++ / Dart 开发者

打开 [网页教材](textbook.html)，或阅读同步生成的 [Markdown 版](textbook.md)。当前版本以熟悉多门语言的资深程序员为读者，压缩通用编程概念，重点介绍 Python 语义差异和模型代码中的工程边界。

## 阅读路线

- **快速迁移：**01 对象绑定 → 07 参数绑定 → 09 模块 → 10 对象协议 → 13 真实脚本 → 14 Tensor → 15 Module。
- **速查：**00 环境、02 文本、11 文件与上下文。
- **系统核对：**按 00—16 顺序，先做语义判断，不确定时展开实验。

17 个单元保留原有锚点；网页新增“全部 / 重点 / 实战”目录筛选。正文按“迁移规则 → 模型代码联系 → 可运行反例 → 代码审查题”组织。主线仍停在 GPT-2 配置与权重检查。

## 执行示例

先按[项目说明](../README.md)准备依赖，以下命令在仓库根目录执行：

```bash
.venv/bin/python python_foundations/examples/07_functions.py
```

现有项目环境已具备依赖。00—12 使用标准库；13 直接读现有课程脚本；14—15 使用 CPU 上的小张量/小模型；16 的配置练习只依赖标准库。

## 文件说明

| 位置 | 内容 |
|---|---|
| `textbook.html` / `textbook.md` | 同步生成的教材，离线可读 |
| `chapters/` | 章节源文稿 |
| `examples/` | 15 个迁移语义程序和包导入示例 |
| `exercises/config_report.py` | 待完成的配置接口模板 |
| `answers/config_report.py` | 第 16 个可运行程序：参考实现 |
| `verification.json` | 实跑记录与源码校验值 |
| `verify_examples.py` | 复跑程序和关键边界检查 |

## 验证与维护

```bash
.venv/bin/python python_foundations/verify_examples.py
```

这会更新验证记录，不下载或修改真实模型。对应的教材构建使用 Node.js 和 `marked`：在能导入 `marked` 的环境运行 `node build_textbook.mjs`，或把 `marked` 的 ESM 文件绝对路径作为唯一参数传入。普通学习无需运行构建工具。

本机验证：Python 3.9.6 / PyTorch 2.2.2 / safetensors 0.7.0。代码中区分语言、标准库与框架行为；不会把运行参考实现等同于学习进度完成。
