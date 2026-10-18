"""生成第五块的代码流程 SVG、定位说明；macOS 加 --png 导出图片。

只读取本项目源码和已归档报告，所有输出及编译缓存都放在项目内。
不导入训练脚本、不执行训练；M/L 编号与行号从当前源码提取。
"""

from html import escape
from pathlib import Path
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "train_dataloader.py"
lines = SOURCE.read_text(encoding="utf-8").splitlines()
anchors = {}
for number, line in enumerate(lines, 1):
    match = re.match(r"\s*# \[([ML]\d+)\]", line)
    if match:
        if match[1] in anchors:
            raise ValueError(f"重复编号：{match[1]}")
        anchors[match[1]] = number
assert set(anchors) == {f"{family}{i}" for family in "ML" for i in range(1, 5)}


def source_line(fragment):
    matches = [i for i, line in enumerate(lines, 1) if fragment in line]
    if len(matches) != 1:
        raise ValueError(f"无法唯一定位：{fragment}")
    return matches[0]


locations = {
    "init": source_line("def __init__(self, token_ids"),
    "preview": source_line("if args.preview_only:"),
    "call": source_line("x, y = train_loader.next_batch()  #"),
    "forward": source_line("optimizer.zero_grad("),
    "loss": source_line("loss = next_token_loss("),
    "update": source_line("loss.backward()"),
    "record": source_line("value = loss.item()"),
    "return": source_line("return x, y"),
    "report": source_line("report = {"),
}
report = json.loads((ROOT / "artifacts/lesson05-dataloader-report.json").read_text())
N = report["data"]["encoded_tokens"]
B, T = report["input_shape"]
assert (N, B, T) == (338025, 4, 32), "图中示例基于当前默认配置，配置改变时请同步版式。"
assert report["target_shape"] == [B, T]
assert [step["batch_start"] for step in report["steps"]] == list(range(0, 6400, 128))

W, H = 2680, 2760
BLUE, TEAL, AMBER = "#285dba", "#087f79", "#a26712"
INK, MUTED = "#172c43", "#526779"
PALE_BLUE, PALE_TEAL, PALE_AMBER = "#eef4ff", "#eaf8f5", "#fff6e5"
background, edges, nodes = [], [], []


def text(x, y, value, size=27, color=INK, weight=400, anchor="start", code=False, target=None):
    target = nodes if target is None else target
    cls = ' class="code"' if code else ""
    target.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
                  f'text-anchor="{anchor}" fill="{color}"{cls}>{escape(str(value))}</text>')


def rect(x, y, w, h, fill="white", stroke="#d6e1e9", radius=20, target=None):
    target = nodes if target is None else target
    target.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')


def card(x, y, w, h, title, content, color=BLUE, fill=PALE_BLUE, line=None, size=27):
    rect(x, y, w, h, fill, color)
    text(x + 24, y + 41, title, size + 2, color, 700)
    if line is not None:
        text(x + w - 23, y + 39, f"第 {line} 行", 20, MUTED, anchor="end")
    for i, item in enumerate(content):
        value, code = item if isinstance(item, tuple) else (item, False)
        text(x + 24, y + 84 + 35 * i, value, size - (2 if code else 0), code=code)


def diamond(cx, cy, w, h, title, detail, color=AMBER, fill=PALE_AMBER):
    points = f"{cx},{cy-h/2} {cx+w/2},{cy} {cx},{cy+h/2} {cx-w/2},{cy}"
    nodes.append(f'<polygon points="{points}" fill="{fill}" stroke="{color}" stroke-width="2.5"/>')
    text(cx, cy - 10, title, 27, color, 700, anchor="middle")
    text(cx, cy + 20, detail, 20, MUTED, anchor="middle")


def edge(points, color=BLUE, dashed=False, arrow=True, bridge=False):
    path = "M" + " L".join(f"{x},{y}" for x, y in points)
    if bridge:
        edges.append(f'<path d="{path}" fill="none" stroke="#f8fafc" stroke-width="11"/>')
    dash = ' stroke-dasharray="10 8"' if dashed else ""
    edges.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3.5" '
                 f'stroke-linejoin="round" stroke-linecap="round"{dash}/>')
    if arrow:
        x, y = points[-1]
        px, py = points[-2]
        dx, dy = x - px, y - py
        length = (dx * dx + dy * dy) ** 0.5
        dx, dy = dx / length, dy / length
        bx, by = x - 15 * dx, y - 15 * dy
        vertices = f"{x},{y} {bx+7*dy},{by-7*dx} {bx-7*dy},{by+7*dx}"
        edges.append(f'<polygon points="{vertices}" fill="{color}"/>')


text(50, 67, "顺序数据加载器 · 从 main() 到下一批数据", 48, weight=750, target=background)
text(52, 116, "阶段二 → 单元 2「复现 GPT-2」→ 2.1 → 知识块 5", 29, MUTED, target=background)
text(52, 159, "对照 train_dataloader.py：M1—M4 是主流程，L1—L4 是 next_batch() 内部步骤。", 27, target=background)
rect(50, 184, 2580, 51, "#edf2f7", "#edf2f7", 12, background)
text(72, 219, "默认：B=4 条序列 · T=32 Token/条 · 每批 128 个监督位置 · CPU / FP32 · 50 次更新", 27, MUTED, target=background)

rect(40, 263, 1235, 2270, "#ffffff", target=background)
rect(1360, 263, 1280, 1295, "#ffffff", target=background)
text(70, 306, "① main()：准备一次，再逐批训练", 31, BLUE, 700, target=background)
text(1390, 306, "② next_batch()：每次调用只返回一批", 31, TEAL, 700, target=background)

# 左列：准备 -> 可选预览 -> 训练循环。循环返回迭代入口，不重建模型或加载器。
card(210, 345, 830, 170, "[M1] 读取全文并编码", [
    "Tiny Shakespeare 全文 → tokenizer.encode(text)",
    "得到 338,025 个整数 ID；读取本地模型配置",
    "参数 / 文件 / 语料校验通过后继续",
], line=anchors["M1"])
card(210, 555, 830, 145, "[M2] 创建一次 DataLoaderLite", [
    ("tokens: torch.long [338025]", True),
    ("current_position = 0", True),
], line=anchors["M2"])
diamond(625, 810, 560, 140, "开启 --preview-only？", f"第 {locations['preview']} 行")
card(965, 740, 280, 165, "是：仅预览", [
    "next_batch() × 3", "打印范围和文本", "return，结束运行",
], color=AMBER, fill=PALE_AMBER, size=21)
card(210, 940, 830, 170, "[M3] 创建一次模型和优化器", [
    "随机初始化 GPT → model.train() → AdamW",
    "输入 / 输出表独立；使用各层默认初始化",
    "后续各步保留模型参数和优化器状态",
], line=anchors["M3"])
diamond(625, 1210, 620, 130, "[M4] 还有训练步数？", f"for step in range(args.steps) · 第 {anchors['M4']} 行")
card(210, 1320, 830, 105, "调用加载器，取得本步 x、y", [
    ("x, y = train_loader.next_batch()", True),
], line=locations["call"])
card(210, 1495, 830, 155, "清空旧梯度 → 前向计算", [
    ("optimizer.zero_grad(set_to_none=True)", True),
    ("logits = model(x)  # [4,32,50257]", True),
], line=locations["forward"])
card(210, 1710, 830, 190, "计算所有位置的下一 Token loss", [
    ("loss = next_token_loss(logits, y)", True),
    "[4,32,50257] + [4,32] → 标量 loss []",
    "128 个位置的交叉熵取平均；检查 loss 有限",
], line=locations["loss"])
card(210, 1960, 830, 150, "反向传播 → 更新参数", [
    ("loss.backward()", True),
    ("optimizer.step()", True),
], line=locations["update"])
card(210, 2170, 830, 145, "记录本步位置与 loss", [
    "记录 batch_start、next_start；打印本步结果",
    "loss 来自更新前的前向，不会因 step() 自动重算",
], line=locations["record"])
card(210, 2380, 830, 120, "步数用完：保存报告，结束", [
    ".cache/runtime-check/lesson05-report.json",
], line=locations["report"])

for ya, yb in [(515, 555), (700, 740), (880, 940), (1110, 1145),
               (1275, 1320), (1650, 1710), (1900, 1960), (2110, 2170)]:
    edge([(625, ya), (625, yb)])
edge([(905, 810), (965, 810)], AMBER)
text(649, 918, "否：进入训练", 23, BLUE)
text(649, 1307, "是", 23, BLUE)
edge([(210, 2243), (130, 2243), (130, 1210), (315, 1210)])
nodes.append(f'<text transform="translate(111 1985) rotate(-90)" font-size="27" fill="{BLUE}" font-weight="600">下一步：保留参数，重新获取下一批数据</text>')
edge([(935, 1210), (1160, 1210), (1160, 2440), (1040, 2440)], AMBER)
text(951, 1189, "否：已完成 50 步", 22, AMBER)
nodes.append(f'<text transform="translate(1194 1660) rotate(90)" font-size="24" fill="{AMBER}">循环退出后才保存报告</text>')

# 右列：先构造本次数据，再推进/重置下一次位置，最后返回原本这批 x/y。
card(1510, 355, 970, 165, "[L1] 从当前位置取 B*T+1 个 ID", [
    ("start = self.current_position", True),
    ("buffer = tokens[start : start + B*T + 1]", True),
    "本批读取 129 个 ID → buffer [129]",
], TEAL, PALE_TEAL, anchors["L1"])
card(1510, 565, 970, 190, "[L2] 错开一位，排成 B 行 T 列", [
    ("x = buffer[:-1].view(B,T)   # [4,32]", True),
    ("y = buffer[1:].view(B,T)    # [4,32]", True),
    "x 和 y 各含 128 个 Token；每个输入位置都有答案",
], TEAL, PALE_TEAL, anchors["L2"])
card(1510, 800, 970, 130, "[L3] 推进下次读取的位置", [
    ("self.current_position += B*T  # 前进 128", True),
], TEAL, PALE_TEAL, anchors["L3"])
diamond(1995, 1045, 830, 155, "[L4] 剩余数据不足下一批及其答案？",
        "current_position + B*T + 1 > len(tokens)", TEAL, PALE_TEAL)
text(1510, 1157, f"第 {anchors['L4']} 行：用 > 判断。", 23, MUTED)
text(1510, 1192, "刚好够一批时，条件为 False。", 23, MUTED)
card(2140, 1220, 440, 115, "是：下次从开头读", [
    ("current_position = 0", True),
], TEAL, PALE_TEAL, size=24)
card(1510, 1400, 970, 125, "返回本次已经构造好的 x、y", [
    "return x, y；位置即使重置，本次数据也不变",
], TEAL, PALE_TEAL, locations["return"])
for ya, yb in [(520, 565), (755, 800), (930, 967.5)]:
    edge([(1995, ya), (1995, yb)], TEAL)
edge([(1995, 1122.5), (1995, 1400)], TEAL)
text(1735, 1243, "否：保留推进后的位置", 23, TEAL)
edge([(2410, 1045), (2605, 1045), (2605, 1180), (2360, 1180), (2360, 1220)], TEAL)
edge([(2360, 1335), (2360, 1370), (1995, 1370)], TEAL, arrow=False)

# 两条跨函数连线；白色间隙表示仅跨越，没有与训练结束通道合并。
edge([(1040, 1372), (1310, 1372), (1310, 437), (1510, 437)], BLUE, bridge=True)
text(1330, 416, "调用：进入 L1", 22, BLUE)
edge([(1510, 1462), (1330, 1462), (1330, 1572), (1040, 1572)], TEAL, dashed=True, bridge=True)
text(1370, 1547, "返回后继续训练", 22, TEAL)

# 学习对照：全部位置都是语料下标，避免把 0..127 误读为 127 个 Token。
rect(1360, 1600, 1280, 330, "#ffffff")
text(1390, 1648, "③ 首批 x/y：范围包含两端，0～127 是 128 个位置", 29, INK, 700)
text(1390, 1688, "下表是语料位置，不是 Token ID 数值；每行都是 32 个 Token。", 25, MUTED)
for x, title in [(1420, "序列"), (1660, "x：输入位置"), (2070, "y：答案位置")]:
    text(x, 1734, title, 25, TEAL, 700)
for i in range(4):
    y = 1772 + i * 35
    text(1420, y, f"第 {i+1} 条", 25)
    text(1660, y, f"{32*i} ～ {32*i+31}", 25, code=True)
    text(2070, y, f"{32*i+1} ～ {32*i+32}", 25, code=True)
text(1390, 1910, "首批 buffer 用 0～128；下一批从 128 开始：上一批末尾答案成为下一批首个输入。", 23, MUTED)

# 只回顾本次问到的前向含义，完整 Attention 计算仍见之前的图。
rect(1360, 1970, 1280, 250, "#ffffff")
text(1390, 2017, "④ 前向回顾：B 条序列，各有 T 份下一 Token 预测", 29, INK, 700)
text(1390, 2062, "ID [4,32] → 嵌入 [4,32,768] → 12 层 Transformer", 26, code=True)
text(1390, 2103, "→ 最后 LayerNorm + lm_head → logits [4,32,50257]", 26, code=True)
text(1390, 2148, "每条序列独立使用 32×32 的下三角因果遮罩；每个位置只看自己及之前。", 25, MUTED)
text(1390, 2187, "128 份预测都参与 loss；本训练流程不取 argmax，也不生成文本。", 25, MUTED)

rect(1360, 2260, 1280, 273, PALE_AMBER, "#e7c78f")
text(1390, 2307, "⑤ 回绕复用的是样本，继续更新的是参数", 29, AMBER, 700)
text(1390, 2354, "一轮：第 1 批 → 第 2 批 → … → 第 2640 批", 28)
text(1390, 2398, "下一轮仍从第 1 批开始；对应批次的 x/y 完全相同。", 27)
text(1390, 2442, "仅重置读取位置，模型和优化器状态继续保留；尾部不足一批就跳过。", 25)
text(1390, 2489, "当前默认只运行 50 步，尚未读完一轮，因此实跑没有触发回绕。", 25, AMBER, 600)

rect(40, 2570, 2600, 140, "#edf2f7", "#edf2f7", 16, background)
text(70, 2615, "读图：蓝色为 main 主流程，绿色为取批逻辑，虚线为函数返回；跨线处白色间隙表示未连接。", 26, target=background)
text(70, 2654, "图展开有效输入、有限 loss 的正常路径。参数 / 文件 / 数值检查失败时会退出；预览模式提前 return。", 24, MUTED, target=background)
text(70, 2690, "源码编号与行号由脚本读取。数据量来自已归档实跑报告。绘制日期：2026-10-18。", 23, MUTED, target=background)

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">
<title id="title">阶段二 · 单元 2 · 2.1 · 知识块 5：顺序数据加载器代码流程</title>
<desc id="desc">左列 main 准备加载器、可选预览、模型和优化器，循环获取下一批、计算 loss 并更新参数；右列展开 next_batch 的取片、错位、前进、条件回绕和返回。下方列出首批四条序列的语料位置及形状。</desc>
<style>text {{font-family:-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif;}} .code {{font-family:Menlo,'SFMono-Regular',Consolas,'PingFang SC',monospace;}}</style>
<rect width="{W}" height="{H}" fill="#f8fafc"/>
''' + "\n".join(background + edges + nodes) + "\n</svg>\n"
ET.fromstring(svg)
svg_path = OUT / "gpt2-dataloader-flow.svg"
svg_path.write_text(svg, encoding="utf-8")

descriptions = {
    "M1": "选项、检查、读取全文与编码", "M2": "只创建一次加载器，保存整段 tokens 和位置",
    "M3": "只创建一次随机模型和 AdamW", "M4": "逐步获取下一批并训练，更新后回到循环入口",
    "L1": "从当前位置取 B*T+1 个 ID", "L2": "错开一位并排成 x/y [B,T]",
    "L3": "下次位置前进 B*T", "L4": "判断下一批是否装得下；不足时重置下次位置",
}
rows = [f"| `{tag}` | {description} | [第 {anchors[tag]} 行](../../train_dataloader.py#L{anchors[tag]}) |"
        for tag, description in descriptions.items()]
for key, label in [("init", "加载器构造函数"), ("preview", "只预览前三批的分支"),
                   ("call", "训练循环调用 next_batch"), ("forward", "清空梯度与前向"),
                   ("loss", "全位置 loss 与有限性检查"), ("update", "反传和更新"),
                   ("record", "记录本步更新前的 loss"), ("return", "返回已经构造的 x/y"),
                   ("report", "循环后保存 JSON 报告")]:
    rows.append(f"| — | {label} | [第 {locations[key]} 行](../../train_dataloader.py#L{locations[key]}) |")
guide = '''# 顺序数据加载器：代码与流程对照

学习位置：**阶段二 · 单元 2「复现 GPT-2」· 2.1 · 知识块 5**。

![顺序数据加载器：代码与流程对照](gpt2-dataloader-flow.png)

[打开可放大的 SVG 原图](gpt2-dataloader-flow.svg)。源代码：[train_dataloader.py](../../train_dataloader.py)，完整讲义：[lesson05.md](../../lesson05.md)。

## 阅读顺序

1. **左列 M1—M3：每次启动只准备一次。** 全文编码；构造加载器并保存位置；未开启预览时，创建随机模型和优化器。`--preview-only` 调用同一个 `next_batch()` 三次，打印后直接退出。
2. **左列 M4：每个训练步调用一次 next_batch。** 蓝线进入右列 L1，绿虚线从 `return x, y` 返回左列，接着清梯度、前向、loss、反传与更新。
3. **右列 L1—L4：本块新增逻辑。** 取 129 个 ID → 得到各含 128 个 ID 的 x/y → 前进 128 → 检查下一批是否装得下。
4. **区分“本次数据”和“下次位置”。** L2 先构造本次 x/y；L4 重置的是下次读取位置；本次返回的仍然是已构造的数据。下一次调用才会从头读取。
5. **更新后回到 M4，不回到 M2/M3。** 模型、优化器、加载器都继续沿用。达到训练步数才保存报告。

图中蓝色为 main 主流程，绿色为加载器，虚线为返回；跨线处白色间隙表示没有连接。图展开正常训练路径，参数、文件或有限性检查失败时会退出。

## 图与源码定位

图中 M/L 编号来自源码注释；第几行由绘图脚本自动提取。源码增删后可重新生成；阅读时优先搜索稳定编号。

| 编号 | 作用 | 源码 |
| --- | --- | --- |
''' + "\n".join(rows) + '''

## 结合本次讨论看图

- **位置与数量：** 位置 0～127 包含两端，共 128 个 Token；x 为位置 0～127，y 为位置 1～128，各排成 `[4,32]`。这些数字是语料位置，不是 ID 数值。
- **每个位置都预测：** `model(x)` 得到 `[4,32,50257]`。每条序列独立做因果注意力，128 个位置都参与交叉熵；`next_token_loss` 已有对应标签，不再右移一次，也不先取 argmax。
- **固定可见范围：** 每条序列内的位置使用 32×32 的因果遮罩。四条序列之间、前后 batch 之间不延续隐藏状态。详见 [Attention 代码](../../gpt2_model.py#L138) 和 [前向调用图](gpt2-forward-call-map.md)。
- **每轮重复：** 只重置读取位置，不重置模型或优化器。无打乱、随机偏移；重新走到同一位置时 x/y 相同，模型参数可以不同。
- **两个循环边界：** 当前数据可取 2640 个完整 batch 才回绕；默认外层只有 50 步，只处理 6400 个监督位置，并没有读完整轮。数据末尾不足一批的部分跳过。
- **日志与更新：** `.step()` 后记录的 `loss.item()` 仍来自本轮更新前的前向，参数更新不会让它自动重新计算。

数据量与首轮起点来自 [50 步归档报告](../../artifacts/lesson05-dataloader-report.json)；这里整理已有代码和讨论，未改变训练行为。上一块的 `train_step.py` 和 `lesson04.md` 继续作为固定批次实验的配套。

## 重新生成

```bash
.venv/bin/python docs/diagrams/render_dataloader_flow.py
```

更新 SVG 与定位说明。macOS 加 `--png` 同时生成 PNG，使用系统 AppKit；缓存与临时 Swift 文件均保存在当前项目的 `.cache/` 中。

```bash
.venv/bin/python docs/diagrams/render_dataloader_flow.py --png
```

脚本只读取源码与已有报告，不导入训练脚本、不启动训练。绘制依据：2026-10-18 的本地代码。
'''
(OUT / "gpt2-dataloader-flow.md").write_text(guide, encoding="utf-8")

if "--png" in sys.argv[1:]:
    if sys.platform != "darwin":
        raise SystemExit("SVG 与说明已生成；--png 需要 macOS 系统 AppKit。")
    runtime = ROOT / ".cache/runtime-check"
    runtime.mkdir(parents=True, exist_ok=True)
    swift_path = runtime / "render-dataloader-svg.swift"
    swift_path.write_text('''import AppKit
import Foundation
let input = URL(fileURLWithPath: CommandLine.arguments[1])
let output = URL(fileURLWithPath: CommandLine.arguments[2])
guard let image = NSImage(contentsOf: input) else { fatalError("Cannot load SVG") }
let width = Int(image.size.width)
let height = Int(image.size.height)
guard let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0), let context = NSGraphicsContext(bitmapImageRep: bitmap) else { fatalError("Cannot create bitmap") }
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = context
NSColor.white.setFill()
NSRect(x: 0, y: 0, width: width, height: height).fill()
image.draw(in: NSRect(x: 0, y: 0, width: width, height: height))
context.flushGraphics()
NSGraphicsContext.restoreGraphicsState()
guard let data = bitmap.representation(using: .png, properties: [:]) else { fatalError("Cannot encode PNG") }
try data.write(to: output)
print("PNG export complete: \\(width) x \\(height)")
''', encoding="utf-8")
    subprocess.run(["/usr/bin/swift", "-module-cache-path", str(ROOT / ".cache/swift-module-cache"),
                    str(swift_path), str(svg_path), str(svg_path.with_suffix(".png"))], check=True)

print(json.dumps({"svg": str(svg_path), "source_anchors": anchors, "locations": locations}, ensure_ascii=False))
