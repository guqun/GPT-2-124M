"""用标准库生成课程流程 SVG，并从源码提取步骤编号与当前行号。"""

from html import escape
from pathlib import Path
import json
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "generate_text.py"
source_lines = SOURCE.read_text(encoding="utf-8").splitlines()
anchors = {}
for number, line in enumerate(source_lines, 1):
    match = re.match(r"\s*# \[([MGS]\d+)\]", line)
    if match:
        if match[1] in anchors:
            raise ValueError(f"重复步骤编号：{match[1]}")
        anchors[match[1]] = number
expected = {f"M{i}" for i in range(1, 8)} | {f"G{i}" for i in range(9)} | {f"S{i}" for i in range(7)}
assert set(anchors) == expected

W, H = 2680, 2240
colors = {
    "M": ("#2563eb", "#eff6ff"),
    "G": ("#0f766e", "#effcf8"),
    "S": ("#7c3aed", "#f7f2ff"),
}
parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">
<title id="title">GPT-2 文本生成：代码与完整流程对照</title>
<desc id="desc">左列 main 准备并输出；中列 generate 包含预算、上下文、前向、最后位置、选取、追加和 EOS 循环；右列展开贪心与采样。节点编号对应 generate_text.py 注释。</desc>
<style>
text {{ font-family: -apple-system, BlinkMacSystemFont, 'PingFang SC', 'Microsoft YaHei', sans-serif; fill:#18283b; }}
.code {{ font-family: 'SFMono-Regular', Menlo, Consolas, monospace; }}
</style>
<rect width="{W}" height="{H}" fill="#f8fafc"/>
''']
edges, nodes = [], []


def text(x, y, value, size=25, color="#18283b", weight=400, anchor="start", code=False, target=None):
    target = nodes if target is None else target
    cls = ' class="code"' if code else ""
    target.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" style="fill:{color}"{cls}>{escape(value)}</text>')


def rect(x, y, w, h, fill="#ffffff", stroke="#dbe3ed", radius=18, target=None):
    target = nodes if target is None else target
    target.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')


def panel(x, w, prefix, title, subtitle):
    color, pale = colors[prefix]
    rect(x, 220, w, 1690, fill="#ffffff", target=parts)
    text(x + 26, 262, title, 30, color, 700, target=parts)
    text(x + 26, 292, subtitle, 20, "#58687b", target=parts)


def card(tag, x, y, w, h, title, lines, size=24):
    color, pale = colors[tag[0]]
    rect(x, y, w, h, pale, color)
    text(x + 20, y + 33, f"[{tag}]  {title}", size, color, 700)
    text(x + w - 17, y + 33, f"L{anchors[tag]}", 18, "#617187", anchor="end", code=True)
    for index, line in enumerate(lines):
        value, code = line if isinstance(line, tuple) else (line, False)
        text(x + 20, y + 68 + 29 * index, value, 22 if code else size - 1, code=code)


def diamond(tag, cx, cy, w, h, title, detail):
    color, pale = colors[tag[0]]
    points = f"{cx},{cy-h/2} {cx+w/2},{cy} {cx},{cy+h/2} {cx-w/2},{cy}"
    nodes.append(f'<polygon points="{points}" fill="{pale}" stroke="{color}" stroke-width="2"/>')
    text(cx, cy - 5, f"[{tag}] {title}", 25, color, 700, anchor="middle")
    text(cx, cy + 25, f"{detail} · L{anchors[tag]}", 20, "#526477", anchor="middle")


def edge(path, color="#658094", dashed=False, arrow=True, bridge=False):
    if bridge:
        edges.append(f'<path d="{path}" fill="none" stroke="#ffffff" stroke-width="10"/>')
    dash = ' stroke-dasharray="9 7"' if dashed else ""
    edges.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"{dash}/>')
    # 显式三角箭头兼容原生 SVG 渲染器，不依赖 SVG 2 的 context-stroke。
    if arrow:
        tokens = re.findall(r"[MHV]|-?\d+(?:\.\d+)?", path)
        i, x, y, previous = 0, 0., 0., (0., 0.)
        while i < len(tokens):
            command = tokens[i]
            previous = (x, y)
            if command == "M":
                x, y = float(tokens[i + 1]), float(tokens[i + 2])
                i += 3
            elif command == "H":
                x = float(tokens[i + 1])
                i += 2
            elif command == "V":
                y = float(tokens[i + 1])
                i += 2
            else:
                raise ValueError(path)
        dx, dy = x - previous[0], y - previous[1]
        length = (dx * dx + dy * dy) ** .5
        dx, dy = dx / length, dy / length
        bx, by = x - 15 * dx, y - 15 * dy
        points = f"{x},{y} {bx+6*dy},{by-6*dx} {bx-6*dy},{by+6*dx}"
        edges.append(f'<polygon points="{points}" fill="{color}"/>')


text(48, 76, "GPT-2 文本生成：代码与完整流程对照", 47, weight=750, target=parts)
text(50, 121, "打开 generate_text.py，搜索节点编号；L 表示本次生成图时的源码行号。", 27, "#526477", target=parts)
for x, prefix, label in [(50, "M", "M：启动与收尾"), (450, "G", "G：逐 Token 生成"), (920, "S", "S：选取下一个 ID")]:
    color, pale = colors[prefix]
    rect(x, 152, 350, 43, pale, color, 12, target=parts)
    text(x + 16, 182, label, 23, color, 600, target=parts)
text(1360, 181, "B=1   V=50257   T=当前输入长度   CPU / FP32   无 KV cache", 24, "#526477", target=parts)

panel(40, 710, "M", "main() · 一次运行", "准备文本、分词器和权重，最后显示与保存结果")
panel(790, 1120, "G", "generate() · 每轮新增一个 Token", "预算检查 → 前向 → 选取 → 追加 → EOS 检查 → 下一轮")
panel(1950, 690, "S", "select_next_token() · G5 的展开", "输入 last_logits [1,V]，返回一个 ID 与概率分布")

# M：入口与收尾。M4 的调用、返回跨到 G 列，其他步骤沿纵向阅读。
card("M1", 80, 330, 630, 140, "读取选项与本地文件", ["prompt / strategy / 生成预算", "配置 + tokenizer.json + 预训练权重"])
card("M2", 80, 520, 630, 145, "编码提示词", [("tokenizer.encode(prompt)", True), "ID 列表 → torch.long 张量 [1,T₀]"])
card("M3", 80, 715, 630, 150, "构建并加载自己的 GPT", ["HF 读取参数 → own.load_hf_weights(...)", "own.eval()；本次计算使用 CPU / FP32"])
card("M4", 80, 935, 630, 120, "调用生成循环", [("generate(own, input_ids, ...)", True), "进入 G0，等待返回完整 ID 与记录"])
card("M5", 80, 1300, 630, 150, "解码并显示", [("tokenizer.decode(output_ids[0])", True), "完整序列一起解码；输出包含提示词"])
card("M6", 80, 1510, 630, 130, "可选参考验证", ["开启 --check-reference → HF 贪心 ID 对照", "未开启 → 直接进入 M7"])
card("M7", 80, 1700, 630, 135, "保存 JSON 报告", ["输入/输出 ID、每步概率、停止原因", "默认写入项目 .cache/runtime-check/"])
for y1, y2 in [(470, 520), (665, 715), (865, 935), (1450, 1510), (1640, 1700)]:
    edge(f"M395 {y1} V{y2}")
rect(90, 1120, 610, 115, "#f8fafc", "#dbe3ed")
text(112, 1158, "模型参数只在 M3 加载", 23, "#526477", 600)
text(112, 1193, "G 区循环增加的是输入长度，权重保持不变。", 23, "#526477")

# G：预算耗尽、上下文上限与 EOS 三条路径，共用右侧的停止通道。
card("G0", 940, 330, 700, 120, "检查与初始化", ["检查 [1,T₀] / long / CPU / eval", "复制输入；创建一次 RNG；默认原因=预算耗尽"])
diamond("G1", 1290, 535, 700, 110, "还有新增次数？", "for step in range(max_new_tokens)")
diamond("G2", 1290, 690, 700, 110, "T ≥ n_positions？", "本课将完整结果也限制在 1024 内")
card("G3", 940, 805, 700, 145, "① 完整前向", [("logits = model(ids)", True), "[1,T] → [1,T,V]；每轮重算完整前缀"])
card("G4", 940, 1000, 700, 115, "② 取最后输入位置", [("last_logits = logits[:, -1, :]", True), "[1,T,V] → [1,V]；利用整个当前前缀"])
card("G5", 940, 1170, 700, 120, "③ 调用选择函数", [("select_next_token(...)", True), "接收 next_id [1,1] 和 probs [1,V]"])
card("G6", 940, 1345, 700, 150, "④ 追加 ID，并记录本轮", [("ids = torch.cat((ids, next_id), dim=1)", True), "[1,T] + [1,1] → [1,T+1]；记录候选概率"])
diamond("G7", 1290, 1595, 700, 110, "⑤ 新 ID 是 EOS？", "GPT-2 EOS=50256；已追加并记录")
card("G8", 940, 1760, 700, 100, "返回结果", [("return ids, steps, stop_reason", True)])
for y1, y2 in [(450, 480), (590, 635), (745, 805), (950, 1000), (1115, 1170), (1290, 1345), (1495, 1540)]:
    edge(f"M1290 {y1} V{y2}", "#0f766e")
text(1310, 619, "是", 21, "#0f766e")
text(1310, 783, "否", 21, "#0f766e")
edge("M1640 535 H1830 V1810 H1640", "#b7791f")
edge("M1640 690 H1830", "#b7791f", arrow=False)
edge("M1640 1595 H1830", "#b7791f", arrow=False)
text(1657, 508, "否：预算耗尽", 21, "#9a641a")
text(1657, 532, "max_new_tokens", 18, "#9a641a", code=True)
text(1657, 663, "是：长度上限", 21, "#9a641a")
text(1657, 687, "context_limit", 18, "#9a641a", code=True)
text(1662, 1580, "是：eos", 22, "#9a641a", code=True)
edge("M940 1595 H850 V535 H940", "#0f766e")
text(875, 1577, "否", 22, "#0f766e")
nodes.append('<text transform="translate(875 1225) rotate(-90)" font-size="25" font-weight="600" style="fill:#0f766e">使用已增长的 ids，回到预算检查</text>')

# S：S0/1 共用入口，左支贪心、右支采样，共同返回 G5。
card("S0", 1990, 330, 610, 115, "检查分数", [("last_logits [1,V]", True), "检查原始分数均为有限值"])
diamond("S1", 2295, 530, 550, 110, "strategy == greedy？", "按策略选择一条分支")
card("S2", 1990, 650, 280, 190, "贪心", [("argmax(dim=-1,", True), ("  keepdim=True)", True), "直接选最高分 ID", "softmax 仅供记录"], size=23)
card("S3", 2320, 650, 280, 145, "温度", [("scores =", True), ("  last_logits / τ", True), "τ = temperature"], size=23)
card("S4", 2320, 850, 280, 190, "top-k", ["保留最高 k 个分数", "其余列设为 -inf", "scatter 保持词表 ID", "k=0 时保留全部"], size=23)
card("S5", 2320, 1095, 280, 135, "转为概率", [("softmax(scores)", True), "probs [1,V]"], size=23)
card("S6", 2320, 1285, 280, 165, "抽一个 ID", [("multinomial(probs,", True), ("  num_samples=1)", True), "使用持续推进的 RNG"], size=23)
rect(1990, 1710, 610, 120, "#f7f2ff", "#7c3aed")
text(2010, 1751, "返回 G5，随后执行 G6 追加", 26, "#7c3aed", 700)
text(2010, 1792, "next_id [1,1] + probs [1,V]", 25, code=True)
edge("M2295 445 V475", "#7c3aed")
edge("M2020 530 H1970 V605 H2130 V650", "#7c3aed")
edge("M2570 530 H2620 V605 H2460 V650", "#7c3aed")
text(2020, 592, "是：greedy", 21, "#7c3aed")
text(2395, 592, "否：sample", 21, "#7c3aed")
edge("M2130 840 V880 H1970 V1770 H1990", "#7c3aed")
for y1, y2 in [(795, 850), (1040, 1095), (1230, 1285), (1450, 1710)]:
    edge(f"M2460 {y1} V{y2}", "#7c3aed")
rect(2010, 965, 265, 415, "#faf8ff", "#ddd2ee")
for index, line in enumerate(["选择时要区分", "", "V=50257：整个词表", "top-k=50：候选数", "num_samples=1：", "每轮抽取一个 ID", "", "候选分数变为概率后，", "每行总和约为 1。", "", "较低概率也可能抽中。"]):
    text(2025, 1002 + index * 32, line, 20, "#675382", 600 if index == 0 else 400)
text(2480, 1520, "[1,1]", 22, "#7c3aed", code=True)

# 跨函数连接。白色描边在相交处留下间隙，表示相交而非合并。
edge("M710 995 H770 V390 H940", "#2563eb")
text(780, 370, "调用 G0", 21, "#2563eb")
edge("M940 1810 H775 V1375 H710", "#2563eb", dashed=True)
text(783, 1710, "返回", 20, "#2563eb")
edge("M1640 1200 H1930 V390 H1990", "#7c3aed", bridge=True)
text(1844, 1183, "调用 S0", 20, "#7c3aed")
edge("M1990 1770 H1920 V1260 H1640", "#7c3aed", dashed=True, bridge=True)
text(1700, 1307, "返回 next_id / probs", 19, "#7c3aed", code=True)

# 使用已保存的真实运行记录，底部展示前三轮，避免手编观测值。
report_path = ROOT / ".cache/runtime-check/lesson03-greedy.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
rect(40, 1950, 2600, 220, "#ffffff", "#dbe3ed", target=parts)
text(70, 1993, "真实运行对照 · 同一模型，输入逐步增长", 29, weight=700)
text(70, 2033, f'提示词：{report["prompt"]}    编码长度：{len(report["prompt_ids"])}', 25, "#526477")
for i, step in enumerate(report["steps"][:3]):
    x = 70 + i * 860
    text(x, 2084, f'第 {i+1} 轮  [1,{step["input_length"]}] → [1,{step["input_length"]},V]', 26, "#0f766e", 600)
    text(x, 2128, f'ID {step["next_token_id"]}  {step["token_text"]!r}  → [1,{step["output_length"]}]', 25, code=True)
text(48, 2210, "实线：执行 / 调用；虚线：跨函数返回。图展开正常流程；参数或数值检查失败时抛错退出。2026-10-18", 23, "#526477")

parts.extend(edges)
parts.extend(nodes)
parts.append("</svg>\n")
svg_path = OUT / "gpt2-generation-flow.svg"
svg_path.write_text("\n".join(parts), encoding="utf-8")

descriptions = {
    "M1": "选项、本地文件、配置与分词器", "M2": "提示词编码与 batch 维", "M3": "加载参考权重并复制到自己的模型",
    "M4": "进入生成循环", "M5": "解码完整 ID 并显示日志", "M6": "可选的参考生成对照", "M7": "保存实验报告",
    "G0": "前置检查、复制输入与初始化随机数", "G1": "新增次数预算", "G2": "总长度边界", "G3": "完整前缀前向计算",
    "G4": "取最后位置的词表分数", "G5": "调用 Token 选择函数", "G6": "追加 ID 与记录观察值", "G7": "新生成 EOS 检查", "G8": "返回 ID、记录与停止原因",
    "S0": "检查分数有效性", "S1": "选择策略", "S2": "贪心 argmax", "S3": "温度缩放", "S4": "top-k 与原词表 ID",
    "S5": "softmax 概率", "S6": "按概率抽样并返回",
}
rows = []
for family, count, start in [("M", 7, 1), ("G", 9, 0), ("S", 7, 0)]:
    for i in range(start, start + count):
        tag = f"{family}{i}"
        rows.append(f"| `{tag}` | {descriptions[tag]} | [第 {anchors[tag]} 行](../../generate_text.py#L{anchors[tag]}) |")
guide = '''# GPT-2 文本生成：代码与完整流程对照

![GPT-2 文本生成：代码与完整流程对照](gpt2-generation-flow.png)

[打开可放大的 SVG 原图](gpt2-generation-flow.svg)

图中每个编号对应 `generate_text.py` 的同名注释，可在 VS Code 中搜索 `[G3]` 等编号。`L` 是本次绘图时的行号；之后若继续增加注释，优先用稳定的编号定位。

## 按执行顺序读图

1. **M1～M4：准备一次运行。** 解析选项，加载本地配置与分词器，把提示词变成 `[1,T₀]` 的整数 ID，加载自己的模型，然后进入 `generate()`。
2. **G0～G2：准备并检查本轮。** 初始化一次随机数生成器。每次迭代先检查新增次数预算，再检查当前长度是否达到本教学实现的总长度上限。
3. **G3～G5：从前缀算出下一个 ID。** 自己的 `GPT.forward()` 输出 `[1,T,V]`，最后位置切片得到 `[1,V]`，将它交给右列 S 区选择。
4. **S0～S6：展开选取函数。** 贪心分支直接 argmax，softmax 仅供报告观察；采样分支依次进行温度缩放、top-k、softmax、multinomial。两条路径都返回 `[1,1]` 的 ID 和 `[1,V]` 的概率。
5. **G6～G8：反馈、停止与返回。** 先追加新 ID 并记录，再检查新 ID 是否为 EOS。未结束时用增长后的输入回到 G1。预算耗尽、达到长度上限或生成 EOS，都会通过 G8 返回调用者。
6. **M5～M7：解码与输出。** 完整序列一起解码；可选地做 Hugging Face 贪心对照；最后写出 JSON 报告。

实线表示执行或调用，虚线表示跨函数返回。跨线处的白色间隙表示两条线没有合并。错误检查失败会抛错退出；图中重点展开正常生成路径。

## 图与源码定位表

| 编号 | 观察内容 | 对应源码 |
| --- | --- | --- |
''' + "\n".join(rows) + '''

## 对照时抓住五件事

- **前向输出的 T 对应已有输入位置。** 最后位置经过因果注意力，已经可以利用完整前缀，因此用它预测接下来追加的 Token。
- **每轮真正反馈的是整数 ID。** `torch.cat(..., dim=1)` 把 `[1,T]` 扩展为 `[1,T+1]`，下一轮重新计算完整前缀；本课未使用 KV cache。
- **选择策略不改变模型权重。** 贪心和采样改变选中的 ID，随后不同前缀会影响之后的预测分布。
- **停止检查有先后顺序。** 预算在循环入口控制；长度在前向前检查；EOS 在追加后检查。因而返回序列包含新生成的 EOS，零预算时不调用前向。
- **展示与生成分开理解。** G6 中保存的前 5 个候选用于观察，与 top-k 的采样候选数量用途不同；最终文本由 M5 对完整 ID 序列解码。

图底部的三轮结果来自项目内保存的真实贪心运行报告。完整教学步骤、采样对比与自检见 [lesson03.md](../../lesson03.md)。前向模型内部结构可继续对照 [上一块的调用图](gpt2-forward-call-map.md)。

## 维护图与代码的对应关系

图由本目录的 `render_generation_flow.py` 生成，SVG 部分只使用 Python 标准库。脚本从源码读取编号和行号，从项目缓存读取已验证的实验前三步；输出在本目录，更新前应保留实验报告。

```bash
.venv/bin/python docs/diagrams/render_generation_flow.py
```

上面的命令更新 SVG 与本说明。macOS 可加 `--png` 同时通过系统 AppKit 导出 PNG，编译缓存和临时文件均放在本项目 `.cache/` 中：

```bash
.venv/bin/python docs/diagrams/render_generation_flow.py --png
```

生成图的本次依据：2026-10-18 本地注释版本；实验数据来自此前保存的 CPU / FP32 贪心报告。源码执行逻辑没有随注释调整而改变。
'''
(OUT / "gpt2-generation-flow.md").write_text(guide, encoding="utf-8")
if "--png" in sys.argv[1:]:
    if sys.platform != "darwin":
        raise SystemExit("SVG 已生成；--png 使用 macOS 系统 AppKit。")
    runtime = ROOT / ".cache/runtime-check"
    runtime.mkdir(parents=True, exist_ok=True)
    swift_path = runtime / "render-generation-svg.swift"
    swift_path.write_text('''import AppKit
import Foundation
let input = URL(fileURLWithPath: CommandLine.arguments[1])
let output = URL(fileURLWithPath: CommandLine.arguments[2])
guard let image = NSImage(contentsOf: input) else { fatalError("Cannot load local SVG") }
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
print("PNG export complete")
''', encoding="utf-8")
    subprocess.run(["/usr/bin/swift", "-module-cache-path", str(ROOT / ".cache/swift-module-cache"),
                    str(swift_path), str(svg_path), str(svg_path.with_suffix(".png"))], check=True)
print(json.dumps({"svg": str(svg_path), "source_anchors": anchors}, ensure_ascii=False))
