"""绘制 DDP / NCCL / All-Reduce 教学流程图。

沿用项目已有的 SVG + macOS AppKit 导出方式。四个梯度元素为手算例子，
仅校验分块求和、收集和平均；不运行分布式训练，也不代表 NCCL 实测轨迹。
"""

import argparse
from html import escape
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 2400, 2940
INK, MUTED, LINE = "#172b40", "#586c7d", "#dce4ec"
BLUE, BLUE_BG = "#355fc0", "#edf2ff"
TEAL, TEAL_BG = "#087e75", "#e8f6f1"
AMBER, AMBER_BG = "#946027", "#fff4e4"
G0, G1 = [2, 4, 6, 8], [10, 20, 30, 40]


def check_example():
    """验证图中每个数；这是普通算术核对，不调用 NCCL。"""
    first = [a + b for a, b in zip(G0[:2], G1[:2])]
    second = [a + b for a, b in zip(G0[2:], G1[2:])]
    gathered = first + second
    average = [x / 2 for x in gathered]
    assert first == [12, 24] and second == [36, 48]
    assert gathered == [a + b for a, b in zip(G0, G1)]
    assert average == [6, 12, 18, 24]
    # DDP 可以先做缩放，再执行 SUM；图上后除 2 是教学展开。
    assert average == [a / 2 + b / 2 for a, b in zip(G0, G1)]
    return first, second, gathered, average


class Drawing:
    def __init__(self):
        self.bg, self.edges, self.fg = [], [], []

    def text(self, x, y, value, size=29, color=INK, weight=400,
             anchor="start", mono=False):
        family = ' font-family="Menlo, monospace"' if mono else ""
        self.fg.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}"{family}>'
            f'{escape(str(value))}</text>')

    def rect(self, x, y, w, h, fill="white", stroke=LINE, radius=20,
             background=False, dashed=False):
        dash = ' stroke-dasharray="10 8"' if dashed else ""
        target = self.bg if background else self.fg
        target.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                      f'rx="{radius}" fill="{fill}" stroke="{stroke}" '
                      f'stroke-width="2"{dash}/>')

    def arrow(self, points, color=MUTED, dashed=False, width=3):
        dash = ' stroke-dasharray="10 8"' if dashed else ""
        path = "M " + " L ".join(f"{x} {y}" for x, y in points)
        self.edges.append(f'<path d="{path}" stroke="{color}" stroke-width="{width}" '
                          f'fill="none" stroke-linejoin="round" '
                          f'stroke-linecap="round"{dash}/>')
        (px, py), (x, y) = points[-2:]
        length = math.hypot(x - px, y - py)
        dx, dy = (x - px) / length, (y - py) / length
        bx, by = x - 16 * dx, y - 16 * dy
        self.edges.append(f'<polygon points="{x},{y} {bx-7*dy},{by+7*dx} '
                          f'{bx+7*dy},{by-7*dx}" fill="{color}"/>')

    def card(self, x, y, w, h, title, lines, fill="white", color=INK):
        self.rect(x, y, w, h, fill)
        self.text(x + 26, y + 45, title, 32, color, 650)
        for i, line in enumerate(lines):
            self.text(x + 26, y + 91 + 40 * i, line, 27, MUTED)

    def step(self, number, y, title, line1, line2, color=INK):
        self.text(88, y + 38, number, 46, color, 700)
        self.text(88, y + 86, title, 31, color, 650)
        self.text(88, y + 128, line1, 25, MUTED)
        self.text(88, y + 165, line2, 25, MUTED)

    def vector(self, x, y, values, color, fill, w=650, cell_h=74):
        """同一行的四个数对应同样的四个参数；竖线标记通信分块。"""
        self.rect(x, y, w, cell_h, fill, stroke="none", radius=13)
        col = w / 4
        for i, value in enumerate(values):
            self.text(x + col * (i + 0.5), y + 49, f"{value:g}",
                      37, color, 600, "middle", mono=True)
        self.fg.append(f'<line x1="{x+w/2}" y1="{y+13}" x2="{x+w/2}" '
                       f'y2="{y+cell_h-13}" stroke="{color}" stroke-width="2" opacity=".45"/>')

    def finish(self):
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<title>DDP 梯度同步：两张 GPU 如何得到同一份梯度</title>
<desc>CPU 上两个训练进程分别驱动 GPU 0 与 GPU 1。两份模型初值相同，分别处理数据 A 与 B。教学梯度为 [2,4,6,8] 和 [10,20,30,40]。通过 Reduce-Scatter，GPU 0 在本卡计算前半和 [12,24]，GPU 1 在本卡计算后半和 [36,48]。All-Gather 后两卡各有 [12,24,36,48]，平均得到 [6,12,18,24]。同步后各自裁剪并更新参数。DDP 负责协调同步，NCCL 执行 GPU 通信与归约，All-Reduce 定义汇总结果每卡均有。分块过程是教学实现，真实算法和缩放位置可能不同。</desc>
<style>text {{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}}</style>
<rect width="{WIDTH}" height="{HEIGHT}" fill="#f8fafc"/>
{''.join(self.bg)}{''.join(self.edges)}{''.join(self.fg)}
</svg>'''


def draw():
    first, second, gathered, average = check_example()
    d = Drawing()
    d.text(64, 96, "DDP 梯度同步：两张 GPU 如何得到同一份梯度", 57, weight=700)
    d.text(64, 151, "各算一批 → 分块求和 → 交换结果 → 求平均 → 各自更新参数", 33, MUTED)
    d.card(64, 185, 728, 165, "DDP · 协调训练同步", [
        "Distributed Data Parallel",
        "发现梯度就绪，组织各卡同步。"], BLUE_BG, BLUE)
    d.card(836, 185, 728, 165, "All-Reduce · 规定汇总结果", [
        "Reduce：对应元素汇总；All：每卡都有。",
        "本图使用 SUM：将对应元素相加。"], TEAL_BG, TEAL)
    d.card(1608, 185, 728, 165, "NCCL · 执行通信与归约", [
        "NVIDIA Collective Communications Library",
        "组织 GPU 间传输，在 GPU 上求和。"], AMBER_BG, AMBER)

    d.rect(64, 380, 2272, 112, fill="white")
    d.text(90, 423, "CPU 上的进程 0 / 进程 1", 30, weight=650)
    d.text(90, 465, "各自运行 Python / DDP，发起 GPU 计算与通信；求和、缩放和参数更新在 GPU 上执行。", 29, MUTED)
    d.arrow([(800, 492), (800, 528)], BLUE, dashed=True)
    d.arrow([(1900, 492), (1900, 528)], TEAL, dashed=True)
    d.rect(400, 530, 800, 80, fill=BLUE_BG)
    d.rect(1500, 530, 800, 80, fill=TEAL_BG)
    d.text(800, 580, "进程 0 → GPU 0", 34, BLUE, 650, "middle")
    d.text(1900, 580, "进程 1 → GPU 1", 34, TEAL, 650, "middle")
    d.text(1350, 558, "通信链路", 26, MUTED, 550, "middle")
    d.text(1350, 594, "NVLink / PCIe 等", 24, MUTED, anchor="middle")

    d.step("01", 644, "本地求梯度", "各处理自己的数据", "模型初始参数相同")
    for x, gpu, grad, data, color, fill in [
        (400, 0, G0, "A", BLUE, BLUE_BG),
        (1500, 1, G1, "B", TEAL, TEAL_BG),
    ]:
        d.rect(x, 642, 800, 273)
        d.text(x + 30, 688, f"完整模型 θ + 本卡数据 {data}", 33, color, 650)
        d.text(x + 30, 731, "前向 → 本地平均 loss → backward()", 29, MUTED)
        d.text(x + 30, 775, f"本地梯度 g{gpu}：同一组 4 个参数的梯度", 27, MUTED)
        d.vector(x + 70, 802, grad, color, fill)
        d.arrow([(x + 400, 915), (x + 400, 1030)], color)
    d.text(1350, 951, "θ 保持不变", 26, MUTED, anchor="middle")

    # 两个阶段合起来是一种 All-Reduce(SUM) 的教学实现。
    d.rect(370, 990, 1960, 760, fill="#fffaf2", stroke="#e2cba8",
           radius=24, background=True, dashed=True)
    d.text(1350, 1022, "All-Reduce (SUM)  =  分块求和 + 交换汇总结果", 28, AMBER, 650, "middle")
    d.step("02", 1050, "分块求和", "Reduce-Scatter", "每张卡算一半的和", AMBER)
    for x, color, fill, title, formula, result in [
        (400, BLUE, BLUE_BG, "GPU 0 负责前半部分", "[ 2,  4 ] + [ 10, 20 ]", first),
        (1500, TEAL, TEAL_BG, "GPU 1 负责后半部分", "[ 30, 40 ] + [ 6,  8 ]", second),
    ]:
        d.rect(x, 1042, 800, 295)
        d.text(x + 30, 1093, title, 33, color, 650)
        d.text(x + 30, 1135, "本卡对应部分 + 收到的对方对应部分", 28, MUTED)
        d.text(x + 400, 1195, formula, 36, color, 550, "middle", mono=True)
        d.rect(x + 190, 1224, 420, 75, fill=fill, stroke="none", radius=13)
        d.text(x + 400, 1274, f"= [ {result[0]}, {result[1]} ]", 39,
               color, 650, "middle", mono=True)
        d.arrow([(x + 400, 1337), (x + 400, 1442)], color)
    d.text(1350, 1152, "前半 [10,20]", 26, TEAL, 550, "middle")
    d.arrow([(1500, 1174), (1200, 1174)], TEAL, width=4)
    d.text(1350, 1241, "后半 [6,8]", 26, BLUE, 550, "middle")
    d.arrow([(1200, 1263), (1500, 1263)], BLUE, width=4)
    d.text(1350, 1387, "加法发生在各自 GPU 上", 27, AMBER, 650, "middle")

    d.step("03", 1450, "交换汇总结果", "All-Gather", "让每张卡都拿全", AMBER)
    for x, color, fill in [(400, BLUE, BLUE_BG), (1500, TEAL, TEAL_BG)]:
        d.rect(x, 1442, 800, 263)
        d.text(x + 30, 1492, "发送自己算好的半份，收齐另外半份", 30, color, 650)
        d.text(x + 30, 1538, "每张卡得到完整的梯度和", 29, MUTED)
        d.vector(x + 70, 1564, gathered, color, fill)
        d.text(x + 400, 1675, "g_sum = g₀ + g₁", 31, color, 550, "middle")
        d.arrow([(x + 400, 1705), (x + 400, 1830)], color)
    d.text(1350, 1548, "前半和 [12,24]", 25, BLUE, 550, "middle")
    d.arrow([(1200, 1570), (1500, 1570)], BLUE, width=4)
    d.text(1350, 1637, "后半和 [36,48]", 25, TEAL, 550, "middle")
    d.arrow([(1500, 1659), (1200, 1659)], TEAL, width=4)
    d.text(1350, 1782, "SUM 完成；每张卡持有同一份和", 28, AMBER, 650, "middle")

    d.step("04", 1848, "求平均", "GPU 上做缩放", "DDP 默认同步结果")
    for x, color, fill in [(400, BLUE, BLUE_BG), (1500, TEAL, TEAL_BG)]:
        d.rect(x, 1830, 800, 235)
        d.text(x + 30, 1880, "梯度和 ÷ 2 张卡 = 平均梯度", 32, color, 650)
        d.text(x + 30, 1926, "ḡ = (g₀ + g₁) / 2", 31, MUTED)
        d.vector(x + 70, 1952, average, color, fill)
        d.arrow([(x + 400, 2065), (x + 400, 2180)], color)
    d.text(1350, 2122, "两张卡的 .grad 一致", 29, TEAL, 650, "middle")

    d.step("05", 2200, "各自更新", "同步后再裁剪", "每卡更新本地模型")
    for x, color, fill in [(400, BLUE, BLUE_BG), (1500, TEAL, TEAL_BG)]:
        d.rect(x, 2180, 800, 245, fill=fill)
        d.text(x + 30, 2229, "输入：θ、平均梯度、优化器 m/v", 31, color, 650)
        d.text(x + 30, 2277, "全局梯度裁剪 → 设置学习率 → AdamW", 29, INK)
        d.text(x + 30, 2323, "各自调用 optimizer.step()", 29, INK)
        d.text(x + 30, 2378, "输出：一致的新参数 θ_next", 34, color, 650)
    d.rect(400, 2460, 1900, 74, fill="white")
    d.text(1350, 2508, "下一轮：清 .grad → 各取新数据 → 使用更新后的同一模型继续训练", 30, INK, 550, "middle")

    d.card(64, 2580, 1108, 210, "什么时候同步？", [
        "backward 中：一组梯度就绪 → DDP 触发 NCCL。",
        "按 bucket 分批同步，可与剩余反向计算重叠。",
        "同步完成后，各卡再进行裁剪和参数更新。"], "white", BLUE)
    d.card(1208, 2580, 1128, 210, "怎样理解图中的先后顺序？", [
        "图中先 SUM 再 ÷2；实际可先缩放，或融合执行。",
        "两阶段分块是教学实现，NCCL 会选择具体算法。",
        "相同参数、m/v、梯度和更新规则 → 相同的新参数。"], "white", TEAL)
    d.text(64, 2842, "示例条件：2 张 NVIDIA GPU，每卡有效 Token 数相同；4 个梯度元素为手算例子。", 27, MUTED)
    d.text(64, 2882, "数据由加载器分配；本图不展开梯度累积。通信分块不改变每张卡持有完整模型这一点。", 26, MUTED)
    d.text(64, 2920, "课程 2.4 · DDP 原理配图 · PyTorch / NVIDIA NCCL 官方文档与阅读说明见同名 .md · 非 CUDA 实测", 23, MUTED)
    return d.finish()


def export_png(svg_path):
    if sys.platform != "darwin":
        raise SystemExit("SVG 已保存；--png 使用 macOS AppKit。")
    runtime = ROOT / ".cache/runtime-check"
    runtime.mkdir(parents=True, exist_ok=True)
    swift = runtime / "render-ddp-allreduce-svg.swift"
    swift.write_text(r'''import AppKit
import Foundation
let input = URL(fileURLWithPath: CommandLine.arguments[1])
let output = URL(fileURLWithPath: CommandLine.arguments[2])
guard let image = NSImage(contentsOf: input) else { fatalError("Cannot load SVG") }
let width = Int(image.size.width), height = Int(image.size.height)
guard let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height,
    bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
    colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0),
    let context = NSGraphicsContext(bitmapImageRep: bitmap) else { fatalError("Cannot create bitmap") }
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = context
NSColor.white.setFill()
NSRect(x: 0, y: 0, width: width, height: height).fill()
image.draw(in: NSRect(x: 0, y: 0, width: width, height: height))
context.flushGraphics()
NSGraphicsContext.restoreGraphicsState()
guard let data = bitmap.representation(using: .png, properties: [:]) else { fatalError("Cannot encode PNG") }
try data.write(to: output)
print("PNG export complete: \(width) x \(height)")
''', encoding="utf-8")
    subprocess.run(["/usr/bin/swift", "-module-cache-path",
                    str(ROOT / ".cache/swift-module-cache"), str(swift),
                    str(svg_path), str(svg_path.with_suffix(".png"))], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", action="store_true", help="在 macOS 导出同尺寸 PNG")
    args = parser.parse_args()
    svg = draw()
    ET.fromstring(svg)
    output = OUT / "gpt2-ddp-allreduce-flow.svg"
    output.write_text(svg, encoding="utf-8")
    if args.png:
        export_png(output)
    print("Teaching arithmetic checked: sum=[12,24,36,48], average=[6,12,18,24].")
    print(output)
