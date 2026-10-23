"""生成 FlashAttention 教学流程图；--png 通过 macOS AppKit 导出图片。

沿用本项目的 SVG 流程图格式。先独立复算课堂数字例子与稳定在线归一化，
再写入图片；不修改模型、不启动训练、不调用 GPU 性能实验。
"""

from html import escape
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 2200, 2320
INK, MUTED, LINE = "#172b40", "#586c7d", "#dce4ec"
BLUE, BLUE_BG = "#355fc0", "#edf2ff"
TEAL, TEAL_BG = "#087e75", "#e8f6f1"
AMBER, AMBER_BG = "#946027", "#fff4e4"
bg, edges, fg = [], [], []


def check_example():
    """验证示意图的普通归一化与带最大值修正的两块累计得到同一个输出。"""
    exponentials = [1, 2, 3, 4]
    values = [10, 20, 30, 40]
    scores = [math.log(value) for value in exponentials]
    reference = sum(e * v for e, v in zip(exponentials, values)) / sum(exponentials)
    simplified = []
    total, weighted = 0, 0
    maximum, denominator, numerator = -math.inf, 0.0, 0.0
    for start in (0, 2):
        block_exp, block_values = exponentials[start:start + 2], values[start:start + 2]
        total += sum(block_exp)
        weighted += sum(e * v for e, v in zip(block_exp, block_values))
        simplified.append((total, weighted))
        block_scores = scores[start:start + 2]
        new_maximum = max(maximum, *block_scores)
        correction = math.exp(maximum - new_maximum)
        scaled = [math.exp(s - new_maximum) for s in block_scores]
        denominator = correction * denominator + sum(scaled)
        numerator = correction * numerator + sum(e * v for e, v in zip(scaled, block_values))
        maximum = new_maximum
    assert simplified == [(3, 50), (10, 300)]
    assert reference == 30
    assert math.isclose(numerator / denominator, reference, abs_tol=1e-12)
    return exponentials, values, simplified, reference


def text(x, y, content, size=27, color=INK, weight=400, anchor="start", target=None):
    layer = fg if target is None else target
    layer.append(
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(content))}</text>'
    )


def rect(x, y, w, h, fill="white", stroke=LINE, radius=20, target=None, dash=False):
    layer = fg if target is None else target
    dash_attr = ' stroke-dasharray="10 8"' if dash else ""
    layer.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2"{dash_attr}/>'
    )


def arrow(points, color=MUTED, dashed=False):
    dash_attr = ' stroke-dasharray="10 8"' if dashed else ""
    path = "M " + " L ".join(f"{x} {y}" for x, y in points)
    edges.append(
        f'<path d="{path}" stroke="{color}" stroke-width="3" fill="none" '
        f'stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
    )
    (px, py), (x, y) = points[-2:]
    length = math.hypot(x - px, y - py)
    dx, dy = (x - px) / length, (y - py) / length
    bx, by = x - 13 * dx, y - 13 * dy
    edges.append(
        f'<polygon points="{x},{y} {bx+6*dy},{by-6*dx} '
        f'{bx-6*dy},{by+6*dx}" fill="{color}"/>'
    )


def card(x, y, w, h, title, lines=(), color=BLUE, fill=BLUE_BG):
    rect(x, y, w, h, fill)
    fg.append(
        f'<path d="M {x+1} {y+20} V {y+h-20}" stroke="{color}" stroke-width="5"/>'
    )
    text(x + 24, y + 40, title, 29, INK, 650)
    for index, line in enumerate(lines):
        text(x + 24, y + 78 + 36 * index, line, 26, MUTED)


def pill(x, y, w, label, fill, color):
    rect(x, y, w, 38, fill, fill, 10)
    text(x + w / 2, y + 27, label, 22, color, 650, "middle")


def matrix(x, y, n=6, cell=12, gap=4, selected=None, color=BLUE):
    """示意注意力矩阵：下三角允许，右上三角被因果遮罩挡住。"""
    for row in range(n):
        for column in range(n):
            if column > row:
                fill = "#f2f5f8"
            elif selected and (row, column) in selected:
                fill = color
            else:
                fill = "#cad7e8"
            rect(x + column * (cell + gap), y + row * (cell + gap),
                 cell, cell, fill, fill, 3)


def draw():
    exponentials, values, partials, result = check_example()

    text(64, 80, "FlashAttention：分块计算，累计输出", 54, weight=750, target=bg)
    text(66, 129, "同一个注意力公式 · 更少的大矩阵读写 · 把省下的搬运时间变成训练速度",
         27, MUTED, target=bg)

    # 在既有 Block 主线中定位优化范围。
    rect(64, 167, 2072, 142, "white", LINE, 22, bg)
    text(88, 204, "在一个 Transformer Block 中", 24, MUTED)
    route = [
        (88, 160, "LN1"),
        (292, 258, "Q / K / V 投影"),
        (594, 490, "注意力：匹配 → 归一化 → 汇总 V"),
        (1128, 246, "多头拼接"),
        (1418, 300, "输出投影 + 残差"),
        (1762, 350, "LN2 → MLP → 残差"),
    ]
    for index, (x, width, label) in enumerate(route):
        highlighted = index == 2
        rect(x, 226, width, 57, TEAL_BG if highlighted else "#f5f7fa",
             TEAL if highlighted else LINE, 12)
        text(x + width / 2, 263, label, 24, TEAL if highlighted else INK,
             650 if highlighted else 500, "middle")
        if index < len(route) - 1:
            arrow([(x + width, 254), (route[index + 1][0], 254)])

    # 左右相同输入输出，中间比较数据驻留与计算组织。
    for x, title, color, note in [
        (64, "01  普通 Attention：显式生成中间矩阵", BLUE, "对照项目当前写法"),
        (1120, "02  FlashAttention：按块融合与累计", TEAL, "下图固定一块 Q，遍历 K / V 块"),
    ]:
        rect(x, 345, 1016, 870, "white", LINE, 22, bg)
        text(x + 26, 393, title, 32, color, 700)
        text(x + 26, 431, note, 24, MUTED)

    # 左侧：完整 S/A 的显存往返。
    card(88, 462, 968, 96, "输入：显存中的 Q、K、V",
         ["分头后的形状均为 [B, H, T, d]"], BLUE, BLUE_BG)
    card(88, 592, 968, 124, "计算完整分数 S",
         ["S = QKᵀ / √d，并遮挡未来位置", "把 [B, H, T, T] 的分数矩阵写入显存"])
    matrix(920, 608)
    card(88, 754, 968, 124, "读取 S → 逐行 softmax → 完整权重 A",
         ["每行在允许的位置归一化", "把 [B, H, T, T] 的权重矩阵写入显存"])
    matrix(920, 770)
    card(88, 916, 968, 98, "读取 A 和 V → 计算 O = AV",
         ["输出为 [B, H, T, d]；后续交给多头拼接"])
    for a, b in [(558, 592), (716, 754), (878, 916)]:
        arrow([(572, a), (572, b)], BLUE)
    rect(88, 1050, 968, 132, AMBER_BG, AMBER_BG, 16)
    text(112, 1093, "主要开销：中间矩阵的存储与往返读写", 29, AMBER, 650)
    text(112, 1133, "T = 1024 时，单个序列、单个头的一张表有 1,048,576 个数。", 25, MUTED)
    text(112, 1168, "这些读写发生在 GPU 的显存与片上存储之间。", 25, MUTED)

    # 右侧：块内工作区、跨块状态、循环边界。
    card(1144, 462, 968, 96, "输入：显存中的 Q、K、V",
         ["分头后的形状均为 [B, H, T, d]"], TEAL, TEAL_BG)
    arrow([(1628, 558), (1628, 588)], TEAL)
    rect(1144, 588, 968, 428, "#f6fbf9", "#9accc0", 18, bg, True)
    pill(1166, 604, 254, "片上高速工作区", TEAL_BG, TEAL)
    text(2086, 631, "容量小 · 当前块算完即复用", 24, MUTED, anchor="end")

    card(1166, 662, 856, 86, "取当前 Q 块，以及一块 K / V",
         ["只把当前需要的小块载入工作区"], TEAL, "white")
    card(1166, 778, 856, 86, "计算局部分数 → 遮罩 → 指数",
         ["当前分数块随即参与后续累计"], TEAL, "white")
    card(1166, 894, 856, 96, "更新跨块状态 m、L、U",
         ["保留统计量和输出累计值，复用当前分数块的空间"], TEAL, TEAL_BG)
    for a, b in [(748, 778), (864, 894)]:
        arrow([(1580, a), (1580, b)], TEAL)
    arrow([(2022, 940), (2062, 940), (2062, 703), (2022, 703)], TEAL, True)
    text(2086, 791, "下", 23, TEAL, 600)
    text(2086, 824, "一", 23, TEAL, 600)
    text(2086, 857, "块", 23, TEAL, 600)
    arrow([(1580, 990), (1580, 1060)], TEAL)
    text(1600, 1043, "这一块 Q 的全部允许位置处理完", 23, TEAL)
    card(1144, 1060, 968, 122, "归一化 O = U / L → 写回输出",
         ["输出仍为 [B, H, T, d]；各 Q 块共同组成完整输出",
          "完整 S / A 无需写入显存"], TEAL, TEAL_BG)

    # 数字例子：与前一轮解释保持相同符号与数值，使用简化累计。
    rect(64, 1251, 2072, 591, "white", LINE, 22, bg)
    text(90, 1299, "03  为什么分块仍能得到同一个结果？", 32, INK, 700)
    text(90, 1340, "教学例子：固定最后一个 Token 的查询，只观察 V 的一维；四个位置均可见，Dropout = 0。",
         26, MUTED)
    text(90, 1380, "s 已含 Q·K / √d 缩放。这里先省略最大值修正，用分子、分母累计解释原理。", 25, MUTED)

    # 输入条与完整权重作对照。
    rect(90, 1410, 2020, 82, "#f5f7fb", "#f5f7fb", 12)
    text(114, 1443, "各位置 exp(s)", 24, MUTED)
    text(114, 1475, "[ 1, 2  |  3, 4 ]", 29, BLUE, 650)
    text(525, 1443, "V 的这一维", 24, MUTED)
    text(525, 1475, "[ 10, 20  |  30, 40 ]", 29, BLUE, 650)
    text(1060, 1443, "若先算完整权重", 24, MUTED)
    text(1060, 1475, "[ 0.1, 0.2, 0.3, 0.4 ]", 29, INK, 650)
    text(1660, 1443, "普通计算", 24, MUTED)
    text(1660, 1475, "1 + 4 + 9 + 16 = 30", 28, INK, 650)

    # 初始化 -> 两块 -> 最终输出，明确每个数的来源。
    rect(90, 1527, 205, 220, "#f5f7fb", LINE, 16)
    text(114, 1570, "初始化", 29, INK, 650)
    text(114, 1630, "L = 0", 33, BLUE, 650)
    text(114, 1681, "U = 0", 33, BLUE, 650)

    rect(341, 1527, 626, 220, BLUE_BG, LINE, 16)
    text(365, 1570, "第 1 块：位置 1、2", 29, BLUE, 700)
    text(365, 1620, "L = 0 + (1 + 2) = 3", 29, INK, 550)
    text(365, 1665, "U = 0 + (1 × 10 + 2 × 20) = 50", 27, INK, 550)
    text(365, 1714, "保留 L = 3、U = 50，继续处理下一块", 25, MUTED)

    rect(1013, 1527, 650, 220, TEAL_BG, LINE, 16)
    text(1037, 1570, "第 2 块：位置 3、4", 29, TEAL, 700)
    text(1037, 1620, "L = 3 + (3 + 4) = 10", 29, INK, 550)
    text(1037, 1665, "U = 50 + (3 × 30 + 4 × 40) = 300", 27, INK, 550)
    text(1037, 1714, "把新块的贡献，加到之前累计的 L、U 上", 25, MUTED)

    rect(1709, 1527, 401, 220, "#173d40", "#173d40", 16)
    text(1733, 1570, "全部块处理完", 29, "#ffffff", 650)
    text(1733, 1622, "o = U / L", 31, "#c7ede4", 600)
    text(1733, 1689, f"300 / 10 = {int(result)}", 44, "#ffffff", 700)
    for x1, x2 in [(295, 341), (967, 1013), (1663, 1709)]:
        arrow([(x1, 1637), (x2, 1637)], TEAL)

    text(90, 1795, "L = 指数值之和（归一化分母）    U = 指数值 × V 的加权和（分子）",
         27, INK, 600)
    text(90, 1827, "实际 V 有 d 维，所以每个查询保留 1 个 L 和 d 维 U；逐维除以同一个 L。", 25, MUTED)

    # 两个必要补充：稳定版与反向。避免把省略 m 的课堂例子画成真实内核。
    rect(64, 1878, 1016, 300, BLUE_BG, LINE, 20, bg)
    text(90, 1924, "真实实现：加上最大值 m，避免指数溢出", 30, BLUE, 700)
    text(90, 1965, "m = 已处理位置的最大分数；指数改为 exp(s − m)。", 25, MUTED)
    text(90, 2007, "新块进入 → 得到新最大值 m′ → α = exp(m − m′)", 26, INK, 550)
    text(90, 2053, "L′ = αL + Σ新块 exp(s − m′)", 28, BLUE, 650)
    text(90, 2097, "U′ = αU + Σ新块 exp(s − m′) · V", 28, BLUE, 650)
    text(90, 2141, "旧累计量与新块换到同一尺度；最终输出仍为 U′ / L′。", 25, MUTED)

    rect(1120, 1878, 1016, 300, "#f5f7fa", LINE, 20, bg)
    text(1146, 1924, "反向传播：按需重算局部权重", 30, INK, 700)
    text(1146, 1965, "前向保留 Q / K / V、输出与归一化统计。", 25, MUTED)
    card(1146, 1991, 360, 104, "收到输出梯度",
         ["读取已保留的数据"], BLUE, "white")
    card(1556, 1991, 554, 104, "重算当前块的分数 / 权重",
         ["随后计算并累计 Q / K / V 的梯度"], TEAL, TEAL_BG)
    arrow([(1506, 2043), (1556, 2043)], TEAL)
    text(1146, 2141, "用局部重算，减少保存与读回整张注意力矩阵的开销。", 25, MUTED)

    text(66, 2227, "范围说明：图示分块与累计原理，具体内核的循环和调度可不同；密集注意力的主要计算量仍随 T² 增长。",
         24, MUTED)
    text(66, 2267, "数学目标相同，浮点运算顺序可能带来小差异。依据：FlashAttention §3.1；Karpathy 视频约 1:59:46—2:06:54。",
         24, MUTED)
    text(2134, 2302, "GPT-2 学习 · 2.2 训练提速", 22, MUTED, anchor="end")

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<title>FlashAttention：普通注意力与分块累计的对照流程</title>
<desc>上方定位 Attention 在 Transformer Block 中的位置，中间对比显存往返与片上分块计算，下方用指数权重一二三四与 V 值十到四十，演示两个块累计得到三十。附稳定在线 softmax 公式与反向重算说明。</desc>
<style>text {{ font-family: "PingFang SC", "Helvetica Neue", Arial, sans-serif; }}</style>
<rect width="{WIDTH}" height="{HEIGHT}" fill="#f8fafc"/>
{''.join(bg)}{''.join(edges)}{''.join(fg)}
</svg>'''


def export_png(svg_path):
    if sys.platform != "darwin":
        raise SystemExit("SVG 已生成；--png 需要 macOS AppKit。")
    runtime = ROOT / ".cache/runtime-check"
    runtime.mkdir(parents=True, exist_ok=True)
    swift = runtime / "render-flash-attention-svg.swift"
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
    subprocess.run(
        ["/usr/bin/swift", "-module-cache-path", str(ROOT / ".cache/swift-module-cache"),
         str(swift), str(svg_path), str(svg_path.with_suffix(".png"))], check=True,
    )


if __name__ == "__main__":
    svg = draw()
    ET.fromstring(svg)
    output = OUT / "gpt2-flash-attention-flow.svg"
    output.write_text(svg, encoding="utf-8")
    if "--png" in sys.argv[1:]:
        export_png(output)
    print(output)
