"""生成本课 BF16 精度流程图；--png 使用 macOS AppKit 导出 PNG。

沿用项目中的 SVG 绘图方式。只读取已有报告，不执行训练。
图内颜色标记 Tensor dtype；A100 内部累加精度作为独立说明。
"""

from html import escape
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
probe = json.loads((ROOT / 'artifacts/lesson09-cpu-dtypes.json').read_text())
training = json.loads((ROOT / 'artifacts/lesson09-cpu-bf16-smoke.json').read_text())
observed = {row['name']: row['dtype'] for row in probe['bf16']['observations']}
for name in ('transformer.wte', 'transformer.h.0.ln_1', 'transformer.h.0',
             'transformer.ln_f', 'cross_entropy'):
    assert observed[name] == 'torch.float32', name
for name in ('transformer.h.0.attn.c_attn', 'transformer.h.0.mlp.c_fc',
             'transformer.h.0.mlp.c_proj', 'lm_head'):
    assert observed[name] == 'torch.bfloat16', name
for step in training['steps']:
    for name in ('parameter_dtype', 'parameter_grad_dtype', 'loss_dtype',
                 'adam_m_dtype', 'adam_v_dtype'):
        assert step[name] == 'torch.float32', name

WIDTH, HEIGHT = 2400, 1900
INK, MUTED, LINE = '#142d3b', '#566b78', '#dce5e9'
BLUE, BLUE_BG = '#305eb8', '#edf3ff'
TEAL, TEAL_BG = '#087b70', '#e8f6f2'
AMBER, AMBER_BG = '#99631b', '#fff5e5'
GRAY, GRAY_BG = '#526773', '#edf1f4'
bg, links, nodes = [], [], []


def text(x, y, content, size=26, color=INK, weight=400, anchor='start', target=None):
    layer = nodes if target is None else target
    layer.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
                 f'text-anchor="{anchor}" fill="{color}">{escape(str(content))}</text>')


def rect(x, y, w, h, fill='white', stroke=LINE, radius=18, target=None, dashed=False):
    layer = nodes if target is None else target
    dash = ' stroke-dasharray="8 7"' if dashed else ''
    layer.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>')


def pill(x, y, label, color, fill, width=112):
    rect(x, y, width, 38, fill, fill, 12)
    text(x + width / 2, y + 27, label, 23, color, 650, 'middle')


def card(x, y, w, h, title, body, dtype=None, color=BLUE, fill=BLUE_BG):
    rect(x, y, w, h, fill)
    nodes.append(f'<path d="M {x+1} {y+22} V {y+h-22}" stroke="{color}" stroke-width="5"/>')
    text(x + 24, y + 43, title, 29, INK, 650)
    if dtype:
        pill(x + w - 134, y + 17, dtype, color, 'white')
    for index, line in enumerate(body):
        text(x + 24, y + 76 + 34 * index, line, 25, MUTED)


def edge(points, color=GRAY, dashed=False, width=3):
    path = 'M ' + ' L '.join(f'{x} {y}' for x, y in points)
    dash = ' stroke-dasharray="9 8"' if dashed else ''
    links.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width}" '
                 f'stroke-linecap="round" stroke-linejoin="round"{dash}/>')
    x, y = points[-1]
    px, py = points[-2]
    length = ((x-px)**2 + (y-py)**2)**0.5
    dx, dy = (x-px)/length, (y-py)/length
    bx, by = x - 14*dx, y - 14*dy
    links.append(f'<polygon points="{x},{y} {bx+6*dy},{by-6*dx} {bx-6*dy},{by+6*dx}" fill="{color}"/>')


text(58, 80, '一次训练更新，精度怎样流动？', 56, weight=750, target=bg)
text(60, 130, 'GPT-2 · FP32 参数 + BF16 autocast · 从输入、前向与反向，到 AdamW 更新',
     28, MUTED, target=bg)
pill(60, 168, 'FP32', BLUE, BLUE_BG)
text(186, 195, '参数 / 部分结果', 24, MUTED)
pill(420, 168, 'BF16', TEAL, TEAL_BG)
text(546, 195, '部分运算的输入 / 输出', 24, MUTED)
pill(850, 168, '混合', AMBER, AMBER_BG)
text(976, 195, '按路径选择类型', 24, MUTED)
pill(1225, 168, 'int64', GRAY, GRAY_BG)
text(1351, 195, 'Token ID / 标签', 24, MUTED)
text(2336, 195, '色块表示 Tensor 的存储类型', 23, MUTED, anchor='end')

for x, w, number, title, detail in (
    (56, 744, '01', '前向计算', 'autocast 区域内'),
    (856, 664, '02', '反向传播', 'autocast 区域外'),
    (1576, 768, '03', '参数更新', 'autocast 区域外'),
):
    rect(x, 280, w, 1290, '#ffffff', LINE, 24, bg)
    text(x+26, 331, number, 30, TEAL, 750)
    text(x+84, 331, title, 32, INK, 700)
    text(x+w-26, 331, detail, 23, MUTED, anchor='end')

# 前向：以主要算子和一个矩阵乘法局部，说明跨运算的类型变化。
card(80, 380, 696, 92, '输入 x / 正确目标 y', ['[B,T] 整数 ID；本步已清空旧梯度'], 'int64', GRAY, GRAY_BG)
card(80, 515, 696, 122, 'Embedding / 位置向量 → LN',
     ['查 FP32 参数表，得到浮点特征 X32', '本例查表、LN 输出保持 FP32'], 'FP32')
card(80, 681, 696, 244, 'Linear / 矩阵乘法', [], 'BF16', TEAL, TEAL_BG)
text(104, 760, '特征 X32 → X16       权重 W32 → W16', 26, TEAL, 600)
text(104, 807, 'BF16 输入 × BF16 权重 → BF16 输出', 27, INK, 600)
text(104, 852, '原始 X32、W32 不因这次转换而改 dtype', 24, MUTED)
text(104, 894, '内部：A100 Tensor Core 可用 FP32 累加¹', 24, MUTED)
card(80, 970, 696, 146, 'Transformer 中反复执行',
     ['线性层输出 BF16；本例 LN / 残差结果 FP32', '按算子选择精度，逐层保留计算图'], '混合', AMBER, AMBER_BG)
card(80, 1160, 696, 109, '最终 LN → lm_head',
     ['FP32 特征 → 输出 logits [B,T,V]'], 'BF16', TEAL, TEAL_BG)
card(80, 1314, 696, 146, '交叉熵 loss',
     ['logits 转 FP32，与 int64 标签 y 计算', '输出标量 L；转换也记录在计算图中'], 'FP32')
for start, end in ((472, 515), (637, 681), (925, 970), (1116, 1160), (1269, 1314)):
    edge([(428, start), (428, end)])

# 反向：沿着前向真实运算与转换回传，不能整列涂成 FP32。
card(880, 380, 616, 114, '起点：loss 标量',
     ['调用 loss.backward()，沿计算图反传'], 'FP32')
card(880, 560, 616, 184, '沿各条计算路径传回梯度',
     ['logits 的梯度：BF16', 'FP32 残差分支：FP32 梯度', '中间梯度类型随对应路径变化'], '混合', AMBER, AMBER_BG)
card(880, 812, 616, 166, '经过参数的类型转换节点',
     ['以线性层的 BF16 权重分支为例：', 'g16 → cast(FP32) → g32', '转换保留计算图与原参数的连接'], color=AMBER, fill=AMBER_BG)
card(880, 1040, 616, 176, '累积到原参数 W.grad',
     ['梯度存储类型与原参数一致：FP32', '同一参数的多条使用路径在这里相加', '共享的 wte / lm_head 也是同一份梯度'], 'FP32')
for start, end in ((494, 560), (744, 812), (978, 1040)):
    edge([(1188, start), (1188, end)])
rect(880, 1314, 616, 146, '#f7f9fb', LINE)
text(904, 1355, '反向结束：得到了更新依据', 27, INK, 650)
text(904, 1396, 'W.grad 已有梯度；原参数 W 的数值', 25, MUTED)
text(904, 1432, '此时仍保持本轮前向时的值。', 25, MUTED)

# 参数与优化器状态持久保留；AdamW 使用本轮梯度更新这些状态。
card(1600, 380, 720, 302, '全程保留的训练状态', [], 'FP32')
text(1624, 473, '原始参数 W', 31, BLUE, 700)
text(1624, 516, '前向的 BF16 权重由它转换得到', 26, MUTED)
text(1624, 576, 'AdamW 历史统计 m、v', 30, BLUE, 700)
text(1624, 618, '首步创建；以后每批沿用并更新', 26, MUTED)
text(1624, 655, '本课这三份持久状态均保持 FP32', 25, MUTED)
card(1600, 875, 720, 270, 'optimizer.step()', [], 'FP32')
text(1624, 963, '输入：W + W.grad + 历史 m / v', 28, INK, 600)
text(1624, 1007, '更新 m、v → 偏差修正 → 计算更新量', 27, MUTED)
text(1624, 1055, 'W_new = (1 − ηλ)W − η m̂ / (√v̂ + ε)', 25, BLUE, 600)
text(1624, 1105, '把新数值写回同一个原始 FP32 参数', 28, INK, 650)
card(1600, 1230, 720, 150, '更新后的原参数 W',
     ['细小变化在 FP32 参数中继续累积', '下一批使用这份新参数和更新后的 m / v'], 'FP32')
card(1600, 1420, 720, 92, '下一批：重新按需转换',
     ['新 W32 → 新 W16，进入下一次前向'], color=TEAL, fill=TEAL_BG)
edge([(1960, 682), (1960, 875)], BLUE)
text(1984, 781, '沿用原始状态', 25, BLUE)
edge([(1960, 1145), (1960, 1230)], BLUE)
edge([(1960, 1380), (1960, 1420)], BLUE)

# 跨阶段连线：loss 进入反向；参数梯度与原始状态在优化器处相遇。
edge([(776, 1387), (838, 1387), (838, 437), (880, 437)], BLUE)
text(826, 1499, 'L → 反向', 23, BLUE, anchor='end')
edge([(1496, 1128), (1548, 1128), (1548, 963), (1600, 963)], BLUE)
text(1552, 1181, '梯度', 23, BLUE, anchor='middle')

# 右侧原始参数仅供前向取值，转换不会覆盖它。
edge([(2320, 473), (2372, 473), (2372, 252), (812, 252), (812, 760), (776, 760)], BLUE, True)
text(1160, 241, '前向从原始 W32 取值；计算用副本保持与原参数的连接', 24, BLUE)

# 下一 batch 回路，保持在主面板外，避免穿过节点。
edge([(1960, 1512), (1960, 1595), (30, 1595), (30, 426), (80, 426)], GRAY, True)
text(1110, 1631, '下一批：保留更新后的参数与 m / v，取新数据，zero_grad 清空旧梯度', 27, MUTED, anchor='middle')

rect(56, 1680, 744, 150, BLUE_BG, BLUE_BG)
text(80, 1724, '参数保存精度 ≠ 运算输入精度', 28, BLUE, 700)
text(80, 1768, '原始参数一直是 FP32；BF16 表示用于部分运算。', 25, MUTED)
text(80, 1803, '优化器更新的对象始终是原始参数。', 25, MUTED)
rect(856, 1680, 664, 150, TEAL_BG, TEAL_BG)
text(880, 1724, 'BF16 → FP32：不会恢复已丢失的细节', 27, TEAL, 700)
text(880, 1768, '1.234375（BF16）→ 1.234375（FP32）', 25, MUTED)
text(880, 1803, '扩展后的低位补零；随后可参与 FP32 更新。', 25, MUTED)
rect(1576, 1680, 768, 150, AMBER_BG, AMBER_BG)
text(1600, 1724, '反向在 autocast 外，仍包含混合精度', 28, AMBER, 700)
text(1600, 1768, '中间梯度沿前向图回传；W.grad 最终为 FP32。', 25, MUTED)
text(1600, 1803, '图示范围：本课 FP32 参数 + BF16 autocast。', 25, MUTED)
text(58, 1871, '¹ 内部累加以 A100 Tensor Core 为例，具体内核依设备与配置而定。节点 dtype 对照本项目 PyTorch 2.2.2 CPU 实测；源码与依据见配套说明。',
     21, MUTED)

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<title>BF16 自动混合精度：前向、反向与参数更新的精度变化</title>
<desc>FP32 原参数在前向按需生成 BF16 计算表示，反向中间梯度按路径使用不同精度，原参数梯度和 AdamW 状态保持 FP32，更新后进入下一批。</desc>
<style>text {{ font-family: "PingFang SC", "Helvetica Neue", Arial, sans-serif; }}</style>
<rect width="{WIDTH}" height="{HEIGHT}" fill="#f8fafb"/>
{''.join(bg)}{''.join(links)}{''.join(nodes)}
</svg>'''
ET.fromstring(svg)
svg_path = OUT / 'gpt2-bf16-precision-flow.svg'
svg_path.write_text(svg, encoding='utf-8')

if '--png' in sys.argv[1:]:
    if sys.platform != 'darwin':
        raise SystemExit('SVG 已生成；--png 需要 macOS AppKit。')
    runtime = ROOT / '.cache/runtime-check'
    runtime.mkdir(parents=True, exist_ok=True)
    swift = runtime / 'render-bf16-precision-svg.swift'
    swift.write_text('''import AppKit
import Foundation
let input = URL(fileURLWithPath: CommandLine.arguments[1])
let output = URL(fileURLWithPath: CommandLine.arguments[2])
guard let image = NSImage(contentsOf: input) else { fatalError("Cannot load SVG") }
let width = Int(image.size.width), height = Int(image.size.height)
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
''', encoding='utf-8')
    subprocess.run(['/usr/bin/swift', '-module-cache-path', str(ROOT / '.cache/swift-module-cache'),
                    str(swift), str(svg_path), str(svg_path.with_suffix('.png'))], check=True)

print(svg_path)
