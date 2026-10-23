"""绘制梯度累积教学图；使用项目既有 SVG / macOS AppKit 导出方式。

数值矩阵是教学例子，不冒充 GPT-2 的实测梯度。--verify-torch 使用一个
CPU/float64 二次损失实验，核对整批与微批累积、裁剪及一次 AdamW 更新。
不会改变模型实现或既有训练入口。
"""

import argparse
from html import escape
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 2400, 3520
INK, MUTED, LINE = "#172b40", "#586c7d", "#dce4ec"
BLUE, BLUE_BG = "#355fc0", "#edf2ff"
TEAL, TEAL_BG = "#087e75", "#e8f6f1"
AMBER, AMBER_BG = "#946027", "#fff4e4"
GRADIENTS = [
    [[2, -2], [4, 0]],
    [[4, 2], [0, 4]],
    [[6, -2], [4, 4]],
    [[8, 2], [0, 8]],
]
K = 4


def matrix_add(a, b):
    return [[x + y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def check_example():
    rows, current = [], [[0.0, 0.0], [0.0, 0.0]]
    for index, gradient in enumerate(GRADIENTS, 1):
        delta = [[x / K for x in row] for row in gradient]
        updated = matrix_add(current, delta)
        rows.append({"micro_step": index, "raw_gradient": gradient,
                     "scaled_contribution": delta, "previous": current,
                     "accumulated": updated})
        current = updated
    assert current == [[5, 0], [2, 4]]
    assert [r["accumulated"][0][0] for r in rows] == [0.5, 1.5, 3.0, 5.0]
    assert 2 * 32 * K == 256
    return rows


class Drawing:
    def __init__(self):
        self.bg, self.edges, self.fg = [], [], []

    def text(self, x, y, content, size=28, color=INK, weight=400,
             anchor="start", mono=False):
        family = ' font-family="Menlo, monospace"' if mono else ""
        self.fg.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}"{family}>'
            f'{escape(str(content))}</text>')

    def rect(self, x, y, w, h, fill="white", stroke=LINE, radius=20,
             background=False, dashed=False):
        target = self.bg if background else self.fg
        dash = ' stroke-dasharray="12 9"' if dashed else ""
        target.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                      f'rx="{radius}" fill="{fill}" stroke="{stroke}" '
                      f'stroke-width="2"{dash}/>')

    def arrow(self, points, color=MUTED, dashed=False):
        dash = ' stroke-dasharray="10 8"' if dashed else ""
        path = "M " + " L ".join(f"{x} {y}" for x, y in points)
        self.edges.append(f'<path d="{path}" stroke="{color}" stroke-width="3" '
                          f'fill="none" stroke-linejoin="round" '
                          f'stroke-linecap="round"{dash}/>')
        (px, py), (x, y) = points[-2:]
        length = math.hypot(x - px, y - py)
        dx, dy = (x - px) / length, (y - py) / length
        bx, by = x - 14 * dx, y - 14 * dy
        self.edges.append(f'<polygon points="{x},{y} {bx - 6 * dy},{by + 6 * dx} '
                          f'{bx + 6 * dy},{by - 6 * dx}" fill="{color}"/>')

    def card(self, x, y, w, h, title, lines, fill="white", color=INK,
             title_size=31, line_size=27, line_height=38):
        self.rect(x, y, w, h, fill=fill)
        self.text(x + 24, y + 43, title, title_size, color, 650)
        for index, line in enumerate(lines):
            self.text(x + 24, y + 86 + line_height * index, line, line_size, MUTED)

    def matrix(self, x, y, values, color=INK, fill="#f4f7fa"):
        self.rect(x, y, 260, 120, fill=fill, radius=12)
        self.fg.append(f'<path d="M {x+30} {y+18} h -10 v 84 h 10 '
                       f'M {x+230} {y+18} h 10 v 84 h -10" '
                       f'fill="none" stroke="{color}" stroke-width="2.5"/>')
        for r in range(2):
            for c in range(2):
                value = f"{values[r][c]:g}".replace("-", "−")
                self.text(x + 78 + 104 * c, y + 47 + 45 * r,
                          value, 34, color, 550, "middle", mono=True)

    def finish(self):
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<title>梯度累积全流程：四次反向，一次参数更新</title>
<desc>单设备、B为2、T为32、累积4个新微批次。展开每个微批次的取数、前向、平均交叉熵、loss除4、反向与逐元素梯度累加。四行2乘2矩阵展示从零累积到[[5,0],[2,4]]。最后统一裁剪、设置学习率并执行一次AdamW，下一轮只清梯度，保留参数和优化器历史。矩阵为教学数字。</desc>
<style>text {{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}}</style>
<rect width="{WIDTH}" height="{HEIGHT}" fill="#f8fafc"/>
{''.join(self.bg)}{''.join(self.edges)}{''.join(self.fg)}
</svg>'''


def draw():
    rows = check_example()
    d = Drawing()
    d.text(64, 96, "梯度累积：4 次反向，1 次参数更新", 60, weight=700)
    d.text(64, 150, "模型、优化器与 loader 在循环外准备一次；进入外层更新循环 s。", 31, MUTED)
    for x, title, body in [
        (64, "B = 2 条序列", "每次送入模型的微批次"),
        (644, "T = 32 个 Token", "每条序列的上下文长度不变"),
        (1224, "K = 4 次累积", "4 次 forward + backward"),
        (1804, "有效 batch = 8 条", "每次更新参考 256 个 Token"),
    ]:
        d.card(x, 185, 532, 110, title, [body], BLUE_BG, BLUE,
               title_size=32, line_size=25)

    d.rect(64, 324, 2272, 1080, background=True)
    d.text(94, 377, "01  控制流程：微批次内部发生什么", 38, weight=650)
    d.card(94, 414, 630, 126, "① 开始一次参数更新：清梯度", [
        "optimizer.zero_grad(set_to_none=True)"], BLUE_BG, BLUE, line_size=25)
    d.card(762, 414, 1544, 126, "整个内层循环，保持同一套参数 θₛ", [
        "仅 .grad 累积；AdamW 的 m、v 和更新计数也保持不变。循环中不调用 step()。"],
        AMBER_BG, AMBER, line_size=27)
    d.arrow([(405, 540), (405, 575)], BLUE)
    d.rect(94, 575, 2212, 665, fill="#f5fbf8", stroke="#9acfc4",
           background=True, dashed=True)
    d.text(126, 625, "② 内层循环：i = 1 → 2 → 3 → 4", 32, TEAL, 650)
    d.text(2268, 625, "每次取新数据，不重复训练同一个微批次", 26, MUTED, anchor="end")

    d.card(130, 665, 650, 165, "a  取新的 xᵢ、yᵢ", [
        "输入 / 目标：[B,T] = [2,32]",
        "y 是对应位置的下一 Token 目标"], title_size=31)
    d.card(850, 665, 650, 165, "b  使用参数 θₛ 前向计算", [
        "Embedding → Blocks → LN_f → lm_head",
        "输出 logits：[2,32,V]；V 是词表大小"], title_size=31, line_size=25)
    d.card(1570, 665, 650, 165, "c  当前微批次的平均交叉熵", [
        "Lᵢ = (1 / 64) × Σ本批64个位置的损失",
        "先在当前微批次内部求均值"], title_size=30, line_size=27)
    d.arrow([(780, 748), (850, 748)])
    d.arrow([(1500, 748), (1570, 748)])
    d.arrow([(1895, 830), (1895, 915)], TEAL)
    d.text(1925, 882, "由 loss 向参数反传", 25, TEAL)

    d.card(1570, 915, 650, 175, "d  先缩放 loss：Lᵢ / 4", [
        "让本次贡献占整个有效 batch 的 1/4",
        "避免把 4 个平均 loss 直接求和"], BLUE_BG, BLUE, line_size=27)
    d.card(850, 915, 650, 175, "e  backward()：链式求导", [
        "Δᵢ = gᵢ/4 = ∂(Lᵢ/4)/∂θ",
        "= (1/256) Σ本批64个位置 ∂ℓ/∂θ",
        "共享参数在不同路径上的贡献也相加"],
        BLUE_BG, BLUE, line_size=25, line_height=31)
    d.card(130, 915, 650, 175, "f  参数的 .grad 接收贡献", [
        "Gᵢ = Gᵢ₋₁ + Δᵢ（对应元素相加）",
        "首次从 None 建立；之后继续相加"], TEAL_BG, TEAL, line_size=27)
    d.arrow([(1570, 1002), (1500, 1002)], TEAL)
    d.arrow([(850, 1002), (780, 1002)], TEAL)
    d.rect(130, 1124, 2090, 78, fill="white")
    d.text(154, 1174, "本次 backward 结束：可释放本批反向所需的中间激活；累计的 .grad 保留下来。", 29, MUTED)
    d.rect(750, 1270, 900, 80, fill=TEAL_BG, stroke="#9acfc4")
    d.text(1200, 1322, "③ 已经累计满 K = 4 个微批次？", 32, TEAL, 650, "middle")
    d.arrow([(455, 1090), (455, 1220), (1200, 1220), (1200, 1270)], TEAL)
    d.arrow([(750, 1310), (110, 1310), (110, 748), (130, 748)], TEAL, dashed=True)
    d.text(152, 1290, "否：i + 1；保留 .grad", 27, TEAL, 550)
    d.arrow([(1650, 1310), (1750, 1310)], TEAL)
    d.text(1772, 1322, "是：进入下方统一更新 ↓", 29, TEAL, 650)

    d.rect(64, 1436, 2272, 1044, background=True)
    d.text(94, 1490, "02  四轮累计展开：旧 .grad + 本次梯度贡献 → 新 .grad", 38, weight=650)
    d.text(94, 1540, "仅观察一张参数 W 的 2×2 梯度矩阵。数值为教学假设；GPT-2 的每张参数矩阵都各自累计。", 28, MUTED)
    for x, text in [(195, "微批次"), (495, "原始梯度 gᵢ"), (910, "贡献 Δᵢ = gᵢ/4"),
                    (1310, "原累计 Gᵢ₋₁"), (1710, "新累计 Gᵢ")]:
        d.text(x, 1597, text, 28, MUTED, 550, "middle")
    notes = [
        ("首次建立 .grad", "None 在此按零起点理解"),
        ("读取上一轮 G₁", "相加，形状仍为 2×2"),
        ("读取上一轮 G₂", "W、m/v 仍未更新"),
        ("得到完整的 G₄", "交给下面的统一裁剪"),
    ]
    for index, row in enumerate(rows):
        y = 1620 + 180 * index
        if index % 2 == 1:
            d.rect(92, y - 3, 2216, 158, fill="#f9fbfd", stroke="none", background=True)
        d.text(195, y + 55, f"第 {index+1} 批", 32, BLUE, 650, "middle")
        d.text(195, y + 100, f"参数仍是 θₛ", 25, MUTED, anchor="middle")
        d.matrix(365, y + 13, row["raw_gradient"], BLUE, BLUE_BG)
        d.text(698, y + 87, "÷ 4", 34, BLUE, 550, "middle")
        d.matrix(780, y + 13, row["scaled_contribution"], BLUE, BLUE_BG)
        d.text(1110, y + 87, "+", 42, TEAL, 500, "middle")
        d.matrix(1180, y + 13, row["previous"])
        d.text(1510, y + 87, "=", 42, TEAL, 500, "middle")
        d.matrix(1580, y + 13, row["accumulated"], TEAL, TEAL_BG)
        d.arrow([(1840, y + 73), (1895, y + 73)], TEAL)
        d.text(1918, y + 60, notes[index][0], 28, TEAL, 650)
        d.text(1918, y + 102, notes[index][1], 24, MUTED)
    d.rect(94, 2350, 2212, 96, fill=TEAL_BG)
    d.text(120, 2391, "G₄ = (g₁ + g₂ + g₃ + g₄) / 4 = ∂[(L₁ + L₂ + L₃ + L₄) / 4] / ∂W", 31, TEAL, 600)
    d.text(120, 2427, "两层汇总：每批内部的 64 个位置先求平均；跨 4 批再平均。它们共同对应 256 个位置的平均损失。", 25, MUTED)

    d.rect(64, 2510, 2272, 690, background=True)
    d.text(94, 2566, "03  统一更新：累计完成后，裁剪、学习率调度、AdamW 各执行一次", 37, weight=650)
    d.card(94, 2620, 650, 440, "④ 裁剪累计梯度", [
        "输入：所有参数各自的累计梯度 G",
        "r = √(Σ所有参数、所有元素 G²)",
        "c = min(1, 1 / (r + 10⁻⁶))",
        "g = c × G",
        "同一个 c 缩放所有参数的梯度。",
        "这里阈值为 1；r 是全局范数。",
        "不是每个微批次各裁剪一次。"],
        TEAL_BG, TEAL, line_size=27, line_height=43)
    d.card(790, 2620, 490, 440, "⑤ 设置当前学习率", [
        "η = get_lr(s)",
        "按参数更新编号 s 调度。",
        "为各参数组写入当前 η。",
        "内层 4 次微批次计算，",
        "不推进 4 次调度。"],
        BLUE_BG, BLUE, title_size=30, line_size=27, line_height=43)
    d.card(1326, 2620, 980, 440, "⑥ optimizer.step()：AdamW 更新", [
        "读入 θₛ、裁剪后 g、历史 m/v、各组衰减系数 λ",
        "t ← t + 1",
        "m ← β₁m + (1−β₁)g",
        "v ← β₂v + (1−β₂)g²",
        "m̂ = m/(1−β₁ᵗ)；v̂ = v/(1−β₂ᵗ)",
        "θₛ₊₁ = (1−ηλ)θₛ − ηm̂/(√v̂+ε)",
        "写回新参数、m/v 和 t；fused 改变执行方式。"],
        AMBER_BG, AMBER, line_size=27, line_height=43)
    d.arrow([(744, 2810), (790, 2810)], TEAL)
    d.arrow([(1280, 2810), (1326, 2810)], BLUE)
    d.rect(94, 3102, 2212, 68, fill="#f3f6fa")
    d.text(122, 3146, "下一次外层更新：s ← s + 1 → 使用新参数与历史 m/v → 回到①只清 .grad → 再取 4 个新微批次。", 28, INK, 550)

    d.card(64, 3232, 1108, 208, "适用条件与显存", [
        "每批有效 Token 数相同，loss 先求均值；否则按 nᵢ/Σn 加权。",
        "逐批释放中间激活；参数、累计梯度、m/v 仍需存储。",
        "随机 Dropout 与浮点运算顺序可能带来数值差异。"],
        line_size=25, line_height=37)
    d.card(1208, 3232, 1128, 208, "一轮调用次数：不要混淆两个循环", [
        "zero_grad：1 次  ·  forward：4 次  ·  backward：4 次",
        "clip_grad_norm_：1 次  ·  optimizer.step：1 次",
        "AdamW 的 m/v、计数与权重衰减，仅在最后 step 时更新。"],
        line_size=25, line_height=37)
    d.text(64, 3480, "单设备教学图 · Karpathy 约 2:35–2:46 / commit 01be6b3 · PyTorch 2.2.2 · 完整来源与矩阵复算见同名 .md", 23, MUTED)
    return d.finish()


def export_png(svg_path):
    if sys.platform != "darwin":
        raise SystemExit("SVG 已生成；--png 需要 macOS AppKit。")
    runtime = ROOT / ".cache/runtime-check"
    runtime.mkdir(parents=True, exist_ok=True)
    swift = runtime / "render-gradient-accumulation-svg.swift"
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


def verify_torch():
    """用可复算的二次损失生成图中梯度；不是完整 GPT 训练实验。"""
    import torch

    torch.set_num_threads(1)
    start = torch.tensor([[1.0, -1.0], [0.5, 2.0]], dtype=torch.float64)
    gs = torch.tensor(GRADIENTS, dtype=torch.float64)
    # 每批两条相同的教学样本。MSE 对四个元素求均值，故取 target=W−2g。
    targets = (start[None, :, :] - 2 * gs).repeat_interleave(2, dim=0)
    full = torch.nn.Parameter(start.clone())
    accum = torch.nn.Parameter(start.clone())
    kwargs = dict(lr=0.01, betas=(0.9, 0.95), weight_decay=0.1,
                  eps=1e-8, foreach=False)
    full_opt = torch.optim.AdamW([full], **kwargs)
    accum_opt = torch.optim.AdamW([accum], **kwargs)
    full_opt.zero_grad(set_to_none=True)
    full_loss = ((full - targets) ** 2).mean()
    full_loss.backward()
    accum_opt.zero_grad(set_to_none=True)
    assert accum.grad is None
    report_rows = []
    rows = check_example()
    for i in range(K):
        loss = ((accum - targets[2*i:2*i+2]) ** 2).mean()
        (loss / K).backward()
        expected = torch.tensor(rows[i]["accumulated"], dtype=torch.float64)
        assert torch.equal(accum.grad, expected)
        assert torch.equal(accum, start)
        assert len(accum_opt.state) == 0  # 第一次 step 前，AdamW 尚未初始化 m/v。
        report_rows.append({**rows[i], "mean_loss": loss.item(),
                            "parameters_unchanged": True,
                            "optimizer_state_not_advanced": True})
    gradient_error = (full.grad - accum.grad).abs().max().item()
    assert gradient_error == 0.0
    full_norm = torch.nn.utils.clip_grad_norm_([full], 1.0).item()
    accum_norm = torch.nn.utils.clip_grad_norm_([accum], 1.0).item()
    clipped_error = (full.grad - accum.grad).abs().max().item()
    assert clipped_error == 0.0
    full_opt.step()
    accum_opt.step()
    update_error = (full - accum).abs().max().item()
    assert update_error == 0.0
    assert accum_opt.state[accum]["step"].item() == 1
    saved_m = accum_opt.state[accum]["exp_avg"].clone()
    saved_v = accum_opt.state[accum]["exp_avg_sq"].clone()
    saved_parameter = accum.detach().clone()
    accum_opt.zero_grad(set_to_none=True)
    assert accum.grad is None
    assert torch.equal(accum, saved_parameter)
    assert torch.equal(accum_opt.state[accum]["exp_avg"], saved_m)
    assert torch.equal(accum_opt.state[accum]["exp_avg_sq"], saved_v)
    report = {
        "scope": "CPU/float64 teaching quadratic loss, one 2x2 parameter; not GPT-2 training or a speed test",
        "torch_version": torch.__version__, "micro_batches": K,
        "samples_per_micro_batch": 2, "rows": report_rows,
        "full_batch_mean_loss": full_loss.item(),
        "gradient_after_clipping": full.grad.detach().clone().tolist(),
        "diagram_accumulated_gradient_before_clipping": rows[-1]["accumulated"],
        "max_gradient_difference_before_clipping": gradient_error,
        "global_norm_before_clipping": {"full": full_norm, "accumulated": accum_norm},
        "max_gradient_difference_after_clipping": clipped_error,
        "max_parameter_difference_after_one_adamw_step": update_error,
        "optimizer_updates": 1, "zero_grad_preserves_parameters_and_moments": True,
        "assumptions": "equal sample counts, same fixed parameters, additive mean loss, no randomness",
    }
    path = ROOT / "artifacts/gradient-accumulation-diagram-check.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--verify-torch", action="store_true")
    args = parser.parse_args()
    svg = draw()
    ET.fromstring(svg)
    output = OUT / "gpt2-gradient-accumulation-flow.svg"
    output.write_text(svg, encoding="utf-8")
    if args.png:
        export_png(output)
    if args.verify_torch:
        verify_torch()
    print(output)
