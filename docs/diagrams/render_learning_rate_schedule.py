"""绘制预热、余弦衰减、下限保持三段规则；阶段宽度仅为教学示意。"""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gpt2-lr-matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


OUT = Path(__file__).resolve().parent
MAX_LR, MIN_LR = 0.0006, 0.00006
WARMUP_END, DECAY_END = 0.18, 0.78


def main():
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ["PingFang SC", "Heiti TC", "Hiragino Sans GB", "Arial Unicode MS", "Noto Sans CJK SC"]
    font = next((name for name in candidates if name in available), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [font, "DejaVu Sans"],
        "axes.unicode_minus": False, "svg.fonttype": "path",
    })

    x1 = np.linspace(0, WARMUP_END, 100)
    y1 = MIN_LR + (MAX_LR - MIN_LR) * x1 / WARMUP_END
    x2 = np.linspace(WARMUP_END, DECAY_END, 500)
    r = (x2 - WARMUP_END) / (DECAY_END - WARMUP_END)
    y2 = MIN_LR + (1 + np.cos(np.pi * r)) / 2 * (MAX_LR - MIN_LR)
    x3 = np.linspace(DECAY_END, 1, 100)
    y3 = np.full_like(x3, MIN_LR)
    assert np.isclose(y1[-1], y2[0]) and np.isclose(y2[-1], y3[0])
    assert np.all(np.diff(y1) > 0) and np.all(np.diff(y2) <= 0)
    assert np.all(y3 == MIN_LR)

    ink, muted, grid = "#172B40", "#596B7C", "#E5EBF1"
    amber, blue, teal = "#9F6A12", "#2563B7", "#087D76"
    fig = plt.figure(figsize=(12, 7), facecolor="white")
    fig.text(0.115, 0.925, "学习率调度的三段规则", fontsize=23, color=ink)
    fig.text(0.115, 0.865, "逐渐增大 → 余弦降低 → 保持最小学习率", fontsize=14, color=muted)

    ax = fig.add_axes([0.115, 0.23, 0.83, 0.55])
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(0, 0.00078)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    for name in ("bottom", "left"):
        ax.spines[name].set_color("#BDC9D3")
        ax.spines[name].set_linewidth(0.9)
    ax.tick_params(axis="both", colors=muted, labelsize=11, length=0, pad=9)
    ax.set_xticks([0, WARMUP_END, DECAY_END, 1])
    ax.set_xticklabels(["训练开始", "预热结束", "衰减结束", "继续训练"])
    ax.set_yticks(np.linspace(0, MAX_LR, 5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.6f}"))
    ax.set_ylabel("学习率 η", fontsize=14, color=ink, labelpad=17)
    ax.set_xlabel("训练进度（阶段宽度仅为示意）", fontsize=12, color=muted, labelpad=16)

    phases = [
        (0, WARMUP_END, amber, "① 线性预热", x1, y1),
        (WARMUP_END, DECAY_END, blue, "② 余弦衰减", x2, y2),
        (DECAY_END, 1, teal, "③ 下限保持", x3, y3),
    ]
    for start, end, color, label, x, y in phases:
        ax.axvspan(start, end, color=color, alpha=0.045, linewidth=0)
        ax.text((start + end) / 2, 0.00072, label, fontsize=14, color=ink, ha="center")
        ax.plot(x, y, color=color, linewidth=3.2, solid_capstyle="round")
    ax.vlines([WARMUP_END, DECAY_END], 0, 0.00067, colors="#C5CFD9", linewidth=1, linestyles=(0, (4, 4)))
    ax.hlines(MIN_LR, 0, DECAY_END, colors="#97A7B8", linewidth=0.9, linestyles=(0, (4, 4)))
    ax.scatter([WARMUP_END, DECAY_END], [MAX_LR, MIN_LR], s=60, c=[blue, teal], edgecolors="white", linewidths=1.6, zorder=4)
    ax.annotate("最大值 0.000600", xy=(WARMUP_END, MAX_LR), xytext=(15, 12), textcoords="offset points", fontsize=12, color=ink)
    ax.annotate("你截图中的余弦公式", xy=(0.48, 0.00033), xytext=(15, 22), textcoords="offset points", fontsize=12, color=ink)
    ax.text(0.89, 0.000145, "保持 0.000060", ha="center", fontsize=12, color=ink)

    fig.text(0.115, 0.085, "上一张图的 r = 0 → 1，仅对应这里的第 ② 段。", fontsize=12, color=ink)
    fig.text(0.115, 0.042, "第 ③ 段用于衰减结束后继续训练；保持学习率时，模型参数仍会更新。", fontsize=11, color=muted)
    for extension in ("png", "svg"):
        path = OUT / f"gpt2-learning-rate-three-phases.{extension}"
        fig.savefig(path, dpi=180, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
