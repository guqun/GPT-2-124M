"""绘制课程中的余弦学习率衰减曲线，导出 PNG 和 SVG。"""

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


def learning_rate(progress):
    """progress 为预热结束后的衰减进度，取值范围为 [0, 1]。"""
    return MIN_LR + (1 + np.cos(np.pi * progress)) / 2 * (MAX_LR - MIN_LR)


def main():
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ["PingFang SC", "Heiti TC", "Hiragino Sans GB", "Arial Unicode MS", "Noto Sans CJK SC"]
    font = next((name for name in candidates if name in available), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
        "svg.fonttype": "path",
    })

    progress = np.linspace(0, 1, 1001)
    rates = learning_rate(progress)
    anchors = np.array([0.0, 0.5, 1.0])
    values = learning_rate(anchors)
    assert np.allclose(values, [0.0006, 0.00033, 0.00006], rtol=0, atol=1e-15)
    assert np.all(np.diff(rates) <= 0)

    ink, muted, grid, blue = "#172B40", "#596B7C", "#E5EBF1", "#2563B7"
    fig = plt.figure(figsize=(12, 7.5), facecolor="white")
    fig.text(0.115, 0.925, "余弦衰减：学习率随进度变化", fontsize=23, color=ink)
    fig.text(
        0.115, 0.835,
        r"$\eta(r)=\eta_{\min}+\frac{1+\cos(\pi r)}{2}(\eta_{\max}-\eta_{\min})$",
        fontsize=23, color=ink,
    )
    fig.text(0.115, 0.78, "最大学习率 0.000600     ·     最小学习率 0.000060", fontsize=12, color=muted)

    ax = fig.add_axes([0.115, 0.16, 0.83, 0.55])
    ax.set_xlim(-0.025, 1.025)
    ax.set_ylim(0, MAX_LR * 1.10)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    for name in ("bottom", "left"):
        ax.spines[name].set_color("#BDC9D3")
        ax.spines[name].set_linewidth(0.9)
    ax.tick_params(axis="both", colors=muted, labelsize=11, length=0, pad=9)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_yticks(np.linspace(0, MAX_LR, 5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.6f}"))
    ax.set_xlabel("衰减阶段的进度 r", fontsize=14, color=ink, labelpad=15)
    ax.set_ylabel("学习率 η(r)", fontsize=14, color=ink, labelpad=17)

    ax.hlines(MIN_LR, 0, 1, color="#97A7B8", linewidth=1, linestyles=(0, (4, 4)))
    ax.vlines(0.5, 0, values[1], color="#C5CFD9", linewidth=1, linestyles=(0, (4, 4)))
    ax.plot(progress, rates, color=blue, linewidth=3.2, solid_capstyle="round")
    ax.scatter(anchors, values, s=65, color=blue, edgecolors="white", linewidths=1.7, zorder=4)

    labels = [
        (0.0, values[0], (18, -44), "left", "0% · 衰减开始\nη = 0.000600"),
        (0.5, values[1], (20, 24), "left", "50% · 衰减中点\nη = 0.000330"),
        (1.0, values[2], (-16, 24), "right", "100% · 衰减结束\nη = 0.000060"),
    ]
    for x, y, offset, align, label in labels:
        ax.annotate(
            label, xy=(x, y), xytext=offset, textcoords="offset points",
            ha=align, va="bottom", fontsize=12, color=ink, linespacing=1.55,
        )

    fig.text(0.115, 0.045, "横轴从预热结束后开始计量；虚线表示最小学习率。", fontsize=11, color=muted)
    for extension in ("png", "svg"):
        path = OUT / f"gpt2-cosine-learning-rate.{extension}"
        fig.savefig(path, dpi=180, facecolor="white")
        print(path)
    plt.close(fig)
    print("Verified: r=0, 0.5, 1 -> 0.000600, 0.000330, 0.000060")


if __name__ == "__main__":
    main()
