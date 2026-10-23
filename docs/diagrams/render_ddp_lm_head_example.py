"""把 DDP 通信与 lm_head 的前向、交叉熵和反向计算连接起来。

这是固定上游特征的 2×2 分类教学模型，仅隔离输出投影路径。真实 GPT-2
的共享词嵌入梯度还包括输入路径的贡献。--verify-torch 用 CPU/float64
核对数学与整批目标，不运行 CUDA / NCCL，也不修改任何课程训练入口。
"""

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from render_ddp_allreduce_flow import (
    Drawing as BaseDrawing, export_png, INK, MUTED, LINE,
    BLUE, BLUE_BG, TEAL, TEAL_BG, AMBER, AMBER_BG,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 2600, 3700
W = [[1.0, 0.0], [0.0, 0.5]]
FEATURES = [[1.0, 2.0], [3.0, 6.0]]
TARGETS = [0, 1]
LR, DECAY, EPS = 0.01, 0.1, 1e-8


def example():
    """从特征和 W 算出图中所有数据，不把假设的梯度填进反向流程。"""
    rows = []
    for h, target in zip(FEATURES, TARGETS):
        z = [sum(w * x for w, x in zip(row, h)) for row in W]
        exp_z = [math.exp(v - max(z)) for v in z]
        p = [v / sum(exp_z) for v in exp_z]
        delta = [v - int(j == target) for j, v in enumerate(p)]
        grad = [[d * x for x in h] for d in delta]
        dh = [sum(delta[j] * W[j][k] for j in range(2)) for k in range(2)]
        rows.append(dict(h=h, target=target, logits=z, probabilities=p,
                         loss=-math.log(p[target]), delta=delta,
                         gradient=grad, feature_gradient=dh))
    summed = [[rows[0]["gradient"][j][k] + rows[1]["gradient"][j][k]
               for k in range(2)] for j in range(2)]
    avg = [[v / 2 for v in row] for row in summed]
    # AdamW 第一次更新：偏差修正后 m_hat=g、v_hat=g²（初始 m=v=0）。
    updated = [[(1 - LR * DECAY) * W[j][k]
                - LR * avg[j][k] / (abs(avg[j][k]) + EPS)
                for k in range(2)] for j in range(2)]
    assert rows[0]["gradient"] == [[-0.5, -1.0], [0.5, 1.0]]
    assert rows[1]["gradient"] == [[1.5, 3.0], [-1.5, -3.0]]
    assert summed == [[1.0, 2.0], [-1.0, -2.0]]
    assert avg == [[0.5, 1.0], [-0.5, -1.0]]
    return dict(initial_weight=W, rows=rows, summed_gradient=summed,
                averaged_gradient=avg, updated_weight=updated)


class Drawing(BaseDrawing):
    def matrix(self, x, y, values, w=300, h=145, color=INK,
               fill="#f4f7fa", highlight=False, digits=None, size=37):
        self.rect(x, y, w, h, fill=fill, stroke="none", radius=14)
        nrow, ncol = len(values), len(values[0])
        dx, dy = (w - 38) / ncol, (h - 22) / nrow
        if highlight:
            self.rect(x + 19, y + 11, dx, dy, AMBER_BG,
                      stroke="#d9bb8d", radius=8)
        self.fg.append(f'<path d="M {x+19} {y+13} h -7 v {h-26} h 7 '
                       f'M {x+w-19} {y+13} h 7 v {h-26} h -7" '
                       f'fill="none" stroke="{color}" stroke-width="2.5"/>')
        for r, row in enumerate(values):
            for c, v in enumerate(row):
                content = f"{v:.{digits}f}" if digits is not None else f"{v:g}"
                self.text(x + 19 + dx * (c + 0.5),
                          y + 11 + dy * (r + 0.5) + size * .34,
                          content.replace("-", "−"), size,
                          AMBER if highlight and r == 0 and c == 0 else color,
                          600, "middle", mono=True)

    def stage(self, n, y, title, lines, color=INK):
        self.text(82, y + 40, n, 47, color, 700)
        self.text(82, y + 91, title, 31, color, 650)
        for i, line in enumerate(lines):
            self.text(82, y + 137 + i * 37, line, 25, MUTED)

    def finish(self):
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<title>DDP 连上神经网络：lm_head 前向、反向、梯度同步与更新</title>
<desc>两张 GPU 各持有完整模型，分别处理样本 A、B。只隔离一个两维特征、两个输出词的 lm_head 教学模型，输入特征给定为 [1,2] 与 [3,6]。相同权重 [[1,0],[0,0.5]] 得到 logits [1,1] 与 [3,3]。目标分别是 ID0、ID1，交叉熵反向产生完整梯度矩阵 [[-0.5,-1],[0.5,1]] 与 [[1.5,3],[-1.5,-3]]。分块汇总同一参数的梯度、收集结果并平均得到 [[0.5,1],[-0.5,-1]]。各卡本地执行一次相同 AdamW，得到相同新权重。橙色格持续跟踪 W[0,0]：参数 1，本地梯度 -0.5 和 1.5，平均梯度 0.5，更新后约 0.989。</desc>
<style>text {{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}}</style>
<rect width="{WIDTH}" height="{HEIGHT}" fill="#f8fafc"/>
{''.join(self.bg)}{''.join(self.edges)}{''.join(self.fg)}
</svg>'''


def draw():
    data = example()
    d = Drawing()
    d.text(64, 96, "让 DDP 连上神经网络：从 lm_head 到梯度同步", 58, weight=700)
    d.text(64, 152, "两张 GPU 各自运行完整模型，处理不同样本；下方展开同一个线性层的具体计算。", 32, MUTED)

    d.rect(64, 188, 2472, 154, fill="white", background=True)
    d.text(88, 228, "在完整 GPT-2 中的位置 · 每张 GPU 上都有这整条路线", 28, MUTED, 550)
    blocks = [(88, 200, "Token ID"), (338, 320, "Embedding"),
              (708, 360, "12 个 Block"), (1118, 220, "LN_f"),
              (1388, 400, "lm_head · 本图展开"), (1838, 670, "交叉熵 loss → backward")]
    for i, (x, w, name) in enumerate(blocks):
        d.rect(x, 250, w, 66, AMBER_BG if i == 4 else "#f1f5f8", radius=12)
        d.text(x + w/2, 294, name, 29, AMBER if i == 4 else INK, 600, "middle")
        if i < len(blocks)-1:
            d.arrow([(x+w, 283), (blocks[i+1][0], 283)])
    d.text(64, 385, "教学缩小：特征维度 2，输出词只有 猫 / ID 0、狗 / ID 1；固定上游 h，仅实算 lm_head 输出路径。", 28, MUTED)

    # 第一步：不同 batch 行交给不同 GPU，而不是把网络层分给不同 GPU。
    d.stage("01", 445, "分配样本", ["总 batch = 2 条", "每卡各 1 条", "各看一个预测位置"])
    lanes = [(370, BLUE, BLUE_BG, "A", "猫"), (1590, TEAL, TEAL_BG, "B", "狗")]
    for i, (x, color, fill, sample, word) in enumerate(lanes):
        row = data["rows"][i]
        d.rect(x, 438, 940, 252, fill=fill)
        d.text(x+30, 488, f"GPU {i} · 完整模型副本 · 样本 {sample}", 36, color, 650)
        d.text(x+30, 538, "本卡前向：Embedding → 12 个 Block → LN_f", 29, INK)
        d.text(x+30, 590, f"给定该位置的输出特征 h{sample} = {row['h']}", 29, INK)
        d.text(x+30, 644, f"正确目标：{word} / ID {i}；目标向量 t = {[int(j == i) for j in range(2)]}", 29, INK)
        d.arrow([(x+470, 690), (x+470, 768)], color)
    d.text(1450, 715, "两份 W 副本", 25, MUTED, 550, "middle")
    d.text(1450, 751, "数值相同，显存独立", 24, MUTED, 550, "middle")

    # 第二步：前向需要完整的两行 W，每张卡都计算两个词的分数。
    d.stage("02", 790, "本卡前向", ["W 的数值相同", "各卡独立存储", "每卡都给两个词打分"])
    for i, (x, color, fill, sample, word) in enumerate(lanes):
        row = data["rows"][i]
        d.rect(x, 768, 940, 408)
        d.text(x+30, 817, f"GPU {i} 显存内：z{sample} = h{sample} Wᵀ", 34, color, 650)
        d.text(x+30, 859, f"W 副本在 GPU {i}；行 = 输出词，列 = 特征维度", 26, MUTED)
        d.matrix(x+40, 885, W, w=300, h=145, color=color, fill=fill, highlight=True)
        h0, h1 = row["h"]
        d.text(x+390, 904, f"猫：{h0:g} × 1 + {h1:g} × 0 = {row['logits'][0]:g}", 29, INK)
        d.text(x+390, 952, f"狗：{h0:g} × 0 + {h1:g} × 0.5 = {row['logits'][1]:g}", 29, INK)
        d.text(x+390, 1005, f"GPU {i}：logits = [{row['logits'][0]:g}, {row['logits'][1]:g}]", 30, color, 650)
        d.text(x+30, 1078, f"GPU {i}：p{sample} = softmax(logits) = [0.5, 0.5]", 29, INK)
        d.text(x+30, 1131, f"GPU {i}：L{sample} = −ln p{sample}[{word}] = 0.693147", 30, color, 600)
        d.arrow([(x+470, 1176), (x+470, 1256)], color)
    d.text(1450, 1214, "本卡 loss 反向", 26, MUTED, 550, "middle")

    # 第三步：交叉熵对 logits 的导数，再对 Linear 的 W 做链式求导。
    d.stage("03", 1280, "本卡反向", ["δ = ∂L/∂z", "G = ∂L/∂W", "每卡算完整梯度矩阵"])
    for i, (x, color, fill, sample, word) in enumerate(lanes):
        row = data["rows"][i]
        d.rect(x, 1256, 940, 440)
        delta = row["delta"]
        d.text(x+30, 1307, f"GPU {i} 本卡反向：δ{sample} = p{sample} − t{sample} = [{delta[0]:g}, {delta[1]:g}]".replace("-", "−"), 30, color, 650)
        d.text(x+30, 1354, f"GPU {i} 上计算 G{sample} = δ{sample}ᵀ h{sample}；下方三份数据都在本卡显存", 27, INK)
        d.text(x+107, 1405, f"δ{sample}ᵀ · GPU {i}", 23, color, 600, "middle")
        d.text(x+370, 1405, f"h{sample} · GPU {i}", 25, color, 600, "middle")
        d.text(x+737, 1405, f"G{sample} · GPU {i}", 25, color, 600, "middle")
        d.matrix(x+35, 1423, [[v] for v in delta], w=145, h=145, color=color, fill=fill, size=34)
        d.text(x+211, 1511, "×", 45, MUTED, 500, "middle")
        d.matrix(x+250, 1457, [row["h"]], w=240, h=80, color=color, fill=fill, size=35)
        d.text(x+531, 1511, "=", 43, MUTED, 500, "middle")
        d.matrix(x+575, 1423, row["gradient"], w=325, h=145, color=color, fill=fill, highlight=True)
        value = row["gradient"][0][0]
        d.text(x+30, 1612, f"GPU {i}：W.grad[0,0] = {value:g}（同步前的本卡贡献）".replace("-", "−"), 28, AMBER, 650)
        dh = row["feature_gradient"]
        d.text(x+30, 1656, f"留在 GPU {i}，传给本卡上游：∂L/∂h = [{dh[0]:g}, {dh[1]:g}]".replace("-", "−"), 26, MUTED)
        d.arrow([(x+470, 1696), (x+470, 1818)], color)
    d.text(1450, 1455, "此处各算各的", 25, MUTED, 600, "middle")
    d.text(1450, 1498, "尚未跨卡交换 G", 25, MUTED, 600, "middle")
    d.text(1450, 1740, "W 仍未更新", 26, AMBER, 600, "middle")

    # 第四步：各卡已经得到完整本地 dW；这时分工汇总梯度的行。
    d.rect(342, 1774, 2216, 821, fill="#fffaf2", stroke="#e2cba8",
           radius=24, background=True, dashed=True)
    d.text(1450, 1806, "DDP 触发 NCCL：汇总的是同一个参数在不同样本上的梯度", 29, AMBER, 650, "middle")
    d.stage("04", 1850, "分块求和", ["Reduce-Scatter", "此时才分工", "汇总梯度的两行"], AMBER)
    for i, x, color, fill, title, local, received, result in [
        (0, 370, BLUE, BLUE_BG, "GPU 0 内相加：汇总第一行梯度",
         "本卡 [−0.5, −1]", "GPU 1 的 [1.5, 3]", "[−0.5, −1] + [1.5, 3] = [1, 2]"),
        (1, 1590, TEAL, TEAL_BG, "GPU 1 内相加：汇总第二行梯度",
         "本卡 [−1.5, −3]", "GPU 0 的 [0.5, 1]", "[−1.5, −3] + [0.5, 1] = [−1, −2]"),
    ]:
        d.rect(x, 1818, 940, 300)
        d.text(x+30, 1871, title, 32, color, 650)
        d.text(x+30, 1924, f"GPU {i} 已有：{local}", 30, INK)
        d.text(x+30, 1976, f"接收到 GPU {i}：来自 {received}", 29, INK)
        d.rect(x+24, 2015, 892, 68, fill=fill, stroke="none", radius=12)
        d.text(x+470, 2060, result, 29, color, 650, "middle")
        d.arrow([(x+470, 2118), (x+470, 2248)], color)
    d.text(1450, 1931, "GPU 1 → GPU 0", 25, TEAL, 600, "middle")
    d.text(1450, 1963, "第一行 [1.5, 3]", 23, TEAL, 550, "middle")
    d.arrow([(1590, 1981), (1310, 1981)], TEAL, width=4)
    d.text(1450, 2044, "GPU 0 → GPU 1", 25, BLUE, 600, "middle")
    d.text(1450, 2076, "第二行 [0.5, 1]", 23, BLUE, 550, "middle")
    d.arrow([(1310, 2094), (1590, 2094)], BLUE, width=4)
    d.text(1450, 2168, "加法在各自 GPU 上执行", 28, AMBER, 650, "middle")

    # 第五步：补齐汇总后的所有行，然后得到每卡相同的平均梯度。
    d.stage("05", 2278, "收齐并平均", ["All-Gather", "每卡拿到完整的和", "再除以 2 张卡"], AMBER)
    for i, (x, color, fill, sample, word) in enumerate(lanes):
        d.rect(x, 2248, 940, 305)
        d.text(x+30, 2299, f"GPU {i} 收齐两行，在本卡显存内求平均", 32, color, 650)
        d.text(x+220, 2347, f"GPU {i} 上的梯度和", 27, MUTED, anchor="middle")
        d.text(x+701, 2347, f"GPU {i} 上的平均梯度 Ḡ", 26, MUTED, anchor="middle")
        d.matrix(x+55, 2370, data["summed_gradient"], w=330, h=145, color=color, fill=fill, highlight=True)
        d.text(x+463, 2456, "÷ 2 =", 34, AMBER, 650, "middle")
        d.matrix(x+535, 2370, data["averaged_gradient"], w=330, h=145, color=color, fill=fill, highlight=True)
        d.arrow([(x+470, 2553), (x+470, 2718)], color)
    d.text(1450, 2319, "GPU 0 → GPU 1", 25, BLUE, 600, "middle")
    d.text(1450, 2351, "第一行和 [1, 2]", 23, BLUE, 550, "middle")
    d.arrow([(1310, 2369), (1590, 2369)], BLUE, width=4)
    d.text(1450, 2449, "GPU 1 → GPU 0", 25, TEAL, 600, "middle")
    d.text(1450, 2481, "第二行和 [−1,−2]", 23, TEAL, 550, "middle")
    d.arrow([(1590, 2499), (1310, 2499)], TEAL, width=4)
    d.text(1450, 2635, "每卡各有一份 W.grad，数值一致；W[0,0] 的平均梯度是 0.5", 28, AMBER, 650, "middle")

    # 第六步：都使用平均梯度更新相同参数；不交换本地更新后的参数。
    d.stage("06", 2750, "各自更新", ["optimizer.step()", "同参数 + 同梯度", "得到同一个新模型"])
    for i, (x, color, fill, sample, word) in enumerate(lanes):
        d.rect(x, 2718, 940, 390, fill=fill)
        d.text(x+30, 2769, f"GPU {i}：用本卡的 W、Ḡ 和 m/v 做 AdamW", 31, color, 650)
        d.text(x+30, 2821, "首次更新，m/v 从 0 开始；本例不裁剪梯度。", 27, INK)
        d.text(x+30, 2867, "η = 0.01，λ = 0.1；ε = 10⁻⁸", 28, INK)
        d.text(x+30, 2922, "追踪 W[0,0]：m̂ = 0.5，v̂ = 0.25", 28, AMBER, 650)
        d.text(x+30, 2975, "1 × 0.999 − 0.01 × 0.5 / (0.5 + ε)", 25, AMBER)
        d.text(x+235, 3041, "≈ 0.989", 44, AMBER, 700, "middle")
        d.text(x+725, 2913, f"新 W · GPU {i}（近似）", 25, MUTED, anchor="middle")
        d.matrix(x+540, 2937, data["updated_weight"], w=365, h=145,
                 color=color, fill="white", highlight=True, digits=4, size=30)

    d.rect(370, 3144, 2160, 77)
    d.text(1450, 3195, "GPU 0、GPU 1 各存一份新模型（数值一致、显存独立）→ 清梯度 → 各取下一批样本", 29, INK, 550, "middle")

    d.card(64, 3270, 1216, 231, "对应到代码：从计算图到通信", [
        "model(input_ids) → logits → cross_entropy(logits, targets)",
        "loss.backward()：本卡求导；梯度就绪时 DDP 触发 NCCL。",
        "optimizer.step()：每卡使用同步后的梯度，更新本地参数。",
        "实际按 bucket 同步，可与仍在进行的反向计算重叠。"], "white", BLUE)
    d.card(1320, 3270, 1216, 231, "如何映射回完整 GPT-2？", [
        "真实 lm_head 的 W 是 [50257,768]；图中缩为 [2,2]。",
        "h 是给定的教学特征；真实 h 由上游网络前向计算。",
        "本例独立训练输出 W，只展开输出投影这条路径。",
        "其余可训练参数也各自在本卡求导，再按同名参数同步。"], "white", TEAL)
    d.text(64, 3547, "权重共享时：输入 Embedding 路径与输出 lm_head 路径的贡献先在本卡加到同一个参数梯度，再跨卡同步。", 27, INK)
    d.text(64, 3590, "通信示意：先求和后平均仅为数学展开；实际可先缩放或融合执行。图中按行分块只是便于阅读，与样本标签无关。", 26, MUTED)
    d.text(64, 3632, "验证范围：CPU/float64 核对两份本地梯度、整批平均梯度和一次 AdamW；未运行 CUDA / NCCL，数字非完整 GPT-2 输出。", 25, MUTED)
    d.text(64, 3675, "阶段二 · 单元 2 · 2.4 DDP 教学补充 · 橙色格始终追踪同一个 W[0,0] · 来源、源码定位与验证报告见同名 .md", 24, MUTED)
    return d.finish()


def verify_torch(data):
    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.set_num_threads(1)
    initial = torch.tensor(W, dtype=torch.float64)

    def make_model():
        layer = nn.Linear(2, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            layer.weight.copy_(initial)
        return layer

    models = [make_model() for _ in range(2)]
    local_grads, local_rows = [], []
    for i, model in enumerate(models):
        h = torch.tensor([FEATURES[i]], dtype=torch.float64, requires_grad=True)
        logits = model(h)
        logits.retain_grad()
        loss = F.cross_entropy(logits, torch.tensor([TARGETS[i]]))
        loss.backward()
        expected = data["rows"][i]
        assert torch.equal(logits.detach()[0], torch.tensor(expected["logits"]))
        assert torch.equal(logits.grad[0], torch.tensor(expected["delta"]))
        assert torch.equal(model.weight.grad, torch.tensor(expected["gradient"]))
        assert torch.equal(h.grad[0], torch.tensor(expected["feature_gradient"]))
        assert torch.equal(model.weight, initial)
        assert math.isclose(loss.item(), expected["loss"], abs_tol=1e-14)
        local_grads.append(model.weight.grad.detach().clone())
        local_rows.append({**expected, "parameters_unchanged_after_backward": True})

    # 普通 CPU 张量加法模拟归约；这里没有启动 DDP，也没有调用 NCCL。
    averaged = (local_grads[0] + local_grads[1]) / 2
    row0_sum = local_grads[0][0] + local_grads[1][0]
    row1_sum = local_grads[1][1] + local_grads[0][1]
    assert torch.equal(torch.stack([row0_sum, row1_sum]) / 2, averaged)
    full = make_model()
    full_loss = F.cross_entropy(full(torch.tensor(FEATURES, dtype=torch.float64)),
                                torch.tensor(TARGETS))
    full_loss.backward()
    error_grad = (full.weight.grad - averaged).abs().max().item()
    assert error_grad == 0.0
    opts = [torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                             weight_decay=DECAY, eps=EPS, foreach=False)
            for model in [*models, full]]
    for model in models:
        model.weight.grad.copy_(averaged)
    for opt in opts:
        opt.step()
    update_errors = [(model.weight-full.weight).abs().max().item() for model in models]
    assert update_errors == [0.0, 0.0]
    assert torch.allclose(full.weight.detach(), torch.tensor(data["updated_weight"], dtype=torch.float64),
                          rtol=0, atol=1e-14)
    for model, opt in zip([*models, full], opts):
        assert opt.state[model.weight]["step"].item() == 1
    report = {"scope": "CPU/float64, fixed h, isolated untied 2x2 output head; no CUDA, NCCL or full GPT run",
              "torch_version": torch.__version__, "initial_weight": W,
              "local_calculations": local_rows,
              "sum": data["summed_gradient"], "average": averaged.tolist(),
              "full_batch_mean_loss": full_loss.item(),
              "max_gradient_error_vs_full_batch": error_grad,
              "max_updated_weight_error_vs_full_batch": update_errors,
              "updated_weight": full.weight.detach().tolist(),
              "optimizer": {"name": "AdamW", "lr": LR, "betas": [0.9, 0.95],
                            "weight_decay": DECAY, "eps": EPS, "updates": 1,
                            "gradient_clipping": False},
              "source_location": "gpt2_model.py GPT.forward -> self.lm_head(x); toy isolates output path",
              "weight_tying_note": "Full GPT-2 also accumulates input-embedding-path contributions before DDP synchronization."}
    path = ROOT / "artifacts/ddp-lm-head-diagram-check.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


def backward_detail(svg):
    """从同一矢量绘图导出反向局部，保留每个操作数的 GPU 归属标签。"""
    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    root = ET.fromstring(svg)
    root.set("width", "2220")
    root.set("height", "480")
    root.set("viewBox", "340 1230 2220 480")
    root.find(f"{{{namespace}}}title").text = "本地反向：每份数据分别在哪张 GPU"
    root.find(f"{{{namespace}}}desc").text = (
        "左侧 GPU 0 显存中保存 delta_A、h_A 和 G_A；右侧 GPU 1 显存中保存 delta_B、h_B 和 G_B。"
        "两张卡分别在本卡执行外积，得到完整的本地权重梯度。此阶段的数值示意尚未跨卡交换 G。"
        "输入特征的梯度也留在原卡，继续传给原卡上游网络。")
    return ET.tostring(root, encoding="unicode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--verify-torch", action="store_true")
    args = parser.parse_args()
    values = example()
    if args.verify_torch:
        verify_torch(values)
    svg = draw()
    ET.fromstring(svg)
    output = OUT / "gpt2-ddp-lm-head-example.svg"
    output.write_text(svg, encoding="utf-8")
    detail_output = OUT / "gpt2-ddp-backward-gpu-labels.svg"
    detail_output.write_text(backward_detail(svg), encoding="utf-8")
    if args.png:
        export_png(output)
        export_png(detail_output)
    print(output)
    print(detail_output)
