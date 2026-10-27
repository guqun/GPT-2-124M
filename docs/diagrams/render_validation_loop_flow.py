"""绘制知识块 13 的实际代码流转；行号与数值从源码、既有报告读取。

不执行训练，不下载数据。SVG 是可编辑源；--png 复用项目的 AppKit 导出器。
"""

import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from render_ddp_allreduce_flow import (
    Drawing as BaseDrawing, export_png, INK, MUTED, LINE,
    BLUE, BLUE_BG, TEAL, TEAL_BG, AMBER, AMBER_BG,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 2400, 3600
PURPLE, PURPLE_BG = "#7461a8", "#f1eef8"
SOURCE = ROOT / "train_validation.py"
REPORT = ROOT / "artifacts/lesson13-cpu-report.json"


def evidence():
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    anchors = {}
    for n, line in enumerate(lines, 1):
        match = re.match(r"\s*# \[([MVT]\d+)\]", line)
        if match:
            assert match[1] not in anchors
            anchors[match[1]] = n
    for fragment, key in [
        ("def main(", "main"),
        ("def train_with_validation(", "train"),
        ("def evaluate_validation(", "eval"),
        ("def record_validation(", "record"),
        ("optimizer.step()", "update"),
    ]:
        hits = [n for n, line in enumerate(lines, 1) if fragment in line and not line.lstrip().startswith("#")]
        assert len(hits) == 1, (fragment, hits)
        anchors[key] = hits[0]
    assert set(f"V{i}" for i in range(1, 7)) <= anchors.keys()
    r = json.loads(REPORT.read_text(encoding="utf-8"))
    assert (r["device_type"], r["world_size"], r["model_size"]) == ("cpu", 1, "tiny")
    assert (r["batch_size_per_rank"], r["seq_len"], r["accumulation_steps"]) == (2, 32, 8)
    assert (r["parameter_updates"], r["validation_every"], r["validation_batches_per_rank"]) == (3, 2, 2)
    assert r["tokens_per_optimizer_update"] == 2 * 32 * 8 == 512
    assert r["train_range_exclusive"] == [0, 304222]
    assert r["val_range_exclusive"] == [304222, 338025]
    train = r["ranks"][0]["training"]
    val = r["ranks"][0]["validation"]
    assert [v["after_updates"] for v in val] == [0, 2, 3]
    assert all(v["local_batch_starts"] == [0, 64] for v in val)
    assert all(v["global_predicted_positions"] == 128 for v in val)
    assert [v["next_start"] for v in train] == [512, 1024, 1536]
    return anchors, r, train, val


class Drawing(BaseDrawing):
    def node(self, x, y, w, h, title, lines=(), color=INK, fill="white", ref=None):
        self.rect(x, y, w, h, fill, radius=17)
        self.text(x + 24, y + 40, title, 29, color, 650)
        for i, line in enumerate(lines):
            self.text(x + 24, y + 79 + 35 * i, line, 25, MUTED)
        if ref:
            self.text(x + w - 4, y - 10, ref, 20, color, anchor="end")

    def diamond(self, x, y, rx, ry, title, sub):
        self.fg.append(f'<polygon points="{x},{y-ry} {x+rx},{y} {x},{y+ry} {x-rx},{y}" '
                       f'fill="{AMBER_BG}" stroke="#d6ba96" stroke-width="2"/>')
        self.text(x, y - 5, title, 28, INK, 650, "middle")
        self.text(x, y + 32, sub, 23, MUTED, anchor="middle")

    def finish(self):
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<title>验证循环：从主训练循环进入验证，再返回训练</title>
<desc>对应 train_validation.py。左侧展示初始化、初始验证、训练微批次累积、参数更新、验证触发和循环结束；右侧展开同一个验证调用，依次切换 eval、重置验证游标、no_grad 下取数与前向、累计 loss、可选跨进程均值、恢复原模式并返回。底部是 CPU tiny 实测的验证 0、训练 1、训练 2、验证 2、训练 3、验证 3。验证指标不传给优化器。</desc>
<style>text {{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}}</style>
<rect width="{WIDTH}" height="{HEIGHT}" fill="#f7f9fc"/>
{''.join(self.bg)}{''.join(self.edges)}{''.join(self.fg)}
</svg>'''


def draw(a, report, train, val):
    d = Drawing()
    ref = lambda key: f"{key} · L{a[key]}" if key[0] in "MVT" else f"L{a[key]}"
    d.text(64, 84, "验证循环：代码如何进入、计算、返回训练", 56, weight=700)
    d.text(64, 137, "知识块 13  /  train_validation.py  /  左侧看执行顺序，右侧展开验证调用", 30, MUTED)
    for x, label, color in [(64, "训练", BLUE), (260, "验证", TEAL), (456, "修改参数", AMBER), (710, "函数调用", PURPLE)]:
        d.rect(x, 162, 16, 16, color, stroke="none", radius=4)
        d.text(x + 27, 178, label, 23, MUTED)
    d.text(2336, 178, "实测配置：CPU · tiny · B=2 · T=32 · R=1", 25, MUTED, anchor="end")

    # main() 在所有训练/验证调用之前完成准备。
    d.node(64, 220, 736, 230, "M1  建立两套独立的数据加载器", [
        "Tiny Shakespeare → Token ID → 按位置切分",
        "train：[0, 304222)；val：[304222, 338025)",
        "各自在自己的范围内构造 x 与右移一位的 y",
    ], BLUE, BLUE_BG, ref("M1"))
    d.node(832, 220, 736, 230, "M2  建立模型与优化器", [
        "随机初始化 GPT → 放到 device → 可选 DDP",
        "建立 AdamW；训练和验证使用同一份当前参数",
        "eval() 只改变模块模式，不创建另一份模型",
    ], BLUE, "white", ref("M2"))
    d.node(1600, 220, 736, 230, "M3  进入主训练函数", [
        "train_with_validation(...)：所有 rank 都执行",
        "输入：model、optimizer、两套 loader 等",
        "最终返回：training / validation 两组记录",
    ], BLUE, "white", ref("M3"))
    d.arrow([(800, 335), (832, 335)], BLUE)
    d.arrow([(1568, 335), (1600, 335)], BLUE)
    d.arrow([(1968, 450), (1968, 475), (604, 475), (604, 500)], BLUE)

    # 两条泳道不是两个并行任务；右侧是左侧 A/B 的函数展开。
    d.rect(64, 500, 1080, 2430, "white", background=True, radius=24)
    d.rect(1240, 500, 1096, 2430, "#fbfdfc", background=True, radius=24)
    d.text(100, 553, "01  主训练循环", 36, BLUE, 700)
    d.text(100, 591, f"train_with_validation() · L{a['train']}", 25, MUTED, mono=True)
    d.text(1276, 553, "02  A / B 都调用这里", 36, TEAL, 700)
    d.text(1276, 591, f"evaluate_validation() · L{a['eval']} · 正常执行路径", 25, MUTED)

    d.node(174, 630, 860, 126, "A  初始验证：record_validation(0)", [
        "model.train() → 先检查随机模型；验证返回后再往下",
    ], PURPLE, PURPLE_BG, ref("T1"))
    d.node(174, 805, 860, 90, "for step in range(steps)  ·  共更新 3 次", [], BLUE, BLUE_BG)
    d.node(174, 945, 860, 100, "optimizer.zero_grad(set_to_none=True)", [
        "开始一次新更新：清空上一次训练留下的梯度",
    ], BLUE, "white", ref("T2"))
    d.rect(174, 1095, 860, 430, BLUE_BG, radius=17, background=True)
    d.text(198, 1138, "训练内循环：for micro_step in range(K)", 29, BLUE, 650)
    d.text(198, 1179, "本次 K=8；下面三步重复 8 次", 25, MUTED)
    d.node(212, 1208, 770, 73, "① train_loader.next_batch() → x, y", [], BLUE)
    d.node(212, 1310, 770, 73, "② forward_and_loss(...) → logits, loss", [], BLUE)
    d.node(212, 1410, 770, 58, "③ (loss / K).backward() → 累加 .grad", [], BLUE)
    d.arrow([(597, 1281), (597, 1310)], BLUE)
    d.arrow([(597, 1383), (597, 1410)], BLUE)
    d.arrow([(982, 1439), (1064, 1439), (1064, 1244), (982, 1244)], BLUE)
    d.text(198, 1512, "多卡：前 K−1 次 no_sync；最后一次反向同步梯度。", 23, MUTED)
    d.node(174, 1560, 860, 152, "裁剪梯度 → 设置 lr(step) → optimizer.step()", [
        "AdamW 用累积梯度更新参数与 m / v / step",
        "一次更新覆盖 8 × 2 × 32 = 512 个预测位置",
    ], AMBER, AMBER_BG, f"参数更新点 · {ref('update')}")
    d.node(174, 1762, 860, 100, "保存训练记录；completed = step + 1", [
        "记录本次训练 loss、学习率、梯度范数；rank 0 打印",
    ], BLUE)
    d.diamond(604, 1980, 405, 82, "到验证间隔，或已是最后一次更新？", "completed % 2 == 0  或  completed == 3")
    d.text(654, 2091, f"是 · {ref('T3')}", 24, TEAL)
    d.node(174, 2110, 860, 115, "B  record_validation(completed)", [
        "用刚更新的参数验证；返回后接着判断训练是否结束",
    ], PURPLE, PURPLE_BG)
    d.diamond(604, 2390, 340, 78, "还有下一次参数更新？", "for step 循环是否继续")
    d.text(649, 2510, "否", 25, MUTED)
    d.node(174, 2530, 860, 154, "return {'training': ..., 'validation': ...}", [
        "回到 main() → gather_records() 汇总各 rank 记录",
        "rank 0 写入 JSON 报告 → 清理进程组 → 结束",
    ], BLUE, BLUE_BG)
    d.node(100, 2740, 1008, 150, "验证结果在这里如何影响训练？", [
        "本代码只记录与打印；没有把 val_loss 传给 AdamW 或调度器。",
        "下一次更新仍取下一批训练数据，学习率仍按更新序号计算。",
    ], INK, "#f5f7fa")
    for start, end in [(756, 805), (895, 945), (1045, 1095), (1525, 1560), (1712, 1762), (1862, 1898), (2062, 2110), (2225, 2312), (2468, 2530)]:
        d.arrow([(604, start), (604, end)], BLUE)
    # 不需要验证时绕过 B，最终在“还有更新？”前合流。
    d.arrow([(1009, 1980), (1090, 1980), (1090, 2280), (604, 2280)], MUTED)
    d.text(1064, 2030, "否", 24, MUTED)
    d.arrow([(264, 2390), (124, 2390), (124, 995), (174, 995)], BLUE)
    d.text(154, 2444, "是：进入下一次更新", 24, BLUE)

    # 两个调用点用紫色虚线汇入同一个 wrapper；返回后才沿左侧实线继续。
    d.node(1320, 630, 936, 126, "record_validation(completed_updates)", [
        "调用同一个 evaluate_validation(...)；使用调用当时的参数",
    ], PURPLE, PURPLE_BG, ref("record"))
    d.arrow([(1034, 693), (1320, 693)], PURPLE, dashed=True)
    d.arrow([(1034, 2167), (1190, 2167), (1190, 693), (1320, 693)], PURPLE, dashed=True)
    d.text(1170, 668, "调用", 23, PURPLE, anchor="middle")

    d.node(1320, 805, 936, 118, "V1  保存原模式，然后 model.eval()", [
        "was_training = model.training；使用当前参数，切换模块行为",
    ], TEAL, TEAL_BG, ref("V1"))
    d.node(1320, 968, 936, 128, "V2  重置验证游标与 loss 累加器", [
        "val_loader.reset()；val_loss_accum = 0",
        "每次从相同验证位置开始；train_loader 游标不动",
    ], TEAL, "white", ref("V2"))

    d.rect(1298, 1144, 1016, 656, TEAL_BG, radius=20, background=True)
    d.text(1324, 1188, "with torch.no_grad():  ·  不记录反向计算图", 29, TEAL, 650)
    d.text(1324, 1225, "for _ in range(validation_batches)  ·  本次读取 2 批", 25, MUTED)
    d.node(1350, 1255, 880, 120, "V3  val_loader.next_batch() → x, y", [
        "x / y：[2, 32]；移动到 runtime.device（本次 CPU）",
    ], TEAL, "white", ref("V3"))
    d.node(1350, 1418, 880, 174, "V4  forward_and_loss(model, x, y, ...)", [
        "model(x) → logits：[2, 32, 50257]",
        "cross_entropy → 64 个正确目标的平均 loss（一个数）",
        "此处只前向；没有 backward()，也没有 optimizer.step()",
    ], TEAL, "white", ref("V4"))
    d.node(1350, 1635, 880, 114, "val_loss_accum += loss / 2", [
        "累计 2 批后：本卡均值 = (loss₁ + loss₂) / 2",
    ], TEAL)
    d.arrow([(1790, 1375), (1790, 1418)], TEAL)
    d.arrow([(1790, 1592), (1790, 1635)], TEAL)
    d.arrow([(2230, 1692), (2276, 1692), (2276, 1315), (2230, 1315)], TEAL)
    d.text(1324, 1783, "循环结束后退出 no_grad；验证只累计 loss 数值，不累计梯度。", 24, TEAL)

    d.diamond(1790, 1900, 335, 72, "runtime.distributed ?", "V5 · 是否多进程 / 多卡")
    d.text(1833, 2017, "是：所有 rank 都参与", 23, TEAL)
    d.node(1350, 2035, 880, 112, "all_reduce(SUM) → val_loss_accum /= R", [
        "汇总每卡的 loss 标量；得到所有验证位置的平均损失",
    ], TEAL, "white", ref("V5"))
    d.arrow([(2125, 1900), (2300, 1900), (2300, 2190), (1790, 2190)], MUTED)
    d.text(2174, 1872, "否：本次 R=1", 22, MUTED)
    d.node(1320, 2235, 936, 112, "检查 loss 有限，准备返回记录字典", [
        "global_mean_loss + 批次起点 + 预测位置数等",
    ], TEAL)
    d.node(1320, 2392, 936, 118, "V6  finally: model.train(was_training)", [
        "先恢复调用前的模式，再把记录交还给调用者",
    ], TEAL, TEAL_BG, ref("V6"))
    d.node(1320, 2555, 936, 137, "回到 record_validation() → 返回 A 或 B", [
        "补上 after_updates → val_records.append(result)",
        "rank 0 打印 → 回到原调用点；沿左侧蓝色实线继续",
    ], PURPLE, PURPLE_BG)
    d.node(1276, 2740, 1024, 150, "验证期间保持什么？", [
        "参数、已有 .grad、AdamW 状态和训练游标保持原样。",
        "try 内若异常也先恢复模式，再向外抛出；不继续正常训练。",
    ], INK, "#f5f7fa")
    for start, end in [(756, 805), (923, 968), (1096, 1144), (1800, 1828), (1972, 2035), (2147, 2235), (2347, 2392), (2510, 2555)]:
        d.arrow([(1790, start), (1790, end)], TEAL)

    # 用归档报告连接代码中的条件与真实执行顺序。
    d.text(64, 3005, "03  把这张图走一遍：本次 CPU 小模型的真实记录", 38, INK, 700)
    d.text(64, 3050, "steps=3 · validation_every=2 · validation_batches=2；以下是时间顺序，不是同时执行。", 27, MUTED)
    events = [
        ("验证 · 更新 0 次", val[0], "初始基线 A", TEAL, TEAL_BG),
        ("训练 · 更新 1 次", train[0], "1 % 2 ≠ 0，跳过 B", BLUE, BLUE_BG),
        ("训练 · 更新 2 次", train[1], "2 % 2 = 0，调用 B", BLUE, BLUE_BG),
        ("验证 · 更新 2 次", val[1], "返回后继续第 3 次训练", TEAL, TEAL_BG),
        ("训练 · 更新 3 次", train[2], "最后一次更新，调用 B", BLUE, BLUE_BG),
        ("验证 · 更新 3 次", val[2], "返回后保存报告，结束", TEAL, TEAL_BG),
    ]
    for i, (title, item, caption, color, fill) in enumerate(events):
        x = 64 + i * 380
        d.rect(x, 3090, 350, 176, fill, radius=17)
        d.text(x + 20, 3133, title, 28, color, 650)
        d.text(x + 20, 3185, f"loss {item['global_mean_loss']:.6f}", 28, INK, 600, mono=True)
        d.text(x + 20, 3235, caption, 23, MUTED)
        if i < 5:
            d.arrow([(x + 350, 3177), (x + 380, 3177)], MUTED)
    d.node(64, 3312, 1120, 166, "三个计数不能混在一起", [
        "steps=3：参数更新次数；K=8：每次更新前的训练微批次数。",
        "validation_batches=2：每次验证的批次数；不会增加更新次数。",
        "验证平均成立的条件：各批 / 各 rank 的有效预测位置数相同。",
    ], INK)
    d.node(1216, 3312, 1120, 166, "训练游标向前走，验证每次回到同一组位置", [
        "train 更新后游标：512 → 1024 → 1536；val 批次起点：[0, 64]。",
        "val 起点相对验证子集；本次验证 2 × 2 × 32 = 128 个位置。",
        "训练 loss 来自更新前；验证使用更新后参数，数据片段也不同。",
    ], INK)
    d.text(64, 3530, "代码：train_validation.py  ·  数值：artifacts/lesson13-cpu-report.json  ·  行号由绘图脚本从当前源码读取", 24, MUTED)
    d.text(64, 3571, "本图对应本地 Tiny Shakespeare 短跑；只保存指标 JSON，不保存 checkpoint；不代表完整 GPT-2 / FineWeb / CUDA 训练。", 24, MUTED)
    return d.finish()


def notes(a, train, val):
    rows = []
    for completed in range(4):
        if completed:
            t = train[completed - 1]
            rows.append(f"| 训练 | {completed} | {t['global_mean_loss']:.6f} | 参数更新一次 |")
        v = next((v for v in val if v["after_updates"] == completed), None)
        if v:
            reason = "初始基线 A" if not completed else ("间隔命中，调用 B" if completed == 2 else "最后一次更新，调用 B")
            rows.append(f"| 验证 | {completed} | {v['global_mean_loss']:.6f} | {reason} |")
    table = "\n".join(rows)
    return f'''# 验证循环：代码流转图

配套 [知识块 13](../../lesson13.md) 与 [train_validation.py](../../train_validation.py)。

[打开 PNG](gpt2-validation-loop-flow.png) · [打开可缩放 SVG](gpt2-validation-loop-flow.svg)

![验证循环代码流转](gpt2-validation-loop-flow.png)

## 怎么读

1. 顶部是 `main()` 的准备：切分数据、建立独立 loader、建立随机模型与 AdamW。
2. 左侧沿蓝色实线读：初始验证 A → 训练微批次累积 → 裁剪 / 学习率 / 参数更新 → 判断是否验证 B → 下一次更新或结束。
3. 紫色虚线表示函数调用：A 和 B 都调用右侧同一个函数。右侧完成后，返回各自的调用点，再沿左侧实线继续；两条泳道不是并行训练与验证。
4. 右侧 `evaluate_validation()`：eval → reset → no_grad 内取数、前向与 loss 累计 → 可选跨进程均值 → finally 恢复模式 → 返回。`record_validation()` 再加入标签、保存记录和打印。
5. 底部按时间排列归档的 CPU 实测记录，让循环条件对应到具体的第 0、2、3 次验证。

## 代码定位

| 图中节点 | 文件中的位置 | 输入 → 输出 |
|---|---|---|
| 主入口 M1—M3 | `train_validation.py`：L{a['main']}；标记 M1 / M2 / M3 | 本地文本、配置 → loader / model / optimizer → 记录 |
| 左侧主循环 | L{a['train']} `train_with_validation()` | 模型、优化器、两套 loader 等 → training / validation 记录 |
| A / B 共用包装函数 | L{a['record']} `record_validation()` | 已完成更新数 → 调用验证、补标签、保存并打印 |
| 验证函数 V1—V6 | L{a['eval']} `evaluate_validation()` | 当前模型、验证 loader、设备信息、批次数、精度 → 平均 loss 与取数信息 |
| 修改参数的位置 | L{a['update']} `optimizer.step()` | 参数、累计梯度、AdamW 历史状态 → 新参数与新状态 |

行号是生成本图时的定位；源码增删后可重新运行绘图脚本。代码中的 M / T / V 标记保持直接对应。

## 本次数字和判断

本图读取 [CPU 报告](../../artifacts/lesson13-cpu-report.json)，不把教学假设写成实测。配置为 `--device cpu --model-size tiny`，R=1、B=2、T=32、K=8，更新 3 次。

| 动作 | 已完成更新数 | loss | 为什么执行 |
|---|---:|---:|---|
{table}

训练 loss 来自该次更新之前的前向；同一更新号的验证 loss 使用该次更新之后的参数。两者的数据片段也不同，不能把相邻两个数视为同一批数据的更新前后损失。

一次训练更新先清梯度，再处理 K=8 个训练微批次；每批 `loss/K` 反向累计到 `.grad`，最后裁剪并更新一次。多进程时前 K−1 次使用 `no_sync()`，最后一次反向同步梯度。训练 loss 另有跨进程归约供日志使用，图中训练主干为突出更新/验证边界而合并展示。

一次验证从 `val_loader.reset()` 开始，读取 2 批。每批 x/y 都是 `[2,32]`，GPT 输出 `[2,32,50257]`；`forward_and_loss()` 在模型返回后计算交叉熵，得到本批 64 个预测位置的平均 loss。两批平均得到 128 个位置的均值。

多进程时各 rank 先算本地均值，再 `all_reduce(SUM)`、除以 R。**这次同步的是 loss 标量，不是参数或梯度。** 所有 rank 都必须参与，仅 rank 0 打印。本例每批、每 rank 的有效位置数相同，所以平均可以分两级计算；有效位置数不同时，应汇总损失之和与有效位置数后相除。

`finally` 会在返回记录交付调用者之前执行。图示正常路径；try 内若异常也先恢复调用前模式，然后向外抛出，不沿正常训练路径继续。

## 数据与状态边界

- Tiny Shakespeare 按 Token **位置**切分：train 为 `[0,304222)`，val 为 `[304222,338025)`；这不是文档划分或文本去重。两边当然可能包含相同 Token ID。
- 每次验证取的相对子集位置固定为 `[0,64]`；训练游标在三次更新后分别为 512、1024、1536。
- 验证使用同一模型的当前参数；没有复制模型，没有 `backward()` 或 `optimizer.step()`，不清除已有 `.grad`，也不推进训练游标或学习率的更新计数。
- `eval()` 改变模块行为；`no_grad()` 关闭该作用域内的反向图记录；两者职责不同。本例训练 Dropout 配置为 0，仍保留模式切换。
- 本代码把验证结果写入指标报告，没有自动早停、最佳 checkpoint 选择或根据 val_loss 调整学习率，也没有保存模型 checkpoint。
- 本图对应本地教学改编。原课程依据、FineWeb 与本地 Tiny Shakespeare 的差异见 [lesson13.md](../../lesson13.md)，CPU 结果不代表 CUDA 性能或完整预训练。

## 重新生成

使用安装好课程依赖的 Python；绘图脚本只读取本地源码和归档报告，不执行训练。SVG 不需要额外绘图库；PNG 导出复用项目现有 macOS AppKit 方法。

```bash
python docs/diagrams/render_validation_loop_flow.py --png
```

脚本会核对来源行号、实验配置、验证触发次数和游标，再输出 SVG、说明 Markdown，以及可选 PNG。
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", action="store_true", help="在 macOS 导出同尺寸 PNG")
    args = parser.parse_args()
    anchors, report, train, val = evidence()
    svg = draw(anchors, report, train, val)
    ET.fromstring(svg)
    output = OUT / "gpt2-validation-loop-flow.svg"
    output.write_text(svg, encoding="utf-8")
    output.with_suffix(".md").write_text(notes(anchors, train, val), encoding="utf-8")
    if args.png:
        export_png(output)
    print("Source anchors, CPU values, validation schedule and data cursors checked.")
    print(output)
