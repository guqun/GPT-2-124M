"""知识块 14 的代码流转图：读取源码行号和已归档报告，不执行训练。

新增配图，保留之前的生成图、验证图和训练代码。
SVG 使用项目已有的绘图组件；--png 复用 macOS AppKit 导出器。
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
WIDTH, HEIGHT = 2400, 3860
PURPLE, PURPLE_BG = '#7461a8', '#f2eef9'
SOURCE = ROOT / 'train_generation.py'
REPORT = ROOT / 'artifacts/lesson14-cpu-report.json'


def evidence():
    lines = SOURCE.read_text(encoding='utf-8').splitlines()
    anchors = {}
    for n, line in enumerate(lines, 1):
        match = re.match(r'\s*# \[([MGT]\d+)\]', line)
        if match:
            assert match[1] not in anchors
            anchors[match[1]] = n
    for fragment, key in [
        ('def main(', 'main'), ('def train_with_generation(', 'train'),
        ('def sample_current_model(', 'sample'), ('def record_generation(', 'record'),
        ('def record_validation(', 'validation'), ('optimizer.step()', 'update'),
    ]:
        hits = [n for n, line in enumerate(lines, 1)
                if fragment in line and not line.lstrip().startswith('#')]
        assert len(hits) == 1, (key, hits)
        anchors[key] = hits[0]
    val_lines = (ROOT / 'train_validation.py').read_text().splitlines()
    anchors['eval'] = next(n for n, s in enumerate(val_lines, 1)
                           if s.startswith('def evaluate_validation('))
    assert set(f'G{i}' for i in range(1, 7)) <= anchors.keys()
    r = json.loads(REPORT.read_text(encoding='utf-8'))
    assert (r['device_type'], r['model_size'], r['world_size']) == ('cpu', 'tiny', 1)
    assert (r['batch_size_per_rank'], r['seq_len'], r['accumulation_steps']) == (2, 32, 8)
    assert r['tokens_per_optimizer_update'] == 512
    assert r['parameter_updates'] == 3 and r['validation_every'] == 2
    assert r['pretrained_weights_loaded'] is False and r['checkpoint_saved'] is False
    g = r['generation']
    assert (g['num_sequences'], g['max_new_tokens'], g['top_k'], g['base_seed']) == (4, 24, 50, 42)
    assert g['every_updates'] == 2 and g['precision'] == 'fp32'
    assert g['stops_on_eos'] is False and g['generated_text_used_for_training'] is False
    assert len(g['prompt_ids']) == 8
    assert g['prompt'] == "Hello, I'm a language model,"
    train, val, samples = [r['ranks'][0][k] for k in ('training', 'validation', 'generation')]
    assert [row['step'] for row in train] == [1, 2, 3]
    assert [row['after_updates'] for row in val] == [0, 2, 3]
    assert [row['after_updates'] for row in samples] == [2, 3]
    assert [row['next_start'] for row in train] == [512, 1024, 1536]
    for sample in samples:
        assert sample['sample_seed'] == 42 and sample['new_tokens_per_sequence'] == 24
        assert sample['stop_reason'] == 'max_new_tokens'
        assert len(sample['token_ids']) == len(sample['texts']) == 4
        assert [row['input_length'] for row in sample['steps']] == list(range(8, 32))
        for ids in sample['token_ids']:
            assert len(ids) == 32 and ids[:8] == g['prompt_ids']
        for step, row in enumerate(sample['steps']):
            assert row['logits_shape'] == [4, 8 + step, 50257]
            assert row['next_token_ids'] == [ids[8 + step] for ids in sample['token_ids']]
    return anchors, r, train, val, samples


class Drawing(BaseDrawing):
    def node(self, x, y, w, h, title, lines=(), color=INK, fill='white', ref=None,
             title_size=29, body_size=25):
        self.rect(x, y, w, h, fill, radius=17)
        self.text(x + 24, y + 41, title, title_size, color, 650)
        for i, line in enumerate(lines):
            self.text(x + 24, y + 82 + 35 * i, line, body_size, MUTED)
        if ref:
            self.text(x + w - 4, y - 12, ref, 20, color, anchor='end')

    def diamond(self, x, y, rx, ry, title, sub, color=AMBER, fill=AMBER_BG):
        self.fg.append(f'<polygon points="{x},{y-ry} {x+rx},{y} {x},{y+ry} {x-rx},{y}" '
                       f'fill="{fill}" stroke="{color}" stroke-width="1.5"/>')
        self.text(x, y - 6, title, 28, INK, 650, 'middle')
        self.text(x, y + 32, sub, 23, MUTED, anchor='middle')

    def finish(self):
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<title>训练中定期生成文本：主循环、生成内部与返回训练</title>
<desc>对应 train_generation.py 的 M1—M3、T1—T4 和 G1—G6。左侧训练主循环只在 AdamW 更新参数后判断是否验证与生成。右侧是同步函数调用的展开，使用当前参数与固定提示词，建立独立随机数生成器，eval 和 no_grad 下逐 Token 生成，恢复模式后返回原训练循环。底部用 CPU tiny 实测更新 0、1、2、3 的记录说明触发规则，并显示更新 2 次时第一轮生成的四个真实 ID。这里只生成观察文本，不用生成文本训练。</desc>
<style>text {{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}}</style>
<rect width="{WIDTH}" height="{HEIGHT}" fill="#f7f9fc"/>
{''.join(self.bg)}{''.join(self.edges)}{''.join(self.fg)}
</svg>'''


def draw(a, report, train, val, samples):
    d = Drawing()
    ref = lambda key: f'{key} · L{a[key]}' if re.match(r'[MGT]\d', key) else f'L{a[key]}'
    d.text(64, 84, '训练中生成：从参数更新到续写，再返回训练', 56, weight=700)
    d.text(64, 137, '知识块 14  /  train_generation.py  /  目的：观察同一提示词的续写怎样随训练变化', 29, MUTED)
    for x, label, color in [(64, '训练', BLUE), (250, '验证', TEAL), (436, '生成 / 调用与返回', PURPLE), (810, '参数更新', AMBER)]:
        d.rect(x, 162, 16, 16, color, stroke='none', radius=4)
        d.text(x + 27, 178, label, 23, MUTED)
    d.text(2336, 178, '本图：CPU · tiny · FP32 · 同步执行', 24, MUTED, anchor='end')

    d.node(64, 220, 736, 210, 'M1  准备训练 / 验证数据与提示词', [
        'Tiny Shakespeare → Token ID → 90% / 10%',
        'train_loader 与 val_loader 各有独立游标',
        '固定提示词另外编码为 8 个 ID，不从 loader 取',
    ], BLUE, BLUE_BG, ref('M1'), body_size=24)
    d.node(832, 220, 736, 210, 'M2  随机建立模型与 AdamW', [
        'GPT → device → 可选 DDP；只初始化一次',
        '训练 / 验证 / 生成使用同一份当前参数',
        '本图 tiny：2 个 Block；没有加载预训练权重',
    ], BLUE, 'white', ref('M2'), body_size=24)
    d.node(1600, 220, 736, 210, 'M3  train_with_generation(...)', [
        '传入模型、优化器、两套 loader 和提示词',
        '训练 3 次更新；验证 / 生成间隔均为 2',
        '返回 training / validation / generation 记录',
    ], BLUE, 'white', ref('M3'), body_size=24)
    d.arrow([(800, 325), (832, 325)], BLUE)
    d.arrow([(1568, 325), (1600, 325)], BLUE)
    d.arrow([(1968, 430), (1968, 460), (580, 460), (580, 490)], BLUE)

    d.rect(64, 490, 1028, 2620, 'white', background=True, radius=24)
    d.rect(1240, 490, 1096, 2620, '#fdfcfe', background=True, radius=24)
    d.text(100, 544, '01  训练主循环', 36, BLUE, 700)
    d.text(100, 585, f'train_with_generation() · L{a["train"]}；m = 已完成更新次数', 24, MUTED)
    d.text(1276, 544, '02  展开一次生成调用', 36, PURPLE, 700)
    d.text(1276, 585, f'sample_current_model() · L{a["sample"]}；右侧结束后才返回左侧', 24, MUTED)

    # 左侧：只有 T2 内部更新参数；验证与生成各自有触发判断。
    d.node(160, 630, 840, 130, 'T1  初始验证：record_validation(0)', [
        'model.train() → 先验证随机模型 → 恢复训练模式',
        'm=0 时跳过生成；接着进入 for step in range(3)',
    ], TEAL, TEAL_BG, ref('T1'))
    d.node(160, 816, 840, 100, 'T2  开始一次更新：optimizer.zero_grad(...)', [
        '清空已有梯度；准备处理本次的 K=8 个训练微批次',
    ], BLUE, BLUE_BG, ref('T2'), title_size=27)
    d.rect(160, 966, 840, 310, BLUE_BG, radius=17, background=True)
    d.text(184, 1010, 'for micro_step in range(K)  ·  以下重复 8 次', 28, BLUE, 650)
    d.node(190, 1035, 760, 58, '① 取真实 x/y：[2,32] → 前向与交叉熵', [], BLUE, title_size=27)
    d.node(190, 1134, 760, 58, '② (loss / 8).backward() → 累加到 .grad', [], BLUE, title_size=27)
    d.arrow([(570, 1093), (570, 1134)], BLUE)
    d.arrow([(950, 1163), (972, 1163), (972, 1064), (950, 1064)], BLUE)
    d.text(184, 1232, '多卡时前 K−1 次 no_sync，最后一次反向同步梯度。', 24, MUTED)
    d.text(184, 1262, '完成 8 批再更新一次；生成循环不在这个累积循环里。', 24, MUTED)
    d.node(160, 1320, 840, 150, '裁剪 → 设置 lr(step) → optimizer.step()', [
        'AdamW 修改参数；本次覆盖 8 × 2 × 32 = 512 个位置',
        '记录训练指标；m = step + 1，此后使用更新后的参数',
    ], AMBER, AMBER_BG, f'参数更新点 · {ref("update")}', title_size=28, body_size=24)
    d.diamond(580, 1590, 370, 78, '到验证间隔，或是最终更新？', 'm % 2 == 0  或  m == 3')
    d.text(228, 1700, f'是 · {ref("T3")}', 23, TEAL)
    d.node(160, 1720, 840, 170, 'record_validation(m)  ·  复用上一节验证函数', [
        'eval → reset 验证游标 → no_grad 前向与 loss 均值',
        '恢复原模式 → 保存验证记录 → 返回主循环',
        '调用 train_validation.py 的 evaluate_validation()',
    ], TEAL, TEAL_BG, f'evaluate_validation · L{a["eval"]}', title_size=26, body_size=24)
    d.diamond(580, 2040, 370, 78, '到生成间隔，或是最终更新？', 'm % 2 == 0  或  m == 3')
    d.text(626, 2150, f'是 · {ref("T4")}', 23, PURPLE)
    d.node(160, 2180, 840, 115, '调用 record_generation(m)  →  右侧展开', [
        '传入当前参数 θₘ 与固定提示词；等待本次生成完成',
    ], PURPLE, PURPLE_BG, ref('record'), title_size=28)
    d.node(160, 2360, 840, 125, '生成返回：已经恢复训练模式', [
        'generation 记录已有文字与 ID；参数仍然是 θₘ',
        '训练游标、已有梯度、AdamW 状态保持原样',
    ], PURPLE, PURPLE_BG, title_size=28)
    d.diamond(580, 2630, 320, 78, '还有下一次参数更新？', 'for step 循环是否继续')
    d.text(628, 2740, '否', 24, MUTED)
    d.node(160, 2760, 840, 145, '返回三组记录 → main() 保存报告', [
        'gather_records() 汇总各 rank 的指标与文字',
        'rank 0 写 JSON → 清理进程组 → 结束；不保存权重',
    ], BLUE, BLUE_BG, title_size=28, body_size=24)
    d.node(100, 2950, 956, 120, '下一次训练从哪里接着执行？', [
        '重新取训练集的下一批真实 x/y。生成文字只供观察，',
        '不会成为下一批输入或标签，也不推进学习率的更新计数。',
    ], INK, '#f5f7fa', title_size=28, body_size=24)
    for start, end in [(760, 816), (916, 966), (1276, 1320), (1470, 1512),
                       (1668, 1720), (1890, 1962), (2118, 2180), (2485, 2552), (2708, 2760)]:
        d.arrow([(580, start), (580, end)], BLUE)
    d.arrow([(950, 1590), (1040, 1590), (1040, 1930), (580, 1930)], MUTED)
    d.text(1000, 1565, '否', 23, MUTED)
    # 未触发生成，跳过调用及返回框，直接进入下一次更新判断。
    d.arrow([(950, 2040), (1060, 2040), (1060, 2520), (580, 2520)], MUTED)
    d.text(1010, 2015, '否', 23, MUTED)
    d.arrow([(260, 2630), (110, 2630), (110, 866), (160, 866)], BLUE)
    d.text(137, 2606, '是', 24, BLUE)
    d.text(137, 2680, '下一次更新', 23, BLUE)

    # 紫色调用与返回经过两个独立通道；不是两条并行任务。
    d.node(1320, 630, 936, 130, 'record_generation(m) → sample_current_model(...)', [
        'raw_model：训练模型的本体；使用同一份当前参数 θₘ',
        'prompt_ids：同一提示词；每次观察都从头开始续写',
    ], PURPLE, PURPLE_BG, ref('record'), title_size=27, body_size=24)
    d.arrow([(1000, 2236), (1140, 2236), (1140, 694), (1320, 694)], PURPLE, dashed=True)
    d.text(1166, 669, '调用', 23, PURPLE)
    d.node(1320, 816, 936, 180, 'G1  独立采样 RNG ＋ 复制提示词', [
        'Generator(device).manual_seed(42 + rank)',
        '“Hello, I\'m a language model,” → 8 个 ID',
        'repeat(4,1)：[8] → [4,8]；4 是生成条数，训练 B=2',
    ], PURPLE, 'white', ref('G1'), body_size=24)
    d.node(1320, 1050, 936, 130, 'G2  保存原模式，再 raw_model.eval()', [
        'was_training = raw_model.training；关闭 Dropout',
        '即将进入 no_grad：只前向，不记录本次反向计算图',
    ], PURPLE, PURPLE_BG, ref('G2'), title_size=28, body_size=24)

    d.rect(1304, 1230, 1008, 1030, '#f4f1fa', radius=20, background=True)
    d.text(1330, 1276, 'with torch.no_grad():', 29, PURPLE, 650)
    d.diamond(1780, 1380, 330, 74, '当前长度 T < 目标长度 32？', 'while ids.size(1) < output_length', PURPLE, 'white')
    d.text(1385, 1494, '是：每条序列再生成一个 Token', 24, PURPLE)
    d.node(1370, 1520, 820, 170, 'G3  完整前向 → 取最后位置的 logits', [
        'raw_model(ids)：[4,T] → [4,T,50257]',
        'logits[:, -1, :] → [4,50257]',
        '目的：用当前全部前缀，为下一个 Token 打分',
    ], PURPLE, 'white', ref('G3'), title_size=27, body_size=24)
    d.node(1370, 1740, 820, 225, 'G4  把分数变成每条序列的一个新 ID', [
        'softmax → probs：[4,50257]',
        'topk(50) → top_probs / top_ids：[4,50]',
        'multinomial(..., generator=sample_rng) → [4,1]',
        'gather(top_ids, ...) → next_ids：[4,1]',
    ], PURPLE, 'white', ref('G4'), title_size=27, body_size=24)
    d.node(1370, 2020, 820, 150, 'G5  ids = cat((ids, next_ids), dim=1)', [
        '[4,T] + [4,1] → [4,T+1]；记录本轮选出的 ID',
        '回到 while 判断：输入变长，参数 θₘ 保持不变',
    ], PURPLE, 'white', ref('G5'), title_size=27, body_size=24)
    d.arrow([(1780, 1454), (1780, 1520)], PURPLE)
    d.arrow([(1780, 1690), (1780, 1740)], PURPLE)
    d.arrow([(1780, 1965), (1780, 2020)], PURPLE)
    d.arrow([(1780, 2170), (1780, 2214), (2270, 2214), (2270, 1380), (2110, 1380)], PURPLE)
    d.text(2158, 1350, '下一轮', 23, PURPLE)
    # while 不满足时先离开 no_grad，再执行 finally，最后返回记录。
    d.arrow([(1450, 1380), (1280, 1380), (1280, 2420), (1320, 2420)], PURPLE)
    d.text(1300, 1350, '否', 24, PURPLE)
    d.node(1320, 2350, 936, 150, 'G6  finally：恢复原模式，返回 ID 与过程记录', [
        'raw_model.train(was_training)；本图恢复为 train 模式',
        '返回完整 ID：[4,32]；24 轮记录；stop_reason',
    ], PURPLE, PURPLE_BG, ref('G6'), title_size=27, body_size=24)
    d.node(1320, 2555, 936, 150, '回到 record_generation(m)，然后返回左侧', [
        'tokenizer.decode → 4 条文字；补上 after_updates=m',
        'sample_records.append(result)；rank 0 打印样本',
    ], PURPLE, PURPLE_BG, ref('record'), title_size=27, body_size=24)
    d.arrow([(1788, 760), (1788, 816)], PURPLE)
    d.arrow([(1788, 996), (1788, 1050)], PURPLE)
    d.arrow([(1788, 1180), (1788, 1230)], PURPLE)
    d.arrow([(1780, 2500), (1780, 2555)], PURPLE)
    d.arrow([(1320, 2630), (1190, 2630), (1190, 2420), (1000, 2420)], PURPLE, dashed=True)
    d.text(1150, 2666, '返回', 23, PURPLE)
    d.node(1276, 2770, 1024, 300, '一次生成期间，哪些东西保持不变？', [
        '参数、已有 .grad、AdamW 状态与训练数据游标保持不变。',
        '只有生成用的 ids 在增长，独立 sample_rng 的状态在推进。',
        '这里没有 y、交叉熵、backward() 或 optimizer.step()。',
        '固定长度生成：不因 EOS 提前停；总长度受位置表限制。',
        '多卡各自生成，不 All-Reduce；本图数值来自单进程 CPU。',
        'try 内若报错：先恢复模式，再向外抛出，不继续正常训练。',
    ], INK, '#f5f7fa', title_size=28, body_size=24)

    # 真实记录将条件判断落到每一轮；不把手算假设写成模型输出。
    d.text(64, 3180, '03  把规则落到实际运行：什么时候生成？', 36, INK, 700)
    d.text(64, 3224, 'steps=3；val_every=2；sample_every=2。验证与生成先后执行，均使用本次更新后的参数。', 26, MUTED)
    positions = [64, 640, 1216, 1792]
    values = [
        ('更新 0 次', TEAL, TEAL_BG, [f'验证 loss  {val[0]["global_mean_loss"]:.6f}', '初始基线：只验证', '跳过随机初值生成']),
        ('更新 1 次', BLUE, BLUE_BG, [f'训练 loss  {train[0]["global_mean_loss"]:.6f}', '1 % 2 ≠ 0，且不是最终更新', '验证 / 生成均跳过']),
        ('更新 2 次', PURPLE, PURPLE_BG, [f'验证 loss  {val[1]["global_mean_loss"]:.6f}', '2 % 2 = 0：验证 → 生成', '用 θ₂ 生成 4 条 × 24 个新 Token']),
        ('更新 3 次', PURPLE, PURPLE_BG, [f'验证 loss  {val[2]["global_mean_loss"]:.6f}', '最终更新：验证 → 生成', '用 θ₃ 从相同提示词重新生成']),
    ]
    for x, (title, color, fill, rows) in zip(positions, values):
        d.node(x, 3270, 544, 205, title, rows, color, fill, title_size=29, body_size=23)
    for x in positions[:-1]:
        d.arrow([(x + 544, 3372), (x + 576, 3372)], MUTED)
    d.text(64, 3520, '对照两种循环：每次参数更新要处理 8 个训练微批次；一次生成要进行 24 轮前向，每轮每条只选 1 个 ID。', 26, MUTED)
    d.node(64, 3560, 2272, 205, '实际展开一个生成轮次：更新 2 次后的第 1 轮', [
        '[4,8] → 模型前向 [4,8,50257] → 最后位置 [4,50257] → top-k 与抽样 → 4 个新 ID',
        f'next_ids = {samples[0]["steps"][0]["next_token_ids"]}ᵀ，形状 [4,1]；分别追加到各自序列 → [4,9]',
        '随后以 [4,9] 再前向，直到 [4,32]。这 24 轮始终使用同一份 θ₂；到下一次训练更新才改变参数。',
    ], PURPLE, 'white', title_size=30, body_size=27)
    d.text(64, 3810, '源码：train_generation.py；数据：artifacts/lesson14-cpu-report.json；行号与数值由本绘图脚本自动核对。', 23, MUTED)
    d.text(64, 3844, '本图对应本地 Tiny Shakespeare / CPU tiny 短跑；只保存指标与文字，不保存模型权重。完整说明见同名 .md。', 23, MUTED)
    return d.finish()


def notes(a, report, train, val, samples):
    first = samples[0]['steps'][0]['next_token_ids']
    return f'''# 训练中定期生成文本：代码流转图

配套 [知识块 14](../../lesson14.md) 与 [train_generation.py](../../train_generation.py)。

[打开 PNG](gpt2-periodic-generation-flow.png) · [打开可缩放 SVG](gpt2-periodic-generation-flow.svg)

![训练中定期生成的完整代码流转](gpt2-periodic-generation-flow.png)

## 怎么读

1. 顶部是 `main()`：准备数据与固定提示词、随机初始化模型、建立 AdamW、进入训练函数。
2. 左侧沿蓝色箭头：初始验证 → 训练微批次累积 → 参数更新 → 判断验证 → 判断生成 → 下一次更新或结束。
3. 紫色虚线表示同步函数调用与返回。左侧调用 `record_generation(m)` 后，等待右侧 G1—G6 完成；右侧不是另一个并行任务。
4. 右侧把生成展开：独立 RNG、重复提示词、eval/no_grad、完整前向、最后位置分数、top-k 抽样、追加 ID、回到长度判断、恢复模式、返回与解码。
5. 底部用 [CPU tiny 实测报告](../../artifacts/lesson14-cpu-report.json) 对照判断条件，再展开一个真实的生成轮次。

## 代码定位

| 图中节点 | 源码位置 | 输入 → 输出 |
|---|---|---|
| 主入口 | `train_generation.py` L{a['main']}；M1 / M2 / M3 | 文本、配置、提示词 → loader / 模型 / 优化器 → 实验记录 |
| 主循环 | L{a['train']} `train_with_generation()` | 当前模型、优化器、数据与观察设置 → 三组记录 |
| 参数更新点 | L{a['update']} `optimizer.step()` | 参数、梯度、AdamW 状态 → 新参数与状态 |
| 验证触发 | T3 · L{a['T3']} | 已完成更新数 m → 必要时调用验证 |
| 验证函数 | `train_validation.py` L{a['eval']} | 当前参数与真实验证 x/y → 平均 loss；返回时恢复模式 |
| 生成触发 | T4 · L{a['T4']} | 已完成更新数 m → 必要时调用生成 |
| 包装与记录 | L{a['record']} `record_generation()` | m → 调用生成、解码、加标签、保存与打印 |
| 生成内部 | L{a['sample']} `sample_current_model()`；G1—G6 | 当前参数、prompt IDs 与采样设置 → 完整 ID / 过程 / 停止原因 |

行号从源码提取；重建图时会重新定位。图中的 G / T / M 标记与源码注释直接对应。

## 两类循环与两次观察

本图使用默认短跑配置的 CPU tiny 记录：B=2、训练 T=32、K=8、R=1，共 3 次参数更新。一次更新前累积 8 个训练微批次，共 512 个预测位置。

生成的 S=4 是每次生成的序列条数，独立于训练 B=2；生成的 T 从 8 增长到 32，并非一直等于训练 T。每次每条新增 24 个 Token，每轮四条各新增一个，因此执行 24 轮模型前向。

| 已更新次数 m | 条件如何判断 | 实际动作 |
|---|---|---|
| 0 | 训练循环之前的特殊基线 | 验证 loss={val[0]['global_mean_loss']:.6f}；不生成 |
| 1 | `1 % 2 != 0` 且 `1 != 3` | 已完成训练更新；跳过验证和生成 |
| 2 | `2 % 2 == 0` | 验证 loss={val[1]['global_mean_loss']:.6f} → 用 θ₂ 生成 |
| 3 | `3 == steps`，满足最终更新条件 | 验证 loss={val[2]['global_mean_loss']:.6f} → 用 θ₃ 重新生成 → 结束 |

两个观察间隔在代码中独立配置；本次恰好都为 2，不代表生成必须由验证结果触发。条件中的 `or` 保证最后一次更新也执行观察。

更新 2 次后的第一轮：输入 `[4,8]`，输出 logits `[4,8,50257]`，取最后位置 `[4,50257]`，实际选出四个新 ID `{first}`，分别追加后得到 `[4,9]`。报告后续 23 轮的输入长度为 9—31，最终输出 `[4,32]`；不会以 `[4,32]` 再做一轮前向。

到更新 3 次后的观察，又从原来的 8 个提示词 ID 开始；不会接着上次的 32 个 ID 继续写。两次观察使用同一个提示词和同一 rank 的种子，但参数已由 θ₂ 变为 θ₃，因此输出可以不同。

## 调用与状态边界

- `raw_model` 是训练中的模型本体；不复制、不重新加载、不重新初始化。每次重新启动脚本会新建随机模型，上一节指标 JSON 不能用于恢复参数。
- `eval()` 切换 Dropout 等行为；`no_grad()` 关闭本次反向图记录。参数、已有 `.grad`、AdamW 状态与训练游标不变；生成的局部 `ids` 和独立 RNG 状态会变。
- `sample_rng = torch.Generator(device=device).manual_seed(42 + rank)` 每次观察重新创建；生成抽样不消耗训练的全局 RNG。相同种子不保证跨设备或版本输出一致。
- G4 中 `multinomial` 返回 top-k 列表内部的下标，`gather` 转换为原词表 ID。截取的 top-k 概率无需再次手动归一化，`multinomial` 按非负权重相对比例抽样。
- 默认目标长度为 `min(8+24, n_positions)`，本次是 32。本例沿用原课固定长度，不因 EOS 提前停止；没有 KV cache，每轮重新计算完整前缀。
- G6 的 `finally` 会在返回结果交给调用者前恢复模式；若 try 内出现异常，恢复后向外抛出，不沿正常路径继续训练。
- 返回 ID 后，包装函数解码、保存 `after_updates` 和样本文字；各 rank 的文字最后随记录汇总，只有 rank 0 打印和写总报告。生成本身无梯度通信，各 rank 使用自己的 raw_model。
- 生成结果只观察和保存，没有交叉熵、反传、参数更新，也没有把生成文字放回训练数据。训练仍取下一批真实 x/y；学习率仍按参数更新次数调度。
- 图中验证 loss 使用更新后的参数；训练日志 loss 来自该次更新前的前向。两者的数据不同，不能把相邻数值解释为同一批数据的前后损失。

## 来源与重建

本图依据当前源码和既有报告，只画本地教学配套的正常执行路径；课程原段落与历史源码、Tiny Shakespeare/FineWeb-Edu、eager/compile 等改编区别见 [lesson14.md](../../lesson14.md)。没有重新训练，也不把 CPU tiny 结果当作完整预训练或 CUDA/NCCL 实测。

```bash
python docs/diagrams/render_periodic_generation_flow.py --png
```

绘图只用 Python 标准库和项目已有 SVG 组件；PNG 导出复用 macOS AppKit。先核对源码标记、实际触发次数、生成长度和选出的 ID，再输出 SVG / Markdown / PNG。旧生成图、旧验证图及训练代码保留原样。
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--png', action='store_true', help='在 macOS 导出同尺寸 PNG')
    args = parser.parse_args()
    anchors, report, train, val, samples = evidence()
    svg = draw(anchors, report, train, val, samples)
    ET.fromstring(svg)
    path = OUT / 'gpt2-periodic-generation-flow.svg'
    path.write_text(svg, encoding='utf-8')
    path.with_suffix('.md').write_text(notes(anchors, report, train, val, samples), encoding='utf-8')
    if args.png:
        export_png(path)
    print('Source lines, update conditions, token shapes and sampled IDs checked.')
    print(path)
