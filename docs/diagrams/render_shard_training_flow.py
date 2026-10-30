"""分片训练代码流转图：从源码定位节点，从归档报告提取实际读取顺序。

不执行训练、不修改训练入口。使用项目已有 SVG 组件与 macOS PNG 导出器。
重建：python docs/diagrams/render_shard_training_flow.py --png
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from render_ddp_allreduce_flow import (
    Drawing as BaseDrawing, export_png, INK, MUTED, LINE,
    BLUE, BLUE_BG, TEAL, TEAL_BG, AMBER, AMBER_BG,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 2560, 4850
PURPLE, PURPLE_BG = '#7461a8', '#f2eef9'
REPORT = ROOT / 'artifacts/lesson12-shard-training-cpu.json'


def evidence():
    specs = {
        'train_shards.py': {
            'main': 'def main(', 'loaders': 'def make_shard_loaders(',
            'train': 'def train_from_shards(', 'eval': 'def evaluate_shards(',
            'M1': '# [M1]', 'M2': '# [M2]', 'M3': '# [M3]', 'M4': '# [M4]',
            'T1': '# [T1]', 'T2': '# [T2]', 'T3': '# [T3]', 'T4': '# [T4]',
            'T5': '# [T5]', 'T6': '# [T6]',
            'zero': 'optimizer.zero_grad(', 'outer': 'for step in range(steps):',
            'micro': 'for micro_step in range(accum_steps):',
            'backward': 'scaled_loss.backward()', 'update': 'optimizer.step()',
        },
        'load_shards.py': {
            'load': 'def load_tokens(', 'switch': 'def _load_usable_shard(',
            'batch': 'def next_batch(', 'S4': '# [S4]', 'S5': '# [S5]',
            'S6': '# [S6]', 'S7': '# [S7]',
        },
        'train_generation.py': {'sample': 'def sample_current_model(',
                                **{f'G{i}': f'# [G{i}]' for i in range(1, 7)}},
    }
    anchors, hashes = {}, {}
    for filename, fragments in specs.items():
        source = (ROOT / filename).read_text(encoding='utf-8')
        ast.parse(source)
        hashes[filename] = hashlib.sha256(source.encode()).hexdigest()
        for key, fragment in fragments.items():
            hits = [n for n, line in enumerate(source.splitlines(), 1)
                    if line.strip().startswith(fragment)]
            assert len(hits) == 1, (filename, key, hits)
            anchors[key] = {'file': filename, 'line': hits[0]}
    r = json.loads(REPORT.read_text(encoding='utf-8'))
    assert (r['device_type'], r['model_size'], r['world_size']) == ('cpu', 'tiny', 1)
    assert (r['batch_size_per_rank'], r['seq_len'], r['accumulation_steps']) == (2, 3, 2)
    assert r['tokens_per_optimizer_update'] == 12 and r['parameter_updates'] == 3
    assert r['training_positions'] == 36 and r['precision'] == 'fp32'
    assert not r['pretrained_weights_loaded'] and not r['checkpoint_saved']
    assert r['validation_every'] == r['generation']['every_updates'] == 2
    assert r['validation_batches_per_rank'] == 2
    train, val, gen = [r['ranks'][0][key] for key in ('training', 'validation', 'generation')]
    assert [x['step'] for x in train] == [1, 2, 3]
    assert [x['after_updates'] for x in val] == [0, 2, 3]
    assert [x['after_updates'] for x in gen] == [2, 3]

    # 按源码的全局边界规则复算文件/位置；数据来自既有报告，不重新训练。
    files = r['inventory']['train']['files']
    assert [f['tokens'] for f in files] == [16, 8]
    index, pos, stride = 0, 0, 6
    for row in train:
        expected = []
        for _ in range(2):
            expected.append({'file': files[index]['file'], 'start': pos})
            assert pos + stride + 1 <= files[index]['tokens']
            pos += stride
            if pos + stride + 1 > files[index]['tokens']:
                index, pos = (index + 1) % len(files), 0
        assert row['local_batches'] == expected
        assert row['next_train_cursor'] == {'file': files[index]['file'], 'start': pos}
    for row in val:
        assert row['train_cursor_before'] == row['train_cursor_after']
        assert row['global_predicted_positions'] == 12
        assert [b['start'] for b in row['local_batches']] == [0, 6]
    for row in gen:
        assert row['cursors_before'] == row['cursors_after']
        assert row['new_tokens_per_sequence'] == 24
        assert len(row['token_ids']) == 4
        assert [s['input_length'] for s in row['steps']] == list(range(8, 32))
        assert all(len(ids) == 32 for ids in row['token_ids'])
    assert len(r['generation']['prompt_ids']) == 8
    assert r['generation']['top_k'] == 50
    doc_report = json.loads((ROOT / 'artifacts/lesson12-data-preparation-demo.json').read_text())
    train_ids = doc_report['documents'][0]['first_ids']
    assert train_ids == [50256, 15496, 995, 13]
    buf = (train_ids * 2)[:7]
    x, y = [buf[:3], buf[3:6]], [buf[1:4], buf[4:7]]
    return anchors, hashes, r, train, val, gen, x, y


class Drawing(BaseDrawing):
    def node(self, x, y, w, h, title, lines=(), color=INK, fill='white', ref='',
             title_size=30, body_size=26):
        self.rect(x, y, w, h, fill, radius=18)
        self.text(x + 24, y + 43, title, title_size, color, 650)
        for i, line in enumerate(lines):
            self.text(x + 24, y + 86 + 37 * i, line, body_size, MUTED)
        if ref:
            self.text(x + w - 4, y - 13, ref, 22, color, anchor='end')

    def diamond(self, x, y, rx, ry, title, sub, color=AMBER, fill=AMBER_BG):
        self.fg.append(f'<polygon points="{x},{y-ry} {x+rx},{y} {x},{y+ry} {x-rx},{y}" '
                       f'fill="{fill}" stroke="{color}" stroke-width="1.5"/>')
        self.text(x, y - 7, title, 29, INK, 650, 'middle')
        self.text(x, y + 31, sub, 25, MUTED, anchor='middle')

    def panel(self, x, y, w, h, title, sub, color, fill='white'):
        self.rect(x, y, w, h, fill, background=True, radius=24)
        self.text(x + 36, y + 54, title, 37, color, 700)
        self.text(x + 36, y + 98, sub, 24, MUTED)

    def finish(self):
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<title>分片读取、微批次累积、参数更新、验证与生成：代码全流程</title>
<desc>左侧 train_from_shards 主循环，右侧展开 next_batch、evaluate_shards 和 sample_current_model。每两个微批次累积一次梯度并由 AdamW 更新一次参数；验证和生成同步返回主循环。底部记录第 2 次更新的两批分别来自不同分片，并保留 CPU tiny 实跑的文件位置和验证损失。</desc>
<style>text {{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}}</style>
<rect width="{WIDTH}" height="{HEIGHT}" fill="#f7f9fc"/>
{''.join(self.bg)}{''.join(self.edges)}{''.join(self.fg)}
</svg>'''


def draw(a, r, train, val, gen, x, y):
    d = Drawing()
    ref = lambda key: f'{key} · L{a[key]["line"]}'
    d.text(64, 86, '分片如何进入训练，又如何回到下一次更新', 57, weight=700)
    d.text(64, 140, '2.4 工程串联  /  train_shards.py  /  沿真实代码读一遍完整循环', 30, MUTED)
    for xx, label, color in [(64, '训练与取数', BLUE), (350, '参数更新', AMBER),
                             (600, '验证', TEAL), (780, '生成', PURPLE)]:
        d.rect(xx, 169, 17, 17, color, stroke='none', radius=4)
        d.text(xx + 30, 187, label, 25, MUTED)
    d.text(2496, 187, '实线：执行顺序    虚线：函数调用 / 返回', 25, MUTED, anchor='end')
    for xx, title, lines, rr in [
        (64, 'M1  选择数据，创建两个 loader',
         ['train：000001.npy（16）＋000002.npy（8）',
          'val：000000.npy（16）；各有自己的游标',
          '目录来自 --data-dir，或新建两句教学 demo'], 'M1'),
        (896, 'M2  模型与优化器只创建一次',
         ['随机初始化 GPT → device → 可选 DDP',
          '建立 AdamW；训练、验证、生成共用当前参数',
          '本图：tiny / CPU / FP32；词表仍为 50257'], 'M2'),
        (1728, 'M3  调用 train_from_shards(...)',
         ['B=2，T=3，R=1；每微批次有 6 个预测位置',
          '目标每次更新 12 个位置 → K=12÷6=2',
          '共更新 3 次；验证 / 生成间隔各为 2'], 'M3'),
    ]:
        d.node(xx, 235, 768, 200, title, lines, BLUE, BLUE_BG if xx == 64 else 'white',
               ref(rr), title_size=29, body_size=25)
    d.arrow([(832, 335), (896, 335)], BLUE)
    d.arrow([(1664, 335), (1728, 335)], BLUE)
    d.arrow([(2112, 435), (2112, 455), (590, 455), (590, 470)], BLUE)

    d.panel(64, 470, 1040, 3230, '01  主循环：什么时候更新参数？',
            f'train_shards.py · train_from_shards() L{a["train"]["line"]}；m = 已完成更新次数', BLUE)
    d.node(160, 620, 860, 135, 'T1  model.train() → 初始验证',
           ['record_validation(0) → 右侧 B 验证流程',
            '返回后进入 for step in range(steps)'], TEAL, TEAL_BG, ref('T1'))
    d.node(160, 815, 860, 135, '开始一次参数更新：清空 .grad',
           ['optimizer.zero_grad(set_to_none=True)',
            '下面累积 K=2 次；期间不再清梯度'], BLUE, BLUE_BG, ref('zero'))
    d.rect(160, 1010, 880, 880, '#f5f8fe', radius=20, background=True)
    d.text(184, 1060, f'for micro_step in range(K)  ·  L{a["micro"]["line"]}', 29, BLUE, 650)
    d.node(180, 1110, 840, 135, 'T2  train_loader.next_batch()  →  右侧 A',
           ['输入：当前训练片与游标', '返回：CPU 上的 x、y，形状均为 [2,3]'],
           BLUE, 'white', ref('T2'), title_size=28)
    d.node(180, 1300, 840, 165, 'x/y.to(device) → forward_and_loss(...)',
           ['本例在 CPU；CUDA 配置才将当前批次送入 GPU',
            'logits [2,3,50257] → 6 个目标位置的交叉熵',
            'loss：这 6 个位置损失的平均值，一个标量'], BLUE, 'white', title_size=27)
    d.node(180, 1520, 840, 135, 'T3  (loss / K).backward()',
           ['每个参数的梯度累加进 .grad；模型参数 θ 不变',
            '多卡：前 K−1 次 no_sync，末次反向同步梯度'],
           BLUE, BLUE_BG, ref('T3'), body_size=25)
    d.diamond(590, 1770, 320, 75, '已完成 K 个微批次？', '本例：micro_step + 1 == 2')
    d.arrow([(270, 1770), (140, 1770), (140, 1178), (180, 1178)], BLUE)
    d.text(181, 1738, '否', 25, BLUE)
    d.text(612, 1874, '是：退出微批次循环', 25, BLUE)
    d.node(160, 1960, 860, 190, 'T4  裁剪 → 设置 lr(step) → AdamW',
           ['clip_grad_norm_ → 各参数组设置学习率',
            'optimizer.step()：用累计梯度更新 θ 与 m/v',
            'm = step+1；本例每次覆盖 2×2×3 = 12 个位置'],
           AMBER, AMBER_BG, f'{ref("T4")}；step() L{a["update"]["line"]}', body_size=25)
    d.diamond(590, 2260, 320, 75, '到验证间隔，或最后一次？', 'm % 2 == 0  或  m == 3')
    d.node(160, 2390, 860, 120, 'T5  record_validation(m) → 右侧 B',
           ['等待验证完成；记录 val_loss；回到这里继续'], TEAL, TEAL_BG, ref('T5'))
    d.diamond(590, 2645, 320, 75, '到生成间隔，或最后一次？', 'm % 2 == 0  或  m == 3')
    d.node(160, 2780, 860, 135, 'T6  sample_current_model(...) → 右侧 C',
           ['输入：raw_model（当前 θ）与固定 prompt IDs',
            '返回：生成 IDs → 解码、记录 → 继续主循环'],
           PURPLE, PURPLE_BG, ref('T6'), title_size=28)
    d.diamond(590, 3110, 320, 75, '还有下一次参数更新？', 'for step 循环是否继续')
    d.node(160, 3270, 860, 145, '返回记录 → main() 写 JSON 报告',
           ['gather_records() → rank 0 保存 → 清理进程组',
            '报告含三类记录与读取位置；本入口未保存权重'],
           BLUE, BLUE_BG, ref('M4'), title_size=28, body_size=25)
    d.node(160, 3500, 860, 140, '换分片与开始新更新，是两件事',
           ['换片只更换 tokens 与文件 / 游标。',
            '模型、AdamW 历史和学习率计数继续沿用。'], INK, '#f4f6f9')
    for yy1, yy2 in [(755, 815), (950, 1010), (1245, 1300), (1465, 1520),
                     (1655, 1695), (1890, 1960), (2150, 2185), (2335, 2390),
                     (2510, 2570), (2720, 2780), (2915, 3035), (3185, 3270)]:
        d.arrow([(590, yy1), (590, yy2)], BLUE)
    d.text(615, 2370, '是', 25, TEAL)
    d.text(615, 2762, '是', 25, PURPLE)
    d.text(615, 3240, '否', 25, MUTED)
    d.arrow([(910, 2260), (1060, 2260), (1060, 2540), (590, 2540)], MUTED)
    d.text(955, 2238, '否', 25, MUTED)
    d.arrow([(910, 2645), (1060, 2645), (1060, 2990), (590, 2990)], MUTED)
    d.text(955, 2623, '否', 25, MUTED)
    d.arrow([(270, 3110), (100, 3110), (100, 882), (160, 882)], BLUE)
    d.text(168, 3068, '是：接着训练游标取数', 25, BLUE)

    # A：next_batch 会先准备下一次的文件/游标，再返回当前批次。
    d.panel(1240, 470, 1256, 1300, 'A  展开分片读取：只管数据与位置',
            f'load_shards.py · next_batch() L{a["batch"]["line"]}；train / val 各有一个对象', BLUE)
    d.node(1280, 620, 1176, 135, '当前分片在 CPU 内存中：tokens [N]',
           ['首次加载 / 换片：np.load → uint16 → int64 → CPU torch.long',
            '整片留在 CPU；next_batch() 本身不把数据送入 GPU'],
           BLUE, BLUE_BG, f'load_tokens() L{a["load"]["line"]}', body_size=27)
    d.node(1280, 810, 1176, 180, '① 取 7 个连续 ID，错开一位得到 x / y',
           ['buf = tokens[p : p + B*T + 1]     本例 B*T+1 = 7',
            'x = buf[:-1].view(2,3)             y = buf[1:].view(2,3)',
            'x 的每个位置都预测 y 的对应 ID → 共 6 个预测位置'],
           BLUE, 'white', f'{ref("S4")} / {ref("S5")}', body_size=27)
    d.node(1280, 1045, 1176, 135, '② 更新“下一次”的起点：p += B*T*R',
           ['本图 R=1：前进 6 个位置；最后一个标签可成为下一批输入',
            '多卡：起点 = 全局 p + rank×B×T；全局 p 每轮前进 B×T×R'],
           BLUE, 'white', ref('S6'), body_size=26)
    d.diamond(1868, 1300, 440, 80, '当前片还能放下下一轮吗？', '更新后的 p + B×T×R + 1 ≤ N', BLUE, BLUE_BG)
    d.node(1510, 1440, 946, 120, '不足：加载下一可用片，p = 0',
           ['末片后绕回首片；跳过整片也不够一轮的文件'],
           AMBER, AMBER_BG, f'{ref("S7")} / _load_usable_shard() L{a["switch"]["line"]}', title_size=29)
    d.node(1280, 1630, 1176, 100, '③ 返回刚才切出的 x / y，回到调用者',
           ['即使已经预载下一片，返回值仍属于旧片；单个 batch 不拼接分片。'],
           BLUE, BLUE_BG, title_size=30, body_size=26)
    for yy1, yy2 in [(755, 810), (990, 1045), (1180, 1220), (1380, 1440), (1560, 1630)]:
        d.arrow([(1868, yy1), (1868, yy2)], BLUE)
    d.text(1900, 1416, '否：不足', 25, AMBER)
    d.arrow([(1428, 1300), (1308, 1300), (1308, 1595), (1868, 1595)], BLUE)
    d.text(1340, 1532, '是：留在本片', 25, BLUE)
    d.arrow([(1020, 1150), (1140, 1150), (1140, 685), (1280, 685)], BLUE, dashed=True)
    d.arrow([(1280, 1690), (1190, 1690), (1190, 1210), (1020, 1210)], BLUE, dashed=True)
    d.text(1120, 1125, '调用 A', 23, BLUE)
    d.text(1114, 1242, '返回 x/y', 23, BLUE)

    # B：只有验证 loader 重置；评估结果只记录，不改变更新策略。
    d.panel(1240, 1850, 1256, 760, 'B  展开验证：用固定验证数据测当前参数',
            f'train_shards.py · evaluate_shards() L{a["eval"]["line"]}；初始 / 间隔 / 最终调用', TEAL, '#fbfefd')
    d.node(1280, 1990, 1176, 135, '保存模式 → model.eval() → val_loader.reset()',
           ['仅 val 回到第一片开头；train 的文件与游标保持原样',
            '先检查验证容量，避免这次验证绕回重复计数'], TEAL, TEAL_BG, title_size=29)
    d.node(1280, 2185, 1176, 190, 'with torch.no_grad()：本例读取 2 个验证微批次',
           ['val_loader.next_batch()（同 A）→ x/y.to(device) → forward_and_loss',
            'val_loss += loss / 2；本例共评估 2×2×3 = 12 个预测位置',
            '多卡还对指标求全局均值；这里没有 backward() 或 optimizer.step()'],
           TEAL, 'white', title_size=28, body_size=26)
    d.node(1280, 2440, 1176, 125, '恢复训练模式 → 返回 val_loss 与读取记录',
           ['本入口只记录验证指标；学习率仍按更新次数调度。'], TEAL, TEAL_BG)
    d.arrow([(1868, 2125), (1868, 2185)], TEAL)
    d.arrow([(1868, 2375), (1868, 2440)], TEAL)
    d.arrow([(1020, 2425), (1140, 2425), (1140, 2048), (1280, 2048)], TEAL, dashed=True)
    d.arrow([(1280, 2520), (1190, 2520), (1190, 2480), (1020, 2480)], TEAL, dashed=True)
    d.text(1120, 2399, '调用 B', 23, TEAL)
    d.text(1108, 2556, '返回指标', 23, TEAL)

    # C：固定 prompt 的自回归生成；不把生成样本变成训练 batch。
    d.panel(1240, 2690, 1256, 1010, 'C  展开生成：观察当前模型能续写什么',
            f'train_generation.py · sample_current_model() L{a["sample"]["line"]}；G1—G6', PURPLE, '#fdfcfe')
    d.node(1280, 2830, 1176, 145, '固定 prompt → 4 条输入 [4,8]；沿用当前 θ',
           ['建立独立抽样 RNG → eval() → no_grad()',
            '不调用 train / val loader；不读分片；不修改参数'],
           PURPLE, PURPLE_BG, f'{ref("G1")} / {ref("G2")}', title_size=29)
    d.node(1280, 3040, 1176, 185, '完整前缀前向 → 取最后位置 logits → 抽样',
           ['[4,t] → [4,t,50257] → [4,50257] → softmax → top-k=50',
            'multinomial → gather 得到词表 ID → cat 追加为 [4,t+1]',
            '每轮每条增加 1 个 Token；输入长度增长，参数 θ 保持不变'],
           PURPLE, 'white', f'{ref("G3")} / {ref("G4")} / {ref("G5")}', title_size=29, body_size=26)
    d.diamond(1868, 3360, 350, 75, '已到目标长度？', '本例目标：8＋24 = 32', PURPLE, PURPLE_BG)
    d.node(1280, 3490, 1176, 145, '恢复原模式 → 返回生成结果 → 调用者解码',
           ['本次循环 24 轮：[4,8] → [4,32]，每条新增 24 个 ID',
            '观察后继续取真实训练数据；生成文字不进入训练 batch'],
           PURPLE, PURPLE_BG, ref('G6'), title_size=29)
    d.arrow([(1868, 2975), (1868, 3040)], PURPLE)
    d.arrow([(1868, 3225), (1868, 3285)], PURPLE)
    d.arrow([(1868, 3435), (1868, 3490)], PURPLE)
    d.text(1900, 3470, '是', 25, PURPLE)
    d.arrow([(2218, 3360), (2473, 3360), (2473, 3132), (2456, 3132)], PURPLE)
    d.text(2300, 3330, '否：继续', 25, PURPLE)
    d.arrow([(1020, 2820), (1140, 2820), (1140, 2878), (1280, 2878)], PURPLE, dashed=True)
    d.arrow([(1280, 3570), (1190, 3570), (1190, 2880), (1020, 2880)], PURPLE, dashed=True)
    d.text(1120, 2795, '调用 C', 23, PURPLE)
    d.text(1108, 2950, '返回 IDs', 23, PURPLE)

    d.text(64, 3790, '02  用实际读取记录，把“一个微批次”和“一次更新”对上', 40, BLUE, 700)
    d.text(64, 3840, '示例来自 CPU tiny 归档：B=2、T=3、R=1、K=2；下面的文件编号省略 edufineweb_train_ 前缀。', 27, MUTED)
    d.node(64, 3880, 768, 290, '一个微批次：7 个 ID → 6 对预测任务', [], BLUE, 'white', title_size=29)
    d.text(88, 3970, '第一片 p=0；x 与 y 的对应格子是一对输入 / 目标', 24, MUTED)
    d.text(88, 4020, 'x [2,3]', 27, BLUE, 650)
    d.text(464, 4020, 'y [2,3]', 27, TEAL, 650)
    for ii in range(2):
        d.text(88, 4075 + 44 * ii, '[ ' + '  '.join(map(str, x[ii])) + ' ]', 26, INK, mono=True)
        d.text(464, 4075 + 44 * ii, '[ ' + '  '.join(map(str, y[ii])) + ' ]', 26, INK, mono=True)
    d.node(864, 3880, 768, 290, '两次反向：累积到同一份 .grad',
           ['第 1 批：.grad = g₁ / 2',
            '第 2 批：.grad += g₂ / 2',
            '得到 (g₁ + g₂) / 2 → 裁剪 → 更新一次',
            'gᵢ 是第 i 批平均 loss 对参数的梯度',
            '两个 g 都使用同一组尚未更新的参数 θ'], BLUE, BLUE_BG, title_size=30, body_size=26)
    d.node(1664, 3880, 832, 290, '三个循环，各自推进不同状态',
           ['读取循环：游标前进；不足时换文件',
            '微批次循环：重复 K 次，累计 .grad',
            '更新循环：AdamW 一次，更新数 m 加一',
            '验证 / 生成：同步调用完成后返回主循环',
            '分片数、微批次数、参数更新数并不相等'], AMBER, AMBER_BG, title_size=30, body_size=26)
    d.rect(64, 4220, 2432, 380, 'white', radius=20)
    columns = [94, 320, 790, 1260, 1740]
    for xx, label in zip(columns, ['参数更新', '微批次 1：文件 : 起点', '微批次 2：文件 : 起点', '下一次训练读取位置', '更新后的观察']):
        d.text(xx, 4274, label, 27, BLUE, 650)
    for i, row in enumerate(train):
        yy = 4322 + i * 88
        if i == 1:
            d.rect(78, yy - 31, 2404, 78, '#fff4e4', stroke='none', radius=10)
        short = lambda p: p['file'].split('_')[-1].removesuffix('.npy') + f' : {p["start"]}'
        values = [f'第 {row["step"]} 次', short(row['local_batches'][0]),
                  short(row['local_batches'][1]), short(row['next_train_cursor'])]
        for xx, value in zip(columns, values):
            d.text(xx, yy + 20, value, 29, INK, 600 if i == 1 else 400)
        observation = '跳过验证与生成' if i == 0 else f'验证 {val[i]["global_mean_loss"]:.6f} → 生成 4×24'
        d.text(columns[-1], yy + 20, observation, 27, TEAL if i else MUTED)
    d.text(94, 4574, f'更新 0 次：先验证 {val[0]["global_mean_loss"]:.6f}。第 2 次更新跨两个分片，但两批梯度仍累积在一起。', 27, AMBER, 600)
    d.text(64, 4662, '返回位置示例：第 2 次更新后，验证 / 生成均不改变 train 游标；第 3 次从 000001 : 6 接着读取。', 28, INK, 600)
    d.text(64, 4720, '数据说明：两句教学文本、train 16＋8 个 ID、val 16 个 ID；本图只说明执行流转。默认完整模型与 tiny 区别见配套讲义。', 25, MUTED)
    d.text(64, 4770, '数值：artifacts/lesson12-shard-training-cpu.json   ·   行号从源码自动提取   ·   未重新训练或修改训练代码', 24, MUTED)
    return d.finish()


def notes(a, r, train, val, gen):
    rows = []
    labels = [('main', '主入口'), ('loaders', '创建两个加载器'), ('train', '更新 / 微批次两层循环'),
              ('batch', '读取、错位、推进、换片'), ('backward', '梯度累积'), ('update', '参数更新'),
              ('eval', '独立验证循环'), ('sample', '生成循环'), ('M4', '汇总并保存报告')]
    for key, label in labels:
        p = a[key]
        rows.append(f'| {label} | [{p["file"]} L{p["line"]}](../../{p["file"]}#L{p["line"]}) |')
    return '''# 分片训练：代码全流程图

[PNG 原图](gpt2-shard-training-flow.png) · [可缩放 SVG](gpt2-shard-training-flow.svg) · [配套讲义](../../lesson12-shard-training.md)

![分片读取、微批次累积、参数更新、验证与生成](gpt2-shard-training-flow.png)

## 按什么顺序读

1. 顶部 `main()`：选择分片目录，创建独立的 train / val 加载器，随机初始化一次模型与 AdamW，再进入主循环。
2. 左侧先初始验证，再清梯度。内层循环每次读一个微批次，搬运 x/y，前向求平均 loss，执行 `(loss/K).backward()`。重复 K 次后才裁剪、设置学习率、调用一次 `optimizer.step()`。
3. 右侧 A 展开 `next_batch()`：当前分片在 CPU；取 B×T+1 个 ID，错开一位形成 x/y，推进下一次游标并判断换片，最后返回本次切出的 x/y。预载下一片不会改写已切出的 batch。
4. 更新完成后分别判断验证与生成。右侧 B 只重置验证加载器，评估后恢复模式；右侧 C 使用固定提示词和当前参数，不调用数据加载器，逐 Token 生成，恢复模式后返回主循环。
5. 下一次更新先清梯度，再从训练游标继续。底部记录第 2 次更新跨两个文件、观察后第 3 次仍从 `000001:6` 继续。

虚线是同步函数的调用 / 返回，不代表并发。左侧初始验证也调用 B，只省略重复的跨图连线。B 内部读取的是 val 对象，使用与 A 相同的类方法。

## 代码定位

| 图中环节 | 源码 |
|---|---|
''' + '\n'.join(rows) + f'''

行号由绘图脚本从源码提取；M/T/S/G 标记对应源码注释。A/B/C 是图中三个展开区域，不是额外的函数。

## 本图的具体计算

- 每个微批次 `x/y=[2,3]`：两行序列，每行三个预测位置，共 6 对输入 / 目标。每个位置用因果上下文预测对应 y；不是只有每行最后一格有 loss。
- 为获得这 6 对监督信号，需要连续 7 个 ID：前 6 个组成 x，后 6 个组成 y。读 7 个 ID 不等于有 7 个预测位置。
- `loss` 已是 6 个位置的平均损失；每次更新要求 12 个位置，因此 `K=12/(2×3×1)=2`。两个微批次都用同一组参数 θ，梯度最终是 `(g₁+g₂)/2`。
- `.grad` 由 `backward()` 累加；代码的 `loss_accum += scaled_loss.detach()` 只是记录 loss 指标，不能替代梯度累积。
- 当前分片长度 16，从 p=0 取完后下一起点为 6，`6+6+1≤16`，留在当前片；从 p=6 取完后下一起点为 12，`12+6+1>16`，预载下一片。末尾不足一批的部分不跨文件拼接。
- 长度 8 的第二片只够从 p=0 取一次，随后绕回首片。一次参数更新的两个微批次可以跨文件，但单个微批次始终来自一个文件。换片不清梯度、不重建优化器、不重置调度。
- 本次验证 / 生成都在更新 2、3 次后触发；两项判断各自独立。验证数据是 `val_000000.npy` 的起点 0、6；生成从固定 8 个提示词 ID 开始，4 条序列各新增 24 个 ID。
- 验证只测量指标，本入口不会根据验证 loss 自动改学习率或早停。生成文字只供观察，不是下一批训练样本。JSON 是实验报告，不是可恢复的权重检查点。

## 范围与复现

数字来自 [CPU tiny 报告](../../artifacts/lesson12-shard-training-cpu.json)：3 次更新、{r['training_positions']} 个训练预测位置。模型为随机初始化的 2 层、64 维教学配置；默认不加 `--model-size tiny` 时入口创建完整 GPT-2。图中两句 demo 不代表 FineWeb 语料，短跑不衡量语言质量。

多卡只作为旁注：每进程的起点加 `rank×B×T`，全局游标步长为 `B×T×R`；每次更新覆盖 `K×B×T×R` 个位置。前 K−1 次 `no_sync()` 同时包住前向与反向，最后反向同步参数梯度；验证中的 all-reduce 只汇总指标。主图数值是 R=1，不能当作 CUDA/NCCL 实测。

原课来源和本地改编边界保留在[讲义第 0 节](../../lesson12-shard-training.md)。本次只绘图，不重跑训练，不修改旧训练代码。

重建（项目已配置的 Python 环境）：

```bash
python docs/diagrams/render_shard_training_flow.py --png
```

SVG / Markdown 只需要标准库及项目绘图组件，PNG 导出复用 macOS AppKit。脚本核对源码定位、分片边界、真实读取顺序、验证 / 生成触发、观察前后的游标与生成长度；结果保存在 [制图核对记录](../../artifacts/lesson12-shard-training-diagram-check.json)。
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--png', action='store_true', help='使用 macOS AppKit 导出 PNG')
    args = parser.parse_args()
    anchors, hashes, report, train, val, gen, x, y = evidence()
    svg = draw(anchors, report, train, val, gen, x, y)
    ET.fromstring(svg)
    path = OUT / 'gpt2-shard-training-flow.svg'
    path.write_text(svg, encoding='utf-8')
    path.with_suffix('.md').write_text(notes(anchors, report, train, val, gen), encoding='utf-8')
    if args.png:
        export_png(path)
    check = {
        'purpose': 'diagram only; source and archived run cross-check',
        'training_run': False, 'source_lines': anchors, 'source_sha256': hashes,
        'report': str(REPORT.relative_to(ROOT)),
        'report_sha256': hashlib.sha256(REPORT.read_bytes()).hexdigest(),
        'shape': {'B': 2, 'T': 3, 'R': 1, 'K': 2, 'positions_per_update': 12},
        'training_trace': train,
        'validation_after_updates': [v['after_updates'] for v in val],
        'generation_after_updates': [g['after_updates'] for g in gen],
        'observer_train_cursors_unchanged': True,
        'first_batch': {'x': x, 'y': y},
        'canvas': [WIDTH, HEIGHT],
        'outputs': [str(path.relative_to(ROOT)), str(path.with_suffix('.md').relative_to(ROOT))],
    }
    if args.png:
        check['outputs'].append(str(path.with_suffix('.png').relative_to(ROOT)))
    (ROOT / 'artifacts/lesson12-shard-training-diagram-check.json').write_text(
        json.dumps(check, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Source anchors, shard boundaries, update trace and observer state checked.')
    print(path)
