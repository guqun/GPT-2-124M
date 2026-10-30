"""绘制当前 HellaSwag 训练代码的 CPU/GPU 泳道图，不运行训练。

行号从源码定位，数字只读取已有 CPU 报告。PNG 使用项目已有的 AppKit 导出器。
重建：python docs/diagrams/render_hellaswag_device_flow.py --png
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import unicodedata
import xml.etree.ElementTree as ET

from render_ddp_allreduce_flow import Drawing as BaseDrawing, export_png

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
W, H = 2560, 4600
INK, MUTED, LINE = '#172c3d', '#566c7d', '#d9e3ec'
CPU, CPU_BG = '#355fc0', '#edf2ff'
GPU, GPU_BG = '#087e75', '#e8f6f1'
MOVE, MOVE_BG = '#a36520', '#fff3e2'
BG = '#f6f8fb'
X1, X2, CW = 96, 1440, 1024


def evidence():
    specs = {
        'T': ('train_hellaswag.py', {
            'main': 'def main(', 'loop': 'def train_with_hellaswag(',
            'record': 'def record_hellaswag(', 'initial': 'record_hellaswag(0)',
            'read': 'x, y = train_loader.next_batch()',
            'transfer': 'x, y = x.to(runtime.device)',
            'forward': 'logits, loss = forward_and_loss(',
            'backward': 'scaled_loss.backward()', 'clip': 'norm = torch.nn.utils.clip_grad_norm_(',
            'lr': 'lr = learning_rate_at(', 'step': 'optimizer.step()',
            'val': 'if (step + 1) % validation_every',
            'eval': 'if (step + 1) % hellaswag_every',
            'generation': 'if (step + 1) % generation_every',
            'decode': 'texts=[tokenizer.decode(', 'initialize': 'raw_model = GPT(',
            'move_model': 'raw_model.to(runtime.device)',
            'opt': 'optimizer = configure_optimizer(', 'gather': 'ranks = gather_records(',
            'save': 'path.write_text(',
        }),
        'H': ('hellaswag_eval.py', {
            'render': 'def render_example(', 'load': 'def load_questions(',
            'score': 'def score_candidates(', 'shift': 'shift_logits = logits',
            'ce': 'losses = F.cross_entropy(', 'count': 'count = shift_mask.sum(',
            'mean': 'means = sums / count', 'argmin': "'prediction': means.argmin().item()",
            'evaluate': 'def evaluate_hellaswag(', 'mode': 'raw_model.eval()',
            'no_grad': 'with torch.no_grad():', 'assign': 'if question.index %',
            'tokens_to': 'tokens = question.tokens.to(', 'mask_to': 'mask = question.mask.to(',
            'forward': 'logits = raw_model(tokens)', 'compare': 'correct += int(',
            'details': "'mean_losses': result['mean_loss'].tolist()",
            'pack': 'counts = torch.tensor(', 'reduce': 'dist.all_reduce(counts',
            'readback': 'global_correct, global_total = counts.tolist()',
            'accuracy': "'accuracy': global_correct / global_total", 'restore': 'raw_model.train(was_training)',
        }),
        'S': ('train_shards.py', {'val': 'def evaluate_shards('}),
        'L': ('load_shards.py', {'batch': 'def next_batch('}),
        'G': ('train_generation.py', {'sample': 'def sample_current_model('}),
        'D': ('train_ddp.py', {'runtime': 'def setup_runtime(', 'opt': 'def configure_optimizer(',
                              'gather': 'def gather_records('}),
    }
    anchors, hashes = {}, {}
    for prefix, (filename, fragments) in specs.items():
        text = (ROOT / filename).read_text()
        ast.parse(text)
        hashes[filename] = hashlib.sha256(text.encode()).hexdigest()
        for key, fragment in fragments.items():
            hits = [n for n, line in enumerate(text.splitlines(), 1)
                    if line.strip().startswith(fragment)]
            assert len(hits) == 1, (filename, key, hits)
            anchors[prefix + '.' + key] = dict(file=filename, line=hits[0], alias=prefix)
    paths = ['artifacts/lesson15-hellaswag-cpu.json', 'artifacts/lesson15-hellaswag-ddp-cpu.json']
    reports = [json.loads((ROOT / p).read_text()) for p in paths]
    for r in reports:
        assert r['device_type'] == 'cpu' and r['model_size'] == 'tiny'
        assert r['parameter_updates'] == 3
        assert not r['pretrained_weights_loaded'] and not r['checkpoint_saved']
        for rank in r['ranks']:
            assert [e['after_updates'] for e in rank['hellaswag']] == [0, 2, 3]
            assert all(e['cursors_before'] == e['cursors_after'] for e in rank['hellaswag'])
    single, dual = reports
    assert single['world_size'] == 1 and single['hellaswag']['examples'] == 8
    assert dual['world_size'] == 2 and dual['backend'] == 'gloo'
    a, b = [r['hellaswag'][0] for r in dual['ranks']]
    assert a['local_question_indices'] == [0, 2] and b['local_question_indices'] == [1]
    assert a['local_correct'] + b['local_correct'] == a['global_correct'] == 0
    assert a['local_total'] + b['local_total'] == a['global_total'] == 3
    return anchors, hashes, reports


class Drawing(BaseDrawing):
    def __init__(self):
        super().__init__()
        self.layout_checks = []

    @staticmethod
    def text_width(value, size):
        return sum(size * (1 if unicodedata.east_asian_width(c) in 'WF' else .58)
                   for c in value)

    def fitted(self, x, y, value, size, color=INK, weight=400, width=960):
        while self.text_width(value, size) > width:
            size -= 1
        assert size >= 23, (value, size)
        self.text(x, y, value, size, color, weight)
        self.layout_checks.append({'text': value, 'size': size, 'estimated_width': self.text_width(value, size),
                                   'available_width': width, 'x': x, 'y': y})

    def node(self, x, y, title, lines, *, color, fill, h, ref='', w=CW):
        self.rect(x, y, w, h, fill, radius=20)
        self.fitted(x + 28, y + 47, title, 34, color, 650, w - 56)
        for i, line in enumerate(lines):
            self.fitted(x + 28, y + 94 + 39 * i, line, 29, MUTED, width=w - 56)
        if ref:
            self.text(x + w - 4, y - 12, ref, 22, MUTED, anchor='end')
        assert not lines or 94 + (len(lines) - 1) * 39 <= h - 20, title

    def section(self, y, h, number, title, subtitle):
        self.rect(48, y, W - 96, h, 'white', radius=26, background=True)
        self.text(88, y + 61, number, 43, CPU, 700)
        self.text(174, y + 61, title, 41, INK, 650)
        self.text(88, y + 108, subtitle, 27, MUTED)

    def lane_headers(self, y):
        for x, title, sub, color, fill in [(X1, 'CPU · Python 进程 / 主内存', '读取、分词、循环、条件判断、记录', CPU, CPU_BG),
                                         (X2, 'GPU · 当前进程对应的 CUDA 设备', '模型张量、前向/反向、评分与归约', GPU, GPU_BG)]:
            self.rect(x, y, CW, 65, fill, stroke='none', radius=13)
            self.text(x + 23, y + 43, title, 31, color, 650)

    def transfer(self, y, labels, right=True):
        self.arrow([(X1 + CW, y), (X2, y)] if right else [(X2, y), (X1 + CW, y)], MOVE, width=4)
        for i, label in enumerate(labels):
            self.text((X1 + CW + X2) / 2, y - 16 - (len(labels) - i - 1) * 32,
                      label, 24, MOVE, 600, anchor='middle')

    def finish(self):
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<title>HellaSwag 训练与评测：CPU 与 GPU 分工及代码调用</title>
<desc>按 CUDA 路径绘制。CPU 读取分片和评测题、分词、准备 mask、控制训练和评测循环；GPU 执行前向、交叉熵、反向、AdamW 和候选 argmin。预测编号通过 item 回到 CPU 与标准答案比较，局部正确数和题数再打包成 GPU Tensor，经 NCCL SUM 后通过 tolist 回到 CPU 计算准确率。当前报告来自 CPU/Gloo，右栏张量算子在 CPU 执行，图不代表 CUDA 实测。</desc>
<style>text {{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}}</style>
<rect width="{W}" height="{H}" fill="{BG}"/>
{''.join(self.bg)}{''.join(self.edges)}{''.join(self.fg)}
</svg>'''


def draw(a, reports):
    d = Drawing()
    def ref(*keys):
        return ' · '.join(f"{a[k]['alias']}:L{a[k]['line']}" for k in keys)
    d.text(72, 86, 'HellaSwag 训练与评测：CPU / GPU 各做什么', 56, INK, 700)
    d.text(72, 143, '从 main() 到训练更新、逐题评分、跨卡汇总，再回到下一次训练', 33, MUTED)
    d.rect(72, 178, 2416, 96, MOVE_BG, stroke='none', radius=17)
    d.text(96, 217, '主图按 --device cuda 绘制：Python 在 CPU 发起调用，Tensor 运算在其 device 上执行。', 30, INK, 600)
    d.text(96, 254, '当前已验证的是 CPU / Gloo：右侧的张量计算也在 CPU；双进程不等于已运行两张 GPU。', 28, MUTED)
    d.text(72, 310, '代码简写：T = train_hellaswag.py     H = hellaswag_eval.py     S = train_shards.py     G = train_generation.py', 25, MUTED)

    d.section(340, 1200, '01', '训练主循环：数据进入设备，参数留在设备',
              '实线为数据 / 执行流；橙色箭头为跨 CPU–GPU 的数据传递；虚线为调用或循环。')
    d.lane_headers(475)
    d.node(X1, 585, 'M1 / M2 · 读取文件，准备输入与模型',
           ['JSONL → 分词 → CPU tokens / mask；label 是整数',
            'npy → 当前 CPU 分片；训练时再切出 x / y',
            'CPU 上创建并随机初始化 GPT；每进程一份'],
           color=CPU, fill=CPU_BG, h=200, ref=ref('H.load', 'H.render', 'T.initialize'))
    d.node(X2, 585, '模型参数留在当前 GPU',
           ['raw_model.to(runtime.device) 搬运模型参数',
            '各轮训练 / 验证 / 评测 / 生成复用当前参数',
            'Python 持有 AdamW 对象；其 m / v 随参数在 GPU'],
           color=GPU, fill=GPU_BG, h=200, ref=ref('T.move_model', 'T.opt'))
    d.transfer(720, ['模型参数', '.to(device)'])
    d.arrow([(608, 785), (608, 855)], CPU)
    d.text(636, 825, '初始验证 / 评测返回后', 25, CPU)

    d.node(X1, 855, 'T2 · 循环 K 次，每次取一个微批次',
           ['train_loader.next_batch() → CPU x / y [B,T]',
            'CPU 管理文件、游标和 micro_step 循环'],
           color=CPU, fill=CPU_BG, h=160, ref=ref('T.read'))
    d.node(X2, 855, 'T3 · 前向求 loss，反向累积 .grad',
           ['model(x) → logits → loss/K → backward()',
            '多卡：最后一个微批次的反向同步累计梯度'],
           color=GPU, fill=GPU_BG, h=160, ref=ref('T.forward', 'T.backward'))
    d.transfer(955, ['x / y 两个张量', '.to(device)'])
    d.arrow([(2464, 970), (2500, 970), (2500, 1050), (62, 1050), (62, 940), (96, 940)], CPU, dashed=True)
    d.text(1275, 1087, '尚未累计满 K 次：CPU 继续取下一微批次', 25, CPU, anchor='middle')

    d.node(X1, 1140, 'T4 · 满 K 次，计算学习率并发起更新',
           ['learning_rate_at(...) 得到 Python 浮点数 lr',
            '设置 param_groups；调用裁剪及 optimizer.step()'],
           color=CPU, fill=CPU_BG, h=160, ref=ref('T.lr', 'T.step'))
    d.node(X2, 1140, 'T4 · GPU 上裁剪梯度，更新参数',
           ['PyTorch 执行裁剪与 AdamW 张量计算',
            'θ、梯度、m / v 都在本卡；参数更新一次'],
           color=GPU, fill=GPU_BG, h=160, ref=ref('T.clip', 'T.step'))
    d.arrow([(X1 + CW, 1230), (X2, 1230)], CPU, dashed=True)
    d.text(1280, 1205, 'CPU 调度算子', 25, CPU, anchor='middle')
    d.arrow([(1952, 1015), (1952, 1140)], GPU)
    d.rect(96, 1355, 2368, 145, '#f3f6fa', stroke='none', radius=17)
    d.text(122, 1397, '进入训练前：初始验证 → 初始评测；完成每次更新后，CPU 依次判断下面三个条件：', 29, INK, 600)
    d.text(122, 1441, 'T5 验证？  →  T7 HellaSwag 评测？（下方展开）  →  T6 生成？  →  下一次更新', 31, INK, 600)
    d.text(122, 1481, f"默认评测时机：已更新 0、2、3 次；末次必评。条件位置 {ref('T.initial', 'T.val', 'T.eval', 'T.generation')}", 25, MUTED)

    d.section(1580, 2250, '02', '展开 T7：一轮评测从哪里进入，又从哪里返回',
              f"record_hellaswag() → evaluate_hellaswag()    {ref('T.record', 'H.evaluate')}    一道题的四个候选在同一进程 / 设备计算。")
    d.lane_headers(1720)
    d.node(X1, 1825, '① · 切换模式，选择属于本进程的题',
           ['eval() / no_grad()；设置模式与梯度记录开关',
            'question.index % world_size == rank 才处理'],
           color=CPU, fill=CPU_BG, h=160, ref=ref('H.mode', 'H.no_grad', 'H.assign'))
    d.node(X2, 1825, '本卡模型保持当前参数',
           ['raw_model 是训练模型本体，不是新加载的模型',
            '整轮评测没有 backward() 和 optimizer.step()'],
           color=GPU, fill=GPU_BG, h=160)
    d.node(X1, 2050, '② · 取出一道题：CPU 张量与标准答案',
           ['question.tokens [4,T]；question.mask [4,T]',
            'question.label 留在 CPU，不参与前向或评分',
            '题目 T 由完整上下文 + 候选决定，不是训练 seq_len'],
           color=CPU, fill=CPU_BG, h=200, ref=ref('H.tokens_to', 'H.mask_to'))
    d.node(X2, 2050, '③ · raw_model(tokens) 前向',
           ['GPU tokens / mask；模型使用已有 GPU 参数',
            '输出 logits [4,T,V]；V = 50,257',
            '每行只读自身前缀；四行之间没有注意力连接'],
           color=GPU, fill=GPU_BG, h=200, ref=ref('H.forward'))
    d.arrow([(608, 1985), (608, 2050)], CPU)
    d.transfer(2150, ['tokens + mask', '.to(device)'])
    d.node(X1, 2280, 'CPU 发起评分函数调用',
           ['label、题号、correct / total 仍是 CPU 数据',
            'logits 留在 GPU，后续评分也使用 GPU 张量',
            '值检查需要 Python 判断时，也可能同步取数'],
           color=CPU, fill='#f6f8fc', h=200, ref=ref('H.score'))
    d.node(X2, 2280, '④ · score_candidates() 计算四个得分',
           ['logits[:, :-1] 对齐 tokens[:, 1:]',
            "cross_entropy(..., reduction='none') → [4,T−1]",
            'shift_mask = mask[:, 1:]；只统计候选结尾',
            'sums = (losses * shift_mask).sum(dim=1)',
            'count = shift_mask.sum(dim=1)；means = sums/count',
            'means [4] → argmin() → GPU 上的标量编号'],
           color=GPU, fill=GPU_BG, h=320, ref=ref('H.shift', 'H.ce', 'H.mean', 'H.argmin'))
    d.arrow([(1952, 2250), (1952, 2280)], GPU)
    d.arrow([(1120, 2375), (1440, 2375)], CPU, dashed=True)
    d.text(1280, 2348, '调用 H3', 25, CPU, anchor='middle')
    d.node(X1, 2690, '⑤ · CPU 判对，累计本卡结果',
           ['prediction 是 .item() 返回的 Python 整数',
            'correct += int(prediction == question.label)',
            'total += 1；记录题号和前四题的得分明细'],
           color=CPU, fill=CPU_BG, h=200, ref=ref('H.compare', 'H.details'))
    d.node(X2, 2690, '把少量结果取回 CPU',
           ['means.argmin().item()：取回一个选项编号',
            '前四题：得分 / Token 数 .tolist() 形成报告',
            '不把整份 logits 或模型参数搬回 CPU'],
           color=GPU, fill=GPU_BG, h=200, ref=ref('H.argmin', 'H.details'))
    d.arrow([(1952, 2600), (1952, 2690)], GPU)
    d.transfer(2825, ['一个编号 / 少量明细', '.item() / .tolist()'], right=False)
    d.arrow([(608, 2890), (608, 2940), (64, 2940), (64, 2140), (96, 2140)], CPU, dashed=True)
    d.text(130, 2974, '本卡还有题：回到②；处理完本卡所有题后，才进入⑥。', 27, CPU)
    d.node(X1, 3040, '⑥ · 本卡题目循环结束，打包两个计数',
           ['Python 整数：[correct, total]',
            'torch.tensor([...], device=runtime.device)'],
           color=CPU, fill=CPU_BG, h=160, ref=ref('H.pack'))
    d.node(X2, 3040, '⑦ · 多卡时，用 NCCL 汇总两个整数',
           ['counts：GPU long Tensor [2]',
            'dist.all_reduce(counts, SUM)；无题的卡也参加'],
           color=GPU, fill=GPU_BG, h=160, ref=ref('H.reduce'))
    d.transfer(3140, ['正确数 + 题数', 'CPU → GPU'])
    d.node(X1, 3290, '⑧ · CPU 计算准确率，恢复模式并返回',
           ['global_correct / global_total → Python float',
            'raw_model.train(was_training) 恢复原模式',
            '返回 record_hellaswag()；rank 0 打印结果',
            'train / val 游标未前进，模型及优化器保持原样'],
           color=CPU, fill=CPU_BG, h=240, ref=ref('H.accuracy', 'H.restore', 'T.record'))
    d.node(X2, 3290, '汇总结果取回 CPU；本卡保留参数',
           ['counts.tolist() 取回全局正确数、全局题数',
            '单卡跳过 all_reduce，其余评分流程相同',
            '多卡：参数 θ、已有 .grad、AdamW 状态仍在本卡'],
           color=GPU, fill=GPU_BG, h=240, ref=ref('H.readback'))
    d.arrow([(1952, 3200), (1952, 3290)], GPU)
    d.transfer(3425, ['全局两个整数', '.tolist()'], right=False)
    d.rect(96, 3590, 2368, 195, '#f3f6fa', stroke='none', radius=18)
    d.text(123, 3635, '返回主循环后，其余两条观察分支也有明确的设备分工：', 31, INK, 650)
    d.text(123, 3680, f"验证：CPU 取 val 批次 → GPU 前向 / loss / 归约 → CPU 记录标量。   {ref('S.val')}", 28, MUTED)
    d.text(123, 3721, f"生成：CPU 调度循环 → GPU 前向 / 抽样 / 追加 ID → CPU 解码文字。   {ref('G.sample', 'T.decode')}", 28, MUTED)
    d.text(123, 3764, f"继续训练；所有更新完成后汇总 Python 记录，rank 0 在 CPU 写 JSON。   {ref('T.gather', 'T.save')}", 27, MUTED)

    d.section(3870, 635, '03', '当前 CPU 实验验证了什么，换成多 GPU 后哪里变化',
              '同一套控制逻辑；设备由 runtime.device 决定。下方题号与计数来自归档的双进程 CPU 报告。')
    d.node(96, 4025, '当前已验证：两个 CPU 进程 + Gloo',
           ['每进程一份模型；主图右栏的张量也在 CPU',
            '进程 0：题 0、2 → [答对 0, 题数 2]',
            '进程 1：题 1     → [答对 0, 题数 1]',
            'Gloo SUM → 两进程各得到 [0,3] → 准确率 0/3'],
           color=CPU, fill=CPU_BG, h=240, w=1136)
    d.node(1328, 4025, '使用 CUDA 时：每个进程驱动一张 GPU',
           ['CPU 仍负责读取、分词、循环、判对和日志',
            '模型、前向/反向、评分张量在对应 GPU',
            '跨 GPU 的梯度 / 计数归约使用 NCCL',
            '实际 GPU 速度、显存、NCCL 性能仍需实测'],
           color=GPU, fill=GPU_BG, h=240, w=1136)
    results = reports[0]['ranks'][0]['hellaswag']
    counts = ' → '.join(f"{r['global_correct']}/{r['global_total']}" for r in results)
    d.text(96, 4330, f'单进程真实题预览：更新 0 / 2 / 3 次后的准确率为 {counts}；题数仅 8，不是完整基准。', 29, INK, 600)
    d.text(96, 4380, '双进程测试还覆盖 2+1 道题与 1+0 道题的分配、汇总和恢复反向；这不代表运行过两张 GPU。', 28, MUTED)
    d.text(96, 4430, '代码与报告：train_hellaswag.py / hellaswag_eval.py；artifacts/lesson15-hellaswag-{cpu,ddp-cpu}.json', 26, MUTED)
    d.text(72, 4560, '位置不是由函数名决定：看 Tensor 的 device，以及 .to() / .item() / .tolist() 在哪里发生。', 31, INK, 600)
    return d.finish(), d.layout_checks


def notes(anchors, reports):
    rows = []
    labels = [('T.main', '主入口和模型准备'), ('H.load', '读取 JSONL / CPU 分词'),
              ('H.render', 'CPU 补齐和 mask'), ('T.move_model', '模型参数搬到 device'),
              ('T.transfer', '训练 x/y 搬到 device'), ('T.forward', '训练前向 / loss'),
              ('T.backward', '反向与梯度累积'), ('T.lr', 'CPU 学习率计算'), ('T.step', '发起 AdamW 更新'),
              ('T.eval', 'CPU 判断评测时机'), ('T.record', '调用和接收评测结果'),
              ('H.tokens_to', '评测 tokens/mask 送往 device'), ('H.forward', '评测前向'),
              ('H.score', '候选逐位置损失和平均'), ('H.argmin', 'GPU argmin / CPU 取标量'),
              ('H.compare', 'CPU 判对、累计计数'), ('H.pack', '把 CPU 计数打包为设备张量'),
              ('H.reduce', '设备上的跨进程 SUM'), ('H.readback', '全局计数回到 CPU'),
              ('H.restore', '恢复模式'), ('T.save', 'rank 0 写报告')]
    for key, label in labels:
        a = anchors[key]
        rows.append(f"| {label} | [{a['file']} L{a['line']}](../../{a['file']}#L{a['line']}) |")
    return '''# HellaSwag 训练与评测：CPU / GPU 代码流程图

[PNG 原图](gpt2-hellaswag-cpu-gpu-flow.png) · [可缩放 SVG](gpt2-hellaswag-cpu-gpu-flow.svg) · [配套讲义](../../lesson15.md)

![HellaSwag 训练与评测的 CPU / GPU 分工](gpt2-hellaswag-cpu-gpu-flow.png)

## 如何阅读

主图按 `--device cuda` 绘制：蓝色是 CPU 的 Python 控制及主内存数据，绿色是本进程 GPU 上的张量运算，橙色是跨设备数据传递。两个泳道不是两个并发执行的 worker；一个进程在 CPU 执行 Python，并向自己的 GPU 提交算子。

1. 上方是训练主循环：CPU 加载分片、准备题目、随机初始化模型，再把参数搬到设备；CPU 每次取 x/y，GPU 完成前向、反向和参数更新。循环、学习率计算和调用发起在 CPU。
2. 中间展开 `record_hellaswag → evaluate_hellaswag`：CPU 按题号选择本进程负责的题，把 tokens 和 mask 送到 GPU；label 一直是 CPU 数据。
3. GPU 前向与评分后，`argmin()` 仍是 GPU 算子；`.item()` 才取回 Python 整数。CPU 比较 prediction 和 label，累计本卡正确数、总题数；返回取下一题。
4. 本卡题目全部处理完后，CPU 把这两个计数建立为设备 Tensor。CUDA/NCCL 在 GPU 上 SUM；`counts.tolist()` 把最终计数取回，CPU 做准确率除法、恢复模式和记录结果。
5. 返回主循环，按条件生成，然后继续训练。整个循环结束后汇总记录，由 rank 0 写 JSON；模型没有因评测搬回 CPU，也没有重建优化器。

## 必须区分的设备细节

- `model.eval()` / `model.train()` 改变模块的模式标志；`no_grad()` 控制反向图记录。它们不会把模型移到其他设备。
- AdamW 对象与参数分组由 CPU Python 管理；CUDA 路径的参数、梯度和一阶/二阶矩张量驻留 GPU，PyTorch 在 GPU 执行更新算术。本图不把所有优化器元数据都笼统画成 GPU 数据。
- 交叉熵、乘 mask、sum、除以候选长度、argmin 都跟随输入张量的 device。原地切片通常是视图，不代表复制回 CPU。
- `.item()` / `.tolist()` 在 CUDA 路径需要让结果对 CPU 可见，可能等待 GPU；图中还省略了有限值、mask 等检查触发的其他标量同步。本图说明职责，不是 CUDA 时间线或性能优化结论。
- 只对本卡前四题保存得分明细时，候选得分及长度会 `.tolist()` 到 CPU；不搬回整个 logits，也不把整个模型来回移动。
- 评测 `all_reduce` 汇总两个计数，区别于训练最后一个微批次反向时的梯度同步。每张卡题数不等时仍须参加最终计数归约，没分到题的卡也一样。
- 评测用 raw_model，避免不等题数的 DDP 前向 buffer 广播。恢复后训练继续使用原 DDP 包装模型。
- 最终 `gather_records()` 管理的是 Python 记录对象，底层分布式后端仍可能做序列化和设备传输；图只把最终 JSON 写盘放在 CPU，不声称对象通信完全不使用 GPU。
- 当前 `--device cpu` 运行时，图中绿色区域的 Tensor 运算也在 CPU；Gloo 完成进程通信。双 CPU 进程能检查分题、归约和恢复训练，不能证明 CUDA/NCCL 性能。

## 代码定位

T/H/S/G 是图上的文件简写，不是新代码入口。下表行号由绘图程序自动提取。

| 图中操作 | 当前源码 |
|---|---|
'''+ '\n'.join(rows)+'''

## 证据与范围

数字读取自已有 [单进程 CPU 报告](../../artifacts/lesson15-hellaswag-cpu.json)和[双进程 CPU/Gloo 报告](../../artifacts/lesson15-hellaswag-ddp-cpu.json)，本轮没有重新训练。单进程 8 道真实验证题在更新 0、2、3 次后分别是 1/8、3/8、3/8；双进程全局 3 题分为 [0,2] 和 [1]，对应本卡总题数为 2、1，全局正确数为 0。这些是小模型流程检查，不是完整基准或 CUDA 实测。

[制图核对记录](../../artifacts/lesson15-device-flow-diagram-check.json)保存源码行号、源码哈希、报告校验、版面检查和输出列表。这里只新增图片与说明，不修改训练/评测算法，不更新学习完成状态。

重建：

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/docs/diagrams/render_hellaswag_device_flow.py --png
```

SVG 与说明使用标准库和项目既有绘图组件；PNG 导出复用 macOS AppKit。
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--png', action='store_true')
    args = parser.parse_args()
    anchors, hashes, reports = evidence()
    svg, layout = draw(anchors, reports)
    ET.fromstring(svg)
    path = OUT / 'gpt2-hellaswag-cpu-gpu-flow.svg'
    path.write_text(svg, encoding='utf-8')
    path.with_suffix('.md').write_text(notes(anchors, reports), encoding='utf-8')
    if args.png:
        export_png(path)
    check = dict(source_anchors=anchors, source_sha256=hashes,
                 canvas=[W, H], text_layout_checks=layout,
                 reports=['artifacts/lesson15-hellaswag-cpu.json','artifacts/lesson15-hellaswag-ddp-cpu.json'],
                 cuda_execution_verified=False, training_rerun=False,
                 outputs=[str(path.relative_to(ROOT)),str(path.with_suffix('.md').relative_to(ROOT))])
    if args.png:
        check['outputs'].append(str(path.with_suffix('.png').relative_to(ROOT)))
    (ROOT / 'artifacts/lesson15-device-flow-diagram-check.json').write_text(
        json.dumps(check,ensure_ascii=False,indent=2)+'\n')
    print(f'Saved {path}; {len(anchors)} code anchors and {len(layout)} text checks verified.')


if __name__ == '__main__':
    main()
