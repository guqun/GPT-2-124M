"""数据准备代码流转图：从实际源码提取行号，从已有 demo 报告读取 ID。

只生成 SVG / Markdown / PNG，不下载语料、不运行模型、不改训练代码。
重建：python docs/diagrams/render_data_preparation_flow.py --png
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
SOURCE = ROOT / "prepare_fineweb.py"
REPORT = ROOT / "artifacts/lesson12-data-preparation-demo.json"
WIDTH, HEIGHT = 2560, 3600
PURPLE, PURPLE_BG = "#7461a8", "#f2eef9"


def evidence():
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    lines = source.splitlines()
    fragments = {
        "main": "def main(", "read": "def read_documents(",
        "encode": "def encode_document(", "write": "def write_shards(",
        "save": "def save_shard(", "generator": "encoded = (encode_document(",
        "invoke": "result = write_shards(", "init": "buffer = np.empty(",
        "outer": "for doc_index, tokens in enumerate(", "count": "count = len(tokens)",
        "offset": "offset = 0", "while": "while offset < count:",
        "take": "take = min(", "copy": "buffer[used:used + take] =",
        "full": "if used == shard_size:", "tail": "if used > 0:",
        "return": 'return {"total_tokens": total',
        "manifest": 'report_path = output_dir / "manifest.json"',
        "entry": 'if __name__ == "__main__":',
    }
    anchors = {}
    for key, fragment in fragments.items():
        hits = [i for i, line in enumerate(lines, 1) if line.strip().startswith(fragment)]
        assert len(hits) == 1, (key, hits)
        anchors[key] = hits[0]
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["source"]["is_fineweb"] is False
    assert report["training_run"] is False
    assert (report["shard_size"], report["total_tokens"], report["eot_id"]) == (5, 8, 50256)
    docs = [row["first_ids"] for row in report["documents"]]
    assert docs == [[50256, 15496, 995, 13], [50256, 10248, 3329, 13]]
    assert [row["tokens"] for row in report["documents"]] == [4, 4]
    assert [(s["split"], s["tokens"]) for s in report["shards"]] == [("val", 5), ("train", 3)]
    ids = [token for doc in docs for token in doc]
    expected_shards = [ids[:5], ids[5:]]

    # 从既有 ID 复算图中三次搬运；同时核对总流区间，避免手填数字漂移。
    trace = []
    buffer = [None] * 5
    used = 0
    for document, tokens in enumerate(docs):
        offset = 0
        while offset < len(tokens):
            old_used, old_offset = used, offset
            take = min(5 - used, len(tokens) - offset)
            buffer[used:used + take] = tokens[offset:offset + take]
            used += take
            offset += take
            full = used == 5
            trace.append(dict(document=document, old_used=old_used, old_offset=old_offset,
                              used=used, offset=offset, take=take, full=full,
                              buffer=buffer.copy(), valid=used))
            if full:
                used = 0
    assert [row["take"] for row in trace] == [4, 1, 3]
    assert [row["full"] for row in trace] == [False, True, False]
    assert used == 3
    assert trace[1]["buffer"] == expected_shards[0]
    assert trace[2]["buffer"][:3] == expected_shards[1]
    for shard, expected in zip(report["shards"], expected_shards):
        assert shard["stream_end"] - shard["stream_start"] == len(expected)
    return anchors, report, docs, trace, expected_shards


class Drawing(BaseDrawing):
    def node(self, x, y, w, h, title, lines=(), color=INK, fill="white", ref=None,
             title_size=28, body_size=25):
        self.rect(x, y, w, h, fill, radius=18)
        self.text(x + 25, y + 43, title, title_size, color, 650)
        for i, line in enumerate(lines):
            self.text(x + 25, y + 83 + 37 * i, line, body_size, MUTED)
        if ref:
            self.text(x + w - 3, y - 12, ref, 20, color, anchor="end")

    def diamond(self, cx, cy, rx, ry, title, sub, ref=None):
        self.fg.append(f'<polygon points="{cx},{cy-ry} {cx+rx},{cy} {cx},{cy+ry} {cx-rx},{cy}" '
                       f'fill="{AMBER_BG}" stroke="#d6ba96" stroke-width="2"/>')
        self.text(cx, cy - 6, title, 29, INK, 650, "middle")
        self.text(cx, cy + 32, sub, 23, MUTED, anchor="middle")
        if ref:
            self.text(cx + rx, cy - ry - 10, ref, 20, AMBER, anchor="end")

    def cells(self, x, y, values, valid, w=580, h=62):
        gap = 7
        width = (w - 4 * gap) / 5
        for i, value in enumerate(values):
            active = i < valid
            color = TEAL if value == 50256 else BLUE
            self.rect(x + i * (width + gap), y, width, h,
                      TEAL_BG if active and value == 50256 else BLUE_BG if active else "#f0f2f5",
                      stroke="none", radius=10)
            self.text(x + i * (width + gap) + width / 2, y + 40,
                      "?" if value is None else value, 25, color if active else "#aeb6bf",
                      600 if active else 400, "middle", mono=True)

    def finish(self):
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">
<title id="title">数据准备：正文如何流转成 Token 分片文件</title>
<desc id="desc">对应 prepare_fineweb.py。左侧 main 读取正文、建立分词器和惰性生成器、调用分片函数、保存清单。右侧 for 每次拉取生成器才编码一篇，检查 uint16 范围；内层 while 按剩余空间与文章剩余长度取较小值，填满时保存并只重置 used，保留 offset。外层结束后保存有效尾片，返回 main。底部用实际 demo 的八个 ID 展开 take 为四、一、三的三轮，得到五个和三个 Token 的两片。只验证教学文本，真实 FineWeb 下载尚未成功；没有模型训练。</desc>
<style>text {{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}}</style>
<rect width="{WIDTH}" height="{HEIGHT}" fill="#f7f9fc"/>
{''.join(self.bg)}{''.join(self.edges)}{''.join(self.fg)}
</svg>'''


def draw(a, report, docs, trace, shards):
    d = Drawing()
    ref = lambda key: f"L{a[key]}"
    d.text(64, 88, "数据准备：正文如何变成 Token 分片文件", 56, weight=700)
    d.text(64, 142, "prepare_fineweb.py  /  沿代码看输入、调用、循环判断与输出", 30, MUTED)
    for x, label, color in [(64, "控制流", BLUE), (270, "编码 / 数据", TEAL), (565, "条件判断", AMBER), (835, "调用 / 返回", PURPLE)]:
        d.rect(x, 172, 16, 16, color, stroke="none", radius=4)
        d.text(x + 28, 190, label, 24, MUTED)
    d.text(2496, 190, "P1—P5 对应源码；L 表示行号", 24, MUTED, anchor="end")
    d.rect(64, 223, 2432, 74, "white", radius=15)
    d.text(91, 270, "输入：多篇正文", 28, BLUE, 650)
    d.text(535, 270, "输出：一维 uint16 的 .npy 分片 + manifest.json", 28, TEAL, 650)
    d.text(2464, 270, "本节不创建模型、不反传、不更新参数", 25, MUTED, anchor="end")

    d.rect(64, 325, 820, 2295, "white", background=True, radius=24)
    d.rect(1008, 325, 1488, 2295, "#fcfdfd", background=True, radius=24)
    d.text(100, 377, "01  main()：调用与收尾", 35, BLUE, 700)
    d.text(100, 410, f"入口 {ref('entry')} → main() {ref('main')}", 23, MUTED)
    d.text(1044, 377, "02  展开 write_shards()：两层循环", 35, BLUE, 700)
    d.text(2470, 410, f"{ref('write')} · 右侧返回后，main 才继续", 23, MUTED, anchor="end")

    # 左侧正常执行顺序。中部淡色说明卡不是 main 的执行节点。
    d.node(100, 450, 748, 140, "解析参数与检查分片容量", [
        "默认：source=demo；S=shard_size=5",
        "FineWeb 小样本模式默认 S=2048",
    ], BLUE, BLUE_BG, ref('main'))
    d.node(100, 650, 748, 250, "P1  read_documents(source, cache_dir)", [
        "demo → 两句教学正文",
        "fineweb-preview → 本地缓存 / 官方 20 条记录",
        "真实记录须通过完整性、正文非空与未截断检查",
        "返回：texts（字符串列表）+ source（来源）",
    ], TEAL, TEAL_BG, ref('read'), title_size=26, body_size=23)
    d.node(100, 960, 748, 120, "准备 GPT-2 分词器与输出目录", [
        'tiktoken.get_encoding("gpt2")；没有模型权重',
    ], BLUE, "white", title_size=27, body_size=23)
    d.node(100, 1140, 748, 160, "创建 encoded 生成器", [
        "(encode_document(text, tokenizer) for text in texts)",
        "这里只建生成器；还没有把所有文章编码出来。",
    ], PURPLE, PURPLE_BG, ref('generator'), body_size=22)
    d.node(100, 1360, 748, 130, "调用 write_shards(encoded, output_dir, S)", [
        "main 等待；右侧用 for 循环逐篇拉取生成器。",
    ], PURPLE, PURPLE_BG, ref('invoke'), title_size=26, body_size=23)
    for start, end in [(590, 650), (900, 960), (1080, 1140), (1300, 1360)]:
        d.arrow([(474, start), (474, end)], BLUE)
    d.arrow([(848, 1425), (942, 1425), (942, 430), (1685, 430), (1685, 450)], PURPLE, dashed=True)
    d.text(900, 1393, "调用", 23, PURPLE, anchor="middle")

    d.text(100, 1550, "读图注释 · 以下是右侧变量与写盘函数的说明", 23, MUTED)
    d.node(100, 1580, 748, 245, "三个核心变量各管什么", [
        "used：当前分片已填的有效 ID 数；满片后归零",
        "offset：当前文章已写出的 ID 数；新文章归零",
        "count：当前文章的 ID 总数，包含前置 EOT",
        "S：分片容量，不是 batch 大小，也不是 T",
    ], INK, "#f5f7fa", body_size=24)
    d.node(100, 1885, 748, 280, "P5  save_shard(count)：两处共用", [
        "index = len(shards)：从 0 开始编号",
        "第 0 片为 val；其他片为 train",
        "np.save(path, buffer[:count])：只保存有效位置",
        "将文件名、Token 数、总流区间与校验值加入清单",
        "本函数不重置 used / offset；由调用处控制",
    ], TEAL, TEAL_BG, ref('save'), title_size=27, body_size=23)
    d.node(100, 2225, 748, 135, "检查失败 → 报错退出", [
        "容量非法、目录非空或数据无效，不沿成功路径继续。",
        "真实样本下载失败，不会自动回退成 demo 成功。",
    ], INK, "#f5f7fa", body_size=23)
    d.node(100, 2440, 748, 130, "返回 main → 保存 manifest.json → 打印", [
        "来源 + 分词设置 + result；本次数据准备结束。",
    ], PURPLE, PURPLE_BG, ref('manifest'), title_size=26, body_size=23)

    # 右侧外层 for 通过 next() 才触发编码，已拿到正文后逐篇处理。
    d.node(1210, 450, 950, 145, "初始化一个分片缓冲区", [
        "buffer = np.empty(S, dtype=uint16)；used=0",
        "total=0；shards=[]；documents=[]；输出目录须为空",
    ], BLUE, BLUE_BG, ref('init'), body_size=24)
    d.diamond(1685, 720, 360, 82, "外层 for：还有下一篇正文？", "enumerate(encoded_documents)", ref('outer'))
    d.arrow([(1685, 595), (1685, 638)], BLUE)
    d.text(1720, 835, "是：拉取生成器，才执行下面的编码", 23, TEAL)
    d.node(1210, 855, 950, 185, "P2 / P3  encode_document(text, tokenizer)", [
        "ids = [50256] + tokenizer.encode_ordinary(text)",
        "先检查 0 ≤ ID < 65536，再转 np.uint16",
        "返回：tokens，一维数组 [count]，含文章开头的 EOT",
    ], TEAL, TEAL_BG, ref('encode'), title_size=28, body_size=24)
    d.arrow([(1685, 802), (1685, 855)], BLUE)
    d.node(1210, 1095, 950, 145, "登记当前文章，开始从它的第 0 个 ID 写", [
        "count=len(tokens)；记录文档区间；total += count",
        "offset=0；保留上一篇留下的 used，继续填同一分片",
    ], BLUE, "white", ref('count'), body_size=24)
    d.arrow([(1685, 1040), (1685, 1095)], BLUE)
    d.diamond(1685, 1370, 340, 82, "内层 while：本文还没有写完？", "offset < count", ref('while'))
    d.arrow([(1685, 1240), (1685, 1288)], BLUE)
    d.text(1720, 1480, "是：本轮只搬运能放下的那一段", 23, BLUE)
    d.node(1210, 1500, 950, 205, "P4  计算 take → 复制 ID → 前进两个游标", [
        "take = min(S - used, count - offset)",
        "buffer[used:used+take] = tokens[offset:offset+take]",
        "used += take；offset += take",
    ], BLUE, BLUE_BG, ref('take'), title_size=28, body_size=25)
    d.arrow([(1685, 1452), (1685, 1500)], BLUE)
    d.diamond(1685, 1830, 300, 75, "当前分片写满了吗？", "used == S", ref('full'))
    d.arrow([(1685, 1705), (1685, 1755)], BLUE)
    d.text(1720, 1940, "是", 24, TEAL)
    d.node(1210, 1960, 950, 145, "P5  save_shard(used) → used=0", [
        "保存完整分片，复用 buffer；不清零 offset",
        "本文剩余 ID 继续写下一片，返回 while 判断",
    ], TEAL, TEAL_BG, f"L{a['full'] + 1}—{a['full'] + 2}", body_size=24)
    d.arrow([(1685, 1905), (1685, 1960)], BLUE)

    # 内层两条分支合流，均返回 while；未满时不保存，等待剩余文章/下一篇。
    d.arrow([(2160, 2032), (2320, 2032), (2320, 1370), (2025, 1370)], BLUE)
    d.arrow([(1985, 1830), (2320, 1830)], BLUE)
    d.text(2190, 1798, "否：不保存", 23, BLUE, anchor="middle")
    d.text(2336, 1540, "回到", 22, BLUE)
    d.text(2336, 1574, "while", 22, BLUE)
    # 本文完结回到外层，保持当前片已填位置；只有全部文章耗尽才进入尾片判断。
    d.arrow([(1345, 1370), (1100, 1370), (1100, 720), (1325, 720)], BLUE)
    d.text(1165, 1310, "否：本文写完", 22, BLUE, anchor="middle")
    d.text(1165, 1344, "取下一篇", 22, BLUE, anchor="middle")
    d.arrow([(2045, 720), (2440, 720), (2440, 2215), (1995, 2215)], MUTED)
    d.text(2220, 690, "否：全部文章已处理", 23, MUTED, anchor="middle")
    d.diamond(1685, 2215, 310, 75, "所有文章结束：还有有效尾部？", "used > 0", ref('tail'))
    d.text(1720, 2317, "是", 23, TEAL)
    d.node(1210, 2335, 950, 75, "P5  save_shard(used)：只写 buffer[:used]", [], TEAL, TEAL_BG,
           f"L{a['tail'] + 1}", title_size=27)
    d.arrow([(1685, 2290), (1685, 2335)], BLUE)
    d.node(1210, 2465, 950, 105, "返回 result：total_tokens / documents / shards", [
        "所有 .npy 已保存；将统计与文件清单交回 main。",
    ], PURPLE, PURPLE_BG, ref('return'), title_size=27, body_size=24)
    d.arrow([(1685, 2410), (1685, 2465)], BLUE)
    d.arrow([(1375, 2215), (1150, 2215), (1150, 2515), (1210, 2515)], MUTED)
    d.text(1170, 2270, "否", 22, MUTED)
    d.text(1170, 2302, "无尾片", 22, MUTED)
    d.arrow([(1210, 2540), (848, 2540)], PURPLE, dashed=True)
    d.text(944, 2510, "返回", 23, PURPLE, anchor="middle")

    # 底部运行轨迹，从实际报告读取 ID，数字逐轮复算；颜色始终强调 EOT。
    d.rect(64, 2680, 2432, 755, "white", radius=24, background=True)
    d.text(100, 2738, "03  对照本次 demo：2 篇正文 → 8 个 Token → 5 + 3 两片", 35, INK, 700)
    d.text(100, 2780, f"文档 0：{docs[0]}     文档 1：{docs[1]}     S=5", 25, MUTED)
    columns = [(100, "本轮"), (395, "按规则计算 take"), (965, "游标怎样前进"), (1360, "buffer：仅彩色格有效"), (2010, "判断与动作")]
    for x, text in columns:
        d.text(x, 2835, text, 25, INK, 650)
    labels = ["① 文档 0", "② 文档 1 · 前段", "③ 文档 1 · 剩余"]
    for i, row in enumerate(trace):
        y = 2860 + i * 123
        if i % 2 == 0:
            d.rect(88, y - 7, 2384, 109, "#f7f9fc", stroke="none", radius=12)
        d.text(100, y + 35, labels[i], 25, BLUE, 650)
        formula = f"min(5 - {row['old_used']}, 4 - {row['old_offset']}) = {row['take']}"
        d.text(395, y + 35, formula, 27, INK, 550, mono=True)
        d.text(965, y + 25, f"used：{row['old_used']} → {row['used']}", 25, MUTED)
        d.text(965, y + 66, f"offset：{row['old_offset']} → {row['offset']}", 25, MUTED)
        d.cells(1360, y + 4, row['buffer'], row['valid'])
        if i == 0:
            action = ["4 ≠ 5，不保存", "本文已完，继续下一篇"]
        elif i == 1:
            action = ["5 == 5，保存 val 第 0 片", "used → 0；offset 保留为 1"]
        else:
            action = ["本文结束，所有文档也结束", "used=3 > 0，保存 train 尾片"]
        for j, text in enumerate(action):
            d.text(2010, y + 26 + 40 * j, text, 23, TEAL if i else MUTED, 550)
    d.text(100, 3248, "灰色格可能仍有旧数据，代码没有清零缓冲区；保存 buffer[:used] 时不会把这些旧值写入尾片。", 25, MUTED)
    d.node(100, 3280, 1140, 120, report['shards'][0]['file'], [
        str(shards[0]) + "  ·  5 个 Token",
    ], TEAL, TEAL_BG, title_size=28, body_size=26)
    d.node(1272, 3280, 1188, 120, report['shards'][1]['file'], [
        str(shards[1]) + "  ·  3 个 Token；不补到 5",
    ], TEAL, TEAL_BG, title_size=28, body_size=26)
    d.text(64, 3490, "数据依据：lesson12-data-preparation-demo.json。上图仅为已实跑的两句教学文本；真实 FineWeb 预览下载尚未成功。", 25, MUTED)
    d.text(64, 3534, "边界：这里会保存尾片；“不足一个 batch 时跳过”属于后续分片读取。下一块才接入 torch.long → x/y → 模型。", 25, MUTED)
    return d.finish()


def notes(a, report, docs, trace, shards):
    return f'''# 数据准备：代码流转图

配套 [prepare_fineweb.py](../../prepare_fineweb.py) 与 [数据准备讲义](../../lesson12-data-preparation.md)。

[打开 PNG](gpt2-data-preparation-flow.png) · [打开可缩放 SVG](gpt2-data-preparation-flow.svg)

![正文到 Token 分片：代码调用与循环判断](gpt2-data-preparation-flow.png)

## 阅读顺序

1. 左侧 `main()` 先读取正文，再创建分词器与生成器，随后调用右侧 `write_shards()`。
2. 紫色虚线表示调用与返回；右侧不是另一个并行任务。左侧的变量说明、写盘函数说明是读图注释，不是 main 中额外执行的步骤。
3. 右侧的外层 `for` 拉取生成器，每次才调用 `encode_document()` 编码一篇；内层 `while` 把当前文章写完。一篇文章可以跨多个分片。
4. `used == S` 时调用 `save_shard(used)`，再只把 used 清零；offset 保留，回到内层判断。`offset == count` 时换下一篇，offset 重置为 0，used 保留。
5. 全部文章处理完才判断 `used > 0`，保存有效尾部，返回 main 写 manifest。恰好写满时 used=0，不多保存一个空文件。
6. 底部按既有 demo 报告展开实际 8 个 ID，三轮 take 为 4、1、3，最终为 5+3 两片。灰色缓冲区位置可能含旧值，只是当前无效，不代表代码真的清零了它们。

## 源码对应

| 图中节点 | prepare_fineweb.py 行号 | 输入 → 输出 |
| --- | --- | --- |
| 主入口 | L{a['entry']} → L{a['main']} | 命令行参数 → 准备与调用 |
| P1 读取正文 | L{a['read']} | 来源选择 → texts、source |
| 创建生成器 | L{a['generator']} | texts、tokenizer → encoded；此时没有遍历编码 |
| 调用分片器 | L{a['invoke']} → L{a['write']} | 逐篇生成器、目录、容量 → 分片结果 |
| 分片初始化 | L{a['init']} | S → 一维 uint16 缓冲区；used=0 |
| 外层 for | L{a['outer']} | 生成器 → 下一篇的 tokens，或耗尽 |
| P2/P3 编码 | L{a['encode']} | 正文 → EOT+普通编码 ID → 范围检查 → uint16 |
| 新文章初始化 | L{a['count']}—{a['offset']} | tokens → count、文档区间、offset=0 |
| 内层 while | L{a['while']} | offset < count → 继续写本文，或取下一篇 |
| P4 取数与复制 | L{a['take']}—{a['copy'] + 2} | 两种剩余量 → take → 修改 buffer、used、offset |
| 完整片判断 | L{a['full']} | used == S → 保存并把 used 清零 |
| P5 写盘函数 | L{a['save']} | buffer[:count] → .npy、文件清单、校验值 |
| 最终尾片判断 | L{a['tail']} | 全部文章结束后的 used → 是否保存短尾片 |
| 返回 / 保存清单 | L{a['return']} → L{a['manifest']} | result → manifest.json 与终端输出 |

行号由绘图脚本从当前源码提取；再次运行绘图脚本会重新定位。P1—P5 与源码说明字符串一致。

## 两个不同的边界

- 文档的边界由前置 EOT 标记；文件的边界由分片容量决定，不会因为换文件而再加一个 EOT。
- 本例第二篇的 EOT 被放在第 0 片，正文位于第 1 片。按照原课，第 0 片为 val，其余为 train；这不保证完整文档级别的 train/val 隔离。
- 准备阶段把短尾片保存下来。训练读取阶段“片内不足一个 batch 时跳过”尚未接入本脚本，不能把它画成这里丢掉剩余 ID。
- 文件存储类型是整数 uint16，不是 BF16 浮点计算；下一块读取后才转换 torch.long、构造 x/y、送入模型。

## 验证与重建

数值来自[既有 demo 报告](../../artifacts/lesson12-data-preparation-demo.json)，默认容量 {report['shard_size']}，合计 {report['total_tokens']} 个 Token。绘图脚本读取 ID 后复算三次搬运并核对分片区间；不下载数据、不运行模型，也不把演示文本当作真实 FineWeb 实跑。源码与数值核验记录见[图示核对结果](../../artifacts/lesson12-data-preparation-diagram-check.json)。

```bash
python docs/diagrams/render_data_preparation_flow.py --png
```

复用项目既有 SVG 组件和 macOS AppKit PNG 导出方式。源码改动后应重建图片；课程与本地改编的来源见[讲义](../../lesson12-data-preparation.md)。
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", action="store_true", help="使用项目已有 macOS AppKit 导出器")
    args = parser.parse_args()
    a, report, docs, trace, shards = evidence()
    svg = draw(a, report, docs, trace, shards)
    ET.fromstring(svg)
    path = OUT / "gpt2-data-preparation-flow.svg"
    path.write_text(svg, encoding="utf-8")
    path.with_suffix(".md").write_text(notes(a, report, docs, trace, shards), encoding="utf-8")
    check = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "report": str(REPORT.relative_to(ROOT)),
        "report_sha256": hashlib.sha256(REPORT.read_bytes()).hexdigest(),
        "source_lines": a, "trace": trace, "shard_ids": shards,
        "fineweb_run": False, "training_run": False,
    }
    (ROOT / "artifacts/lesson12-data-preparation-diagram-check.json").write_text(
        json.dumps(check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.png:
        export_png(path)
    print("Checked source locations, three copy iterations, offsets and final shard IDs.")
    print(path)


if __name__ == "__main__":
    main()
