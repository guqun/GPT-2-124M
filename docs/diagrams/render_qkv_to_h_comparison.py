"""QKV → H 的可复算 MHA/GQA 对照；只演示注意力核心，不加载模型。"""
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from render_ddp_allreduce_flow import Drawing, export_png

OUT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 2400, 4580
BLUE, TEAL, INK, MUTED = '#355fc0', '#087e75', '#172b40', '#586c7d'
Q = [[[1, 0], [1, 0]], [[0, 1], [0, 1]],
     [[1, 1], [1, 1]], [[-1, 1], [-1, 1]]]
K = [[[1, 0], [0, 1]], [[0, 1], [1, 0]],
     [[1, 1], [0, 1]], [[1, 0], [1, 1]]]
V = [[[10, 0], [0, 20]], [[2, 4], [6, 8]],
     [[1, 2], [3, 4]], [[5, 0], [0, 5]]]


def attention(q, k, v):
    scores = [[sum(a*b for a, b in zip(qi, kj))/math.sqrt(2)
               for kj in k] for qi in q]
    weights, output = [], []
    for i, row in enumerate(scores):
        m = max(row[:i+1])
        exps = [math.exp(x-m) if j <= i else 0.0 for j, x in enumerate(row)]
        a = [x/sum(exps) for x in exps]
        weights.append(a)
        output.append([sum(a[j]*v[j][d] for j in range(2)) for d in range(2)])
        assert math.isclose(sum(a), 1.0)
        assert all(a[j] == 0 for j in range(i+1, 2))
    assert output[0] == v[0]
    return {'S': scores, 'A': weights, 'H': output}


def examples():
    mha = [attention(Q[h], K[h], V[h]) for h in range(4)]
    gqa = [attention(Q[h], K[2*(h//2)], V[2*(h//2)]) for h in range(4)]
    p = 1/(1+math.exp(-1/math.sqrt(2)))
    assert math.isclose(mha[1]['H'][1][0], p*2+(1-p)*6)
    assert math.isclose(gqa[1]['H'][1][1], p*20)
    # 用未归一化 exp 直接求值，独立核对稳定 softmax 的展开结果。
    for model, mapping in [(mha, [0, 1, 2, 3]), (gqa, [0, 0, 2, 2])]:
        for h, result in enumerate(model):
            exps = [math.exp(s) for s in result['S'][1]]
            expected = [sum(exps[j]*V[mapping[h]][j][d] for j in range(2))/sum(exps)
                        for d in range(2)]
            assert all(math.isclose(a, b) for a, b in zip(expected, result['H'][1]))
    return mha, gqa


def draw():
    mha, gqa = examples()
    d = Drawing()
    def text(x, y, s, size=29, color=INK, bold=400):
        d.text(x, y, s, size, color, bold)
    def band(y, s):
        d.rect(60, y, 2280, 66, '#eef2f6', stroke='none', radius=10)
        text(85, y+45, s, 33, INK, 700)
    def panel(x, y, h, title, color):
        d.rect(x, y, 1100, h, 'white', radius=16)
        text(x+28, y+49, title, 34, color, 700)
    def matrix(x, y, rows, color, cw=150):
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                d.rect(x+j*cw, y+i*60, cw-5, 55,
                       '#edf2ff' if color == BLUE else '#e8f6f1', stroke='none', radius=4)
                label = f'{value:.4f}' if isinstance(value, float) else str(value)
                d.text(x+j*cw+(cw-5)/2, y+i*60+37, label, 27, color, 550, 'middle', mono=True)
        for xx, sign in [(x-12, 1), (x+len(rows[0])*cw+4, -1)]:
            d.fg.append(f'<path d="M {xx+sign*8} {y-5} H {xx} V {y+len(rows)*60} H {xx+sign*8}" fill="none" stroke="{color}" stroke-width="3"/>')
    def vec(v):
        return '['+', '.join(f'{x:.4f}' for x in v)+']'

    text(60, 84, '从 Q、K、V 到 H：GPT-2 与 Llama 3 怎样计算？', 49, INK, 700)
    text(60, 143, '相同的计算规则：QKᵀ → 缩放 → 因果遮罩 → softmax 得 A → AV 得 H', 33, MUTED)
    text(60, 191, '不同的是 K/V 归谁使用。蓝色：GPT-2 风格 MHA；绿色：Llama 3 风格 GQA。', 29, MUTED)
    band(230, '00 · 先确认输入在哪里：Q/K/V 是中间数据，W 才是可训练参数')
    for x, name, color in [(60, 'GPT-2 风格 · 4 Q 头 + 4 K/V 头', BLUE), (1240, 'Llama 3 风格 · 4 Q 头 + 2 K/V 头', TEAL)]:
        panel(x, 320, 275, name, color)
        lines = (['前面：U → Qflat=UWQ+bQ；K、V 同样线性投影',
                  'WQ/WK/WV 各 [8,8]；bias 各 [8]；再拆头',
                  '本图输入 Q、K、V 各 [1,4,2,2]',
                  '位置信息已在网络入口加入；此处没有 RoPE。'] if color == BLUE else
                 ['前面：U → Qflat=UWQ；K/V 投影后拆头',
                  'WQ:[8,8]；WK/WV:[8,4]；无 bias',
                  '本图输入 Q′:[1,4,2,2]；K′/V:[1,2,2,2]',
                  'Q′、K′ 已完成 RoPE；下文简记 Q、K，V 不旋转。'])
        for i, s in enumerate(lines): text(x+28, 419+i*44, s, 27)
    text(85, 642, '教学配置 B=1、T=2、D=8、每头 dₕ=2；下面直接指定投影/位置处理后的数据，不反推 W。', 29, INK, 600)
    text(85, 687, '这些不是实际 GPT-2 / Llama 3 权重或测评结果；两边故意选相同 Q，用来隔离 K/V 分组的区别。', 27, MUTED)
    band(725, '01 · 全部输入数值：每个 2×2 矩阵的两行分别是 Token 位置 0、1')
    text(85, 837, '两边使用相同 Q：每个头下方的两行 = q(位置0)、q(位置1)；两列 = 该头的两个特征。', 28)
    for h in range(4):
        xx = 180+h*560
        text(xx, 892, f'Q 头 {h}', 30, INK, 700)
        matrix(xx, 920, Q[h], BLUE, 150)
    for x, color, title in [(60, BLUE, 'MHA：4 组各自的 K/V'), (1240, TEAL, 'GQA：只保存 2 组 K/V')]:
        panel(x, 1085, 355, title, color)
        text(x+28, 1181, '记法 [[第0行]; [第1行]]；行号都是 Token 位置', 25, MUTED)
        groups = range(4) if color == BLUE else range(2)
        for g in groups:
            idx = g if color == BLUE else g*2
            text(x+28, 1230+g*48, f'组 {g}   K={K[idx]}    V={V[idx]}', 25, color, 600)
        if color == TEAL:
            text(x+28, 1350, '共用的是整组 K/V；Q、注意力权重 A 和输出 H 不共用。', 27, TEAL, 600)
    band(1480, '02 · 对应关系：为每个 Q 头选择 K/V；这一步不再乘投影矩阵')
    for x, color, maps in [(60, BLUE, [0,1,2,3]), (1240, TEAL, [0,0,1,1])]:
        panel(x, 1570, 175, '逐头计算；本图完整展开头 1', color)
        text(x+28, 1671, '   '.join(f'Q{h} → KV{g}' for h,g in enumerate(maps)), 29, color, 600)
        text(x+28, 1715, '选出的 K 和 V 都是 [2,2]；Q头1 也是 [2,2]。', 27, MUTED)
    band(1780, '03 · 头 1：Q × Kᵀ 得匹配分数，再除 √2（因为每头有 2 维）')
    for x, color, idx, data in [(60, BLUE, 1, mha), (1240, TEAL, 0, gqa)]:
        panel(x, 1870, 355, '输入 → 矩阵乘法 → 缩放分数 S [2,2]', color)
        text(x+32, 1960, 'Q头1', 26, MUTED); text(x+390, 1960, 'Kᵀ（转置后）', 26, MUTED)
        matrix(x+32, 1980, Q[1], color, 110)
        text(x+284, 2054, '×', 34)
        matrix(x+365, 1980, [list(r) for r in zip(*K[idx])], color, 110)
        text(x+618, 2054, '/ √2 =', 30)
        matrix(x+770, 1980, data[1]['S'], color, 145)
        dot = '0×0+1×1=1；0×1+1×0=0' if color == BLUE else '0×1+1×0=0；0×0+1×1=1'
        text(x+28, 2158, '位置 1 的 q 与两个位置的 k 点积：'+dot, 26)
        text(x+28, 2200, 'S 的行 = 查询位置 i；列 = 被读取位置 j。', 27, MUTED)
    band(2260, '04 · 加遮罩：i 只能看 j≤i；再对每行 softmax，把分数变成读取比例')
    for x, color, data in [(60, BLUE, mha), (1240, TEAL, gqa)]:
        panel(x, 2350, 390, 'S + M → softmax → A [2,2]', color)
        text(x+28, 2444, 'M=[[0, −∞]; [0, 0]]；未来分数置 −∞，权重变 0。', 27)
        sm = [[data[1]['S'][0][0], '−∞'], data[1]['S'][1]]
        matrix(x+40, 2475, sm, color, 180)
        text(x+440, 2545, 'softmax →', 29)
        matrix(x+700, 2475, data[1]['A'], color, 180)
        text(x+28, 2640, '位置 0 只能看自己，所以 A 第 0 行必为 [1,0]。', 27)
        numerator = 'exp(0.7071)' if color == BLUE else 'exp(0)'
        text(x+28, 2682, f'第 1 行第 0 项：{numerator} / (2.028115+1)', 26)
        text(x+28, 2722, '第 1 行两项和为 1；分母是本行所有允许位置的指数之和。', 25, MUTED)
    band(2775, '05 · A × V 得 H：每个位置按自己的 A，把各位置的 V 加权相加')
    for x, color, idx, data in [(60, BLUE, 1, mha), (1240, TEAL, 0, gqa)]:
        panel(x, 2865, 420, '输入 A、V → 输出 H头1 [2,2]', color)
        matrix(x+30, 2960, data[1]['A'], color, 165)
        text(x+390, 3035, '×', 33)
        matrix(x+450, 2960, V[idx], color, 110)
        text(x+705, 3035, '=', 33)
        matrix(x+760, 2960, data[1]['H'], color, 150)
        a = data[1]['A'][1]
        text(x+28, 3130, f'位置 1：{a[0]:.6f} × {V[idx][0]} + {a[1]:.6f} × {V[idx][1]}', 27, color, 600)
        for dim in range(2):
            text(x+28, 3177+dim*43, f'第 {dim} 维：{a[0]:.6f}×{V[idx][0][dim]} + {a[1]:.6f}×{V[idx][1][dim]} ≈ {data[1]["H"][1][dim]:.6f}', 28)
        text(x+28, 3260, 'H 的行仍是 Token 位置；列是输出特征维度。', 26, MUTED)
    band(3320, '06 · 所有头用同一规则算完：两边仍各有 4 个 H 头，并没有少算 Q 头')
    for x, color, data in [(60, BLUE, mha), (1240, TEAL, gqa)]:
        panel(x, 3410, 380, '完整输出 H [1,4,2,2]；逐头列出两行', color)
        text(x+28, 3502, '头号        位置 0 输出                  位置 1 输出', 27, MUTED)
        for h, result in enumerate(data):
            text(x+28, 3553+h*47, f'H{h}       {vec(result["H"][0])}        {vec(result["H"][1])}', 26, color, 600)
        text(x+28, 3761, '后续：每个位置拼接四头 → C [1,2,8] → 输出投影。', 27)
    band(3830, '07 · 优势与代价：GQA 减少 K/V 投影和缓存，MHA 保留更多独立 K/V 表示')
    d.rect(60, 3920, 2280, 370, 'white', radius=16)
    rows = [
        ('本例统计（只比较头数；不计 bias）', 'MHA：4 KV 头', 'GQA：2 KV 头'),
        ('WK + WV 的权重元素', '2×8×8 = 128', '2×8×4 = 64，减少 50%'),
        ('每层 KV Cache：2×B×T×KV头数×dₕ', '2×1×2×4×2 = 32', '2×1×2×2×2 = 16，减少 50%'),
        ('同为 FP32 的 KV 缓存字节（不含开销）', '32×4 = 128 字节', '16×4 = 64 字节'),
        ('Q 头数 / 每层 H 形状', '4 / [1,4,2,2]', '4 / [1,4,2,2]，不变'),
        ('QKᵀ、AV 的理论乘加规模', '按 4 个 Q 头计算', '仍按 4 个 Q 头计算，不减半'),
    ]
    for i, row in enumerate(rows):
        yy = 3975+i*53
        for xx, val, col in zip([90, 1070, 1640], row, [INK, BLUE, TEAL]):
            text(xx, yy, val, 27, col, 650 if i==0 else 400)
    text(85, 4340, 'GQA：更少的 K/V 投影输出与缓存读写，尤其有利于逐 Token 解码；缓存可保留 2 头，无需存成 4 头。', 29, TEAL, 600)
    text(85, 4386, 'MHA：每个 Q 头都有独立 K/V，表示自由度更多；GQA 用共享换取效率，质量取决于训练，不保证输出相同。', 28)
    text(85, 4431, '注意：本例 50% 是 K/V 的节省比例，不是整模型参数/显存/耗时减半；整段加速需要同条件实测。', 28, '#98601f', 600)
    text(85, 4481, '本图从注意力核心输入开始；不把 RoPE 的位置优势混算进 GQA。数字显示取舍，不证明哪种模型更准确。', 27, MUTED)
    text(85, 4532, '依据：OpenAI GPT-2 attn；Meta Llama 3 Attention / repeat_kv；GQA 论文（2023）。省略 Dropout；显示四舍五入。', 25, MUTED)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}"><title>QKV 到 H：GPT-2 MHA 与 Llama GQA 数值计算对照</title><style>text {{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}}</style><rect width="{WIDTH}" height="{HEIGHT}" fill="#f5f8fb"/>'+''.join(d.bg+d.edges+d.fg)+'</svg>'
    ET.fromstring(svg)
    return svg, {'scope': 'synthetic post-projection attention inputs; GQA Q/K after RoPE',
                 'Q': Q, 'MHA_K': K, 'MHA_V': V, 'GQA_K': [K[0],K[2]],
                 'GQA_V': [V[0],V[2]], 'GQA_mapping': [0,0,1,1],
                 'MHA_results': mha, 'GQA_results': gqa,
                 'kv_cache_elements': {'MHA': 32, 'GQA': 16}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--png', action='store_true')
    args = parser.parse_args()
    svg, report = draw()
    dest = OUT/'gpt2-llama3-qkv-to-h-detailed.svg'
    dest.write_text(svg)
    (OUT.parents[1]/'artifacts/qkv-to-h-detailed-check.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    if args.png:
        export_png(dest)
    print(dest)
