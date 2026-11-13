"""QK-Norm 数值流转：保留原值与计算 r 两条支路，汇合后交给 RoPE。"""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from render_ddp_allreduce_flow import Drawing, export_png
from render_llama3_qwen3_comparison import calculate

OUT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 2400, 2940
BLUE, TEAL, INK, MUTED, RED = '#355fc0', '#087e75', '#172b40', '#586c7d', '#cd4141'


def draw():
    d = Drawing()
    def text(x, y, s, size=28, color=INK, bold=400):
        d.text(x, y, s, size, color, bold)
    def box(x, y, w, h, title, lines, color=TEAL, fill='white', stroke='#dce4ec', size=28):
        d.rect(x, y, w, h, fill, stroke=stroke, radius=18)
        text(x+24, y+43, title, 29, color, 700)
        for i, s in enumerate(lines):
            text(x+24, y+88+i*38, s, size)
    def arrow(points, color=TEAL):
        d.arrow(points, color, width=4)
    def band(y, title):
        d.rect(60, y, 2280, 64, '#eaf0f5', stroke='none', radius=10)
        text(84, y+44, title, 32, INK, 700)
    def fmt(row):
        return '['+', '.join(f'{v:.4f}' for v in row)+']'

    text(60, 84, 'QK-Norm：原来的数，究竟怎样走到 RoPE？', 49, INK, 700)
    text(60, 140, '沿箭头读数：原值分成两路 → 一路保留，一路算 r → 汇合做除法 → 乘 γ → 交给 RoPE', 29, MUTED)
    box(60, 185, 2280, 150, '先确定起点：这里的 q、k 已经完成投影和拆头', [
        'Block 输入 → RMSNorm → U → Q=U WQ、K=U WK → 拆成各头 → 本图的 q、k → QK-Norm → RoPE',
        '下方数字是人工指定的投影后数据，不是 Token ID、原始 Embedding 或权重。r 是每个向量算出的一个数。'], size=28)
    box(60, 365, 2280, 145, '对照 · 原始 Llama 3 跳过整个 QK-Norm，直接送入 RoPE', [
        'q₁=[3,4]  →  RoPE(q₁, 位置1)；k₀=[1,0]  →  RoPE(k₀, 位置0)；k₁=[0,2]  →  RoPE(k₁, 位置1)',
        'Qwen3 则先走下面三条数值通道。下标表示 Token 位置；这里观察同一个 Q 头及其对应的 K 头。'], BLUE, '#edf2ff', size=27)
    band(550, 'Qwen3 · 三个向量各算各的 r；下方不跨位置求平均，也不把 Q 与 K 混在一起')

    vectors = [[3., 4.], [1., 0.], [0., 2.]]
    names = ['q₁ · 查询位置 1', 'k₀ · 被读取位置 0', 'k₁ · 被读取位置 1']
    originals = ['[3, 4]', '[1, 0]', '[0, 2]']
    means = [sum(v*v for v in row)/2 for row in vectors]
    rms = [math.sqrt(m) for m in means]
    normalized = [[v/r for v in row] for row, r in zip(vectors, rms)]
    results = {'llama3': calculate(False), 'qwen3': calculate(True)}

    for i, (x, row, r, norm) in enumerate(zip([60, 840, 1620], vectors, rms, normalized)):
        box(x, 650, 720, 130, '① 起始向量：'+names[i], [originals[i]+'    ← 两个特征数值'], TEAL, '#e8f6f1', size=35)
        # 左路只保留原值；右路计算一个标量 r。两路在除法处汇合。
        arrow([(x+180,780),(x+135,820),(x+135,1030)], BLUE)
        d.rect(x+10, 848, 245, 48, '#f5f8fb', stroke='none', radius=5)
        text(x+24, 881, '左路：原值保留', 25, BLUE, 650)
        box(x+10, 1030, 245, 170, '留到除法时使用', [originals[i], '不平方、不改值'], BLUE, '#edf2ff', size=27)
        arrow([(x+135,1200),(x+135,1440)], BLUE)
        text(x+20, 1350, '分子', 26, BLUE, 650)
        arrow([(x+540,780),(x+510,820)])
        a,b = [int(v) for v in row]
        steps = [
            ('② 每一维平方', f'[{a}², {b}²] = [{a*a}, {b*b}]'),
            ('③ 两个平方相加', f'{a*a} + {b*b} = {a*a+b*b}'),
            ('④ 除以维数 2', f'{a*a+b*b} ÷ 2 = {means[i]:g}'),
            ('⑤ 开平方，得到 r', f'√{means[i]:g} ≈ {r:.6f}'),
        ]
        for j,(title,line) in enumerate(steps):
            y=820+j*145
            box(x+290,y,430,115,title,[line],size=28)
            if j<3: arrow([(x+505,y+115),(x+505,y+145)])
        arrow([(x+505,1370),(x+505,1440)])
        text(x+300, 1410, '分母 r →', 25, TEAL, 650)
        box(x,1440,720,165,'⑥ 原来的两维，各自除以同一个 r',[
            f'[{a} ÷ {r:.6f},  {b} ÷ {r:.6f}]',
            '= '+fmt(norm)],TEAL,'white',RED,30)
        arrow([(x+360,1605),(x+360,1660)])
        gamma='γQ' if i==0 else 'γK'
        box(x,1660,720,160,'⑦ 逐维乘参数 '+gamma+'=[1,1]',[
            f'[1 × {norm[0]:.4f},  1 × {norm[1]:.4f}]',
            '= '+fmt(norm)],TEAL,'white',RED,29)
        arrow([(x+360,1820),(x+360,1875)])
        box(x,1875,720,150,'⑧ 最终输出 = RoPE 的向量输入',[
            ('q̂₁ = ' if i==0 else ('k̂₀ = ' if i==1 else 'k̂₁ = '))+fmt(norm),
            '仍是 2 维；此时还没有旋转。'],TEAL,'#e8f6f1',TEAL,29)
        arrow([(x+360,2025),(x+360,2110)])
        text(x+390,2076,'把这两个数传下去',25,TEAL)
        pos=0 if i==1 else 1
        rotated=results['qwen3']['rotated_q'] if i==0 else results['qwen3']['rotated_k'][i-1]
        box(x,2110,720,225,'下一环节 · RoPE（不属于 QK-Norm）',[
            f'向量输入：{fmt(norm)}；位置：{pos}',
            f'本例二维头，旋转角度 = {pos} 弧度。',
            '旋转后的输出：'+fmt(rotated),
            '随后用旋转后的 Q、K 算注意力分数。'],TEAL,'white',size=26)

    band(2380,'读图时抓住三件事')
    text(84,2490,'1  蓝色支路的原值一直保留：平方值只用来算 r，绝不能把 [9,16] 当作除法的分子。',30,INK,650)
    text(84,2546,'2  三个 r 分别是 3.535534、0.707107、1.414214。r 不是权重，也不会作为第三个向量送入 RoPE。',29)
    text(84,2602,'3  γQ 与 γK 是可学习参数，本例都取 [1,1]。K 的两行共用 γK，但各自计算 r；V 绕过本环节。',29)
    box(60,2650,2280,160,'参数、精度与目的',[
        '真实计算：先给平方均值加 ε>0，再开根号。本例非零向量为便于手算忽略 ε；所有展示小数都经过舍入。',
        'γ 从零训练时通常初始化为全 1，加载预训练模型时使用学到的值。此步控制 Q/K 尺度，同时保留逐维学习能力。'],size=28)
    text(60,2863,'来源：Meta Llama 3；Transformers Qwen3Attention / Qwen3RMSNorm；Raschka Qwen3。教学数值，非真实模型实测。',25,MUTED)
    text(60,2910,'RoPE 的二维计算：[a cosφ − b sinφ, a sinφ + b cosφ]；输入 a、b 就是上面第⑧步输出的两维。',26,MUTED)

    assert means == [12.5,.5,2.]
    for v in normalized: assert math.isclose(sum(a*a for a in v)/2,1.)
    for a,b in zip(normalized[0],results['qwen3']['q']): assert math.isclose(a,b)
    for a,b in zip(normalized[1:],results['qwen3']['k']):
        assert all(math.isclose(v,w) for v,w in zip(a,b))
    results['step03']={'inputs':vectors,'mean_squares':means,'rms':rms,'normalized':normalized,
        'gamma_q':[1,1],'gamma_k':[1,1],'epsilon_omitted_for_example':True,
        'alternative_gamma_q':[2,.5],'alternative_q':[normalized[0][0]*2,normalized[0][1]*.5]}
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}"><title>QK-Norm 原始 q/k、尺度 r 与 RoPE 输入的数值流转</title><style>text {{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}}</style><rect width="{WIDTH}" height="{HEIGHT}" fill="#f5f8fb"/>'+''.join(d.bg+d.edges+d.fg)+'</svg>'
    ET.fromstring(svg)
    return svg,results


if __name__=='__main__':
    svg,report=draw()
    path=OUT/'llama3-qwen3-qknorm-step03-worked.svg'
    path.write_text(svg)
    (OUT.parents[1]/'artifacts/qknorm-step03-worked-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    export_png(path)
    print(path)
