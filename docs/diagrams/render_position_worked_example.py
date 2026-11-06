"""位置编码的逐项数值计算图；两个局部算例，非端到端模型比较。"""
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from render_ddp_allreduce_flow import Drawing, export_png
from render_rope_position_comparison import rotate

OUT=Path(__file__).resolve().parent
BLUE,TEAL,INK,MUTED='#355fc0','#087e75','#172b40','#586c7d'
PALE,AMBER='#fff3df','#98601f'
W,H=2400,3830


def draw():
    d=Drawing()
    def txt(x,y,s,size=30,color=INK,bold=400):d.text(x,y,s,size,color,bold)
    def band(y,title,color=TEAL):
        d.rect(60,y,2280,70,'#e8f6f1' if color==TEAL else '#edf2ff',stroke='none',radius=12)
        txt(85,y+47,title,34,color,700)
    def vector(y,label,values,color=TEAL):
        txt(85,y+45,label,30,color,650)
        for i,value in enumerate(values):
            x=660+i*405
            d.rect(x,y,380,69,'white',radius=10)
            d.text(x+190,y+46,str(value),37,color,650,'middle',mono=True)
    def matrix(x,y,rows,cw=190,color=TEAL):
        for i,row in enumerate(rows):
            for j,value in enumerate(row):
                d.rect(x+j*cw,y+i*65,cw-6,59,'#e8f6f1',stroke='none',radius=6)
                d.text(x+j*cw+(cw-6)/2,y+i*65+40,str(value),28,color,550,'middle',mono=True)
        # 方括号明确表示矩阵，而非独立数字卡片。
        for xx,sign in [(x-10,1),(x+len(rows[0])*cw+2,-1)]:
            d.fg.append(f'<path d="M {xx+sign*8} {y-5} L {xx} {y-5} L {xx} {y+len(rows)*65-1} L {xx+sign*8} {y+len(rows)*65-1}" fill="none" stroke="{color}" stroke-width="3"/>')

    txt(60,80,'位置编码怎么算？直接看数字、矩阵和结果',51,INK,700)
    txt(60,138,'GPT-2：两个向量逐项相加。RoPE：每两维乘一个由位置决定的旋转矩阵。',33,MUTED)
    txt(60,185,'下面是两个独立局部算例：GPT-2 的入口输出 X，与 RoPE 的投影后输出 Q′/K′，不在同一网络环节。',27,MUTED)

    band(230,'A · GPT-2：位置 t=1，查位置表后逐维相加',BLUE)
    txt(85,347,'输入：Token ID=42、位置 t=1。可训练参数：Token 表 E、位置表 P；下面取它们当前的一行。',29)
    vector(380,'Token 向量 E[42]',[1,2,3,4],BLUE)
    vector(470,'＋ 位置向量 P[1]',['0.1','−0.2','0.3','−0.4'],BLUE)
    vector(560,'＝ 输出 X[1]',['1.1','1.8','3.3','3.6'],BLUE)
    txt(690,675,'1+0.1=1.1          2−0.2=1.8          3+0.3=3.3          4−0.4=3.6',30,BLUE,600)
    d.arrow([(1200,696),(1200,735)],BLUE,width=4)
    d.rect(60,740,2280,105,'#edf2ff',stroke='none')
    txt(85,783,'X 交给后续 Block：归一化 → Q/K/V 投影 → 注意力；GPT-2 不再对 Q/K 做 RoPE。',30,BLUE,600)
    txt(85,822,'P[1] 的四个数是模型学到的参数；位置编号 1 只是查哪一行的索引，不参与训练更新。',27,MUTED)

    band(895,'B · Llama 3 RoPE：从已经投影、拆头的 Q/K 开始')
    txt(85,1014,'输入：一个 4 维头，位置 1 的 q=[1,2,3,4]；为比较位置效果，令位置 0 和 1 的原始 k 都为 [2,1,0,1]。',28)
    txt(85,1060,'这些是独立选定的教学 Q/K，不是由上面的 GPT-2 X 算出；真实不同位置的 K 通常不相同。',28,MUTED)
    d.rect(60,1090,2280,114,PALE,stroke='none')
    txt(85,1136,'配置：θ=500000（原始 Llama 3）；dₕ=4。配对编号 r=0、1；角度 φ=t×θ^(−2r/4)，单位是弧度。',29,AMBER,600)
    txt(85,1178,'RoPE 本身没有可训练权重；sin/cos 来自角度计算。Q/K 来自前面的可训练投影，V 不旋转。',28,AMBER)
    vector(1230,'位置 1 的输入 q',[1,2,3,4])
    txt(720,1342,'第 0 对：(1,2)',31,TEAL,650)
    txt(1530,1342,'第 1 对：(3,4)',31,TEAL,650)
    d.arrow([(1060,1300),(1060,1366),(610,1366),(610,1405)],TEAL)
    d.arrow([(1870,1300),(1870,1366),(1790,1366),(1790,1405)],TEAL)

    q=[1,2,3,4];k=[2,1,0,1];theta=500000
    qr=rotate(q,1);k0=rotate(k,0);k1=rotate(k,1)
    for r,x in enumerate([60,1240]):
        phi=theta**(-2*r/4);c,s=math.cos(phi),math.sin(phi);a,b=q[2*r:2*r+2]
        d.rect(x,1405,1100,570,'white',radius=18)
        txt(x+24,1453,f'第 {r} 对：输入 [{a},{b}]',33,TEAL,700)
        txt(x+24,1501,'φ = 1×500000⁰ = 1 弧度' if r==0 else 'φ = 1×500000^(−1/2) ≈ 0.001414214 弧度',28)
        txt(x+24,1546,f'cosφ ≈ {c:.9f}；sinφ ≈ {s:.9f}',27,MUTED)
        txt(x+24,1593,'旋转矩阵 R(φ)           × 输入       ＝ 输出',27,TEAL,600)
        matrix(x+30,1620,[[f'{c:.6f}',f'{-s:.6f}'],[f'{s:.6f}',f'{c:.6f}']],180)
        txt(x+409,1695,'×',36)
        matrix(x+465,1620,[[a],[b]],80)
        txt(x+585,1695,'=',36)
        matrix(x+650,1620,[[f'{qr[2*r]:.6f}'],[f'{qr[2*r+1]:.6f}']],290)
        txt(x+24,1805,'第一行：输入 a×cosφ − 输入 b×sinφ',27,MUTED)
        txt(x+24,1848,f'{a}×{c:.6f} − {b}×{s:.6f} ≈ {qr[2*r]:.6f}',31,TEAL,650)
        txt(x+24,1900,'第二行：输入 a×sinφ ＋ 输入 b×cosφ',27,MUTED)
        txt(x+24,1943,f'{a}×{s:.6f} + {b}×{c:.6f} ≈ {qr[2*r+1]:.6f}',31,TEAL,650)
    d.arrow([(610,1975),(610,2010),(1200,2010),(1200,2045)],TEAL)
    d.arrow([(1790,1975),(1790,2010),(1200,2010),(1200,2045)],TEAL)
    vector(2050,'拼回 q′（仍 4 维）',[f'{v:.6f}' for v in qr])
    txt(85,2173,'注意：这里的 2×2 矩阵来自 cos/sin，不是 Wq 或 Wk；旋转矩阵的两个行各算出一个新特征。',29,MUTED)

    band(2220,'C · K 也这样算，但使用 K 自己所在的位置')
    for x,pos,vals in [(60,0,k0),(1240,1,k1)]:
        d.rect(x,2330,1100,317,'white')
        txt(x+24,2377,f'位置 {pos}：原始 k=[2,1,0,1]',32,TEAL,700)
        if pos==0:
            lines=['两对的角度都是 0：cos=1，sin=0',
                   '第一对：(2×1−1×0, 2×0+1×1) = (2,1)',
                   '第二对：(0×1−1×0, 0×0+1×1) = (0,1)',
                   '输出 k′(位置 0) = [2,1,0,1]']
        else:
            lines=['角度与上面 q 相同：第一对 1，第二对约 0.001414214',
                   '第一对：(2c−1s, 2s+1c) ≈ (0.239134,2.223244)',
                   '第二对：(0c−1s, 0s+1c) ≈ (−0.001414,0.999999)',
                   '输出 k′(位置 1) ≈ [0.239134,2.223244,−0.001414,0.999999]']
        for i,l in enumerate(lines):txt(x+24,2430+i*49,l,25 if i==3 and pos else 27,INK)
    txt(85,2704,'c、s 分别表示当前特征对的 cosφ、sinφ；不同特征对的角度不同，不能把第一对的 c/s 用到第二对。',28,AMBER,600)
    d.arrow([(1200,2728),(1200,2780)],TEAL,width=4)

    band(2784,'D · 旋转后的结果交给点积：比较不同位置的 K')
    dot=lambda a,b:sum(u*v for u,v in zip(a,b))
    s0,s1=dot(qr,k0)/2,dot(qr,k1)/2
    for x,pos,score in [(60,0,s0),(1240,1,s1)]:
        d.rect(x,2896,1100,345,'white')
        txt(x+24,2944,f'同一个 q′(位置 1) 读取 k′(位置 {pos})',31,TEAL,700)
        if pos==0:
            lines=['逐维相乘再求和：',
                   '(−1.142640)×2 + 1.922076×1',
                   '+ 2.994340×0 + 4.004239×1 ≈ 3.641035',
                   '再除以 √dₕ=√4=2：分数 ≈ 1.820517']
        else:
            lines=['逐维相乘再求和：',
                   '(−1.142640)×0.239134 + 1.922076×2.223244',
                   '+ 2.994340×(−0.001414) + 4.004239×0.999999 ≈ 8',
                   '再除以 √4=2：分数 = 4（未舍入数值）']
        for i,l in enumerate(lines):txt(x+24,2999+i*49,l,27,INK if i<3 else TEAL,650 if i==3 else 400)
    d.rect(60,3295,2280,195,PALE,stroke='none')
    txt(85,3344,'原始 k 明明一样，为什么两个分数不同？',33,AMBER,700)
    txt(85,3393,'因为两个 K 的位置不同：位置 0 旋转 0，位置 1 按自己的位置旋转；与 Q 的相对角度因此不同。',29)
    txt(85,3441,'同一位置的 Q/K 共同旋转，点积保持原值；不同位置通常会改变点积。分数还不是概率，随后做 mask / softmax。',28)
    band(3535,'输入 / 输出 / 参数：对照两种位置编码',BLUE)
    txt(85,3650,'GPT-2：输入 Token ID 与位置 t → 查可训练的 E、P → 相加 → 输出入口 X，维度不变。',30,BLUE,650)
    txt(85,3702,'RoPE：输入投影后的 Q/K 与位置 t → 按固定 θ 算旋转 → 输出 Q′/K′，维度不变；V 不旋转。',30,TEAL,650)
    txt(85,3754,'示例数值人为选定，保留 6 位小数展示；底层计算用未舍入值。无模型训练，不宣称位置越远分数一定越小。',26,MUTED)
    txt(85,3798,'依据：Meta Llama 3 官方 model.py 的旋转规则；图中的每头维度缩小为 4，θ=500000。',26,MUTED)
    assert math.isclose(s1,4,abs_tol=1e-12)
    assert math.isclose(s0,1.8205174541612212,abs_tol=1e-12)
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><title>位置编码：矩阵与具体数值计算</title><style>text {{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}}</style><rect width="{W}" height="{H}" fill="#f5f8fb"/>'+''.join(d.bg+d.edges+d.fg)+'</svg>'
    ET.fromstring(svg)
    return svg,{'q_position_1':qr,'k_position_0':k0,'k_position_1':k1,'score_position_0':s0,'score_position_1':s1,'gpt_embedding':[1,2,3,4],'gpt_position':[.1,-.2,.3,-.4],'gpt_sum':[1.1,1.8,3.3,3.6]}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--png',action='store_true');args=p.parse_args()
    svg,report=draw();dest=OUT/'gpt2-llama3-position-worked-example.svg'
    dest.write_text(svg,encoding='utf-8')
    (OUT.parents[1]/'artifacts/position-worked-example-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if args.png:export_png(dest)
    print(dest)
