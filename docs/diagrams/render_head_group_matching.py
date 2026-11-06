"""MHA/GQA 头对应关系与两位置注意力算例；输入是投影后的教学值。"""
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from render_ddp_allreduce_flow import Drawing, export_png

OUT = Path(__file__).resolve().parent
W, H = 2400, 4700
INK, MUTED = '#172b40', '#586c7d'
BLUE, TEAL, AMBER = '#355fc0', '#087e75', '#98601f'


def draw():
    d = Drawing()
    def text(x,y,s,size=29,color=INK,bold=400):
        d.text(x,y,s,size,color,bold)
    def band(y,s,color=TEAL):
        d.rect(60,y,2280,66,'#e8f6f1' if color==TEAL else '#edf2ff',stroke='none',radius=10)
        text(86,y+44,s,33,color,700)
    def matrix(x,y,rows,cw=140,ch=60,color=TEAL):
        for i,row in enumerate(rows):
            for j,val in enumerate(row):
                d.rect(x+j*cw,y+i*ch,cw-6,ch-6,'#e8f6f1',stroke='none',radius=4)
                d.text(x+j*cw+(cw-6)/2,y+i*ch+39,str(val),29,color,600,'middle',mono=True)
        hh=len(rows)*ch
        for xx,sgn in [(x-10,1),(x+len(rows[0])*cw+2,-1)]:
            d.fg.append(f'<path d="M {xx+sgn*8} {y-5} H {xx} V {y+hh-1} H {xx+sgn*8}" fill="none" stroke="{color}" stroke-width="3"/>')
    def node(x,y,w,s,color):
        d.rect(x,y,w,58,'white',stroke=color,radius=10)
        d.text(x+w/2,y+39,s,28,color,650,'middle')

    text(60,79,'Q/K/V 已经算完了，为什么还要“对应头”？',49,INK,700)
    text(60,135,'这一环节只安排：哪个 Q 头读取哪组已有 K/V。新的点积与加权汇总在后面发生。',31,MUTED)
    text(60,181,'教学缩小配置：B=1、T=2、D=8；4 个 Q 头，每头 2 维。图中头编号与 Token 位置编号分开。',27,MUTED)

    band(220,'01 · 前面已经完成：投影 → 拆头 →（Llama）RoPE',BLUE)
    for x,title,color,lines in [
        (60,'GPT-2 风格 · MHA',BLUE,[
            '归一化后的 U [1,2,8]；WQ/WK/WV 各 [8,8]。',
            'Qflat=U WQ+bQ；K/V 同理；各 bias 为 [8]。',
            '拆头后 Q、K、V 各 [1,4,2,2]：各有 4 个头。',
            '这里不再投影；GPT-2 也不额外对 Q/K 做 RoPE。']),
        (1240,'Llama 风格 · GQA',TEAL,[
            '归一化后的 U [1,2,8]；WQ [8,8]，WK/WV 各 [8,4]。',
            'Qflat=U WQ；Kflat=U WK；Vflat=U WV；无 bias。',
            '拆头后 Q [1,4,2,2]；K/V 各 [1,2,2,2]。',
            'Q/K 经 RoPE 后形状不变；以下写 Q′、K′。'])]:
        d.rect(x,310,1100,280,'white')
        text(x+25,356,title,34,color,700)
        for i,line in enumerate(lines):text(x+25,408+i*45,line,27)
    text(85,641,'权重矩阵属于前面的投影步骤；本步新增可训练参数 = 0。K/V 从一开始就只投影出各自需要的头数。',29,AMBER,600)
    d.arrow([(1200,663),(1200,703)],TEAL,width=4)

    band(710,'02 · 当前步骤：为每个 Q 头选择对应 K/V，不重新算 Q/K/V')
    for x,title,color in [(60,'MHA：一对一',BLUE),(1240,'GQA：两个 Q 头共用一组 K/V',TEAL)]:
        d.rect(x,804,1100,466,'white')
        text(x+25,851,title,34,color,700)
        text(x+30,898,'已经算好的 Q 头',26,MUTED)
        text(x+650,898,'读取已有的 K/V',26,MUTED)
        for h in range(4):
            y=929+h*76
            node(x+30,y,220,f'Q′ 头 {h}',color)
            g=h if x==60 else h//2
            if x==60:
                node(x+650,y,350,f'K′ 头 {g}、V 头 {g}',color)
                d.arrow([(x+250,y+29),(x+650,y+29)],color)
            else:
                target_y=967+g*152
                if h%2==0:node(x+650,target_y,350,f'K′ 头 {g}、V 头 {g}',color)
                d.arrow([(x+250,y+29),(x+480,y+29),(x+480,target_y+29),(x+650,target_y+29)],color)
    text(85,1323,'MHA：g(h)=h',30,BLUE,650)
    text(1265,1323,'GQA：g(h)=⌊h/2⌋ → [0,0,1,1]',30,TEAL,650)
    d.rect(60,1350,2280,138,'#fff3df',stroke='none')
    text(85,1395,'逻辑对齐：Q′=[Q′₀,Q′₁,Q′₂,Q′₃]；K*=[K′₀,K′₀,K′₁,K′₁]；V*=[V₀,V₀,V₁,V₁]。',30,AMBER,600)
    text(85,1444,'这不是再次乘 W：重复引用同一组数值；高效实现可以按组读取。上方每个头都包含两个 Token 位置。',29)
    d.arrow([(1200,1490),(1200,1531)],TEAL,width=4)

    band(1540,'03 · 具体数值：看 GQA 第 0 组，两个 Q 读取同一份 K/V')
    text(85,1650,'只观察查询位置 t=1：它允许读取位置 s=0、1。以下 Q′/K′ 是 RoPE 后直接选定的教学值。',28)
    text(85,1695,'不是从上方示意权重实算得到；下面省略撇号。此例 eval 模式，无 Dropout；不再计算投影或 RoPE。',27,MUTED)
    d.rect(60,1730,650,290,'white')
    text(85,1776,'当前查询位置的两个 Q 向量',30,TEAL,650)
    text(85,1832,'Q 头 0：q₀=[1,0]',31)
    text(85,1890,'Q 头 1：q₁=[0,1]',31)
    text(85,1963,'下标 0/1 是头号；二者位置都是 t=1。',25,MUTED)
    for x,title,rows in [(760,'同一份 K 头 0',[[1,0],[0,1]]),(1570,'同一份 V 头 0',[[10,0],[0,20]])]:
        d.rect(x,1730,770,290,'white')
        text(x+25,1776,title,30,TEAL,650)
        text(x+25,1842,'位置 s=0 →',26,MUTED)
        text(x+25,1902,'位置 s=1 →',26,MUTED)
        matrix(x+250,1803,rows,160)
        text(x+25,1963,'行 = Token 位置；列 = 当前头的特征维。',25,MUTED)

    band(2060,'04 · 下一步才做计算：各自打分 → 各自 softmax → 各自汇总 V')
    p=math.exp(1/math.sqrt(2))/(math.exp(1/math.sqrt(2))+1)
    outputs=[[10*p,20*(1-p)],[10*(1-p),20*p]]
    for h,x in enumerate([60,1240]):
        d.rect(x,2160,1100,655,'white')
        text(x+25,2207,f'Q 头 {h}：使用 K 头 0、V 头 0',32,TEAL,700)
        q=[1,0] if h==0 else [0,1]
        text(x+25,2266,'① 点积得到两个位置的分数：q × K₀ᵀ / √2',28,INK,600)
        matrix(x+35,2298,[q],90)
        text(x+240,2340,'×',33)
        matrix(x+300,2298,[[1,0],[0,1]],90)
        text(x+520,2340,'/ √2 =',29)
        matrix(x+690,2298,[['0.7071','0'] if h==0 else ['0','0.7071']],175)
        text(x+25,2465,'② mask 均为 0；softmax：A[s]=exp(S[s])/Σ exp(S)',27)
        a=[p,1-p] if h==0 else [1-p,p]
        text(x+25,2512,f'A{h} ≈ [{a[0]:4.4f}, {a[1]:4.4f}]    ← 对位置 0、1 的权重',29,TEAL,600)
        text(x+25,2575,'③ 用自己的权重，汇总同一个 V₀：',29,INK,600)
        text(x+25,2624,f'H{h} = {a[0]:.4f} × [10,0] + {a[1]:.4f} × [0,20]',29)
        text(x+25,2680,f'     ≈ [{outputs[h][0]:.4f}, {outputs[h][1]:.4f}]',35,TEAL,700)
        text(x+25,2762,'输出：当前头、当前查询位置的 2 维新向量。',27,MUTED)
    d.rect(60,2860,2280,133,'#fff3df',stroke='none')
    text(85,2906,'同一份 K/V，不等于相同输出：q₀ ≠ q₁ → 注意力权重不同 → H₀ ≠ H₁。',33,AMBER,700)
    text(85,2954,'每个 Q 头仍独立产生结果；这里只展开第 0 组。Q 头 2、3 会用第 1 组 K/V 做同样的计算。',29)
    d.arrow([(1200,2995),(1200,3030)],TEAL,width=4)
    band(3035,'05 · 放大头 0：Q 与 K 如何一步一步得到 A？')
    text(85,3143,'输入：当前查询 q=[1,0]；位置 0 的 k=[1,0]，位置 1 的 k=[0,1]；每头 dₕ=2。',30)
    d.rect(60,3180,1100,365,'white')
    d.rect(1240,3180,1100,365,'white')
    text(85,3228,'① 点积 → 缩放 → 因果遮罩',32,TEAL,700)
    text(85,3290,'位置 0：1×1 + 0×0 = 1；再除以 √2 ≈ 0.7071',29)
    text(85,3343,'位置 1：1×0 + 0×1 = 0；再除以 √2 = 0',29)
    text(85,3401,'当前查询位于 t=1，两位置均可读，mask 都是 0。',27)
    text(85,3460,'输出分数 S ≈ [0.7071, 0]；还不是概率。',30,TEAL,650)
    text(1265,3228,'② softmax：取指数 → 求和 → 各自除以总和',31,TEAL,700)
    text(1265,3290,'exp(0.7071) ≈ 2.0281；exp(0) = 1',29)
    text(1265,3343,'分母：2.0281 + 1 = 3.0281',29)
    text(1265,3401,'位置 0：2.0281 / 3.0281 ≈ 0.6698',29)
    text(1265,3460,'位置 1：1 / 3.0281 ≈ 0.3302',29)
    text(85,3600,'输出 A ≈ [0.6698,0.3302]：两个位置的读取权重，总和为 1。算 A 时不使用 V。',31,TEAL,650)
    d.arrow([(1200,3620),(1200,3655)],TEAL,width=4)
    band(3665,'06 · A 与 V 如何得到 H？每个位置的整条 V 乘权重，再逐维相加')
    text(85,3770,'输入：A ≈ [0.6698,0.3302]；V 的行 0=[10,0]、行 1=[0,20]。以下先用舍入后的权重展示乘法。',28)
    text(85,3820,'矩阵乘法：',29,INK,600)
    matrix(330,3790,[['0.6698','0.3302']],190)
    text(755,3832,'×',38)
    matrix(825,3790,[[10,0],[0,20]],150)
    text(1190,3832,'=',38)
    matrix(1280,3790,[['6.698','6.604']],230)
    text(1830,3825,'A [1,2] × V [2,2]',28,MUTED)
    text(1830,3870,'→ H [1,2]',28,TEAL,650)
    for x,caption,calc in [
        (60,'位置 0 的贡献','0.6698 × [10,0] = [6.698,0]'),
        (1240,'位置 1 的贡献','0.3302 × [0,20] = [0,6.604]')]:
        d.rect(x,3940,1100,145,'white')
        text(x+25,3988,caption,31,TEAL,700)
        text(x+25,4044,calc,33)
    text(85,4140,'逐维相加：H = [6.698+0, 0+6.604] = [6.698,6.604]',34,TEAL,700)
    text(85,4200,'第 0 维：0.6698×10 + 0.3302×0；第 1 维：0.6698×0 + 0.3302×20。',30)
    text(85,4255,'每个权重乘该位置的全部特征维；A 的元素对应位置，不是对应 V 的特征维。',30,AMBER,650)
    text(85,4310,'H 属于当前查询位置：汇总多个位置的信息，得到一条向量。一般规则 H[t,d] = Σₛ A[t,s] V[s,d]。',29)
    text(85,4365,'使用未舍入权重，H ≈ [6.6976,6.6048]，与上方结果一致。完整 A 对每个查询位置分别计算一行。',28,MUTED)
    d.arrow([(1200,4390),(1200,4420)],TEAL,width=4)
    band(4425,'输出交给哪里：各头输出 → 沿特征维拼接 → 输出投影 → 残差相加',BLUE)
    text(85,4540,'全部位置的输出仍是 [B,Hq,T,dₕ]=[1,4,2,2]；拼接后 [1,2,8]，再交给输出投影 WO。',29)
    text(85,4585,'目的：保留 4 个 Q 头的独立查询，同时减少 K/V 投影参数与生成时的 KV Cache。',29,TEAL,600)
    text(85,4630,'图中 Q′/K′/V 是中间数据，W 才是参数。逻辑对齐不等于实际 KV Cache 也存成 4 头。',27,MUTED)
    text(85,4675,'依据：原始 Llama 3 model.py 的 repeat_kv / Attention；数值为教学选值。所有完整精度结果由脚本复算。',25,MUTED)
    assert math.isclose(sum(a),1)
    assert math.isclose(outputs[0][0]+outputs[1][0],10)
    assert math.isclose(outputs[0][1]+outputs[1][1],20)
    assert math.isclose(p,0.6697615493266569)
    rounded_contributions = [[round(p,4)*10,0],[0,round(1-p,4)*20]]
    assert math.isclose(rounded_contributions[0][0],6.698)
    assert math.isclose(rounded_contributions[1][1],6.604)
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><title>MHA 与 GQA：已有 QKV 如何对应头并计算注意力</title><style>text {{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}}</style><rect width="{W}" height="{H}" fill="#f5f8fb"/>'+''.join(d.bg+d.fg+d.edges)+'</svg>'
    ET.fromstring(svg)
    return svg,{'query_position':1,'key_positions':[0,1],'q0':[1,0],'q1':[0,1],'K0':[[1,0],[0,1]],'V0':[[10,0],[0,20]],'A0':[p,1-p],'A1':[1-p,p],'outputs':outputs,'gqa_mapping':[0,0,1,1],'head0_scores':[1/math.sqrt(2),0],'head0_exponents':[math.exp(1/math.sqrt(2)),1],'rounded_contributions':rounded_contributions}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--png',action='store_true');args=parser.parse_args()
    svg,report=draw();dest=OUT/'gpt2-llama3-head-matching-flow.svg'
    dest.write_text(svg,encoding='utf-8')
    (OUT.parents[1]/'artifacts/head-matching-example-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if args.png:export_png(dest)
    print(dest)
