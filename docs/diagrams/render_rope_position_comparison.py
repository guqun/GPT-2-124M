"""位置编码对照：GPT-2 入口相加与 Llama 3 RoPE，含可复算教学数值。"""
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from render_ddp_allreduce_flow import Drawing, export_png

OUT=Path(__file__).resolve().parent
BLUE,TEAL,INK,MUTED='#355fc0','#087e75','#172b40','#586c7d'
AMBER,PALE='#98601f','#fff3df'
WIDTH,HEIGHT=2400,3370


def rotate(x,t,theta=500000):
    out=[]
    for r in range(len(x)//2):
        a=t*theta**(-2*r/len(x)); c,s=math.cos(a),math.sin(a)
        u,v=x[2*r:2*r+2]
        out.extend([u*c-v*s,u*s+v*c])
    return out


def draw():
    q,k=[1,2,3,4],[2,1,0,1]
    qr,kr=rotate(q,1),rotate(k,0)
    dot=lambda a,b:sum(x*y for x,y in zip(a,b))
    score=dot(qr,kr)/2
    assert math.isclose(dot(q,q),dot(qr,qr),abs_tol=1e-10)
    assert math.isclose(dot(rotate(q,2),rotate(k,1)),dot(qr,kr),abs_tol=1e-10)
    assert math.isclose(score,1.8205174541612212,abs_tol=1e-12)
    d=Drawing()
    def text(x,y,s,size=28,color=MUTED,bold=400):d.text(x,y,s,size,color,bold)
    def card(x,y,title,lines,color,fill='white',h=None):
        h=h or 85+len(lines)*40
        d.rect(x,y,1100,h,fill,radius=17)
        text(x+24,y+43,title,31,color,650)
        for i,line in enumerate(lines):text(x+24,y+88+40*i,line,26,INK if line.startswith(('输入','输出')) else MUTED)
        return h
    def arrow(x,y1,y2,col):d.arrow([(x+550,y1),(x+550,y2)],col,width=4)

    text(60,78,'位置信息在哪里进入计算？GPT-2 与 Llama 3 RoPE',47,INK,700)
    text(60,132,'蓝色：入口相加的位置向量    绿色：每层注意力中的旋转    橙色：差异、位置输入与配置',29)
    text(60,181,'教学形状：B=1，T=2，模型维度 D=4；本图只取一个 4 维头，不展开多头共享与 KV Cache。',27)
    d.rect(60,214,2280,65,PALE,stroke='none')
    text(83,256,'可训练参数 ≠ 本次输入计算出的 Q/K ≠ 位置编号 t ≠ 固定配置 θ；下面分别标注。',29,AMBER,650)

    for x,col,fill,name,is_rope in [(60,BLUE,'#edf2ff','GPT-2：学习位置表 P',False),(1240,TEAL,'#e8f6f1','Llama 3：由位置计算旋转',True)]:
        text(x+16,331,name,37,col,700)
        card(x,357,'01 入口：Token ID 先变成向量',[
            '输入：ID 序列 I=[17,42]，形状 [1,2]；位置 t=[0,1]',
            '可训练参数：Token 表 E [词表大小,4]'+('；位置表 P [Cmax,4]' if not is_rope else '；没有位置表 P'),
            '计算：X[0,t,:] = E[I[t]]'+(' + P[t]' if not is_rope else ''),
            '输出：X [1,2,4]；'+('位置信息已在入口加入' if not is_rope else '位置 t 留给每层 RoPE')],col,fill,h=253)
        arrow(x,610,663,col)
        card(x,663,'02 每层 Block：归一化 → QKV 投影 → 拆头',[
            '输入：当前层的 X；首层来自入口，后续层来自前一层',
            '计算：U = '+('LayerNorm(X)' if not is_rope else 'RMSNorm(X)'),
            '计算：Q=U Wq'+(' + bq；K=U Wk + bk；V=U Wv + bv' if not is_rope else '；K=U Wk；V=U Wv'),
            '参数：Wq/Wk/Wv 各 [4,4]；'+('γ、β 及各路 bias 可训练' if not is_rope else 'γ 可训练；投影无 bias'),
            '输出：Q、K、V 各 [1,1,2,4] = [B,头数,T,每头维]'],col,h=286)
        arrow(x,949,1004,col)
        if not is_rope:
            lines=['输入：已经含有上游位置影响的 Q、K、V',
                   '计算：Q′=Q；K′=K；此处不做位置旋转',
                   '位置参数 P 已在入口使用，此处不再查 P',
                   '输出：Q′、K′、V 形状均不变；交给 Q′K′ᵀ']
        else:
            lines=['输入：Q、K，以及各自 Token 的位置 t；V 旁路保留',
                   '固定配置：θ=500000；每头维度 dₕ=4；无新增可训练参数',
                   '计算：每头相邻两维配对，按 t 算角度，再旋转每一对',
                   '输出：Q′、K′ 形状不变；V 不旋转；交给 Q′K′ᵀ']
        card(x,1004,'03 差异发生处：是否旋转 Q/K',lines,col,PALE,h=251)

    d.rect(60,1305,2280,112,PALE,stroke='none')
    text(84,1347,'下面只隔离比较“旋转这一步”：人为固定同一组 Q/K，不是两模型从同一文本得到的完整输出。',29,AMBER,650)
    text(84,1390,'查询位置 i=1（第 2 个 Token）读取位置 j=0（第 1 个 Token）；所有向量数值为教学选值。',27,AMBER)

    for x,col,fill,is_rope in [(60,BLUE,'#edf2ff',False),(1240,TEAL,'#e8f6f1',True)]:
        card(x,1464,'04 数值输入：一个查询向量与一个键向量',[
            '位置 1 的 q = [1,2,3,4]；位置 0 的 k = [2,1,0,1]',
            '配对：q 的 (1,2)、(3,4)；k 的 (2,1)、(0,1)',
            '配对在每个头内部进行，不是再拆出新的注意力头。'],col,fill,h=205)
        arrow(x,1669,1720,col)
        if is_rope:
            lines=['角度公式：φ(t,r)=t × θ^(−2r/dₕ)，r=0,1；单位为弧度',
                   'q 位于 t=1：第 0 对 φ=1；第 1 对 φ≈0.001414214',
                   'k 位于 t=0：两对角度都是 0，cos(0)=1，sin(0)=0',
                   '位置属于向量自身；不会把位置 0 的 k 按位置 1 旋转。']
        else:
            lines=['不计算旋转角度；没有 sin/cos 运算',
                   '这一步将 q、k 原样传给注意力打分',
                   '完整 GPT-2 的位置作用已发生在入口 E + P',
                   '此处原样传递，不等于 GPT-2 没有位置信息。']
        card(x,1720,'05 位置编号 → 每对特征的角度',lines,col,PALE if is_rope else 'white',h=250)
        arrow(x,1970,2022,col)
        if is_rope:
            lines=['旋转规则：(a,b) → (a cosφ − b sinφ, a sinφ + b cosφ)',
                   '第一对：cos(1)≈0.540302；sin(1)≈0.841471',
                   '1×cos(1) − 2×sin(1) = −1.142640',
                   '1×sin(1) + 2×cos(1) =  1.922076',
                   '第二对：φ≈0.001414214，同样计算 (3,4)',
                   '(3,4) → (2.994340,4.004239)',
                   'k 两对都旋转 0 弧度：(2,1)→(2,1)，(0,1)→(0,1)']
        else:
            lines=['Q 路：q′ = q = [1,2,3,4]',
                   'K 路：k′ = k = [2,1,0,1]',
                   '无需在这一步乘任何位置矩阵或增加偏置。',
                   '为什么不用继续加位置？',
                   '入口的 X=E+P 已经进入归一化和 Q/K 投影。',
                   'RoPE 则改变的是每层 Q/K 的配对特征。',
                   '两种方法的位置注入位置和计算方式都不同。']
        card(x,2022,'06 执行运算：逐对计算两个新数',lines,col,h=375)
        arrow(x,2397,2446,col)
        qtext='[−1.142640, 1.922076, 2.994340, 4.004239]' if is_rope else '[1,2,3,4]'
        card(x,2446,'07 本环节输出：仍然是四维向量',[
            '输出 q′ = '+qtext,
            '输出 k′ = [2,1,0,1]；V 原样保留',
            '完整张量 Q′、K′ 仍为 [1,1,2,4]；这里只展示一行。'],col,fill,h=208)
        arrow(x,2654,2703,col)
        lines=(['点积：q′·k′ = (−1.142640)×2 + 1.922076 + 4.004239',
                '点积 ≈ 3.641035；缩放分数 s[1,0] ≈ 1.820517',
                '这是一个位置对的分数，不是概率；随后参与 softmax。'] if is_rope else
               ['点积：q′·k′ = 1×2 + 2×1 + 3×0 + 4×1 = 8',
                '缩放分数 s[1,0] = 8 / √4 = 4',
                '分母由每头维度决定；本例 √4=2。'])
        card(x,2703,'08 交给注意力：s[i,j] = q′ᵢ·k′ⱼ / √dₕ',lines,col,h=211)

    d.rect(60,2962,2280,240,PALE,stroke='none')
    text(84,3008,'参数与数据：一定要分清',32,AMBER,700)
    text(84,3052,'可训练：GPT-2 的位置表 P；两边的 E、投影 W、归一化参数。θ、位置编号 t、sin/cos 表不是训练参数。',27,INK)
    text(84,3095,'本次计算结果：X、U、Q/K/V、旋转后的 Q′/K′、注意力分数；它们随输入变化，不是权重表。',27,INK)
    text(84,3138,'RoPE 的目的：角度差 = (j−i) × θ^(−2r/dₕ)，使点积包含相对位置关系；不保证分数随距离单调降低。',27,INK)
    text(84,3181,'旋转保留向量长度，改变配对方向；有多个位置时，分数改变会影响读取各位置 V 的权重。',27,INK)
    text(65,3255,'范围：原始 Llama 3，相邻两维配对；θ 沿用官方配置，维度与向量是教学缩小示例。未运行模型训练。',26)
    text(65,3298,'来源：Meta llama3/llama/model.py 中 precompute_freqs_cis / apply_rotary_emb；GPT-2 对照项目既有实现。',25)
    text(65,3341,'数值核验：旋转前后平方长度相同；q 在位置 2、k 在位置 1 时，与位置 1/0 的点积相同（固定原始 q/k）。',25)
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}"><title>GPT-2 与 Llama 3 位置编码计算流程</title><style>text {{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}}</style><rect width="{WIDTH}" height="{HEIGHT}" fill="#f5f8fb"/>'+''.join(d.bg+d.edges+d.fg)+'</svg>'
    ET.fromstring(svg)
    report={'scope':'isolated rotation, not full-model comparison','q_position':1,'k_position':0,'head_dim':4,'theta':500000,'q':q,'k':k,'q_rotated':qr,'k_rotated':kr,'k_rotated_at_1':rotate(k,1),'score_without_rope':dot(q,k)/2,'score_with_rope':score,'norm_preserved':True,'relative_shift_invariance_checked':True}
    return svg,report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--png',action='store_true');args=p.parse_args()
    svg,report=draw();dest=OUT/'gpt2-llama3-rope-position-flow.svg'
    dest.write_text(svg,encoding='utf-8')
    (OUT.parents[1]/'artifacts/rope-position-example-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.png:export_png(dest)
    print(dest)
