"""Llama 3 / Qwen3 前向对照，复用项目 SVG 绘图与 PNG 导出。"""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from render_ddp_allreduce_flow import Drawing, export_png

OUT = Path(__file__).resolve().parent
BLUE, TEAL, AMBER = '#355fc0', '#087e75', '#9b601f'
INK, MUTED = '#172b40', '#586c7d'
WIDTH, HEIGHT = 2400, 4260


def norm(x):
    r = math.sqrt(sum(a*a for a in x)/len(x))
    return [a/r for a in x]


def rotate(x, pos):
    c, s = math.cos(pos), math.sin(pos)
    return [x[0]*c-x[1]*s, x[0]*s+x[1]*c]


def calculate(use_norm):
    q, keys, values = [3.,4.], [[1.,0.],[0.,2.]], [[10.,0.],[0.,20.]]
    nq = norm(q) if use_norm else q
    nk = [norm(k) for k in keys] if use_norm else keys
    qr, kr = rotate(nq,1), [rotate(k,i) for i,k in enumerate(nk)]
    scores = [sum(a*b for a,b in zip(qr,k))/math.sqrt(2) for k in kr]
    exps = [math.exp(s-max(scores)) for s in scores]
    weights = [e/sum(exps) for e in exps]
    h = [sum(weights[j]*values[j][d] for j in range(2)) for d in range(2)]
    assert math.isclose(sum(weights),1.)
    # 独立用相对旋转核对点积，确保位置没有混用。
    for j,k in enumerate(nk):
        rel=rotate(k,j-1)
        assert math.isclose(scores[j],sum(a*b for a,b in zip(nq,rel))/math.sqrt(2))
    return dict(q=nq,k=nk,rotated_q=qr,rotated_k=kr,scores=scores,A=weights,H=h)


def draw():
    d=Drawing()
    def txt(x,y,s,size=28,color=INK,bold=400):
        d.text(x,y,s,size,color,bold)
    def band(y,s):
        d.rect(60,y,2280,60,'#eaf0f5',stroke='none',radius=10)
        txt(84,y+41,s,31,INK,700)
    def card(x,y,h,title,lines,color,highlight=False):
        d.rect(x,y,1100,h,'#fff4e4' if highlight else 'white',stroke='#dfb776' if highlight else '#dce4ec',radius=16)
        txt(x+28,y+46,title,32,color,700)
        for i,s in enumerate(lines): txt(x+28,y+91+i*40,s,27)
    def pair(y,h,title,left,right,highlight=False):
        band(y,title)
        card(60,y+78,h,'Llama 3',left,BLUE)
        card(1240,y+78,h,'Qwen3 · 稠密模型',right,TEAL,highlight)
        if y<2220:
            for x,col in [(610,BLUE),(1790,TEAL)]:
                d.arrow([(x,y+78+h),(x,y+78+h+23)],col)
    def ring(x,y,w,h,dashed=False):
        # 原生矢量圈注，描边放在文字周围，不遮盖公式。
        dash=' stroke-dasharray="13 9"' if dashed else ''
        color='#a76617' if dashed else '#d43d3d'
        d.fg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="none" stroke="{color}" stroke-width="4"{dash}/>')
    fmt=lambda x:'['+', '.join(f'{v:.4f}' for v in x)+']'
    txt(60,87,'从 Llama 3 到 Qwen3：前向计算哪里不同？',51,INK,700)
    txt(60,146,'红色实线圈 = 结构/参数共享差异；棕色虚线圈 = 按具体型号比较的配置；未圈 ≠ 参数或结果相同。',28,MUTED)
    txt(60,195,'蓝：Llama 3；绿：Qwen3 稠密。统一教学 B=1、T=2、D=8、4Q/2KV、每头2维，不代表真实配置相同。',27,MUTED)
    pair(230,180,'01 · 输入与第一次归一化｜相同计算类型',[
        'Token ID 形状 [1,2] → 查 E[Vocab,8] → X [1,2,8]',
        '每层输入 X → U = RMSNorm₁(X) [1,2,8]',
        '参数 E、γ₁[8]；词表/D/ε 按型号；无位置向量相加。'],[
        'Token ID 形状 [1,2] → 查 E[Vocab,8] → X [1,2,8]',
        '每层输入 X → U = RMSNorm₁(X) [1,2,8]',
        '参数 E、γ₁[8]；词表/D/ε 按型号；无位置向量相加。'])
    pair(515,220,'02 · Q/K/V 投影并拆头｜Q、K、V 是计算结果，W 才是参数',[
        'Qflat = UWQ；Kflat = UWK；Vflat = UWV',
        'WQ[8,8]；WK/WV[8,4]；无 bias',
        '拆头：Q [1,4,2,2]；K/V [1,2,2,2]',
        '头数、KV 头数、每头维度与投影宽度按型号配置。'],[
        'Qflat = UWQ；Kflat = UWK；Vflat = UWV',
        '本教学例 WQ[8,8]；WK/WV[8,4]；无 bias',
        '拆头：Q [1,4,2,2]；K/V [1,2,2,2]',
        '真实型号须读取 head_dim，不能一律假设 D/头数。'])
    pair(840,300,'03 · 关键新增：对每个头的 Q、K 再做 RMSNorm',[
        '此处没有 QK-Norm。',
        'Q、K 直接进入下一步 RoPE；V 暂时保留。',
        '没有本环节的 γQ、γK 参数。',
        '本步骤不改变 Q/K/V 的数值或形状。',
        '注意：前面的 U 已归一化，',
        '但乘 WQ、WK 后，Q/K 的尺度仍可能改变。'],[
        '每个位置、每个头单独计算 r(x)=√(mean(x²)+ε)',
        'Q̂ = γQ ⊙ Q / r(Q)；K̂ = γK ⊙ K / r(K)',
        '新参数：γQ[2]、γK[2]；各层独立，无 bias。',
        '同层 Q 头共用 γQ；K 头共用 γK；两者分别学习。',
        '每个向量的 r 单独计算；只沿头内特征维归一化。',
        '输出形状不变；V 不经过这个新增归一化。'],True)
    pair(1245,260,'04 · RoPE → 匹配 → 汇总｜后续规则相同，参与计算的数据不同',[
        'Q′=RoPE(Q)；K′=RoPE(K)；V 不旋转',
        'Q头 0/1 → KV组0；Q头 2/3 → KV组1',
        'S = Q′K′ᵀ / √2 + M；A = softmax(S)',
        'RoPE θ / 上下文范围按型号；M 对未来置 −∞。',
        'H = AV → [1,4,2,2]；每个 Q 头有自己的 A/H'],[
        'Q′=RoPE(Q̂)；K′=RoPE(K̂)；V 不旋转',
        '同样按 GQA 分组；使用归一化、旋转后的 Q′/K′',
        'S = Q′K′ᵀ / √2 + M；A = softmax(S)',
        'RoPE θ / 上下文范围按型号；仍需缩放与遮罩。',
        'H = AV → [1,4,2,2]；仍使用前面投影得到的 V'])
    pair(1610,180,'05 · 多头拼接 → 输出投影 → 第一次残差｜同在 Block 内',[
        'H 重排并拼接 → C [1,2,8]',
        'O = CWO；WO[8,8]，无 bias → O [1,2,8]',
        'R = X + O → [1,2,8]；加回该 Block 的原输入。'],[
        'H 重排并拼接 → C [1,2,8]',
        'O = CWO；本例 WO[8,8]，无 bias → O [1,2,8]',
        'R = X + O → [1,2,8]；QK-Norm 不改残差规则。'])
    pair(1895,220,'06 · 第二次归一化 → SwiGLU → 第二次残差｜计算类型相同',[
        'Z = RMSNorm₂(R)；独立参数 γ₂[8]',
        'Fout = [SiLU(ZWg) ⊙ (ZWv)] Wd',
        '教学 F=24：Wg/Wv[8,24]；Wd[24,8]，无 bias',
        'Y = R + Fout [1,2,8]；Block 重复层数也按型号。'],[
        'Z = RMSNorm₂(R)；独立参数 γ₂[8]',
        'Fout = [SiLU(ZWg) ⊙ (ZWv)] Wd',
        '本例也取 F=24；不代表两种真实型号宽度相同。',
        'Y = R + Fout [1,2,8]；Block 重复层数也按型号。'])
    pair(2220,220,'07 · 所有 Block 后：最终归一化 → lm_head｜共享与否要看型号',[
        'Hf = RMSNorm_f(XL) [1,2,8]；参数 γf[8]',
        'Wout[Vocab,8]；logits = Hf Woutᵀ → [1,2,Vocab]',
        '原始 Llama 3：输出表 Wout 与输入 E 独立。',
        'logits 是分数；训练算交叉熵，生成选择 Token。'],[
        'Hf = RMSNorm_f(XL) [1,2,8]；参数 γf[8]',
        'Wout[Vocab,8]；logits = Hf Woutᵀ → [1,2,Vocab]',
        '原始 Qwen3：0.6B/1.7B/4B 共享 E；8B 等不共享。',
        '此差异独立于 QK-Norm；以具体模型配置为准。'])

    # 两种结构差异：A 为必经新增运算，B 的共享设置依型号而定。
    for x in (60,1240):
        ring(x+10,918+57,1080,241)
        ring(x+16,2437,1068,43)
    txt(907,964,'差异 A',27,'#d43d3d',700)
    txt(2087,964,'差异 A',27,'#d43d3d',700)
    txt(907,2344,'差异 B',27,'#d43d3d',700)
    txt(2087,2344,'差异 B',27,'#d43d3d',700)
    # 按型号而异的配置，统一教学尺寸不能被理解为真实规格一致。
    for x in (60,1240):
        ring(x+16,445,1068,43,True)  # E/D/epsilon
        ring(x+16,770,1068,43,True)  # head dimensions
        ring(x+16,1505,1068,40,True) # RoPE settings
        ring(x+16,2110,1068,82,True) # FFN width and depth

    band(2560,'08 · 具体数值：追踪一个 Q 头、查询位置 1，直到输出 H')
    txt(85,2670,'两边人为指定同一组“投影并拆头后的数据”；只隔离 QK-Norm 效果，不表示真实模型参数相同。',28)
    txt(85,2715,'q₁=[3,4]；k₀=[1,0]，k₁=[0,2]；V 的两行 v₀=[10,0]，v₁=[0,20]。下标是 Token 位置。',28)
    txt(85,2760,'为便于手算：γQ=γK=[1,1]，忽略 ε。两维头 RoPE 角度 φ₀=0、φ₁=1 弧度；使用完整精度计算。',27,MUTED)
    report={}
    for x,use_norm,title,col in [(60,False,'Llama 3 路径：不加 QK-Norm',BLUE),(1240,True,'Qwen3 路径：先加 QK-Norm',TEAL)]:
        r=calculate(use_norm);report['qwen3' if use_norm else 'llama3']=r
        card(x,2800,920,title,[],col)
        textlines=(['① 进入 RoPE 前：保持原数据',
            'q₁=[3,4]；k₀=[1,0]；k₁=[0,2]',
            '② RoPE：同一公式、按各自位置旋转'] if not use_norm else [
            '① r(q₁)=√12.5；r(k₀)=√0.5；r(k₁)=√2',
            'q̂₁=[0.8485,1.1314]；k̂₀=[1.4142,0]',
            'k̂₁=[0,1.4142]；② 然后按位置做 RoPE'])
        for i,s in enumerate(textlines):txt(x+28,2900+i*45,s,27)
        txt(x+28,3052,'旋转规则：[a,b] → [a cosφ−b sinφ, a sinφ+b cosφ]',26)
        txt(x+28,3100,'q′₁ = '+fmt(r['rotated_q']),29,col,650)
        txt(x+28,3147,'k′₀ = '+fmt(r['rotated_k'][0]),29)
        txt(x+28,3194,'k′₁ = '+fmt(r['rotated_k'][1]),29)
        txt(x+28,3254,'③ 分数 sⱼ = (q′₁ · k′ⱼ) / √2',29,INK,650)
        txt(x+28,3300,'s = '+fmt(r['scores']),29,col,650)
        txt(x+28,3345,'查询位置 1：可读位置 0、1；这两项都不遮罩。',26,MUTED)
        txt(x+28,3400,'④ A = softmax(s) = '+fmt(r['A']),28,INK,650)
        txt(x+28,3455,'⑤ H = A₀[10,0] + A₁[0,20]',29,INK,650)
        txt(x+28,3500,'      = '+fmt(r['H']),34,col,700)
        txt(x+28,3570,'形状：q [2]、K/V [2,2] → A [2] → H [2]',26)
        txt(x+28,3615,'V 保持相同；A 的变化使加权汇总结果改变。',27)
        txt(x+28,3660,'只追踪一个头/一个位置，不是整个模型输出。',26,MUTED)

    band(3770,'09 · 为什么增加这一步？优势、代价与边界')
    txt(85,3880,'问题：U 已归一化，但乘 WQ/WK 后 Q/K 仍可能放大；q、k 各放大 10 倍，点积会放大 100 倍。',28)
    txt(85,3930,'作用：对投影后的每头向量控制尺度，降低注意力分数过大、softmax 过于极端的风险，帮助稳定训练。',28,TEAL,650)
    txt(85,3980,'代价：每层增加两组缩放参数（本例共 4 个）及归一化计算；不减少 KV Cache，也不保证推理更快。',28)
    txt(85,4030,'未圈部分只表示计算类型相同：真实权重分别训练，中间数据与最终输出并不相同；本例不证明质量高低。',28)
    txt(85,4100,'范围：原始 Llama 3 与 Qwen3 稠密架构；省略 Dropout。示例 D/F、头数和矩阵均为教学配置。',26,MUTED)
    txt(85,4150,'来源：Meta Llama 3 model.py；Qwen3 Technical Report §2；Transformers Qwen3；Raschka Qwen3 配套实现。',25,MUTED)
    txt(85,4200,'圈注 A：新增 QK-Norm；圈注 B：部分 Qwen3 型号共享输出表；虚线：词表、维度、RoPE、宽度及层数。',25,MUTED)
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}"><title>Llama 3 与 Qwen3 前向计算和 QK-Norm 数值对照</title><style>text {{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}}</style><rect width="{WIDTH}" height="{HEIGHT}" fill="#f5f8fb"/>'+''.join(d.bg+d.edges+d.fg)+'</svg>'
    ET.fromstring(svg)
    return svg,report


if __name__=='__main__':
    svg,report=draw()
    path=OUT/'llama3-vs-qwen3-forward-qknorm.svg'
    path.write_text(svg)
    (OUT.parents[1]/'artifacts/llama3-qwen3-qknorm-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    export_png(path)
    print(path)
