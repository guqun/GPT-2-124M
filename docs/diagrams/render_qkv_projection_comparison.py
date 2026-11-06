"""QKV 投影与拆头的双泳道教学图；形状沿用前向对照的缩小配置。"""
from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET
from render_ddp_allreduce_flow import Drawing, export_png

OUT = Path(__file__).resolve().parent
W, H = 2400, 2580
BLUE, TEAL, INK, MUTED = '#355fc0', '#087e75', '#172b40', '#586c7d'
AMBER, PALE = '#98601f', '#fff3df'


def draw():
    d = Drawing()
    d.text(60, 78, 'Q / K / V 如何投影、拆头：GPT-2 与 Llama 3', 49, INK, 700)
    d.text(60, 132, '同一份完整输入 → 三路线性投影 → 按输出特征拆头 → 交换维度 → 交给注意力', 30, MUTED)
    d.text(60, 179, '教学配置：B=2 条样本，T=5 个 Token，D=16；Q 头数=4，每头维度 dₕ=4。不是官方模型规模。', 27, MUTED)
    d.rect(60, 208, 2280, 70, PALE, stroke='none')
    d.text(85, 252, '差异 C：GPT-2 的 Q/K/V 各 4 头；Llama 的 Q 为 4 头、K/V 各 2 头。   差异 D：有无 bias。', 29, AMBER, 650)

    def box(x, y, w, h, title, lines, color, fill='white', size=27):
        d.rect(x, y, w, h, fill, radius=15)
        d.text(x+20, y+39, title, 29, color, 650)
        for i, s in enumerate(lines):
            d.text(x+20, y+83+i*37, s, size, MUTED)

    for base, name, col, fill, kv_heads in [(60, 'GPT-2 · MHA', BLUE, '#edf2ff', 4), (1240, 'Llama 3 · GQA', TEAL, '#e8f6f1', 2)]:
        d.text(base+20, 332, name, 37, col, 700)
        box(base, 357, 1100, 130, '① 输入：归一化后的 U [2,5,16]',
            ['每个 Token 是完整 16 维；三个投影都读取这 16 维。两模型各自的 U 数值可以不同。'], col, fill, 25)
        centers = [base+175, base+550, base+925]
        for cx in centers:
            d.arrow([(base+550,487),(base+550,525),(cx,525),(cx,574)],col)
        for i, letter in enumerate('QKV'):
            x=base+i*375; cx=centers[i]
            n=16 if letter=='Q' or kv_heads==4 else 8
            heads=n//4
            bias=kv_heads==4
            changed=letter!='Q' and kv_heads==2
            box(x,574,350,236,f'② {letter} 投影：16 → {n}',[
                f'W{letter.lower()} [16,{n}]',
                f'b{letter.lower()} [{n}]，逐位置广播' if bias else '无 bias',
                f'{letter}₀ = U W{letter.lower()}'+(f' + b{letter.lower()}' if bias else ''),
                f'输出 {letter}₀ [2,5,{n}]'],col,PALE if changed else 'white',26)
            d.arrow([(cx,810),(cx,865)],col)
            box(x,865,350,184,'③ reshape：拆输出特征',[
                f'{n} 维 = {heads} 个头 × 4 维',
                f'[2,5,{n}] → [2,5,{heads},4]',
                '不乘矩阵，不产生新参数'],col,PALE if changed else 'white',25)
            d.arrow([(cx,1049),(cx,1105)],col)
            box(x,1105,350,184,'④ transpose：交换轴',[
                '[B,T,H,4] → [B,H,T,4]',
                f'{letter} [2,{heads},5,4]',
                f'输出：{heads} 头，每头 4 维'],col,PALE if changed else fill,25)
        d.text(base+20, 1344, '拆头的具体含义：以下只展示一个 Token 的投影结果', 29, col,650)
        for row, letter in enumerate('QKV'):
            yy=1370+row*83
            heads=4 if letter=='Q' else kv_heads
            d.text(base+20,yy+39,letter,31,col,700)
            for h in range(heads):
                xx=base+80+h*250
                d.rect(xx,yy,237,61,fill,stroke='none',radius=9)
                d.text(xx+118,yy+39,f'头 {h}：特征 {h*4}～{h*4+3}',24,col,550,'middle')
            if heads==2:
                d.text(base+610,yy+39,'本来只投影出 8 维，不是删掉后两头。',24,AMBER,550)
        box(base,1643,1100,194,'⑤ 接入注意力时：Q 头如何对应 K/V 头',[
            'Q₀ → K₀ / V₀     Q₁ → K₁ / V₁     Q₂ → K₂ / V₂     Q₃ → K₃ / V₃' if kv_heads==4 else
            'Q₀、Q₁ → K₀ / V₀                  Q₂、Q₃ → K₁ / V₁',
            '每个 Q 头有对应的独立 K/V 头。' if kv_heads==4 else '两组 Q 头：每组共享一个 K 头、一个 V 头；Q 仍然不同。',
            '共享的是 K/V 张量，不是把两个 Q 头合并，也不是让 Wk 与 Wv 相同。' if kv_heads==2 else 'K 与 V 由不同权重计算；即使形状相同，内容也不相同。'],col,fill,25)
        box(base,1895,1100,202,'一个输出数值，具体怎样算出来？',[
            '例如 K₀[b,t,j] = Σ(d=0…15) U[b,t,d] × Wk[d,j]'+(' + bk[j]' if kv_heads==4 else ''),
            '每一个输出特征，都使用该 Token 的全部 16 个输入特征。',
            ('j=0…15：算 16 个输出，再拆成 4 个头。' if kv_heads==4 else 'j=0…7：算 8 个输出，再拆成 2 个头。'),
        ],col,'white',25)
        d.text(base+20,2147,'本例 QKV 参数：816 = 3×16×16 + 3×16' if kv_heads==4 else
               '本例 QKV 参数：512 = 16×16 + 2×16×8',29,col,650)

    d.rect(60,2200,2280,215,PALE,stroke='none')
    d.text(85,2248,'读图时最关键的三点',32,AMBER,700)
    d.text(85,2296,'1. 先用完整 16 维输入做投影，再把输出拆头；不是先把输入切成 4 维，再分别投影。',29,INK)
    d.text(85,2343,'2. reshape / transpose 只改变分组和轴顺序；头数减少来自更窄的 K/V 投影矩阵。',29,INK)
    d.text(85,2390,'3. GQA 减少 K/V 投影与 KV Cache 的数据量；Q 头数仍为 4，不等于把所有注意力计算减半。',29,INK)
    d.text(65,2464,'范围：本图聚焦投影与拆头。Llama 3 在匹配 Q/K 前还需 RoPE；此处未展开旋转、mask、softmax 与 AV。',25,MUTED)
    d.text(65,2507,'公式采用 W[in,out]；PyTorch Linear.weight 存储为 [out,in]。GPT-2 可将三路合并为一次投影，数学上等价。',25,MUTED)
    d.text(65,2550,'来源：项目 GPT-2 前向实现、Meta llama3/llama/model.py；B/T/D 与头数为沿用前图的教学配置。',25,MUTED)
    s=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><title>QKV 投影与拆头对照</title><style>text {{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}}</style><rect width="{W}" height="{H}" fill="#f5f8fb"/>'+''.join(d.bg+d.edges+d.fg)+'</svg>'
    ET.fromstring(s)
    return s


def draw_example():
    """使用可复算的小矩阵，明确区分参数、投影输出与拆头。"""
    u=[1,2,3,4]
    wq=[[1,0,1,0],[0,1,0,1],[1,0,-1,0],[0,1,0,-1]]
    wk=[[1,0,1,0],[0,1,1,0],[1,0,0,1],[0,1,0,1]]
    wv=[[1,0,0,1],[0,1,1,0],[0,1,0,-1],[1,0,-1,0]]
    lk=[[1,1],[0,1],[1,0],[-1,1]]
    lv=[[1,0],[1,1],[0,-1],[1,0]]
    cases=[('GPT-2 · MHA',BLUE,'#edf2ff',[wq,wk,wv],[[1,0,0,1],[0,1,0,-1],[-1,0,1,0]]),
           ('Llama 3 · GQA',TEAL,'#e8f6f1',[wq,lk,lv],None)]
    d=Drawing(); report={}
    d.text(60,78,'数值例子：同一个输入，实际算出 Q / K / V',49,INK,700)
    d.text(60,131,'单独缩小为 B=1、T=1、D=4；Q 有 2 头，每头 2 维；GPT-2 的 K/V 各 2 头，Llama 各 1 头。',28,MUTED)
    d.text(60,179,'输入 U=[1,2,3,4] 视为归一化模块的输出；本例从投影开始，不演示上游归一化。',28,MUTED)
    d.rect(60,209,2280,100,PALE,stroke='none')
    d.text(84,251,'权重和 bias 均为人为选定的教学数值，不是预训练参数；不是把 GPT-2 权重裁剪后变成 Llama。',28,AMBER,600)
    d.text(84,288,'两边故意采用同一个 Wq，以隔离 Q 路 bias 的影响；真实模型没有这种跨模型权重共享约束。',26,AMBER)
    for ci,(name,col,fill,weights,biases) in enumerate(cases):
        base=60+1180*ci
        d.rect(base,345,1100,104,fill)
        d.text(base+20,387,name+'：输入 U [1,1,4]',32,col,700)
        d.text(base+20,426,'当前唯一 Token：u = [1, 2, 3, 4]',29,INK)
        outs=[]
        for k,(letter,weight) in enumerate(zip('QKV',weights)):
            x=base+k*375;cx=x+175;n=len(weight[0]);yy=521
            d.arrow([(base+550,449),(base+550,478),(cx,478),(cx,yy)],col)
            d.rect(x,yy,350,366,'white')
            d.text(x+20,yy+42,f'参数 W{letter.lower()} [4,{n}]',29,col,650)
            d.text(x+20,yy+78,'行=输入维；列=输出维',23,MUTED)
            left=x+38;top=yy+104;cw=66 if n==4 else 100
            for r,row in enumerate(weight):
                for j,value in enumerate(row):
                    d.rect(left+j*cw,top+r*43,cw-4,39,fill,stroke='none',radius=5)
                    d.text(left+j*cw+(cw-4)/2,top+r*43+28,str(value),25,col,550,'middle')
            bias=biases[k] if biases else [0]*n
            d.text(x+20,yy+320,('bias = '+str(bias)) if biases else '无 bias',25,MUTED)
            product=[sum(u[r]*weight[r][j] for r in range(4)) for j in range(n)]
            out=[a+b for a,b in zip(product,bias)];outs.append(out)
            d.arrow([(cx,887),(cx,936)],col)
            d.rect(x,936,350,198,fill)
            d.text(x+20,977,f'投影：{letter}₀ = uW'+(' + b' if biases else ''),28,col,650)
            d.text(x+20,1017,'uW = '+str(product),24,MUTED)
            d.text(x+20,1057,'输出 '+str(out),25,INK,650)
            d.text(x+20,1100,f'形状 [1,1,{n}]',25,MUTED)
            d.arrow([(cx,1134),(cx,1184)],col)
            d.rect(x,1184,350,202,'white')
            d.text(x+20,1225,'拆头：每头连续取 2 维',25,col,650)
            for h in range(n//2):
                d.text(x+20,1270+h*39,f'{letter} 头 {h} = {out[h*2:h*2+2]}',27,INK,550)
            d.text(x+20,1361,f'[1,1,{n//2},2] → [1,{n//2},1,2]',22,MUTED)
        report[name]={'input':u,'weights':weights,'biases':biases,'outputs':dict(zip('QKV',outs))}
        d.rect(base,1437,1100,199,PALE,stroke='none')
        d.text(base+22,1482,'展开一个数：K 的第 0 个输出怎么来？',30,AMBER,650)
        d.text(base+22,1528,'取 u 的全部 4 维，与 Wk 第 0 列逐项相乘求和。',27,INK)
        d.text(base+22,1572,'1×1 + 2×0 + 3×1 + 4×0 + bias[0](=0) = 4' if ci==0 else
               '1×1 + 2×0 + 3×1 + 4×(−1) = 0',29,col,650)
        d.text(base+22,1613,'其他输出同理：换一列权重，输入仍用完整 u。',25,MUTED)
        d.rect(base,1685,1100,220,fill)
        d.text(base+22,1732,'下一环节使用哪些头？',30,col,650)
        lines=['Q 头 0 [5,6]   → K 头 0 [4,7]；V 头 0 [4,5]',
               'Q 头 1 [−2,−1] → K 头 1 [3,6]；V 头 1 [−1,−2]'] if ci==0 else [
               'Q 头 0 [4,6]   → K 头 0 [0,7]；V 头 0 [7,−1]',
               'Q 头 1 [−2,−2] → 同一个 K 头 [0,7]；同一个 V 头 [7,−1]']
        for i,line in enumerate(lines):d.text(base+22,1780+i*47,line,27,INK)
        d.text(base+22,1871,'对应关系是按头建立；每个头内部仍有自己的二维向量。',25,MUTED)
    assert report['GPT-2 · MHA']['outputs']=={'Q':[5,6,-2,-1],'K':[4,7,3,6],'V':[4,5,-1,-2]}
    assert report['Llama 3 · GQA']['outputs']=={'Q':[4,6,-2,-2],'K':[0,7],'V':[7,-1]}
    d.text(65,1975,'轴顺序：[B,T,H,每头维] → [B,H,T,每头维]。本例 B=T=1，交换轴不改变这些数值，只改变索引方式。',27,MUTED)
    d.text(65,2025,'本例仅计算投影与拆头；位置取 t=0，RoPE 旋转角为 0。注意力打分、softmax 与 AV 未在此展开。',27,MUTED)
    d.text(65,2075,'本例只有一组 K/V，是 GQA 的单组特例（也称 MQA）；原流程图用两组，实际 Llama 3 有多组。',26,MUTED)
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="2120" viewBox="0 0 2400 2120"><style>text {{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}}</style><rect width="2400" height="2120" fill="#f5f8fb"/>'+''.join(d.bg+d.edges+d.fg)+'</svg>'
    return svg,report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--png',action='store_true');args=p.parse_args()
    out=OUT/'gpt2-llama3-qkv-projection-flow.svg'
    numeric,report=draw_example()
    detail=OUT/'gpt2-llama3-qkv-numerical-example.svg'
    detail.write_text(numeric,encoding='utf-8')
    root=ET.fromstring(draw())
    root.set('height',str(H+2120));root.set('viewBox',f'0 0 {W} {H+2120}')
    child=ET.fromstring(numeric);child.set('y',str(H));root.append(child)
    ET.register_namespace('','http://www.w3.org/2000/svg')
    out.write_text(ET.tostring(root,encoding='unicode'),encoding='utf-8')
    (OUT.parents[1]/'artifacts/qkv-projection-example-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.png:
        export_png(out)
        export_png(detail)
    print(out)
