"""生成 SwiGLU 实践图：代码位置、近似等参数宽度、完整数值与训练接入。"""
import ast
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from render_ddp_allreduce_flow import Drawing, export_png

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def draw():
    spans={}
    for filename in ('swiglu_model.py','swiglu_training.py'):
        tree=ast.parse((ROOT/filename).read_text())
        spans[filename]={n.name:[n.lineno,n.end_lineno] for n in tree.body
                         if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    def at(name,file='swiglu_model.py'):
        return f'{name} · L{spans[file][name][0]}–{spans[file][name][1]}'
    g=[1/(1+math.exp(-1)),-2/(1+math.exp(2))]
    u=[g[0]*2,g[1]*6]; result=[u[0]+u[1],u[0]-u[1]]
    d=Drawing(); ink='#172b40';blue='#355fc0';teal='#087e75';red='#cf4141'
    def text(x,y,s,size=28,color=ink):d.text(x,y,s,size,color)
    def box(x,y,w,h,title,rows,color=blue,fill='white',size=28,changed=False):
        d.rect(x,y,w,h,fill)
        d.text(x+26,y+45,title,32,color,700)
        for i,s in enumerate(rows):text(x+26,y+92+i*42,s,size)
        if changed:d.rect(x+6,y+6,w-12,h-12,'none',red,16)
    def arrow(points,color=teal):d.arrow(points,color,width=4)
    d.text(60,80,'SwiGLU 实践：只替换 MLP，追踪每一步数据',48,ink,700)
    text(60,137,'参照主干已包含 RMSNorm、RoPE、无 bias；红框标出改动。两边随机起步，不加载上轮训练权重。')
    box(60,185,1100,205,'A · train_swiglu_reference.py',[
        "swiglu_training.main('gelu') → run → build_model",
        'GELU：128 → 512 → 128；两块权重。',
        '整模型 6,826,752 参数。'],blue,'#eef2ff',27)
    box(1240,185,1100,205,'B · train_swiglu.py',[
        "swiglu_training.main('swiglu') → run → build_model",
        'SwiGLU：两路 128 → 341，再 341 → 128。',
        '整模型 6,826,496 参数；三块权重。'],teal,'#e8f6f1',27)
    arrow([(610,390),(610,435)],blue);arrow([(1790,390),(1790,435)])
    box(60,435,2280,205,'01  参数预算：'+at('experiment_settings'),[
        'GELU：D×4D + 4D×D = 8D²；SwiGLU：D×F + D×F + F×D = 3DF。',
        '令 3DF≈8D² → F≈8D/3；D=128 → 理想值 341.333…，取最近整数 341。',
        '每层 131,072 对 130,944：少 128（约 0.098%），近似匹配；不是严格相等，也未做 GPU 倍数对齐。'],blue)
    arrow([(1200,640),(1200,685)])
    box(60,685,2280,205,'02  进入 MLP 前，两边经过相同类型的网络环节',[
        'Token ID [2,128] → Embedding → 每层 Norm1 → QKV/RoPE → 注意力/输出投影 → 第一次残差。',
        '得到主干 R[2,128,128] → z = block.ln_2(R)，仍为 [2,128,128]。',
        '接下来调用 block.mlp(z)。不同模型的数值可不同；相同的是主干结构与 MLP 以外的参数初值。'],blue)
    arrow([(610,890),(610,935)],blue);arrow([(1790,890),(1790,935)])
    box(60,935,1100,350,'03 A · 原 MLP.forward（gpt2_model.py）',[
        '输入 z[2,128,128]；本课所有线性层无 bias。',
        'a = c_fc(z) → [2,128,512]',
        'g = GELU(a) → [2,128,512]，tanh 近似。',
        'out = c_proj(g) → [2,128,128]',
        '权重存储：[512,128]、[128,512]。',
        'Dropout(p=0) → 返回分支输出。'],blue,size=27)
    box(1240,935,1100,350,'03 B · '+at('SwiGLU'),[
        '同一个 z，分别交给 gate_proj 和 up_proj。',
        'a = gate_proj(z)；v = up_proj(z)',
        'g = SiLU(a)；u = g * v',
        'out = down_proj(u) → [2,128,128]',
        '权重：[341,128] 两块；[128,341] 一块。',
        'Dropout(p=0) → 返回分支输出。'],teal,'#fff5e7',27,True)
    arrow([(1790,1285),(1790,1330)])
    box(60,1330,2280,160,'04  展开 B：每个位置独立算；下面的数组均是激活值，不是权重',[
        'gate_proj：z[2,128,128] → a[2,128,341]；up_proj：同一个 z → v[2,128,341]。',
        'Linear 计算 input @ weight.T；所有位置共用权重，不在不同 Token 之间混合。'],teal)
    arrow([(610,1490),(610,1535)]);arrow([(1790,1490),(1790,1535)])
    box(60,1535,1100,205,'门控支路 · [S1] → [S2]',[
        'a = gate_proj(z)',
        'g = F.silu(a) = a / (1 + exp(−a))',
        'g[2,128,341]；可为负、可大于 1。'],teal,'#fff5e7',28,True)
    box(1240,1535,1100,205,'内容支路 · [S1]',[
        'v = up_proj(z)',
        'v[2,128,341]；使用独立的可训练矩阵。',
        '这里的 v 是 MLP 内容，不是注意力 V。'],teal,'#fff5e7',28,True)
    arrow([(610,1740),(610,1770),(1200,1770),(1200,1810)])
    arrow([(1790,1740),(1790,1770),(1200,1770)])
    box(60,1810,2280,205,'05  [S3] 逐元素乘 → [S4] 输出投影 → 回到 Block',[
        'u[b,t,j] = g[b,t,j] × v[b,t,j]；u[2,128,341]，这个 * 不是矩阵乘法。',
        'out = down_proj(u) = u @ down_proj.weight.T → [2,128,128]；MLP 返回 out。',
        'Block.forward 随后计算 R + out → [2,128,128]；这里加的是保留的 R，不是归一化后的 z。'],teal,'#fff5e7',28,True)
    arrow([(1200,2015),(1200,2060)])
    box(60,2060,2280,290,'06  具体数值：另取一个位置 D=2、F=2；教学选值，不用于参数量对照',[
        '输入 z=[1,2]。数学右乘矩阵 Wg=[[1,0],[0,−1]]；Wu=[[2,0],[0,3]]；Wd=[[1,1],[1,−1]]。',
        'a=zWg=[1,−2]；v=zWu=[2,6]。PyTorch .weight 存储上述矩阵的转置。',
        f'g=[1/(1+exp(−1)), −2/(1+exp(2))] ≈ [{g[0]:.6f}, {g[1]:.6f}]。',
        f'u=g*v=[{g[0]:.6f}×2, {g[1]:.6f}×6] ≈ [{u[0]:.6f}, {u[1]:.6f}]。',
        f'out=uWd=[u₀+u₁, u₀−u₁] ≈ [{result[0]:.6f}, {result[1]:.6f}]；分支输出依旧是 2 维。'],teal,'#e8f6f1',27)
    arrow([(1200,2350),(1200,2395)])
    box(60,2395,2280,245,'07  回到训练 · swiglu_training.py：'+at('train_update','swiglu_training.py'),[
        '固定配置/原文/90%–10% 切分 → 构造模型并完成替换 → 创建 AdamW → 验证初始 loss。',
        '取 x/y[2,128] → 完整前向/交叉熵 → (loss/4).backward()；四个微批的梯度累加到各参数 .grad。',
        '乘法反向同时流向 gate/up 两路；三块权重均得到梯度 → 裁剪 → AdamW 更新一次 → 清梯度。',
        '100 次更新，每 20 次及末次固定窗口验证；保存独立 report/metrics/checkpoint。旧代码和结果保留。'],blue, size=27)
    arrow([(1200,2640),(1200,2685)])
    box(60,2685,2280,205,'08  compare_swiglu_runs.py：相同协议后再比较',[
        '核对数据、配置、MLP 外共同初始权重、设备、源码和实际微批次顺序。MLP 本身不声称初值相同。',
        'A/B 分别顺序实跑；比较同窗口验证 loss、参数量和 Token/s。CPU 速度不代表 GPU 速度。',
        '能力：学会接入门控 MLP 与控制容量；边界：单种子短跑不保证改善，不是完整 Llama 模型。'],blue, size=27)
    text(60,2960,'来源：GLU Variants Improve Transformer；Meta Llama 3 FeedForward。数值例子和本机配置是教学改编。',27)
    svg='<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="3020" viewBox="0 0 2400 3020"><title>SwiGLU 与 GELU 代码和数值对照</title><style>text{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}</style><rect width="2400" height="3020" fill="#f5f8fb"/>'+''.join(d.bg+d.edges+d.fg)+'</svg>'
    ET.fromstring(svg)
    return svg,{'source_lines':spans,'example_is_synthetic':True,'gate':g,'gated':u,'output':result,
                'gelu_mlp_parameters':131072,'swiglu_mlp_parameters':130944}


if __name__=='__main__':
    svg,check=draw();path=OUT/'swiglu-code-comparison.svg';path.write_text(svg)
    (ROOT/'artifacts/swiglu-diagram-check.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
    export_png(path)
    print(path)
