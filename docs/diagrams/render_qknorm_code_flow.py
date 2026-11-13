"""QK-Norm 实现对照：从 U/W 的手算数据到 RMS、RoPE、缓存与 H。"""
import ast
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from render_ddp_allreduce_flow import Drawing,export_png
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
EPS=1e-5


def worked_example():
    u0,u1=[1.,0.,0.,0.],[0.,1.,0.,0.]
    wq=[[1.,3.,0.,0.],[2.,4.,0.,0.]]
    wk=[[4.,0.,0.,0.],[3.,5.,0.,0.]]
    wv=[[2.,0.,0.,0.],[0.,4.,0.,0.]]
    project=lambda u,w:[sum(a*b for a,b in zip(u,row)) for row in w]
    q=project(u1,wq);ks=[project(u,wk) for u in (u0,u1)];vs=[project(u,wv) for u in (u0,u1)]
    def norm(x):
        r=math.sqrt(sum(v*v for v in x)/len(x)+EPS)
        return [v/r for v in x],r
    def rotate(x,t):
        return [x[0]*math.cos(t)-x[1]*math.sin(t),x[0]*math.sin(t)+x[1]*math.cos(t)]
    outputs={}
    for enabled,key in ((False,'reference'),(True,'qknorm')):
        qn,rq=norm(q);knr=[norm(k) for k in ks]
        if not enabled: qn=q;kn=[k for k in ks]
        else: kn=[pair[0] for pair in knr]
        qr=rotate(qn,1);kr=[rotate(k,i) for i,k in enumerate(kn)]
        scores=[sum(a*b for a,b in zip(qr,k))/math.sqrt(2) for k in kr]
        e=[math.exp(s-max(scores)) for s in scores];a=[v/sum(e) for v in e]
        h=[sum(weight*v[j] for weight,v in zip(a,vs)) for j in range(2)]
        outputs[key]={'q_norm':qn,'k_norm':kn,'q_r':rq,'k_r':[r for _,r in knr],
                      'q_rope':qr,'k_rope':kr,'scores':scores,'a':a,'h':h}
    return {'eps':EPS,'u0':u0,'u1':u1,'wq_head1':wq,'wk':wk,'wv':wv,
            'q':q,'k':ks,'v':vs,'outputs':outputs}


def draw():
    locations={}
    for file in ('qknorm_model.py','qknorm_experiment.py','gqa_model.py','compare_qknorm_runs.py'):
        tree=ast.parse((ROOT/file).read_text());items={}
        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
                items[node.name]=[node.lineno,node.end_lineno]
                if isinstance(node,ast.ClassDef):
                    for f in node.body:
                        if isinstance(f,ast.FunctionDef):items[node.name+'.'+f.name]=[f.lineno,f.end_lineno]
        locations[file]=items
    def at(name,file='qknorm_model.py'):return f'{file} · {name} · L{locations[file][name][0]}'
    ex=worked_example();a=ex['outputs']['reference'];b=ex['outputs']['qknorm']
    vec=lambda values:'['+', '.join(f'{v:.6f}' for v in values)+']'
    d=Drawing();ink='#172b40';blue='#355fc0';teal='#087e75';red='#cf4141'
    def tx(x,y,s,size=27,color=ink,bold=400):d.text(x,y,s,size,color,bold)
    def box(x,y,w,h,title,rows,color=blue,changed=False):
        d.rect(x,y,w,h,'#fff8ed' if changed else 'white',red if changed else '#d9e4ed')
        tx(x+24,y+44,title,31,color,700)
        for i,s in enumerate(rows):tx(x+24,y+89+40*i,s)
    def band(y,title):
        d.rect(60,y,2280,60,'#eaf0f5',stroke='none',radius=12);tx(82,y+40,title,30,ink,700)
    def arrows(y):
        d.arrow([(610,y),(610,y+34)],blue);d.arrow([(1790,y),(1790,y+34)],teal)
    tx(60,78,'QK-Norm 实践：原始 Q/K → 头内归一化 → RoPE → KV Cache',44,ink,700)
    tx(60,132,'左：原 GQA 参照   右：新增 QK-Norm   红框标差异；两边均为 4 Q / 2 KV，缓存容量不变。',28)
    box(60,174,1100,176,'A · run_qknorm_reference.py → main(False)',[
        'q_norm / k_norm = Identity；前课 GQA 计算不变。',
        '相同随机起点、合成 ID、CPU / FP32。'],blue)
    box(1240,174,1100,176,'B · run_qknorm.py → main(True)',[
        '每层新增 gamma_Q[32]、gamma_K[32]，初始全 1。',
        '两层共 128 个新参数；无 beta，V 不归一化。'],teal,True)
    band(388,'01  共同投影与拆头 · '+at('QKNormAttention.step'))
    box(60,466,2280,216,'[N2] 输入是当前 Block 入口归一化后的 U；Q/K/V 是投影结果，不是权重',[
        '默认：U[B,T,128]；WQ[128,128]，WK/WV[64,128]（PyTorch [out,in]）。',
        'q=U WQᵀ → [B,4,T,32]；k=U WKᵀ、v=U WVᵀ → 各 [B,2,T,32]。',
        '入口 RMSNorm 沿 128 维；新增 Q/K-Norm 沿每头 32 维。原 U 已归一化，不代表投影后的 q/k 尺度不变。'],blue)
    band(720,'02  缩小成可手算的输入：D=4、Hq=2、Hkv=1、d=2；只展示 Q 头1，位置0与位置1')
    box(60,798,2280,296,'教学数值：从输入和参数算出原始 q/k/v；不代表默认模型实测',[
        'U₀=[1,0,0,0]；U₁=[0,1,0,0]。以下 W 是模型参数，均按 [out,in] 列出。',
        'WQ头1=[[1,3,0,0],[2,4,0,0]]；q₁=U₁WQ头1ᵀ=[0×1+1×3, 0×2+1×4]=[3,4]。',
        'WK=[[4,0,0,0],[3,5,0,0]]；k₀=U₀WKᵀ=[4,3]；k₁=U₁WKᵀ=[0,5]。',
        'WV=[[2,0,0,0],[0,4,0,0]]；v₀=[2,0]；v₁=[0,4]。',
        '下标0/1在这里表示 Token 位置；γQ=γK=[1,1]，ε=0.00001；此图计算保留 ε。'],teal)
    band(1132,'03  在此分叉 · [N3] 调用 '+at('HeadRMSNorm.forward'))
    box(60,1210,1100,376,'A · Identity：原值直接作为 RoPE 输入',[
        'q₁=[3,4]。',
        'k₀=[4,3]；k₁=[0,5]。',
        '没有本步骤的 gamma，也没有均方根计算。',
        '注意：上游 Block 入口的 RMSNorm 仍然存在。',
        'V 不经过 QK-Norm，也不经过 RoPE。',
        '原 V₀=[2,0]、V₁=[0,4] 一直保留到 AV。'],blue)
    box(1240,1210,1100,376,'B · 每条向量独立算 r，再除 r、乘 gamma',[
        'q₁²=[9,16] → 和=25 → 均值=12.5。',
        f'rq=√(12.5+0.00001)={b["q_r"]:.6f}。',
        f'q₁/rq × [1,1] = {vec(b["q_norm"])}。',
        'k₀²=[16,9]；k₁²=[0,25] → 各自均值也为12.5。',
        f'k₀/rk₀ × [1,1] = {vec(b["k_norm"][0])}。',
        f'k₁/rk₁ × [1,1] = {vec(b["k_norm"][1])}。',
        '三条 r 本例恰好相同；实际必须逐条算，不能共用。'],teal,True)
    band(1624,'04  RoPE 接收上一步的输出 · [N4]；d=2 的唯一频率为1，位置1旋转1弧度')
    box(60,1702,1100,296,'A · 旋转未经 QK-Norm 的原值',[
        'cos(1)=0.540302；sin(1)=0.841471。',
        'q′₁=[3cos−4sin, 3sin+4cos]。',
        f'q′₁={vec(a["q_rope"])}。',
        f'k′₀={vec(a["k_rope"][0])}（位置0不转）。',
        f'k′₁={vec(a["k_rope"][1])}。'],blue)
    box(1240,1702,1100,296,'B · 旋转归一化/缩放后的值',[
        '令 a=0.848528、b=1.131370（上一步的 q）。',
        'q′₁=[a cos−b sin, a sin+b cos]；角度仍为1。',
        f'q′₁={vec(b["q_rope"])}。',
        f'k′₀={vec(b["k_rope"][0])}。',
        f'k′₁={vec(b["k_rope"][1])}。'],teal,True)
    band(2036,'05  旧缓存 + 新 K/V · [N4] 沿位置轴 cat；这里只展开一层、一个 KV 头')
    box(60,2114,1100,216,'A · K 缓存保存原投影再旋转的结果',[
        f'旧行 k′₀={vec(a["k_rope"][0])}。',
        f'追加新行 k′₁={vec(a["k_rope"][1])}。',
        'K两行；V两行=[2,0]、[0,4]；当前位置 q′₁ 不缓存。'],blue)
    box(1240,2114,1100,216,'B · K 缓存保存归一化再旋转的结果',[
        f'旧行 k′₀={vec(b["k_rope"][0])}，直接复用。',
        f'追加新行 k′₁={vec(b["k_rope"][1])}。',
        '旧 K 不重复 Norm/RoPE；V 与左侧完全相同。'],teal,True)
    band(2368,'06  GQA 的 QK → A → H 规则不变 · [N5] grouped_attention；本例两个位置均可见')
    for x,values,title,color,changed in ((60,a,'A · 分数与输出',blue,False),(1240,b,'B · 尺度改变，分数与输出随之改变',teal,True)):
        box(x,2446,1100,296,title,[
            '分数=[q′₁·k′₀, q′₁·k′₁] / √2。',
            f'scores={vec(values["scores"])}。',
            f'A=softmax(scores)={vec(values["a"])}。',
            'H=A₀×[2,0]+A₁×[0,4]。',
            f'H={vec(values["h"])}；图中显示值已舍入。'],color,changed)
    band(2780,'07  完整模型继续流转 · gqa_model.GroupedDecoder；训练走普通模型 forward')
    box(60,2858,2280,216,'每个 Q 头各自得到 H → 拼头/输出投影 → 残差 → Norm/SwiGLU/残差 → 下一层',[
        '推理：各层只处理新 Token；最终 Norm → 末位置 lm_head → logits → 选 ID → 下一轮（携带各层缓存）。',
        '训练检查：独立副本输入完整序列 → 全位置 logits → 交叉熵均值 → backward → AdamW.step，无 KV Cache。',
        '新增 gamma 已注册为 Parameter，优化器包含它们；推理原模型不更新，训练副本用完丢弃。'],teal)
    band(3112,'08  默认规模的参数、缓存与核验 · '+at('main','qknorm_experiment.py'))
    box(60,3190,1100,256,'A · 原 GQA：4 Q / 2 KV，2层',[
        '参数：6,793,728；不含新增 gamma。',
        '提示32 + 生成16 → 末次缓存长47。',
        '每层 K/V 各 [1,2,47,32]。',
        '全部层 KV 有效字节 48,128。'],blue)
    box(1240,3190,1100,256,'B · GQA + QK-Norm',[
        '新增：2层 × (32个γQ + 32个γK) = 128。',
        '参数：6,793,856；同层同类头共享 gamma。',
        '每条向量独立统计；缓存形状仍 [1,2,47,32]。',
        '全部层 KV 仍 48,128 字节；K 的数值改变。'],teal,True)
    box(60,3484,2280,176,'compare_qknorm_runs.py：各模型内部检查缓存等价；两个版本之间不要求 logits 相同',[
        '核对共同参数初值 → 各自重算/缓存分数与贪心 ID → 参数 +128 / 缓存字节不变 → 副本 gamma 梯度与更新。',
        'gamma=1 仍执行 x/r，不等于 Identity。单步合成数据检查只验证训练链路，不证明质量提升或完整训练完成。'],blue)
    tx(70,3720,'来源：Transformers v4.51.3 Qwen3Attention / Qwen3RMSNorm；本课沿用相邻配对 RoPE、共享 lm_head、ε=1e-5。',26)
    tx(70,3762,'只新增 QK-Norm，不声称复现完整 Qwen3。每层 Norm 参数独立；修改模型或上下文后必须重新建立缓存。',26)
    for y in (350,682,1094,1586,1998,2330,2742,3074,3446):arrows(y)
    svg='<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="3820" viewBox="0 0 2400 3820"><title>QK-Norm 代码与数值对照</title><style>text{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}</style><rect width="2400" height="3820" fill="#f5f8fb"/>'+''.join(d.bg+d.edges+d.fg)+'</svg>'
    ET.fromstring(svg)
    return svg,{'source_lines':locations,'example':ex,'gamma_parameters_added':128,'cache_bytes_both':48128}


if __name__=='__main__':
    svg,check=draw();path=OUT/'qknorm-code-comparison.svg';path.write_text(svg)
    (ROOT/'artifacts/qknorm-diagram-check.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
    export_png(path);print(path)
