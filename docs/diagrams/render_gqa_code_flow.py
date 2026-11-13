"""第21块：MHA/GQA 与缓存的单张对照图；源码位置从 AST 提取，例子可复算。"""
import ast
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from render_ddp_allreduce_flow import Drawing, export_png

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def draw():
    locations={}
    for filename in ('gqa_model.py','gqa_experiment.py','kv_cache_experiment.py','compare_gqa_runs.py'):
        tree=ast.parse((ROOT/filename).read_text())
        items={}
        for n in tree.body:
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)):
                items[n.name]=[n.lineno,n.end_lineno]
                if isinstance(n,ast.ClassDef):
                    for method in n.body:
                        if isinstance(method,ast.FunctionDef):
                            items[n.name+'.'+method.name]=[method.lineno,method.end_lineno]
        locations[filename]=items
    def at(name,file='gqa_model.py'):
        return f'{file} · {name} · L{locations[file][name][0]}'
    def calc(q,k,v):
        s=[sum(a*b for a,b in zip(q,row))/math.sqrt(2) for row in k]
        e=[math.exp(t-max(s)) for t in s];a=[t/sum(e) for t in e]
        h=[sum(w*row[j] for w,row in zip(a,v)) for j in range(2)]
        return s,a,h
    k0=[[1,0],[0,1]];k1=[[3,0],[0,3]]
    v0=[[10,0],[0,20]];v1=[[2,4],[6,8]]
    kg=[[2,0],[0,2]];vg=[[6,2],[3,14]]
    ms,ma,mh=calc([0,1],k1,v1)
    gs,ga,gh=calc([0,1],kg,vg)
    _,ga0,gh0=calc([1,0],kg,vg)
    d=Drawing();ink='#172b40';muted='#51697c';blue='#355fc0';teal='#087e75';red='#cf4141'
    def tx(x,y,s,size=27,color=ink,bold=400):d.text(x,y,s,size,color,bold)
    def box(x,y,w,h,title,rows,color=blue,changed=False):
        d.rect(x,y,w,h,'#fff8ed' if changed else 'white',red if changed else '#d9e4ed')
        tx(x+24,y+44,title,31,color,700)
        for i,s in enumerate(rows): tx(x+24,y+88+40*i,s)
    def band(y,title):
        d.rect(60,y,2280,60,'#eaf0f5',stroke='none',radius=12)
        tx(82,y+40,title,30,ink,700)
    def pair_arrow(y):
        d.arrow([(610,y),(610,y+35)],blue)
        d.arrow([(1790,y),(1790,y+35)],teal)
    def mat(x,y,values,color):
        # 标准二维矩阵；每行=Token位置，每列=头内特征。
        cw=94;rh=52
        for i,row in enumerate(values):
            for j,value in enumerate(row):
                d.rect(x+j*cw,y+i*rh,cw-4,rh-4,'#eef3fa',stroke='none',radius=5)
                tx(x+j*cw+cw/2,y+i*rh+35,f'{value:g}',29,color,600)
        d.fg.append(f'<path d="M{x-4},{y-4} h-10 v{len(values)*rh+4} h10 M{x+len(values[0])*cw},{y-4} h10 v{len(values)*rh+4} h-10" fill="none" stroke="{color}" stroke-width="3"/>')
    tx(60,78,'GQA 代码实践：保留 4 个 Q 头，减少 K/V 头与缓存',46,ink,700)
    tx(60,132,'左：MHA 参照   右：GQA 改动   红色边框 = 差异；两边都可完整重算，也都可使用 KV Cache。',29)
    box(60,174,1100,176,'A · generate_gqa_reference.py → main(4)',[
        'Q = 4 头；K = 4 头；V = 4 头。',
        '独立新入口；不覆盖上一课 generate_kv_cache.py。'],blue)
    box(1240,174,1100,176,'B · generate_gqa.py → main(2)',[
        'Q = 4 头；K = 2 头；V = 2 头。',
        '本课随机权重、合成 ID、CPU / FP32；不训练。'],teal,True)
    band(388,'01  构建对照 · '+at('main','gqa_experiment.py'))
    box(60,466,1100,216,'相同起点 → 分开 Q/K/V 投影',[
        '同 seed/spec 构建 RoPE + RMSNorm + SwiGLU 模型。',
        '复制原 Q、K、V；其余权重保持不变。',
        'D=128，2 层，Hq=4，d=32，最大长度 256。'],blue)
    box(1240,466,1100,216,'[A0] 在构建时平均组内 K/V 权重',[
        '新 WK₀=(旧 WK₀+旧 WK₁)/2；WV 同样处理。',
        '新 WK₁=(旧 WK₂+旧 WK₃)/2；Q 和其余权重复制。',
        '只做一次；不是推理时平均缓存，也不保证质量不变。'],teal,True)
    band(720,'02  只投影本轮输入 · '+at('GroupedAttention.step'))
    tx(80,818,'共同输入：prefill ID[1,32] 或 decode ID[1,1] → Embedding → 当前 Block 的 RMSNorm → U[1,T,128]。',28)
    box(60,848,1100,256,'[A1] MHA：三个投影输出同宽',[
        'Qflat=U WQᵀ；Kflat=U WKᵀ；Vflat=U WVᵀ。',
        'PyTorch 参数 WQ/WK/WV 各 [128,128]；无 bias。',
        '三个输出各 [1,T,128] → reshape + transpose。',
        'Q / K / V 各 [1,4,T,32]。'],blue)
    box(1240,848,1100,256,'[A1] GQA：K/V 输出宽度缩小',[
        'Qflat=U WQᵀ；Kflat=U WKᵀ；Vflat=U WVᵀ。',
        'WQ[128,128]；WK/WV 各 [64,128]；无 bias。',
        '输出 Qflat[1,T,128]；Kflat/Vflat 各 [1,T,64]。',
        'Q[1,4,T,32]；K/V 各 [1,2,T,32]。'],teal,True)
    band(1142,'03  RoPE 与缓存追加 · [A2–A3]；以下展开 S=32、T=1 的第一次 decode')
    box(60,1220,1100,256,'MHA：每层 4 组 K/V',[
        '新 Q/K 用位置 32 旋转，V 不旋转。',
        '旧 K[1,4,32,32] + 新 K[1,4,1,32]',
        'cat(dim=2) → K[1,4,33,32]；V 同理。',
        '旧 K 已旋转，直接复用；每层分别保存。'],blue)
    box(1240,1220,1100,256,'GQA：每层仅保存 2 组 K/V',[
        '新 Q/K 同样用位置 32；头数不同也能旋转。',
        '旧 K[1,2,32,32] + 新 K[1,2,1,32]',
        'cat(dim=2) → K[1,2,33,32]；V 同理。',
        '位置一个不少；缓存不重复成 4 组。'],teal,True)
    band(1514,'04  对应头 → 分数 → A → H · '+at('grouped_attention'))
    box(60,1592,1100,296,'MHA：每个 Q 头有自己的一组 K/V',[
        'Q₀→KV₀；Q₁→KV₁；Q₂→KV₂；Q₃→KV₃。',
        '分组视图 Q[1,4,1,1,32]；每组只有 1 个 Q。',
        'Sₕ = Qₕ Kₕᵀ / √32；遮罩未来位置为 −∞。',
        'Aₕ = softmax(Sₕ + M)，沿 33 个位置归一化。',
        'Hₕ = Aₕ Vₕ；合计 H[1,4,1,32]。'],blue)
    box(1240,1592,1100,296,'GQA：同组共享 K/V，分别计算 A 和 H',[
        'Q₀/Q₁→KV₀；Q₂/Q₃→KV₁；g(h)=floor(h/2)。',
        'Q[1,2,2,1,32]；K 加组内轴 [1,2,1,33,32]。',
        'Sₕ = Qₕ K_g(h)ᵀ / √32；遮罩与 softmax 相同。',
        'A 的分组形状 [1,2,2,1,33]，共 4 份权重。',
        'Hₕ = Aₕ V_g(h) → H[1,4,1,32]；没有少一头。'],teal,True)
    band(1926,'05  数值展开：只看第 0 组的两个 Q；d=2、位置=0/1，当前 query 在位置 1')
    tx(80,2025,'下列是教学选取的投影/RoPE 后数据，并非默认模型实测。Q₀=[1,0]，Q₁=[0,1]；两个历史位置都可见。',27)
    box(60,2055,1100,405,'MHA · Q₁ 使用原 KV₁',[],blue)
    tx(90,2151,'K₁：',28,blue,700);mat(185,2115,k1,blue)
    tx(460,2151,'V₁：',28,blue,700);mat(555,2115,v1,blue)
    tx(90,2268,'Q₁ K₁ᵀ / √2 = [0,3] / √2 = [0,2.121320]。')
    tx(90,2310,f'A₁ = softmax([0,2.121320]) = [{ma[0]:.6f}, {ma[1]:.6f}]。')
    tx(90,2352,'H₁ = A₁,₀ × [2,4] + A₁,₁ × [6,8]。')
    tx(90,2406,f'输出 H₁ = [{mh[0]:.6f}, {mh[1]:.6f}]',31,blue,700)
    box(1240,2055,1100,405,'GQA · Q₁ 改为使用共享 KV₀',[],teal,True)
    tx(1270,2151,'K₀：',28,teal,700);mat(1365,2115,kg,teal)
    tx(1640,2151,'V₀：',28,teal,700);mat(1735,2115,vg,teal)
    tx(1270,2268,'Q₁ K₀ᵀ / √2 = [0,2] / √2 = [0,1.414214]。')
    tx(1270,2310,f'A₁ = softmax([0,1.414214]) = [{ga[0]:.6f}, {ga[1]:.6f}]。')
    tx(1270,2352,'H₁ = A₁,₀ × [6,2] + A₁,₁ × [3,14]。')
    tx(1270,2406,f'输出 H₁ = [{gh[0]:.6f}, {gh[1]:.6f}]',31,teal,700)
    box(60,2494,2280,216,'这个例子如何得到共享数据？同一层、相同输入下的教学对照',[
        '原 KV₀：K=[[1,0],[0,1]]，V=[[10,0],[0,20]]；原 KV₁ 如左图。对应元素平均得到右图 K/V。',
        '代码实际平均的是 WK/WV 权重；同输入的线性投影和相同位置旋转允许此例在激活上展示平均结果。',
        f'共享同一 V 的 Q₀ 得到 A₀=[{ga0[0]:.6f},{ga0[1]:.6f}] → H₀=[{gh0[0]:.6f},{gh0[1]:.6f}]，仍与 H₁ 不同。'],teal)
    band(2748,'06  拼头、Block、选择下一 Token · '+at('GroupedDecoder.forward'))
    box(60,2826,2280,216,'[A5 / D1] 两边输出形状相同，数值不要求相同',[
        'H[1,4,1,32] → transpose/reshape → C[1,1,128] → C WOᵀ → 注意力分支[1,1,128]。',
        '加回 Block 输入 → RMSNorm → SwiGLU → 第二次残差 → 下一 Block（使用该层自己的缓存）。',
        '最终 RMSNorm → 末位置 lm_head → logits[1,1,50257] → argmax 选 ID → 追加输出；下一轮只输入此 ID。'],blue)
    band(3080,'07  实际缓存与检查 · '+at('compare','compare_gqa_runs.py'))
    box(60,3158,1100,256,'MHA · 默认提示 32，生成 16 个 Token',[
        'prefill 一次 + decode 15 次 → 最后缓存长 47。',
        '每层 K / V 各 [1,4,47,32]，共 2 层。',
        '字节 = 2×2×1×4×47×32×4 = 96,256。',
        '最后刚选出的 Token 未再输入，输出总长 48。'],blue)
    box(1240,3158,1100,256,'GQA · 同样保留全部 47 个位置',[
        '每层 K / V 各 [1,2,47,32]，共 2 层。',
        '字节 = 2×2×1×2×47×32×4 = 48,128。',
        '存活 KV 张量减半；不等于整个模型内存减半。',
        '注意力仍有 4 个 Q 头，不能推导耗时必减半。'],teal,True)
    tx(70,3478,'核验：各模型内部，缓存与完整重算 logits 近似一致；跨 MHA/GQA 只比参数、缓存和计时，不要求输出相同。',28,ink,600)
    tx(70,3522,'边界：动态 cat 有复制开销；分组广播没有显式 repeat 缓存，但内核可有临时工作区。仅实跑 CPU，不外推 GPU/质量。',27,muted)
    tx(70,3566,'来源：GQA 论文 §2 的组内权重平均、Meta Llama 3 Attention；本图为本地教学实现，未执行论文适配训练。',27,muted)
    for y in (350,682,1104,1476,1888,2710,3042):
        pair_arrow(y)
    svg='<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="3620" viewBox="0 0 2400 3620"><title>GQA 与 MHA 代码和缓存对照</title><style>text{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}</style><rect width="2400" height="3620" fill="#f5f8fb"/>'+''.join(d.bg+d.edges+d.fg)+'</svg>'
    ET.fromstring(svg)
    return svg, {'source_lines':locations,'toy_inputs':{'q1':[0,1],'mha_k':k1,'mha_v':v1,'gqa_k':kg,'gqa_v':vg},
                 'toy_outputs':{'mha_scores':ms,'mha_a':ma,'mha_h':mh,'gqa_scores':gs,'gqa_a':ga,'gqa_h':gh},
                 'default_cache_bytes':[96256,48128], 'synthetic_not_measured':True}


if __name__=='__main__':
    svg,check=draw()
    p=OUT/'gqa-code-comparison.svg';p.write_text(svg)
    (ROOT/'artifacts/gqa-diagram-check.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
    export_png(p)
    print(p)
