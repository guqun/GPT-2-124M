"""绘制缓存与重算对照：输入位置、逐层 K/V、RoPE 偏移、遮罩和数值例子。"""
import ast
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from render_ddp_allreduce_flow import Drawing,export_png

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def draw():
    spans={}
    for filename in ('kv_cache_model.py','kv_cache_experiment.py'):
        tree=ast.parse((ROOT/filename).read_text())
        spans[filename]={n.name:[n.lineno,n.end_lineno] for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    def at(name,file='kv_cache_model.py'):
        return f'{name} · L{spans[file][name][0]}–{spans[file][name][1]}'
    scores=[v/math.sqrt(2) for v in [1,0,1,2]]
    exps=[math.exp(s-max(scores)) for s in scores];a=[v/sum(exps) for v in exps]
    values=[[1,0],[0,2],[1,1],[2,0]]
    result=[sum(w*v[j] for w,v in zip(a,values)) for j in range(2)]
    d=Drawing();ink='#172b40';blue='#355fc0';teal='#087e75';red='#cf4141'
    def text(x,y,s,size=28,color=ink):d.text(x,y,s,size,color)
    def box(x,y,w,h,title,rows,color=blue,fill='white',changed=False):
        d.rect(x,y,w,h,fill);d.text(x+26,y+44,title,32,color,700)
        for i,s in enumerate(rows):text(x+26,y+90+41*i,s,27)
        if changed:d.rect(x+6,y+6,w-12,h-12,'none',red,16)
    def arrow(points,color=teal):d.arrow(points,color,width=4)
    d.text(60,80,'KV Cache：相同权重，复用历史 K/V',48,ink,700)
    text(60,136,'只优化推理执行；保留 RoPE / RMSNorm / SwiGLU / MHA。缓存是激活值，不是参数，不是磁盘模型文件。')
    box(60,185,1100,205,'A · generate_cache_reference.py',[
        '每一轮重新送入完整 Token 前缀。',
        'ABC → 选 D；ABCD → 选 E；ABCDE → 选 F。',
        '调用 generate(..., use_cache=False)。'],blue,'#eef2ff')
    box(1240,185,1100,205,'B · generate_kv_cache.py',[
        '第一次输入完整提示，之后只送最新 Token。',
        'ABC → 选 D；D + 缓存 → 选 E；E + 缓存 → 选 F。',
        '调用 generate(..., use_cache=True)。'],teal,'#e8f6f1',True)
    arrow([(610,390),(610,435)],blue);arrow([(1790,390),(1790,435)])
    box(60,435,2280,205,'01  共同准备 · kv_cache_experiment.main',[
        '默认相同 seed=42 随机小模型；也可两边明确指定同一份前课 SwiGLU 检查点。',
        'decoder = CachedDecoder(model).to(device).eval()；inference_mode 下不建反向图，不更新参数。',
        '默认合成提示 ID [1,2,…,32]，形状 [1,32]；固定贪心生成 16 个新 ID，不评价文本质量。'],blue)
    arrow([(1200,640),(1200,685)])
    box(60,685,2280,205,'02  首轮 prefill · '+at('generate','kv_cache_experiment.py'),[
        '两边完整处理提示。每层：Norm1 → QKV/RoPE → 注意力 → 残差 → Norm2/SwiGLU → 残差。',
        '缓存版分别保留每一层的 (已旋转 K, 未旋转 V)，各 [1,4,32,32]；没有保存旧 Q。',
        '最终 RMSNorm → 末位置 lm_head → logits[1,1,50257] → argmax 选第一个新 Token。'],teal)
    arrow([(610,890),(610,935)],blue);arrow([(1790,890),(1790,935)])
    box(60,935,1100,245,'03 A · 下一轮完整前缀重算',[
        '输入 ids[1,33]：提示 32 个 + 刚选出的 1 个。',
        '每层重新投影全部 33 个位置的 Q/K/V。',
        '分数形状 [1,4,33,33]；旧位置 MLP 也重算。',
        '最后仍然只取末位置的词表 logits。'],blue)
    box(1240,935,1100,245,'03 B · 下一轮 decode',[
        '输入 ids[:,-1:][1,1]，携带各层旧缓存。',
        '每层只投影这个新位置：Q/K/V[1,4,1,32]。',
        '旧长度 S=32 → 新 Q/K 用位置 32 做 RoPE。',
        '不是从位置 0 开始；V 不旋转。'],teal,'#fff5e7',True)
    arrow([(1790,1180),(1790,1225)])
    box(60,1225,2280,245,'04  展开缓存注意力 · '+at('attention_step'),[
        '[K1–K2] 新 x[1,1,128] → QKV → 新 Q/K 使用 cos[32:33]、sin[32:33]。',
        '[K3] 旧 K[1,4,32,32] + 新 k[1,4,1,32] → 沿位置轴 cat → K[1,4,33,32]；V 同理。',
        '[K4] 新 Q @ 全部 Kᵀ / √32 → 分数[1,4,1,33] → 因果遮罩/softmax → A。',
        'A @ 全部 V → H[1,4,1,32] → 拼头/投影[1,1,128]；更新该层缓存，送回当前 Block。'],teal,'#fff5e7',True)
    arrow([(1200,1470),(1200,1515)])
    box(60,1515,1100,245,'05  每层独立保存 · CachedDecoder.forward',[
        'cache[0] = (第 0 层 K, 第 0 层 V)',
        'cache[1] = (第 1 层 K, 第 1 层 V)',
        '每层只处理新位置的主干、残差和 MLP。',
        '新请求 cache=None；更新权重后也须丢弃缓存。'],teal)
    box(1240,1515,1100,245,'06  矩形遮罩：历史 3，新增 2 个位置',[
        '                 key：0  1  2  3  4',
        'query 位置 3：      1  1  1  1  0',
        'query 位置 4：      1  1  1  1  1',
        '规则 key≤query；query 编号加历史长度。'],teal,'#fff5e7',True)
    arrow([(1200,1760),(1200,1805)])
    box(60,1805,2280,330,'07  小数值例子：一层、一个 2 维头，已有 3 个位置，再加入位置 3',[
        '人工选值，Q/K 已经过位置处理：旧 K=[[1,0],[0,1],[1,1]]，旧 V=[[1,0],[0,2],[1,1]]。',
        '新 q=[1,0]、新 k=[2,1]、新 v=[2,0]；缓存拼接 K→4 行、V→4 行。',
        '新 q 与四行 K 点积=[1,0,1,2]，除 √2；位置 3 能看位置 0…3，不遮挡已有行。',
        'softmax → A≈['+', '.join(f'{v:.6f}' for v in a)+']。',
        f'H[0]=A₀×1+A₁×0+A₂×1+A₃×2 ≈ {result[0]:.6f}。',
        f'H[1]=A₀×0+A₁×2+A₂×1+A₃×0 ≈ {result[1]:.6f}；旧 V 的内容仍参与输出。'],teal,'#e8f6f1')
    arrow([(1200,2135),(1200,2180)])
    box(60,2180,2280,205,'08  继续生成 · 新 Token 再回到第 03 步',[
        '当前新位置经过所有 Block → 最终 RMSNorm → lm_head → 贪心选下一 ID，拼到完整输出序列。',
        '循环共 16 次：prefill 1 次 + decode 15 次。两边都只对末位置执行 lm_head。',
        '最后一次选出的 Token 未再输入，所以缓存长度 47，最终输出序列长度 48。'],blue)
    arrow([(1200,2385),(1200,2430)])
    box(60,2430,2280,245,'09  先验正确性，再比较时间和缓存 · compare_kv_cache_runs.py',[
        '同权重、同输入、同设备与代码协议 → 各轮完整词表 logits 近似相同 → 贪心 ID 相同。',
        '每层累计处理位置：重算 32+33+…+47=632；缓存 32+15=47。新 Q 仍需读取历史 K/V。',
        '末次 K/V 有效字节：2(K/V)×2层×1batch×4头×47位置×32维×4字节 = 96,256。',
        '预热 1 次，测量 3 次；分开记录 prefill/decode。缓存内存随长度增长，不计为模型参数。'],blue)
    text(60,2740,'边界：FP32、等长无 padding、固定容量；动态 cat 有复制开销，不含滑窗/缓存淘汰。下一小块才减少 KV 头数。',27)
    text(60,2790,'来源：Hugging Face 缓存说明、Meta Llama 3 Attention；本地图与数字为教学改编。',27)
    svg='<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="2850" viewBox="0 0 2400 2850"><title>KV Cache 代码与数值流转</title><style>text{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}</style><rect width="2400" height="2850" fill="#f5f8fb"/>'+''.join(d.bg+d.edges+d.fg)+'</svg>'
    ET.fromstring(svg)
    return svg,{'source_lines':spans,'synthetic_example':True,'weights':a,'head_output':result,
                'processed_positions':[632,47],'kv_tensor_bytes':96256}


if __name__=='__main__':
    svg,check=draw();p=OUT/'kv-cache-code-flow.svg';p.write_text(svg)
    (ROOT/'artifacts/kv-cache-diagram-check.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
    export_png(p);print(p)
