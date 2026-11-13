"""从当前 RoPE 代码提取定位，绘制位置方案、数值旋转与训练接入图。"""
import ast
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from render_ddp_allreduce_flow import Drawing, export_png

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def draw():
    tree=ast.parse((ROOT/'rope_model.py').read_text())
    spans={n.name:[n.lineno,n.end_lineno] for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    def at(name):return f'{name} · L{spans[name][0]}–{spans[name][1]}'
    q=[1.,2.,3.,4.]; k=[2.,1.,0.,1.]
    def rotate(x,t):
        result=[]
        for r in range(2):
            a=t*10000**(-2*r/4);c,s=math.cos(a),math.sin(a)
            result.extend([x[2*r]*c-x[2*r+1]*s,x[2*r]*s+x[2*r+1]*c])
        return result
    rq,rk=rotate(q,1),rotate(k,1)
    dot0=sum(a*b for a,b in zip(rq,k));dot1=sum(a*b for a,b in zip(rq,rk))
    assert abs(dot1-8)<1e-12
    assert abs(sum(x*x for x in rq)-30)<1e-12
    d=Drawing();blue='#355fc0';teal='#087e75';ink='#172b40';red='#cf4141'
    def text(x,y,s,size=28,color=ink):d.text(x,y,s,size,color)
    def box(x,y,w,h,title,rows,color=blue,fill='white',size=28):
        d.rect(x,y,w,h,fill);d.text(x+26,y+45,title,32,color,700)
        for i,s in enumerate(rows):text(x+26,y+92+42*i,s,size)
    def arrow(points,color=teal):d.arrow(points,color,width=4)
    def redbox(x,y,w,h):d.rect(x+10,y+10,w-20,h-20,'none',red,16)
    d.text(60,80,'RoPE 接入代码：位置表 → 每层 Q/K 旋转',48,ink,700)
    text(60,138,'参照不是原始 GPT-2：两边已经都有 RMSNorm / 无 bias。本节只比较位置方案；红框标出改动。')
    box(60,190,1100,195,'A · train_rope_reference.py',[
        "rope_training.main('absolute')",
        'RMSNorm/no-bias + 学习式绝对位置表。',
        '随机参数起点；共 6,859,520 个参数。'],blue,'#eef2ff')
    box(1240,190,1100,195,'B · train_rope.py',[
        "rope_training.main('rope')",
        '相同结构 + RoPE；移除 256×128 位置表。',
        '共同权重初值相同；共 6,826,752 个参数。'],teal,'#e8f6f1')
    arrow([(610,385),(610,435)],blue);arrow([(1790,385),(1790,435)])
    box(60,435,1100,245,'01  输入端：Token + Position',[
        '输入：Token ID [B,T]，位置编号 0…T−1。',
        '参数：wte[50257,128]、wpe[256,128]。',
        '计算：x = wte(ids) + wpe(positions)。',
        '输出：x[B,T,128]；位置已混入向量。'],blue)
    box(1240,435,1100,245,'01  输入端：'+at('RotaryGPT'),[
        '输入：相同形状的 Token ID [B,T]。',
        "构造时 del transformer['wpe']，不留闲置参数。",
        '计算：x = wte(ids)，没有输入位置向量相加。',
        '输出：x[B,T,128]；位置稍后进入每层 Q/K。'],teal,'#fff5e7',27)
    redbox(1240,435,1100,245)
    arrow([(610,680),(610,730)],blue);arrow([(1790,680),(1790,730)])
    box(60,730,2280,200,'02  每个 Block：RMSNorm → QKV 投影 → 拆头（两边共同）',[
        '输入：[B,T,128] → c_attn 权重 [384,128] → split：Q、K、V 各 [B,T,128]。',
        'reshape + transpose → Q、K、V 各 [B,4,T,32]。每头 32 维；旋转前的 q/k 已经是数据激活值。',
        '共同权重初值相同，不代表 q/k 数值相同：A 的输入带位置向量，B 的输入不带。'],blue)
    arrow([(610,930),(610,980)],blue);arrow([(1790,930),(1790,980)])
    box(60,980,1100,245,'03  A：直接使用投影后的 Q/K',[
        'Q′=Q，K′=K，V 保留。',
        '没有本节新增的三角函数或旋转运算。',
        '形状：Q/K/V 都是 [B,4,T,32]。',
        '注意力继续计算 QKᵀ/√32。'],blue)
    box(1240,980,1100,245,'03  B：'+at('RotaryAttention'),[
        'q,k = self.rope(q,k)，V 不旋转。',
        '32 维拆成 16 对，相邻两维一组。',
        '每对按位置 t 旋转，Q/K 形状保持不变。',
        '每层重新投影，再旋转本层的新 Q/K。'],teal,'#fff5e7')
    redbox(1240,980,1100,245)
    arrow([(1790,1225),(1790,1275)])
    box(60,1275,2280,245,'展开 B：'+at('RotaryEmbedding')+' → '+at('apply_rotary'),[
        '参数：没有新增可训练参数；theta=10000 为固定教学配置。r=0…15，d=32。',
        '角度 φ[t,r]=t×theta^(−2r/d) → cos/sin 缓存 [256,16] → 本批取前 T 行 [T,16]。',
        '缓存是 buffer，随模型移动设备；不同 batch/头广播共用，位置按每条序列 0…T−1 编号。',
        '拆出 even=x[...,0::2]、odd=x[...,1::2] → 两路旋转乘加 → stack/flatten 恢复原维序。'],teal)
    arrow([(1200,1520),(1200,1570)])
    box(60,1570,2280,200,'四维手算例子：仅用于看数据流；不是本机 32 维头的实测激活',[
        '输入：位置 1 的 q=[1,2,3,4]；位置 0、1 的原始 k 都选为 [2,1,0,1]，便于隔离位置效果。',
        'd=4、theta=10000 → 两对频率 [1,0.01] → 位置 1 的角度 [1,0.01] 弧度。',
        'even=[1,3]，odd=[2,4]；cos≈[0.540302,0.999950]，sin≈[0.841471,0.010000]。'],teal,'#e8f6f1')
    arrow([(610,1770),(610,1820)]);arrow([(1790,1770),(1790,1820)])
    box(60,1820,1100,245,'第一对 (1,2)：旋转 1 弧度',[
        '新偶维 = 1×0.540302 − 2×0.841471',
        f'            ≈ {rq[0]:.6f}',
        '新奇维 = 1×0.841471 + 2×0.540302',
        f'            ≈ {rq[1]:.6f}'],teal,size=27)
    box(1240,1820,1100,245,'第二对 (3,4)：旋转 0.01 弧度',[
        '新偶维 = 3×0.999950 − 4×0.010000',
        f'            ≈ {rq[2]:.6f}',
        '新奇维 = 3×0.010000 + 4×0.999950',
        f'            ≈ {rq[3]:.6f}'],teal,size=27)
    arrow([(610,2065),(610,2100),(1200,2100),(1200,2140)])
    arrow([(1790,2065),(1790,2100),(1200,2100)])
    box(60,2140,2280,245,'旋转输出 → 送入注意力分数计算',[
        'q′(位置1) ≈ ['+', '.join(f'{x:.6f}' for x in rq)+']。',
        'k′(位置0)=[2,1,0,1]；k′(位置1)≈['+', '.join(f'{x:.6f}' for x in rk)+']。',
        f'q′₁·k′₀≈{dot0:.6f}；q′₁·k′₁={dot1:.6f}。相同原始 k，因为位置不同，匹配分数不同。',
        '每个二维旋转保持向量长度；同位置同时旋转 q/k 保持原点积。V 始终走原来的支路。'],teal,'#e8f6f1',27)
    arrow([(1200,2385),(1200,2435)])
    box(60,2435,2280,200,'04  回到共同主干：缩放 → 因果遮罩 → softmax → AV',[
        '实训 d=32：S=Q′K′ᵀ/√32 → 遮挡未来位置 → A=softmax(S) [B,4,T,T]。',
        'H=A@V [B,4,T,32] → 拼接 [B,T,128] → 输出投影 → 第一次残差 → RMSNorm/MLP/第二次残差。',
        '继续后续 Block → 最终 RMSNorm → 共享 lm_head → logits [B,T,50257]。'],blue)
    arrow([(1200,2635),(1200,2685)])
    box(60,2685,2280,245,'05  rope_training.py：两边独立运行同一套训练协议',[
        '固定原文与 train/val → 每微批 x/y [2,128] → 前向/CE → (loss/4).backward()，累积四次。',
        '累计梯度 → 全局范数裁剪 → AdamW 更新一次 → 清梯度；默认 100 次，共 102,400 个预测位置。',
        '每 20 次及末次固定窗口验证；计时排除验证/保存；初始两次更新只排除出吞吐统计。',
        '输出 .cache/rope-comparison/{absolute,rope}/ 下 report、metrics、checkpoint；旧实验文件保留。'],blue)
    arrow([(1200,2930),(1200,2980)])
    box(60,2980,2280,155,'06  compare_rope_runs.py：比较之前先核对协议',[
        '共同初始参数指纹排除 A 独有的 wpe；其余配置、数据、设备、源码和实际取批顺序必须匹配。',
        '比较同窗口验证 loss 与吞吐；不保证 RoPE 更优，也不能据此推断长上下文效果。'],blue)
    text(60,3200,'本课最大长度仍为 256、训练长度 128；不含 KV cache 偏移、长上下文外推或 YaRN。theta 不等于模型参数。',27)
    text(60,3250,'红框为代码改动；四维数值是教学输入。旋转公式与复数实现、因果性及反向传播均有行为测试。',27)
    svg='<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="3310" viewBox="0 0 2400 3310"><title>RoPE 代码与数值流转</title><style>text{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}</style><rect width="2400" height="3310" fill="#f5f8fb"/>'+''.join(d.bg+d.edges+d.fg)+'</svg>'
    ET.fromstring(svg)
    return svg,{'source_lines':spans,'q_position1':rq,'k_position1':rk,'dot_with_k0':dot0,'dot_with_k1':dot1,'example_is_synthetic':True}


if __name__=='__main__':
    svg,check=draw();path=OUT/'rope-code-comparison.svg';path.write_text(svg)
    (ROOT/'artifacts/rope-diagram-check.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
    export_png(path)
    print(path)
