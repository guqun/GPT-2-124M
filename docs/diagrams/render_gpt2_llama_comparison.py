"""逐步绘制 GPT-2/Llama 3 的输入、计算、输出和结构差异。
标准库生成 SVG；--png 使用项目已有 macOS 导出器。不执行模型训练。
"""
import argparse
from pathlib import Path
import unicodedata
import xml.etree.ElementTree as ET
from render_ddp_allreduce_flow import Drawing, export_png

OUT = Path(__file__).resolve().parent
W = 2600
INK, MUTED = '#183044', '#526a7b'
BLUE, TEAL, ORANGE = '#355fc0', '#087e75', '#a66522'
# 每个条目明确给出两侧输入、计算及输出，参数矩阵均按行向量记法。
ROWS = [
('00','整数 ID 作为共同起点','相同接口',[
 ['输入：已分词的两条文本，每条 5 个 Token',
  '计算：把 Token ID 排成整数张量 I',
  '输出：I [2,5]；示例 ID 范围为 0…99',
  '目的：供 Embedding 查表，ID 本身不参与梯度更新']]*2),
('01','查表，建立初始向量','差异 A · 位置信息入口',[
 ['输入：I [2,5]；位置 t = 0,1,2,3,4',
  '参数：Token 表 E [100,16]；位置表 P [Cmax,16]',
  '计算：X₀[b,t,d] = E[I[b,t],d] + P[t,d]',
  '输出：X₀ [2,5,16]；P[t] 在 batch 维广播',
  '目的：同时把 Token 内容与绝对位置送入网络'],
 ['输入：I [2,5]',
  '参数：Token 表 E [100,16]；没有可学习的位置表 P',
  '计算：X₀[b,t,d] = E[I[b,t],d]',
  '输出：X₀ [2,5,16]；位置 t 留给第 05 步 RoPE',
  '目的：先取得内容向量，再在注意力中注入位置']]),
('02','保存当前 Block 的输入 X','相同 · 残差旁路起点',[
 ['输入：首层取 X₀；后续层取上一层输出 Y',
  '计算：X = 当前层输入；保留 X 供第 12 步相加',
  '输出：主分支 X [2,5,16]；旁路 X [2,5,16]',
  '目的：原输入绕过归一化与注意力，保留直接通路']]*2),
('03','第一次归一化','差异 B · LayerNorm / RMSNorm',[
 ['输入：X [2,5,16]；每次只处理一个 Token 的 16 维',
  '参数：γ₁、β₁ 各 [16]；ε 是固定的小正数',
  '计算：μ = (Σd Xd)/16；σ² = Σd(Xd−μ)²/16',
  '计算：Ud = γ₁,d × (Xd−μ)/√(σ²+ε) + β₁,d',
  '输出：U [2,5,16]；不混合不同 Token 或样本',
  '目的：先居中，再缩放，并做可学习的仿射变换'],
 ['输入：X [2,5,16]；每次只处理一个 Token 的 16 维',
  '参数：γ₁ [16]；没有平移参数 β₁',
  '计算：r = √((Σd Xd²)/16 + ε)',
  '计算：Ud = γ₁,d × Xd/r；不减去均值',
  '输出：U [2,5,16]；不混合不同 Token 或样本',
  '目的：控制向量尺度，并保留可学习的逐维缩放']]),
('04','Q / K / V 投影并拆成头','差异 C · 头数；差异 D · bias',[
 ['输入：U [2,5,16]',
  '参数：Wq、Wk、Wv 各 [16,16]；bq、bk、bv 各 [16]',
  '计算：Q₀ = U Wq + bq；K₀ = U Wk + bk；V₀ = U Wv + bv',
  '计算：各 [2,5,16] → reshape [2,5,4,4]',
  '计算：交换位置轴与头轴 → [2,4,5,4]',
  '输出：Q、K、V 都是 [2,4,5,4]；4 头，每头 4 维'],
 ['输入：U [2,5,16]',
  '参数：Wq [16,16]；Wk、Wv 各 [16,8]；没有 bias',
  '计算：Q₀ = U Wq；K₀ = U Wk；V₀ = U Wv',
  '计算：Q₀ [2,5,16] → [2,5,4,4] → Q [2,4,5,4]',
  '计算：K₀/V₀ [2,5,8] → [2,5,2,4] → [2,2,5,4]',
  '输出：4 个 Q 头、2 个 K/V 头；每头仍为 4 维']]),
('05','在 Q / K 中处理位置','差异 A · 每层 RoPE',[
 ['输入：Q、K [2,4,5,4]；V [2,4,5,4]',
  '计算：Q′ = Q；K′ = K；不额外旋转',
  '原因：位置表 P 已在第 01 步加入网络输入',
  '输出：Q′、K′、V 仍为 [2,4,5,4]'],
 ['输入：Q [2,4,5,4]；K、V [2,2,5,4]；位置 t',
  '计算：每头按 (x₀,x₁)、(x₂,x₃) 两对处理，r=0,1',
  '角度：φ = t × θ^(−2r/4)；θ 为配置常数',
  '旋转：x′₂r = x₂r cosφ − x₂r+1 sinφ',
  '旋转：x′₂r+1 = x₂r sinφ + x₂r+1 cosφ',
  '输出：Q′、K′ 形状不变；V 不旋转；无新位置参数表']]),
('06','把 Q 头与 K/V 头对应起来','差异 C · MHA / GQA',[
 ['输入：Q′、K′、V，各 4 个头；头编号 h=0…3',
  '计算：K* = K′；V* = V，每个 Q 头使用同号 K/V 头',
  '对应：Q₀→KV₀；Q₁→KV₁；Q₂→KV₂；Q₃→KV₃',
  '输出：Q′、K*、V* 都是 [2,4,5,4]'],
 ['输入：Q′ 有 4 头；K′、V 各有 2 头',
  '计算：g(h)=floor(h/2)；K*[:,h]=K′[:,g(h)]',
  '计算：V*[:,h]=V[:,g(h)]；同组使用相同 K/V 数值',
  '对应：Q₀/Q₁→KV₀；Q₂/Q₃→KV₁',
  '输出：逻辑对齐后的 Q′、K*、V* 均为 [2,4,5,4]',
  '说明：扩展是数学展示；高效内核可直接按组计算']]),
('07','计算位置之间的匹配分数','相同公式 · 输入来自各自模型',[
 ['输入：Q′、K* [2,4,5,4]',
  '计算：S[b,h,i,j] = Σd Q′[b,h,i,d] K*[b,h,j,d] / 2',
  '其中：d=0…3；2=√4；i 为查询位置，j 为被读取位置',
  '输出：S [2,4,5,5]，每个头得到一个 5×5 分数矩阵',
  '目的：决定每个位置与其他位置的匹配程度']]*2),
('08','遮挡未来位置，并归一化权重','相同 · 因果 mask + softmax',[
 ['输入：S [2,4,5,5]；固定因果遮罩 Mask [5,5]',
  '规则：Mask[i,j]=0（j≤i）；Mask[i,j]=−∞（j>i）',
  '计算：S̃ = S + Mask；Mask 广播到每个样本与头',
  '计算：A[b,h,i,j] = exp(S̃[b,h,i,j]) / Σk exp(S̃[b,h,i,k])',
  '输出：A [2,4,5,5]；每行和为 1，未来位置权重为 0',
  '实现：softmax 先减去该行最大值，避免指数溢出']]*2),
('09','汇总 V，拼接四个头','相同公式 · V* 的头共享方式不同',[
 ['输入：A [2,4,5,5]；V* [2,4,5,4]',
  '计算：H[b,h,i,d] = Σj A[b,h,i,j] × V*[b,h,j,d]',
  '中间：H [2,4,5,4]；各头按自己的权重汇总 V',
  '拼接：交换头/位置轴 → [2,5,4,4] → 合并为 [2,5,16]',
  '输出：C [2,5,16]；最后 16 维按头顺序排列',
  '目的：把各头读到的上下文信息放进同一个向量']]*2),
('10','输出投影：混合不同头的特征','差异 D · 有无 bias',[
 ['输入：C [2,5,16]',
  '参数：Wₒ [16,16]；bₒ [16]',
  '计算：O[b,t,d] = Σk C[b,t,k] Wₒ[k,d] + bₒ[d]',
  '输出：O [2,5,16]；作为注意力分支的结果',
  '目的：在 16 个拼接特征间做可学习的线性混合'],
 ['输入：C [2,5,16]',
  '参数：Wₒ [16,16]；没有 bₒ',
  '计算：O[b,t,d] = Σk C[b,t,k] Wₒ[k,d]',
  '输出：O [2,5,16]；作为注意力分支的结果',
  '目的：在 16 个拼接特征间做可学习的线性混合']]),
('11','准备第一次残差的两个操作数','相同 · 取旁路 X 与分支 O',[
 ['输入：第 02 步保留的 X，以及第 10 步算出的 O',
  '检查：两者都为 [2,5,16]，可以逐元素相加',
  '计算：直接把 X、O 交给加法；不做拼接或查表',
  '输出：两个同形状张量 X、O；此步不引入参数']]*2),
('12','第一次残差相加','相同 · 保留 Block 原输入',[
 ['输入：X [2,5,16]；O [2,5,16]',
  '计算：R[b,t,d] = X[b,t,d] + O[b,t,d]',
  '输出：R [2,5,16]；同时保留 R 供第 15 步使用',
  '目的：把注意力分支的结果加到原有表示上',
  '注意：加回的是 X，不是归一化后的 U']]*2),
('13','第二次归一化','差异 B · 独立的一组归一化参数',[
 ['输入：R [2,5,16]；参数 γ₂、β₂ 各 [16]',
  '计算：μR = (Σd Rd)/16；σR² = Σd(Rd−μR)²/16',
  '计算：Zd = γ₂,d × (Rd−μR)/√(σR²+ε) + β₂,d',
  '输出：Z [2,5,16]；交给 MLP',
  '说明：γ₂、β₂ 与第 03 步的参数分别学习'],
 ['输入：R [2,5,16]；参数 γ₂ [16]，没有 β₂',
  '计算：rR = √((Σd Rd²)/16 + ε)',
  '计算：Zd = γ₂,d × Rd/rR',
  '输出：Z [2,5,16]；交给 MLP',
  '说明：γ₂ 与第 03 步的 γ₁ 是不同参数']]),
('14','前馈网络：逐位置加工特征','差异 E · GELU / SwiGLU；差异 D · bias',[
 ['输入：Z [2,5,16]',
  '参数：W₁ [16,64]、b₁ [64]；W₂ [64,16]、b₂ [16]',
  '扩展：a = Z W₁ + b₁ → [2,5,64]',
  '激活：g = 0.5a × (1+tanh(√(2/π)×(a+0.044715a³)))',
  '说明：上式为逐元素 GELU 的 tanh 近似；g [2,5,64]',
  '缩回：Fout = g W₂ + b₂ → [2,5,16]',
  '目的：线性混合 → 非线性变换 → 线性混合'],
 ['输入：Z [2,5,16]；教学中间宽度设 F=48',
  '参数：Wg、Wv 各 [16,48]；Wd [48,16]；无 bias',
  '两路扩展：a = Z Wg；v = Z Wv → 都为 [2,5,48]',
  '门控激活：g = SiLU(a) = a/(1+exp(−a)) → [2,5,48]',
  '逐元素相乘：u = g ⊙ v → [2,5,48]',
  '缩回：Fout = u Wd → [2,5,16]',
  '目的：用一条分支调节另一条分支，再混合回 D 维']]),
('15','第二次残差相加','相同 · 加回的是 R',[
 ['输入：第 12 步保留的 R；第 14 步输出 Fout',
  '计算：Y[b,t,d] = R[b,t,d] + Fout[b,t,d]',
  '输出：Y [2,5,16]，作为当前 Block 的最终输出',
  '注意：这里加回 R，不是再次加 X，也不是加 Z']]*2),
('16','继续后续 Block','相同 · 同构但参数独立',[
 ['输入：当前层的 Y [2,5,16]',
  '计算：令下一层 X=Y，重复第 02—15 步',
  '参数：每层拥有自己的归一化、注意力和 MLP 参数',
  '输出：最后一个 Block 的结果 XL [2,5,16]',
  '说明：L 是层数；本图只展开其中一层']]*2),
('17','最终归一化','差异 B · Block 之外的最后一次归一化',[
 ['输入：XL [2,5,16]；参数 γf、βf 各 [16]',
  '计算：μf = (Σd XL,d)/16；σf² = Σd(XL,d−μf)²/16',
  '计算：Hf,d = γf,d × (XL,d−μf)/√(σf²+ε) + βf,d',
  '输出：Hf [2,5,16]；交给 lm_head',
  '目的：在词表打分前调整最终表示的尺度'],
 ['输入：XL [2,5,16]；参数 γf [16]，没有 βf',
  '计算：rf = √((Σd XL,d²)/16 + ε)',
  '计算：Hf,d = γf,d × XL,d/rf',
  '输出：Hf [2,5,16]；交给 lm_head',
  '目的：在词表打分前调整最终表示的尺度']]),
('18','lm_head：给每个词表 ID 打分','差异 F · 输出表是否与输入表共享',[
 ['输入：Hf [2,5,16]；输入 Embedding 表 E [100,16]',
  '参数：复用 E；lm_head 不带 bias',
  '计算：logits[b,t,j] = Σd Hf[b,t,d] × E[j,d]',
  '输出：logits [2,5,100]；j=0…99，代表候选 Token ID',
  '用途：训练计算交叉熵；生成用末位置 logits 选择下一 Token'],
 ['输入：Hf [2,5,16]；独立输出表 Wout [100,16]',
  '参数：Wout 与 E 分别学习；lm_head 不带 bias',
  '计算：logits[b,t,j] = Σd Hf[b,t,d] × Wout[j,d]',
  '输出：logits [2,5,100]；这些分数还不是概率',
  '范围：此处指原始 Llama 3；其他型号的共享设置须单独看配置']]),
]

SECTIONS=[('输入、归一化与 QKV',0,7,'part1'),('注意力计算与第一次残差',7,13,'part2'),('MLP、第二次残差与词表打分',13,19,'part3')]


def render(rows,title,part=False):
    d=Drawing(); checks=[]
    def txt(x,y,s,size=28,color=MUTED,bold=400,width=990):
        estimate=lambda sz:sum(sz*(1 if unicodedata.east_asian_width(c) in 'WF' else .58) for c in s)
        while estimate(size)>width:size-=1
        assert size>=23,(s,size)
        checks.append((s,size))
        d.text(x,y,s,size,color,bold)
    txt(65,80,title,48,INK,700,2470)
    txt(65,135,'蓝色 GPT-2 ｜ 绿色 Llama 3 ｜ 橙色差异带：明确标出改变了什么',31,INK,width=2470)
    txt(65,187,'教学设置：B=2、T=5、D=16、词表 Vocab=100；Q 头=4；GPT-2 KV 头=4，Llama KV 头=2；每头 4 维。',28,width=2470)
    txt(65,231,'按推理 eval 模式展开公共前向，不画 Dropout/KV Cache；MLP 宽度 64 vs 48 仅为教学，并非等参数量实验。',28,width=2470)
    txt(65,275,'索引：b 样本，t/i/j 位置（lm_head 的 j 为词表 ID），h 头，d/k 特征；⊙ 逐元素乘，Σ 求和；各模型参数独立。',27,width=2470)
    y=320; boxes={}
    for number,name,diff,pair in rows:
        changed=diff.startswith('差异')
        d.rect(60,y,2480,67,'#fff0d9' if changed else '#eaf0f5',stroke='none',radius=12)
        txt(82,y+44,f'{number}  {name}',32,INK,700,1120)
        txt(1240,y+44,diff,29,ORANGE if changed else MUTED,650,1250)
        cy=y+84; h=87+len(max(pair,key=len))*41
        for base,col,label,lines in [(0,BLUE,'GPT-2',pair[0]),(1300,TEAL,'Llama 3',pair[1])]:
            x=base+82; w=1090
            d.rect(x,cy,w,h,'#fffaf2' if changed else 'white',stroke='#e8c38e' if changed else '#d7e3ed',radius=18)
            txt(x+24,cy+42,label,31,col,700)
            for i,line in enumerate(lines):
                txt(x+24,cy+87+i*41,line,28,INK if line.startswith(('输入','输出')) else MUTED,550 if line.startswith(('输入','输出')) else 400,w-48)
            boxes[(number,base)]=(x,cy,w,h)
        y=cy+h+57
    # 绕线只在完整图中跨行相连；分图上的卡片仍明确标注跨图输入来源。
    if not part:
        for base,color in [(0,BLUE),(1300,TEAL)]:
            for src,target,label in [('02','12','X 旁路'),('12','15','R 旁路')]:
                x,sy,w,sh=boxes[(src,base)];_,ty,_,th=boxes[(target,base)]
                route=base+1215
                d.arrow([(x+w,sy+sh-30),(route,sy+sh-30),(route,ty+th-30),(x+w,ty+th-30)],color,width=4)
                d.text(route-10,ty+th-49,label,23,color,650,anchor='end')
    txt(65,y+5,'读图：同一行横向比较；纵向按编号追踪张量。名称带 W、E、P、γ、β、b 的参数，与本次前向算出的张量要分清。',26,width=2470)
    txt(65,y+49,'来源：Raschka GPT→Llama 教程、Meta Llama 3 官方 model.py、本项目 GPT-2 前向；Llama 3 不含 Qwen3 的 QK-Norm。',25,width=2470)
    height=y+82
    s=f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}">
<title>{title}</title><desc>逐步标注输入、参数、计算、输出形状和差异 A 到 F；数字为缩小教学配置。</desc>
<style>text {{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}}</style>
<rect width="{W}" height="{height}" fill="#f5f8fb"/>{''.join(d.bg)}{''.join(d.edges)}{''.join(d.fg)}</svg>'''
    ET.fromstring(s)
    return s,checks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--png',action='store_true');args=p.parse_args()
    jobs=[('gpt2-vs-llama3-forward-flow',ROWS,'GPT-2 vs Llama 3：逐步计算与输入输出',False)]
    jobs += [('gpt2-vs-llama3-'+slug,ROWS[a:b],f'{i+1}/3  {title}：输入 → 计算 → 输出',True) for i,(title,a,b,slug) in enumerate(SECTIONS)]
    for name,rows,title,part in jobs:
        svg,checks=render(rows,title,part);target=OUT/(name+'.svg');target.write_text(svg,encoding='utf-8')
        if args.png:export_png(target)
        print(f'{target.name}: {len(rows)} steps, {len(checks)} fitted text lines')

if __name__=='__main__':main()
