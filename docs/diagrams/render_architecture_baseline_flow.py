"""从当前源码及已有报告生成基线准备/单次更新图；不运行训练。"""
import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from render_ddp_allreduce_flow import Drawing, export_png

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'prepare_architecture_baseline.py'
WIDTH,HEIGHT=2400,4280
INK,MUTED,BLUE,TEAL,AMBER,RED='#172b40','#586c7d','#355fc0','#087e75','#946027','#cd4141'


def draw():
    source=SOURCE.read_text()
    tree=ast.parse(source)
    spans={node.name:[node.lineno,node.end_lineno] for node in ast.walk(tree)
           if isinstance(node,ast.FunctionDef)}
    def at(name):
        start,end=spans[name]
        return f'{name}() · L{start}–{end}'
    def line(fragment):
        return next(i for i,s in enumerate(source.splitlines(),1) if fragment in s)
    spec=json.loads((ROOT/'configs/architecture-baseline-local.json').read_text())
    report=json.loads((ROOT/'artifacts/architecture-baseline-local-check.json').read_text())
    assert spec==report['config']
    assert report['model_source_sha256']==hashlib.sha256((ROOT/'gpt2_model.py').read_bytes()).hexdigest()
    t=spec['training'];B,T,K=t['batch_size'],t['seq_len'],t['accumulation_steps']
    assert (B,T,K)==(2,128,4)
    N=B*T;cut=report['data']['train_range'][1];total=report['data']['total_tokens']
    assert (cut,total)==(304222,338025)
    assert N*K==report['smoke']['training_tokens']==1024
    d=Drawing()
    def txt(x,y,s,size=28,color=INK,weight=400):d.text(x,y,s,size,color,weight)
    def box(x,y,w,h,title,lines,color=BLUE,fill='white',size=28):
        d.rect(x,y,w,h,fill)
        txt(x+26,y+44,title,31,color,700)
        for j,s in enumerate(lines):txt(x+26,y+88+39*j,s,size)
    def arrow(points,color=BLUE,dashed=False):d.arrow(points,color,dashed,width=4)
    def band(y,title,color=INK):
        d.rect(60,y,2280,62,'#eaf0f5',stroke='none',radius=10)
        txt(84,y+42,title,32,color,700)

    txt(60,80,'短训练基线：从配置到 batch，再到一次参数更新',49,INK,700)
    txt(60,136,'代码对应 prepare_architecture_baseline.py｜实线跟踪数据与执行顺序；下标/区间是 Token 位置，不是 Token ID',27,MUTED)
    box(60,180,2280,105,'当前实际设备：全部在 CPU',[
        '磁盘文件 → CPU 内存；x/y 为 int64，模型/梯度为 FP32。本入口没有 GPU 搬运、DDP 或完整 100 步训练。'],TEAL,'#e8f6f1',28)

    band(325,'01  读取配置与核对预算 · '+at('main'))
    box(60,415,1460,190,at('load_spec')+' → '+at('describe'),[
        '输入：--profile local（默认） → architecture-baseline-local.json',
        '检查：正整数；128 可被 4 头整除；T=128 ≤ 位置表容量 256。',
        '输出 spec + summary：2 层 / 128 维 / 4 头；6,862,464 参数；B=2、T=128、K=4。'],size=28)
    box(1600,415,740,190,'旁路：--profile formal',[
        '读取 124M/FineWeb 方案 → pending',
        '不读取教学 demo、不创建模型。',
        '直接输出缺项与 JSON 报告后结束。'],AMBER,'#fff4e4',27)
    arrow([(790,605),(790,655)])
    box(60,655,2280,160,'02  校验原文与分词 · '+at('prepare_local_data'),[
        '输入：Tiny Shakespeare 原文 + 两份 GPT-2 缓存；SHA-256 必须匹配，否则停止。',
        'encode_ordinary(原文) → ids：338,025 个整数 ID；词表 50,257 项；本档不额外插 EOT。'],BLUE,size=29)
    arrow([(1200,815),(1200,860)])
    band(860,'03  先切分，再分别构造 x/y · '+at('split_tokens'))
    arrow([(1200,922),(610,960)])
    arrow([(1200,922),(1790,960)],TEAL)
    box(60,960,1100,155,'训练区 train：原始区间 [0,304222)',[
        '304,222 个 ID；模型反传只使用这段数据。',
        '切分点 cut = floor(338025 × 0.9) = 304222。'],BLUE,'#edf2ff')
    box(1240,960,1100,155,'验证区 val：原始区间 [304222,338025)',[
        '33,803 个 ID；后面仅用于 eval + no_grad。',
        '两区先分开，训练标签不会越过 cut 读到 val。'],TEAL,'#e8f6f1')
    arrow([(610,1115),(610,1170)])
    arrow([(1790,1115),(1790,1170)],TEAL)
    box(60,1170,1460,320,'04  组成首批 · '+at('first_batch'),[
        '输入：train、B=2、T=128、start=0；需要 N+1=257 个连续 ID。',
        'buf = train[0:257]     →     x=buf[:-1]，y=buf[1:]',
        'x = [位置0…127；位置128…255]  →  [2,128]',
        'y = [位置1…128；位置129…256]  →  [2,128]',
        '首行 x[:8] = [5962,22307,25,198,8421,356,5120,597]',
        '首行 y[:8] = [22307,25,198,8421,356,5120,597,2252]'],size=28)
    box(1600,1170,740,320,'同一 first_batch() 也用于 val',[
        'val 内部 start=0 → 取 val[0:257]',
        '对应原文 [304222,304479)。',
        '得到 vx/vy，各为 [2,128]。',
        'smoke 只观察这一个验证首批；',
        '配置中的 8 批完整验证尚未执行。',
        '注意：区间右端均不包含。'],TEAL,size=27)
    arrow([(790,1490),(790,1540)])
    box(60,1540,1460,130,'05  main 判断：是否传入 --smoke-step？',[
        '否：保存配置、数据指纹、首批形状和 ID；报告 training_run=false，然后结束。'],AMBER,'#fff4e4',27)
    box(1600,1540,740,130,'默认运行到这里就结束',[
        '没有模型前向，没有梯度或参数更新。'],AMBER,'#fff4e4',27)
    arrow([(1520,1605),(1600,1605)],AMBER)
    arrow([(790,1670),(790,1740)],TEAL)
    txt(820,1716,'是：进入 smoke_step()，只更新一次 ↓',28,TEAL,650)

    band(1740,'06  可选的一次更新检查 · '+at('smoke_step'),TEAL)
    box(60,1830,1100,215,'S1  构造全新模型与 AdamW',[
        'torch.manual_seed(42)；GPT(...).cpu().float()',
        '随机参数 θ₀；wte/lm_head 共享；没有加载预训练权重。',
        'AdamW：矩阵衰减 0.1；一维参数衰减 0。',
        '本次 lr=0.0006/10=0.00006（第 1 步预热值）。'],TEAL,size=27)
    box(1240,1830,1100,215,'更新前：'+at('validation_probe'),[
        '同一验证首批 vx/vy → model.eval() + no_grad()',
        f"交叉熵 loss = {report['smoke']['validation_probe_before']:.6f}",
        '只记录结果：不 backward，不写参数梯度。',
        '验证结束，回到 model.train()，进入下面训练路径。'],TEAL,'#e8f6f1',27)
    arrow([(1160,1930),(1240,1930)],TEAL)
    arrow([(1790,2045),(1790,2095)],TEAL)
    box(60,2095,2280,120,f"S2  进入循环前清梯度 · L{line('model.train(); optimizer.zero_grad')}",[
        'optimizer.zero_grad(set_to_none=True) → 各参数 .grad=None；随后 4 个微批次之间不清梯度、不调用 step。'],TEAL,size=29)
    arrow([(1200,2215),(1200,2265)],TEAL)
    box(60,2265,2280,200,f"每轮的固定计算 · for micro in range(4) · L{line('for micro in range(k):')}–{line('losses.append(loss.item())')}",[
        'first_batch(train,2,128,start=micro×256) → x/y [2,128] → model(x) → logits [2,128,50257]',
        'cross_entropy(logits.flatten(0,1), y.flatten()) → Lₘ：这 256 个预测位置的平均 loss（标量）。',
        '(Lₘ/4).backward() → PyTorch 把本轮梯度加进每个参数的 .grad；参数 θ₀ 始终不变。'],TEAL,size=29)
    arrow([(1200,2465),(1200,2520)],TEAL)

    # 表内梯度是符号关系；每个 g_m 表示整套参数的梯度张量集合。
    cols=[60,240,670,1240,1970,2340]
    headers=['micro','读取 train 区间','本轮反传贡献','反传后的 .grad','参数']
    for j,h in enumerate(headers):
        d.rect(cols[j],2520,cols[j+1]-cols[j],65,'#dceee8',radius=0)
        txt(cols[j]+18,2563,h,29,TEAL,700)
    for i in range(K):
        y=2585+i*78
        vals=[str(i),f'[{i*N},{(i+1)*N+1})，257 个 ID',f'∇θ(L{i}/4) = g{i}/4',
              '('+' + '.join('g'+str(j) for j in range(i+1))+') / 4','θ₀ 不变']
        for j,v in enumerate(vals):
            d.rect(cols[j],y,cols[j+1]-cols[j],78,'white' if i%2==0 else '#f0f6f4',radius=0)
            txt(cols[j]+18,y+49,v,27,TEAL if j>=3 else INK,600 if j>=3 else 400)
    txt(84,2945,'gₘ = ∂Lₘ/∂θ₀；同一套参数接收 4 次贡献。总共 1,024 个预测位置，不是建立 4 份模型。',29,INK,650)
    box(60,2980,2280,160,'单个参数的手算示意（下面不是模型实测梯度）',[
        '假设四批对同一个参数的 g 为 [2,−1,3,0]：每次加进 .grad 的量为 [0.5,−0.25,0.75,0]。',
        '累计值依次是 0.5 → 0.25 → 1.00 → 1.00；最终等于 (2−1+3+0)/4。每个参数都按同样规则累计。'],TEAL,'#e8f6f1',29)
    arrow([(1200,3140),(1200,3190)],TEAL)
    box(60,3190,2280,170,f"S3 · 先裁剪累计后的梯度 · clip_grad_norm_(parameters,1.0) · L{line('norm=torch.nn.utils.clip_grad_norm_')}",[
        f"输入：最终 .grad = (g₀+g₁+g₂+g₃)/4；本次所有梯度合起来的 L2 范数 = {report['smoke']['grad_norm_before_clip']:.6f}。",
        '范数大于 1 → 所有梯度统一缩小，整体范数约为 1；小于阈值则保持。此时参数仍未更新。'],TEAL,size=29)
    arrow([(1200,3360),(1200,3410)],TEAL)
    box(60,3410,2280,215,f"S3 · 真正更新参数，然后清梯度 · optimizer.step(); optimizer.zero_grad(...) · L{line('optimizer.step(); optimizer.zero_grad')}",[
        '输入：旧参数 θ₀、裁剪后的 .grad；AdamW 更新一阶/二阶统计 m、v，并做初始偏差修正。',
        '对每个参数：θ₁ = (1−lr×λ)θ₀ − lr×m̂/(√v̂+ε)；矩阵 λ=0.1，一维参数 λ=0。',
        '输出：新参数 θ₁ 与优化器统计；随后 .grad 设为 None（不会清掉 m、v）。',
        '整段只调用一次 step；Embedding、注意力、MLP、LayerNorm 等参与求导的参数在这里一起更新。'],TEAL,'#e8f6f1',28)
    arrow([(1200,3625),(1200,3675)],TEAL)
    box(60,3675,2280,170,'S4  更新后再次观察同一验证首批 → 返回 smoke 报告',[
        f"θ₁ + 同一 vx/vy → eval + no_grad → 验证首批 loss = {report['smoke']['validation_probe_after']:.6f}；不反传、不再更新。",
        f"返回：4 个微批次、1 次更新、1,024 个训练预测位置；训练平均 loss={report['smoke']['train_loss']:.6f}。"],TEAL,size=29)
    arrow([(1200,3845),(1200,3895)],TEAL)
    box(60,3895,2280,170,'07  回到 main：保存 JSON 报告，程序结束',[
        '配置、数据指纹、首批预览 + 可选 smoke 结果 → .cache/runtime-check/architecture-baseline-local.json',
        'smoke 模式 training_run=true，但 baseline_completed=false；不保存模型权重，不进入第 2 次更新。'],BLUE,size=29)
    txt(60,4120,'读图提醒：100 次更新、8 批验证是暂定配置；本入口尚无完整训练/测速循环。formal 分支也不会进入本图训练部分。',27,AMBER,650)
    txt(60,4170,'图中 loss/范数来自已归档 CPU 检查；梯度 [2,−1,3,0] 仅为教学示意。gpt2_model.py 和训练入口均未因绘图改动。',26,MUTED)
    txt(60,4220,'代码定位：函数名与行号从当前源码提取；数据区间、形状、Token 预算与已有配置/报告核对。',26,MUTED)

    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}"><title>短训练基线配置、数据切分、batch 与梯度累积的代码流程</title><style>text {{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}}</style><rect width="{WIDTH}" height="{HEIGHT}" fill="#f5f8fb"/>'+''.join(d.bg+d.edges+d.fg)+'</svg>'
    ET.fromstring(svg)
    check={'source':'prepare_architecture_baseline.py','source_sha256':hashlib.sha256(source.encode()).hexdigest(),
           'function_lines':spans,'config_matches_run_report':True,'microbatch_ranges':[[i*N,(i+1)*N+1] for i in range(K)],
           'training_prediction_positions':N*K,'updates':1,'gradient_example_only':[2,-1,3,0],
           'gradient_example_accumulated':[.5,.25,1.,1.], 'new_training_run':False}
    return svg,check


if __name__=='__main__':
    svg,check=draw()
    path=OUT/'architecture-baseline-code-flow.svg'
    path.write_text(svg)
    (ROOT/'artifacts/architecture-baseline-diagram-check.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
    export_png(path)
    print(path)
