"""精确代码对照图；从源码提取函数行号，不启动训练。"""
import ast
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from render_ddp_allreduce_flow import Drawing, export_png

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BLUE, TEAL, INK, RED = '#355fc0', '#087e75', '#172b40', '#cf4141'


def draw():
    spans = {}
    for filename in ('architecture_training.py', 'architecture_variants.py', 'compare_architecture_runs.py'):
        spans[filename] = {node.name: [node.lineno, node.end_lineno]
                           for node in ast.walk(ast.parse((ROOT/filename).read_text()))
                           if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    def loc(name, file='architecture_training.py'):
        start, end = spans[file][name]
        return f'{name} · L{start}–{end}'
    d = Drawing()
    def text(x, y, s, size=29, color=INK): d.text(x, y, s, size, color)
    def box(x, y, w, h, title, rows, color=BLUE, fill='white', size=28):
        d.rect(x, y, w, h, fill)
        d.text(x+28, y+46, title, 32, color, 700)
        for i, s in enumerate(rows): text(x+28, y+94+42*i, s, size)
    def arrow(points, color=TEAL): d.arrow(points, color, width=4)
    def band(y, title):
        d.rect(60, y, 2280, 64, '#eaf0f5', stroke='none', radius=10)
        d.text(84, y+44, title, 32, INK, 700)

    d.text(60, 82, '两份可运行代码：GPT-2 与 RMSNorm / 无 bias 对照', 47, INK, 700)
    text(60, 140, '每次只运行一边；共同训练循环、数据顺序与参数初值相同。红框标出结构变化，后续分别保存结果。', 28)
    box(60, 190, 1100, 155, 'A · train_baseline_gpt2.py', [
        "main('gpt2') → 原始 GPT-2 风格小模型",
        'LayerNorm + 线性 bias；6,862,464 个参数。'], BLUE, '#eef2ff')
    box(1240, 190, 1100, 155, 'B · train_baseline_rmsnorm.py', [
        "main('rmsnorm_no_bias') → 两项改动的组合",
        'RMSNorm + 无线性 bias；6,859,520 个参数。'], TEAL, '#e8f6f1')
    arrow([(610,345),(610,385),(1200,385),(1200,425)], BLUE)
    arrow([(1790,345),(1790,385),(1200,385)], TEAL)
    band(425, '01  共用 architecture_training.py · '+loc('main')+' → '+loc('run'))
    box(60, 520, 2280, 210, '读取 local 配置，固定本次协议', [
        '输入：architecture-baseline-local.json + 命令行参数；默认 100 次更新、seed=42、CPU/FP32、4 线程。',
        '模型：2 层 / 128 维 / 4 头；词表 50,257；B=2、T=128、K=4；无 Dropout、无 compile、无 DDP。',
        '输出：spec、device、独立输出目录。共同源代码/数据/初值指纹写入报告；已有结果不覆盖。'])
    arrow([(1200,730),(1200,780)])
    box(60, 780, 2280, 215, '02  复用 prepare_local_data → split_tokens → first_batch', [
        'CPU：原文校验 → GPT-2 分词 → 338,025 个整数 ID → train[0:304222] / val[304222:338025]。',
        '取 B×T+1=257 个 ID；x=buf[:-1].view(2,128)，y=buf[1:].view(2,128)。x/y 为 int64。',
        '输出：训练列表、验证列表。训练游标按 256 前进；验证始终从自身开头读 8 批，共 2,048 个预测位置。'])
    arrow([(1200,995),(1200,1040)])
    band(1040, '03  architecture_variants.py · '+loc('build_model','architecture_variants.py')+'：先相同初始化，再分支')
    box(60, 1135, 1100, 330, 'A：保留 GPT，不替换模块', [
        '输入：相同 seed=42、相同 GPTConfig。',
        '每个 Block：LN1 → Attention → 残差',
        '                   LN2 → GELU MLP → 残差',
        '所有 Block 后还有一个最终 LayerNorm。',
        'LN(x)=γ⊙(x−mean(x))/√(var(x)+ε)+β',
        '线性层：y=xWᵀ+b；lm_head 原本就无 bias。'], BLUE, 'white', 27)
    box(1240, 1135, 1100, 330, 'B：replace_norm_and_bias（创建优化器前）', [
        '输入：与 A 完全相同的初始 GPT 参数。',
        '替换 5 处 LN → RMSNorm；保留 gamma 初值 1。',
        'RMS(x)=γ⊙x/√(mean(x²)+ε)，没有 beta。',
        '移除注意力/MLP 的 8 组线性 bias：y=xWᵀ。',
        '输出：共同权重原值保留的改动模型。',
        '仍是绝对位置表、MHA、GELU、共享 lm_head。'], TEAL, '#fff5e7', 27)
    d.rect(1252,1147,1076,306,'none',RED,16)
    box(60, 1510, 2280, 170, '归一化手算对照：x=[1,2,3,4]，γ=1、β=0；仅此例忽略 ε', [
        'A：均值=2.5，方差=1.25 → (x−2.5)/√1.25 → [−1.3416,−0.4472,0.4472,1.3416]。',
        'B：平方均值=(1+4+9+16)/4=7.5 → x/√7.5 → [0.3651,0.7303,1.0954,1.4606]。'], TEAL)
    arrow([(610,1465),(610,1510)], BLUE); arrow([(1790,1465),(1790,1510)], TEAL)
    arrow([(1200,1680),(1200,1730)])
    box(60, 1730, 2280, 210, '04  创建各自 AdamW → 初始验证 · '+loc('evaluate'), [
        '模型 .to(device)；默认全部 CPU。选 CUDA/MPS 时，参数在设备，ID 列表仍在 CPU。',
        'AdamW：二维参数 weight decay=0.1，一维=0；β=(0.9,0.95)，ε=1e−8；参数共享只注册一次。',
        '固定 8 批 val → eval + no_grad → 总交叉熵 / 2,048 → val_loss；恢复原模式，不改训练游标。'])
    arrow([(1200,1940),(1200,1990)])
    band(1990, '05  外层循环 update=1…100：设置学习率 → 同步设备 → 开始计时')
    box(60, 2085, 2280, 170, '学习率预热与计时预热是两件事 · '+loc('learning_rate'), [
        '学习率：第 1…10 次为 0.0006×update/10；此后余弦降至第 100 次的 0.00006。',
        '计时：前 2 次正常训练，但不计入稳定吞吐；这 2 次仍消耗训练数据并更新参数。'])
    arrow([(1200,2255),(1200,2305)])
    box(60, 2305, 2280, 255, '06  '+loc('train_update')+'：先 zero_grad，再循环 4 个微批次', [
        'next_batch() → CPU x/y [2,128] → .to(device) → model(x) → logits [2,128,50257]。',
        'PyTorch cross_entropy → 本微批 256 个位置的平均损失 Lm（标量）。',
        '(Lm/4).backward() → 梯度加入同一套参数的 .grad；4 轮内参数 θ 不变。',
        '第 1 次更新读取 train 区间：[0,257)、[256,513)、[512,769)、[768,1025)。'], TEAL, '#e8f6f1')
    arrow([(1200,2560),(1200,2610)])
    box(60, 2610, 2280, 210, '展开累计：gₘ=∂Lₘ/∂θ；每个参数都各自累加', [
        '.grad：g₀/4 → (g₀+g₁)/4 → (g₀+g₁+g₂)/4 → (g₀+g₁+g₂+g₃)/4。',
        '单参数示例 g=[2,−1,3,0]：加进 [0.5,−0.25,0.75,0] → 最终 .grad=1。（教学值）',
        '四轮共 1,024 个预测位置；loss 除以 4，不是把四次参数更新合并。'], TEAL)
    arrow([(1200,2820),(1200,2870)])
    box(60, 2870, 2280, 210, '07  裁剪 → 更新一次参数 → 清梯度 → 同步设备并结束计时', [
        'clip_grad_norm_(全部参数,1.0)：按全局范数统一缩放累计梯度，再 optimizer.step()。',
        'AdamW 更新 m/v 并修正初始偏差：θ←(1−lr×λ)θ−lr×m̂/(√v̂+ε)；zero_grad 不清 m/v。',
        '输出：新参数、训练 loss、梯度范数、取批起点、耗时；本步 Token/s=1,024/秒数。'], TEAL, '#e8f6f1')
    arrow([(1200,3080),(1200,3130)])
    box(60, 3130, 2280, 210, '08  每 20 次与最后一次：固定窗口验证；尚未到 100 次则回到 05', [
        '验证不在训练计时区间内，不 backward、不 step，也不根据 val_loss 改学习率。',
        '下一次更新继续从 train 游标取数；不足一批时跳过短尾、回到 train 开头，不跨入 val。',
        '默认验证记录：update=0、20、40、60、80、100。metrics.jsonl 逐步写入，便于观察。'])
    arrow([(2340,3230),(2375,3230),(2375,2022),(2340,2022)], BLUE)
    arrow([(1200,3340),(1200,3390)])
    band(3390, '09  完成后各自保存 → compare_architecture_runs.py 核对协议并比较')
    box(60, 3485, 1100, 210, 'A：.cache/architecture-comparison/gpt2/', [
        'report.json：配置、loss、时间、初值/数据指纹。',
        'metrics.jsonl：每次训练与验证记录。',
        'checkpoint.pt：模型/优化器/配置/更新次数。'], BLUE, '#eef2ff', 27)
    box(1240, 3485, 1100, 210, 'B：同父目录 / rmsnorm_no_bias/', [
        '同样保存三份文件，与 A 分开。',
        '默认 100 次更新 = 102,400 个训练预测位置。',
        '检查点可加载；本入口尚无断点续训选项。'], TEAL, '#e8f6f1', 27)
    arrow([(610,3695),(610,3730),(1200,3730),(1200,3770)], BLUE)
    arrow([(1790,3695),(1790,3730),(1200,3730)], TEAL)
    box(60, 3770, 2280, 210, '比较条件一致才输出对照表 · '+loc('compare','compare_architecture_runs.py'), [
        '核对配置/数据/共同权重初值/源码/设备/实际取批顺序 → 不一致就停止比较。',
        'loss：比较同一验证窗口的最终值；吞吐：测量步总 Token / 测量步总秒数，排除验证与保存。',
        '这是 RMSNorm + 去 bias 的组合实验；需后续单因素、多种子实验，才能拆分贡献与判断稳健性。'])
    text(60, 4050, '范围：本机小模型、从随机参数开始；不是加载 GPT-2 预训练权重，不等同完整 Llama，也不保证 B 更快。', 27)
    text(60, 4105, '公式与协议由本项目组织；矩阵计算、自动求导和 AdamW 由 PyTorch 执行。图中行号由当前源码提取。', 27)
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="4170" viewBox="0 0 2400 4170"><title>两份代码的结构与训练对照流程</title><style>text{font-family:"PingFang SC","Helvetica Neue",Arial,sans-serif;}</style><rect width="2400" height="4170" fill="#f5f8fb"/>'+''.join(d.bg+d.edges+d.fg)+'</svg>'
    ET.fromstring(svg)
    return svg, spans


if __name__ == '__main__':
    svg, spans = draw()
    path = OUT/'architecture-comparison-training-flow.svg'
    path.write_text(svg)
    (ROOT/'artifacts/architecture-comparison-diagram-map.json').write_text(json.dumps(spans,indent=2)+'\n')
    export_png(path)
    print(path)
