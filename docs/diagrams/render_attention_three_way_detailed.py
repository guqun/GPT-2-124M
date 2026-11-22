"""三列逐步骤对照图；数值由同一组 X/W 复算，旧课程代码保持不变。"""
from pathlib import Path
import argparse
import json
import math
import unicodedata
import xml.etree.ElementTree as ET
from html import escape

import torch
from render_ddp_allreduce_flow import export_png

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
STEM = 'attention-three-way-parameters-detailed'


def example():
    X = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
    Wq = torch.tensor([[1., 0.], [1., 1.]], dtype=X.dtype)
    Wk = torch.eye(2, dtype=X.dtype)
    Wv = torch.tensor([[10., 2.], [4., 8.]], dtype=X.dtype)
    Wo = torch.tensor([[1., 1.], [0., 1.]], dtype=X.dtype)
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = Q @ K.T / math.sqrt(2)
    distances = torch.arange(3)[:, None] - torch.arange(3)[None, :]
    result = dict(X=X, W_Q=Wq, W_K=Wk, W_V=Wv, W_O=Wo, Q=Q, K=K, V=V, scores=scores)
    for name, allowed in [('global', distances >= 0), ('window', (distances >= 0) & (distances < 2))]:
        A = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        H = A @ V
        result[name] = dict(allowed=allowed, A=A, H=H, O=H @ Wo)
    qp, kp = torch.nn.functional.elu(Q) + 1, torch.nn.functional.elu(K) + 1
    S, z = torch.zeros(2, 2, dtype=X.dtype), torch.zeros(2, dtype=X.dtype)
    steps = []
    for i in range(3):
        delta = torch.outer(kp[i], V[i])
        S, z = S + delta, z + kp[i]
        numerator, denominator = qp[i] @ S, qp[i] @ z
        h = numerator / denominator
        weights = qp[i] @ kp[:i+1].T
        weights = weights / weights.sum()
        torch.testing.assert_close(h, weights @ V[:i+1], rtol=1e-12, atol=1e-12)
        steps.append(dict(delta=delta, S=S, z=z, numerator=numerator,
                          denominator=denominator, H=h, O=h @ Wo, weights=weights))
    result['linear'] = dict(Qbar=qp, Kbar=kp, steps=steps)
    torch.testing.assert_close(steps[-1]['H'], torch.tensor([9.92, 6.88], dtype=X.dtype))
    def plain(obj):
        if isinstance(obj, torch.Tensor): return obj.tolist()
        if isinstance(obj, dict): return {k: plain(v) for k, v in obj.items()}
        if isinstance(obj, list): return [plain(v) for v in obj]
        return obj
    return plain(result)


def draw(r):
    width = 3000
    parts = []
    colors = ['#245bbb', '#a75b11', '#0b7662']
    fills = ['#f1f6ff', '#fff6eb', '#edf9f4']
    xs, cw = [60, 1040, 2020], 920
    def text(x, y, s, size=28, color='#172b40', bold=False):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{700 if bold else 400}">{escape(s)}</text>')
    def rect(x, y, w, h, fill, stroke='#dae3ec'):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    def lines_wrap(lines):
        out = []
        for line in lines:
            current, units = '', 0
            for ch in line:
                n = 2 if unicodedata.east_asian_width(ch) in 'WF' else 1
                if units + n > 58:
                    out.append(current); current, units = '  ', 2
                current += ch; units += n
            out.append(current)
        return out
    def band(y, label):
        rect(60, y, 2880, 64, '#e6edf5')
        text(84, y+44, label, 32, bold=True)
    def row(y, title, columns):
        band(y, title)
        lines = [lines_wrap(col) for col in columns]
        h = 44 + 41 * max(map(len, lines))
        for x, color, fill, col in zip(xs, colors, fills, lines):
            rect(x, y+82, cw, h, fill, color)
            for i, line in enumerate(col):
                text(x+24, y+123+41*i, line, 28, color if line.startswith(('参数', '差异', '输出')) else '#172b40', line.startswith(('参数', '差异', '输出')))
            cy = y+82+h
            parts.append(f'<path d="M{x+cw/2},{cy+3} v24" stroke="{color}" stroke-width="3" marker-end="url(#arrow)"/>')
        return y + 82 + h + 42
    v = lambda a: '[' + ', '.join(f'{x:.6f}' for x in a) + ']'
    text(60, 76, '三种注意力逐步骤对照：输入特征 → 可训练参数 → 中间计算 → 输出', 46, bold=True)
    text(60, 127, '阶段二 · 单元3 · 3.9｜同一组 X 与参数快照｜单头，B=1、T=3、D=2｜主线追踪 C 位置', 30)
    rect(60, 160, 2880, 240, '#ffffff')
    for i, line in enumerate([
        '边界：这是注意力子层的完整对照算例，不是完整 Transformer Block；未展开 Embedding、Norm、RoPE、残差、MLP 和词表输出。',
        'X 是进入子层的特征，不是 Token ID。参数为手工设定的演示快照，未训练；三条路径从同一组参数值出发。',
        '记号：W 是可训练模型参数；Q/K/V、权重 A、状态 S/z 和 H/O 是运行时结果。图中矩阵 [[a,b],[c,d]] 表示两行。',
        '本图统一按数学约定 XW；PyTorch nn.Linear.weight 按 [out,in] 保存，实际使用 X @ weight.T，即存储矩阵是本图 W 的转置。',
        '三列纵向读流程、横向比同一步。每个步骤都注明参数；状态更新 ≠ 优化器更新参数。数值用 FP64 复算，显示值四舍五入。',
    ]): text(84, 202+i*41, line, 27)
    for x, c, title in zip(xs, colors, ['① 全局因果 softmax', '② 滑动窗口 softmax · W=2', '③ 基础线性注意力 · ELU+1']):
        rect(x, 428, cw, 82, c, c); text(x+25, 482, title, 36, '#ffffff', True)
    y = 542
    shared = [
        '输入 X [3,2] = [[1,0],[0,1],[1,1]]',
        '行分别对应 A、B、C；x_C=[1,1]。',
        '参数 W_Q [2,2] = [[1,0],[1,1]]',
        '参数 W_K [2,2] = [[1,0],[0,1]]',
        '参数 W_V [2,2] = [[10,2],[4,8]]',
        '参数 W_O [2,2] = [[1,1],[0,1]]',
        '共16个可训练标量；无 bias。0/1也可训练。']
    y = row(y, '01｜输入与参数：三列完全相同，后面比较的是计算规则', [shared]*3)
    projection = [
        '参数：W_Q、W_K、W_V；所有位置共享。',
        'Q=XW_Q；K=XW_K；V=XW_V',
        '形状：[3,2] × [2,2] → [3,2]',
        'Q=[[1,0],[1,1],[2,1]]',
        'K=[[1,0],[0,1],[1,1]]',
        'V=[[10,2],[4,8],[14,10]]',
        'C展开：q_C=[1×1+1×1, 1×0+1×1]',
        '=[2,1]；k_C=[1,1]；v_C=[14,10]。']
    y = row(y, '02｜投影：相同输入通过相同参数，得到相同 Q、K、V', [projection]*3)
    y = row(y, '03｜进入核心计算：softmax 路径保留原 Q/K；线性路径先变换 Q/K', [
        ['参数：本步骤无新增参数。','输入 q_C=[2,1]；K、V 都为 [3,2]。','Q、K、V 直接参与后续点积。','不应用线性列的 φ 特征变换。','V 始终为 [[10,2],[4,8],[14,10]]。'],
        ['参数：本步骤无新增参数。','输入与左列相同；W=2 包含当前位置。','Q、K、V 直接参与后续点积。','窗口 W 是固定配置，不是训练参数。','C 的可见位置为 B、C。'],
        ['参数：φ 为固定函数，无可训练参数。','φ(x)=ELU(x)+1；x≥0 时 x+1，','x<0 时 exp(x)。本例均非负，逐元素加1。','Q̄=[[2,1],[2,2],[3,2]] [3,2]','K̄=[[2,1],[1,2],[2,2]] [3,2]','V 不变；当前 q̄_C=[3,2]。']])
    y = row(y, '04｜历史如何进入计算：逐位置匹配 vs 将 K/V 写入固定形状状态', [
        ['参数：无新增；分数由 Q/K 计算。','q_C Kᵀ：[1,2]×[2,3]→[1,3]','对 A：2×1+1×0=2','对 B：2×0+1×1=1','对 C：2×1+1×1=3','缩放：[2,1,3]/√2','=[1.414214,0.707107,2.121320]','C 的因果遮罩允许 A、B、C；分数不变。','历史表示：各位置的 K、V 分别保存。'],
        ['参数：无新增；与左列相同点积及缩放。','分数=[1.414214,0.707107,2.121320]','可见规则：0 ≤ 查询位置−键位置 < 2','C 对 A 的距离=2，不满足规则。','遮罩后=[−∞,0.707107,2.121320]','A 的指数变成0，不能直接参与加权。','历史表示：递推时仅需近期 K/V。','此处列出被遮分数方便对照；优化实现','可直接跳过窗口外的点积。'],
        ['参数：无新增；S/z 是状态，不是参数。','初始 S=zeros[2,2]，z=zeros[1,2]。','每步 ΔS=k̄ᵀv：[2,1]×[1,2]→[2,2]','A：[[2],[1]]×[10,2]','ΔS_A=[[20,4],[10,2]]','B：[[1],[2]]×[4,8]','ΔS_B=[[4,8],[8,16]]','C：[[2],[2]]×[14,10]','ΔS_C=[[28,20],[28,20]]','累加 S=[[52,32],[46,38]]；','z=[2,1]+[1,2]+[2,2]=[5,5]。']])
    y = row(y, '05｜归一化依据：softmax 指数总和 vs 查询与累计键的匹配总和', [
        ['参数：softmax 无可训练参数。','exp(s) ≈ [4.113250,2.028115,8.342145]','指数总和 ≈ 14.483510','a_j=exp(s_j)/Σexp(s)','输出权重 a_C [1,3]：',v(r['global']['A'][2]),'三项权重之和为1。'],
        ['参数：softmax 无可训练参数。','exp(遮罩分数) ≈ [0,2.028115,8.342145]','指数总和 ≈ 10.370260','a_j=exp(s_j)/Σexp(s)','输出权重 a_C [1,3]：',v(r['window']['A'][2]),'B/C 重新归一化，不只删除 A 的旧权重。'],
        ['参数：归一化运算无新增参数。','输入 q̄_C=[3,2]，z=[5,5]。','分母 q̄_C zᵀ=3×5+2×5=25（标量）','它是当前 Q 对所有历史的匹配总和。','核对：q̄_C·k̄_A=8，·k̄_B=7，·k̄_C=10。','等效权重=[8/25,7/25,10/25]。','递推读取无需实际构造逐位置权重。']])
    y = row(y, '06｜得到汇总特征 H：三列输出形状相同，数值不同', [
        ['参数：加权求和无新增参数。','H_C=a_C V：[1,3]×[3,2]→[1,2]','≈0.283995[10,2]+0.140029[4,8]',' +0.575975[14,10]','输出 H_C='+v(r['global']['H'][2]),'整段 H [3,2] 的前两行为 [10,2]、[7,5]。'],
        ['参数：加权求和无新增参数。','H_C=a_C V：[1,3]×[3,2]→[1,2]','≈0.195570[4,8]+0.804430[14,10]','输出 H_C='+v(r['window']['H'][2]),'前两位置未超窗口，H_A/H_B 与左列相同。'],
        ['参数：状态读取无新增参数。','分子=q̄_C S：[1,2]×[2,2]→[1,2]','=[3×52+2×46, 3×32+2×38]','=[248,172]；H_C=分子/25','输出 H_C=[9.92,6.88]','核对：(8[10,2]+7[4,8]+10[14,10])/25','=[9.92,6.88]；与状态读取完全相同。']])
    cols=[]
    for key in ['global','window','linear']:
        o=r[key]['O'][2] if key!='linear' else r[key]['steps'][2]['O']
        cols.append(['参数：W_O=[[1,1],[0,1]]，可训练。','输入 H_C [1,2]；O_C=H_C W_O','形状：[1,2]×[2,2]→[1,2]','若 H_C=[a,b]，则 O_C=[a,a+b]。','输出 O_C='+v(o)])
    y = row(y, '07｜输出投影：重新使用可训练参数 W_O；单头无需多头拼接', cols)
    y = row(y, '08｜下一个 Token 到来时：谁增加长度，谁只更新数值？', [
        ['参数 W 不因新 Token 到来而改变。','D 的输入特征经同一组 W 得到 q_D/k_D/v_D。','缓存从 A/B/C 追加为 A/B/C/D。','K 与 V 的形状各由 [3,2] 变为 [4,2]。','新 Q 匹配全部可见历史，再输出。'],
        ['参数 W 不因新 Token 到来而改变。','D 的特征同样经 W 生成 q_D/k_D/v_D。','窗口从 B/C 移动到 C/D。','缓存最多保存计算所需的 W 个位置。','新 Q 只匹配窗口内历史。'],
        ['参数 W 不因新 Token 到来而改变。','D 的特征经 W、φ 得到 q̄_D/k̄_D/v_D。','S_D=S_C+k̄_Dᵀv_D；z_D=z_C+k̄_D。','再计算 H_D=(q̄_D S_D)/(q̄_D z_Dᵀ)。','S 仍 [2,2]，z 仍 [1,2]；不增加 Token 行。','每层/头/独立序列分别维护状态；新序列清零。']])
    y = row(y, '09｜训练时的共同路径：可训练参数相同，但前向规则改变，梯度也会改变', [
        ['O → 后续网络 → loss → backward()','梯度经 W_O、AV、softmax、QK 回传。','optimizer 更新 W_Q/W_K/W_V/W_O。','无新增训练参数不代表这一步不参与反传。'],
        ['O → 后续网络 → loss → backward()','梯度经过窗口内的计算路径回传。','本例 C 输出对被遮分数的梯度为0。','同一个参数仍可能经其他位置/路径收到梯度。'],
        ['O → 后续网络 → loss → backward()','梯度经过读取、累计状态、φ 与 QKV 投影。','优化器更新 W；不把 S/z 当参数优化。','固定推理状态不代表训练全部内存固定。']])
    rect(60, y, 2880, 225, '#ffffff')
    for i, s in enumerate([
        '本图比较：①全部历史的 softmax ②窗口内的 softmax ③ELU+1 核匹配的线性状态递推。第三列不是前两列的等价加速。',
        '线性状态的行对应 K 特征、列对应 V 特征，不对应 Token；历史贡献相互叠加，固定形状不保证无损保存全部细节。',
        '当前仅展开注意力子层；参数总数16只属于本教学子层，不是完整模型参数量。未设 loss/标签，因此不虚构数值梯度或训练效果。',
        '依据：Katharopoulos et al., 2020, §3.2–3.3；https://proceedings.mlr.press/v119/katharopoulos20a.html',
        '配套 JSON 保存全部位置的 FP64 结果；SVG 为原生矢量文字和形状，可编辑，未嵌入位图。',
    ]): text(84, y+39+i*39, s, 27)
    height=y+260
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<title>全局、滑动窗口、基础线性注意力：参数与特征逐步骤对照</title>
<desc>相同 X 和四个投影参数，三列九步骤横向比较，包含全部中间数值、固定状态与参数更新的区别。</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#61758a"/></marker></defs>
<style>text{{font-family:"PingFang SC","Noto Sans CJK SC","Microsoft YaHei",sans-serif;}}</style>
<rect width="{width}" height="{height}" fill="#f7f9fc"/>
{''.join(parts)}</svg>'''


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--png',action='store_true')
    args=parser.parse_args()
    report=example()
    svg=draw(report)
    ET.fromstring(svg)
    path=OUT/(STEM+'.svg')
    path.write_text(svg,encoding='utf-8')
    (ROOT/'artifacts'/(STEM+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.png: export_png(path)
    print(path)
    print('Verified: per-position linear state output equals explicit normalized kernel attention.')
