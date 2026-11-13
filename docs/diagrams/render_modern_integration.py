"""第23块：真实代码索引 + 默认张量尺寸；输出 SVG/PNG。"""
import ast
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from render_ddp_allreduce_flow import Drawing,export_png
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def main():
    positions={}
    for file in ('modern_model.py','train_modern.py','architecture_training.py','prepare_architecture_baseline.py'):
        tree=ast.parse((ROOT/file).read_text())
        for n in tree.body:
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)):
                positions[file+':'+n.name]=n.lineno
                if isinstance(n,ast.ClassDef):
                    for f in n.body:
                        if isinstance(f,ast.FunctionDef): positions[file+':'+n.name+'.'+f.name]=f.lineno
    def at(file,name):return f'{file} · {name} · L{positions[file+":"+name]}'
    d=Drawing();blue='#355fc0';teal='#087e75';ink='#172b40'
    def text(x,y,s,size=29,color=ink):d.text(x,y,s,size,color)
    def box(x,y,w,h,title,rows,color=blue):
        d.rect(x,y,w,h,'white','#dce4ec')
        d.text(x+25,y+44,title,32,color,700)
        for i,s in enumerate(rows):text(x+25,y+90+39*i,s,27)
    def arrow(y):d.arrow([(1200,y),(1200,y+34)],teal)
    d.text(60,80,'现代模型整合：同一个 ModernLM，完成训练、保存和缓存生成',43,ink,700)
    text(60,132,'第23块 · 新文件保留全部前课 · 直接随机初始化 · 不是加载或合并 Llama/Qwen 的预训练权重',28)
    box(60,170,2280,225,'01  配置与数据 | CPU 负责读取、分词、切分',[
        at('train_modern.py','load_config')+' → prepare_local_data → SequentialBatches.next_batch',
        '模型：V=50257，D=128，L=2，Hq=4，Hkv=2，d=32，FFN=341；共享 E / lm_head。',
        '莎剧 → GPT-2 IDs → 前90% train / 后10% val；先切分再错位，标签不跨 train/val 边界。',
        '每微批取257个ID：x=前256个 → [2,128]；y=后256个 → [2,128]；标签比输入向后1位。'])
    arrow(395)
    box(60,430,2280,185,'02  创建完整参数，再创建 AdamW | '+at('modern_model.py','ModernLM.__init__'),[
        '直接建立 Embedding、2个Block、最终RMSNorm、lm_head；不先构造GPT-2、不平均旧MHA权重。',
        'gamma 初值=1；矩阵正态初始化；输出残差投影缩放。已有模块按代码构造，不加载检查点。',
        'make_optimizer：二维矩阵做权重衰减，一维gamma不衰减；共享E只交给优化器一次。'])
    arrow(615)
    box(60,650,2280,145,'03  模型入口 | '+at('modern_model.py','ModernLM.forward'),[
        'x [2,128] → E查表 [2,128,128]；没有学习式位置表相加；位置在每层的RoPE进入Q/K。',
        '默认device=cpu：以下运算都在CPU；显式选cuda/mps时，模型和当前batch一起移动到该设备。'])
    arrow(795)
    d.rect(45,830,2310,1010,'#eef5f8','#355fc0')
    d.text(75,875,'04  一个 Block 内部 · 两层各有独立参数；输入与输出都为 [2,128,128]',33,blue,700)
    box(75,905,1100,265,'B1 → A1/A2：归一化、投影、位置',[
        'X → RMSNorm① → U [2,128,128]。',
        'Q=U Wqᵀ → [2,4,128,32]。',
        'K/V=U Wkᵀ / U Wvᵀ → [2,2,128,32]。',
        'Q/K头内RMSNorm：x / √(mean(x²)+ε) × γ。',
        '再RoPE旋转Q/K；V不归一化、不旋转。'])
    box(1225,905,1100,265,'A3/A4：分组注意力与输出投影',[
        'Q0/Q1读KV0，Q2/Q3读KV1；每个Q有自己的A/H。',
        'A=softmax(QKᵀ/√32 + 因果mask)。',
        'H=A×V → [2,4,128,32]。',
        '拼接四头 → [2,128,128]；O=C Woᵀ。',
        '推理才接入旧K/V；训练没有跨批缓存。'],teal)
    d.arrow([(1175,1035),(1225,1035)],teal)
    box(75,1205,2250,185,'第一次残差 | '+at('modern_model.py','ModernBlock.forward'),[
        'R = X + O → [2,128,128]；相同位置、相同维度逐元素相加。',
        '这里的X是Block入口原向量，不是RMSNorm的输出U。',
        '目的：主干保留原表示，并加入注意力读取到的上下文信息。'],teal)
    box(75,1425,2250,225,'第二条分支：RMSNorm② → SwiGLU → 第二次残差 | '+at('modern_model.py','ModernMLP.forward'),[
        'Z=RMSNorm②(R) → [2,128,128]；保留R作为第二次残差主干。',
        'a=Z Wgateᵀ、v=Z Wupᵀ → 各 [2,128,341]；两路都读取完整Z。',
        'u=SiLU(a) ⊙ v → [2,128,341]；F=u Wdownᵀ → [2,128,128]。',
        'Y=R+F → [2,128,128]；交给下一个Block。MLP的v不是注意力V。'],teal)
    text(85,1710,'调用顺序：ModernLM.forward → ModernBlock.forward → ModernAttention.forward / ModernMLP.forward',27)
    text(85,1760,'数学原语复用：HeadRMSNorm、apply_rotary、grouped_attention；前课文件保持原样。',27)
    arrow(1840)
    box(60,1875,2280,145,'05  全部 Block 后：最终 RMSNorm → lm_head',[
        '最终特征 [2,128,128]；logits = Hfinal Eᵀ → [2,128,50257]。',
        '训练：每个位置都预测下一个Token；生成：只取最后位置，得到 [B,1,50257] 分数。'])
    arrow(2020)
    box(60,2055,1100,340,'06  训练：无 KV Cache',[
        at('architecture_training.py','train_update'),
        'logits与y → 256个位置的平均交叉熵loss。',
        'loss/4.backward()；本次.grad加上历史.grad。',
        '重复4个微批；中间不更新任何模型参数。',
        '全局梯度裁剪 → AdamW.step() → 清梯度。',
        '一次更新处理2×128×4=1024个预测位置。',
        '共100次更新=102400个预测位置（默认）。'])
    box(1240,2055,1100,340,'07  验证：固定窗口、只观察',[
        at('architecture_training.py','evaluate'),
        '初始、每20次更新、最后一次更新后执行。',
        'eval + no_grad → 同一val窗口的平均loss。',
        '不backward、不更新、不改变训练游标。',
        '结束恢复原训练模式；继续下一次更新。',
        'PyTorch：矩阵运算、求导、裁剪、AdamW。',
        '本项目：取批、循环、模式切换、记录。'],teal)
    d.arrow([(1160,2225),(1240,2225)],teal)
    arrow(2395)
    box(60,2430,2280,185,'08  保存 → 新建模型 → 严格加载 | '+at('train_modern.py','load_checkpoint'),[
        'checkpoint.pt：模型配置 + 参数 + AdamW状态 + 更新次数 + 数据游标；不保存KV Cache。',
        '重建ModernLM，再load_state_dict(strict=True)；重载前后的同一提示词logits须完全相同。',
        '生成入口可单独重载；本课未实现训练断点续跑（缺少完整恢复协议），不宣称可无缝续训。'],teal)
    arrow(2615)
    box(60,2650,2280,265,'09  同一训练后模型生成 | '+at('modern_model.py','generate'),[
        'eval + inference_mode；prefill输入完整提示词[B,P]，保存每层K/V [B,2,P,32]。',
        '最后位置logits → argmax得到新ID；下一轮只输入该新ID[B,1]。',
        '每层新Q/K先Norm再RoPE（位置偏移S）；新K/V追加旧缓存 → [B,2,S+1,32]。',
        '读全部历史K/V算新位置H → 两次残差/MLP → 最终Norm/lm_head → 下一个ID。',
        '生成N个ID后缓存长度P+N−1：最后选出的ID尚未再次送入模型；缓存不进入参数更新。'],teal)
    arrow(2915)
    box(60,2950,2280,185,'10  核对与输出 | '+at('train_modern.py','generation_check'),[
        '同一模型分别做缓存生成与完整重算：贪心ID一致；沿相同前缀逐轮比较logits的浮点误差。',
        'report.json记录配置、数据指纹、loss、gamma变化、缓存字节与检查点重载结果。',
        '本课是整合流程验证；单次小语料短跑不证明优于GPT-2/Llama/Qwen，不保证流畅生成。'])
    text(60,3200,'来源：既有单元3实践 + Transformers v4.51.3 Qwen3 的模块顺序；沿用本地RoPE配对、词表与共享输出表。',26)
    svg='<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="3250" viewBox="0 0 2400 3250"><title>现代模型整合代码流程</title><style>text{font-family:PingFang SC,Helvetica Neue,Arial,sans-serif}</style><rect width="2400" height="3250" fill="#f8fafc"/>'+''.join(d.bg+d.edges+d.fg)+'</svg>'
    ET.fromstring(svg)
    path=OUT/'modern-integration-flow.svg';path.write_text(svg)
    export_png(path)
    (ROOT/'artifacts/modern-integration-diagram-map.json').write_text(json.dumps(positions,indent=2)+'\n')


if __name__=='__main__':main()
