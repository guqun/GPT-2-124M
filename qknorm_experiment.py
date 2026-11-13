"""第22块：GQA 与 GQA+QK-Norm；缓存核验、数据探针、独立副本的一步梯度检查。

随机小模型、合成 Token ID、FP32；不是预训练模型加载，也不以一次更新衡量质量。
"""
import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path
import torch
from torch.nn import functional as F
from prepare_architecture_baseline import ROOT,load_spec,sha
from swiglu_training import resolve_device
from gqa_model import GroupedDecoder
from kv_cache_experiment import generate,parameter_digest
from qknorm_model import build_model
from rope_model import apply_rotary

SOURCES=('qknorm_model.py','qknorm_experiment.py','gqa_model.py','kv_cache_experiment.py',
         'kv_cache_model.py','swiglu_model.py','rope_model.py','architecture_variants.py',
         'gpt2_model.py','swiglu_training.py','prepare_architecture_baseline.py')


def is_gamma(name):
    return '.q_norm.weight' in name or '.k_norm.weight' in name


def common_digest(model):
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        if is_gamma(name): continue
        digest.update(name.encode());digest.update(str(tuple(p.shape)).encode())
        digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def first_layer_probe(model,ids):
    """[E2] 第一层、头0、末位置；保存完整头向量而非仅显示的前4维。

两版本第一层输入相同，适合看单一改动；后续层输入会因注意力改变而不同。
"""
    block=model.transformer.h[0];a=block.attn
    u=block.ln_1(model.transformer.wte(ids))
    b,t,_=u.shape;d=a.head_dim
    q=a.q_proj(u).view(b,t,a.n_head,d).transpose(1,2)
    k=a.k_proj(u).view(b,t,a.n_kv_heads,d).transpose(1,2)
    v=a.v_proj(u).view(b,t,a.n_kv_heads,d).transpose(1,2)
    qn,kn=a.q_norm(q),a.k_norm(k)
    qr,kr=apply_rotary(qn,a.rope.cos[:t],a.rope.sin[:t]),apply_rotary(kn,a.rope.cos[:t],a.rope.sin[:t])
    _,cache=a.step(u)
    torch.testing.assert_close(cache[0],kr,rtol=0,atol=0)
    torch.testing.assert_close(cache[1],v,rtol=0,atol=0)
    select=lambda x:x[0,0,-1].cpu().tolist()
    return {'position':t-1,'head':0,'input_u':u[0,-1].cpu().tolist(),
            'q_raw':select(q),'k_raw':select(k),'v_raw':select(v),
            'q_r':float((q[0,0,-1].square().mean()+model.config.layer_norm_epsilon).sqrt()),
            'k_r':float((k[0,0,-1].square().mean()+model.config.layer_norm_epsilon).sqrt()),
            'q_norm':select(qn),'k_norm':select(kn),'q_rope':select(qr),'cached_k':select(kr),
            'cached_v':select(cache[1]),'cache_matches_post_norm_rope':True}


def gradient_probe(model):
    """[E4] 在独立副本上做一次完整序列反传/AdamW，推理原模型不变。

输入 1..8，目标 2..9；loss 对8个预测位置取平均。无缓存、无采样。
优化器包含全部参数；weight_decay=0，排除仅靠衰减让 gamma 变化的误判。
"""
    before=parameter_digest(model)
    probe=copy.deepcopy(model).train()
    device=next(probe.parameters()).device
    n=min(8,probe.config.n_positions)
    data=(torch.arange(n+1,device=device) % (probe.config.vocab_size-1)+1)[None,:]
    x,y=data[:,:-1],data[:,1:]
    optimizer=torch.optim.AdamW(probe.parameters(),lr=1e-3,weight_decay=0.)
    optimizer.zero_grad(set_to_none=True)
    logits=probe(x)  # 普通模型 forward，不调用 inference_mode 的 GroupedDecoder。
    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1))
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in probe.parameters()):
        raise ValueError('梯度检查遇到非有限值。')
    gamma={}
    saved={}
    for name,p in probe.named_parameters():
        if is_gamma(name):
            if p.grad is None or not bool(p.grad.abs().max()>0):
                raise ValueError('gamma 没有得到非零梯度。')
            saved[name]=p.detach().clone()
            gamma[name]={'gradient_l2':float(p.grad.norm()),'before_first4':p.detach()[:4].cpu().tolist()}
    optimizer.step()
    for name,p in probe.named_parameters():
        if name in gamma:
            delta=float((p-saved[name]).abs().max())
            if delta<=0: raise ValueError('gamma 未更新。')
            gamma[name].update(max_abs_update=delta,after_first4=p.detach()[:4].cpu().tolist())
    assert parameter_digest(model)==before
    return {'loss_before_update':float(loss.detach()),'input_ids':x.cpu().tolist(),
            'target_ids':y.cpu().tolist(),'optimizer':'AdamW','learning_rate':.001,'weight_decay':0.,
            'updates_on_copy':1,'gamma':gamma,'original_model_unchanged':True,
            'scope':'gradient plumbing check; not short training or quality evaluation'}


def main(enabled):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda','mps','auto'],default='cpu')
    parser.add_argument('--prompt-length',type=int,default=32)
    parser.add_argument('--new-tokens',type=int,default=16)
    parser.add_argument('--threads',type=int,default=4)
    variant='qknorm' if enabled else 'reference'
    parser.add_argument('--output',type=Path,default=ROOT/'.cache/qknorm-comparison'/variant)
    args=parser.parse_args()
    try:
        if min(args.prompt_length,args.new_tokens,args.threads)<1: raise ValueError('参数须为正整数。')
        output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
        if any(output.iterdir()): raise ValueError('输出目录非空，请使用 --output 指定新目录。')
        torch.set_num_threads(args.threads);torch.set_float32_matmul_precision('highest')
        if torch.cuda.is_available(): torch.backends.cuda.matmul.allow_tf32=False
        device=resolve_device(args.device)
        spec=load_spec('local');spec['position_experiment']={'theta':10000.,'pairing':'adjacent','max_length':spec['model']['n_positions']}
        if spec['model']['n_head']!=4: raise ValueError('本对照固定4个Q头、2个KV头。')
        if args.prompt_length+args.new_tokens-1>spec['model']['n_positions']: raise ValueError('超过固定上下文容量。')
        # [E1] 原始结构/参数一致；新增 gamma 初始全1，不等于关闭归一化。
        model=build_model(spec,enabled).to(device).eval()
        initial=parameter_digest(model);common=common_digest(model)
        ids=(torch.arange(args.prompt_length,device=device) % (model.config.vocab_size-1)+1)[None,:]
        probe=first_layer_probe(model,ids)
        decoder=GroupedDecoder(model).eval()
        # [E3] 每个模型内部比较完整重算与缓存，不要求两个不同模型的输出相同。
        full,lf,rf=generate(decoder,ids,args.new_tokens,False)
        cached,lc,rc=generate(decoder,ids,args.new_tokens,True)
        if not torch.equal(full,cached): raise ValueError('同模型贪心路径分岔，不能继续比较不同前缀分数。')
        torch.testing.assert_close(lf,lc,rtol=1e-4,atol=1e-5)
        assert parameter_digest(model)==initial and all(p.grad is None for p in model.parameters())
        grad=gradient_probe(model)
        dim=model.config.n_embd//model.config.n_head
        length=args.prompt_length+args.new_tokens-1
        expected=2*model.config.n_layer*1*2*length*dim*4
        assert rc[-1]['cache_bytes']==expected
        torch.save({'full':lf.cpu(),'cached':lc.cpu()},output/'logits.pt')
        protocol={'spec':spec,'common_parameters_sha256':common,'prompt_ids':ids.cpu().tolist(),
                  'new_tokens':args.new_tokens,'device':str(device),'torch':str(torch.__version__),
                  'platform':platform.platform(),'threads':args.threads,'precision':'fp32','tf32':False,
                  'sources':{n:sha(ROOT/n) for n in SOURCES}}
        report={'completed':True,'variant':variant,'protocol':protocol,'parameters_sha256':initial,
                'parameters':sum(p.numel() for p in model.parameters()),
                'gamma_parameters':sum(p.numel() for name,p in model.named_parameters() if is_gamma(name)),
                'max_abs_cache_logit_difference':float((lf-lc).abs().max()),'same_greedy_ids':True,
                'generated_ids':cached[:,args.prompt_length:].cpu().tolist(),
                'cache_shape':[1,2,length,dim],'kv_tensor_bytes':expected,'cache_steps':rc,
                'first_layer_probe':probe,'gradient_probe':grad,'inference_parameters_unchanged':True,
                'logits_sha256':sha(output/'logits.pt'),
                'limitations':'synthetic IDs/random weights; no performance or quality claim; not full Qwen3'}
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(f'{variant}：4 Q / 2 KV；{device}/FP32；推理参数不变。')
        print(f'缓存/重算核验通过，最大 logits 差 {report["max_abs_cache_logit_difference"]:.3g}。')
        print(f'参数 {report["parameters"]:,}；新增 gamma {report["gamma_parameters"]}；缓存 {expected:,} 字节。')
        print(f'第一层 q：r={probe["q_r"]:.6f}；原值前4维={probe["q_raw"][:4]}')
        label='归一化后' if enabled else 'Identity 跳过后'
        print(f'{label}前4维={probe["q_norm"][:4]}；RoPE后前4维={probe["q_rope"][:4]}')
        gamma_message=f'{len(grad["gamma"])} 组 gamma 有梯度并更新' if enabled else '参照模型没有新增 gamma'
        print(f'独立副本反传检查：loss={grad["loss_before_update"]:.6f}；{gamma_message}。')
        print(f'报告：{output/"report.json"}')
    except (ValueError,OSError,KeyError,RuntimeError,AssertionError) as error:
        parser.exit(1,str(error)+'\n')
