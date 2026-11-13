"""KV Cache 推理实验：不训练，不反传；默认合成 Token ID，重点核对计算等价。

可用 --checkpoint 指定前课 SwiGLU checkpoint.pt；两边必须加载同一文件。
默认随机小模型足以验证缓存行为，不能用它评价生成文本的质量。
"""
import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
import torch
from prepare_architecture_baseline import ROOT, load_spec, sha
from swiglu_model import build_model
from kv_cache_model import CachedDecoder, cache_bytes
from swiglu_training import resolve_device, synchronize


def parameter_digest(model):
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        digest.update(name.encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def generate(decoder, prompt, new_tokens, use_cache):
    """生成固定 N 个 Token；prefill 产生第一个，N−1 次 decode 产生其余。"""
    ids=prompt.clone(); cache=None; rows=[]; logits_history=[]
    device=prompt.device
    for step in range(new_tokens):
        # [G1] 第一轮两边都输入整个 prompt；有 cache 后只输入末尾新 Token。
        current=ids if not use_cache or cache is None else ids[:,-1:]
        before=0 if cache is None else cache[0][0].shape[2]
        synchronize(device);started=time.perf_counter()
        logits,cache=decoder(current,cache,use_cache=use_cache)
        # [G2] 固定用贪心，排除随机采样干扰；最后选择的 Token 尚未进入缓存。
        token=logits[:,-1,:].argmax(dim=-1,keepdim=True)
        ids=torch.cat((ids,token),dim=1)
        synchronize(device);seconds=time.perf_counter()-started
        logits_history.append(logits[:,-1,:])
        rows.append({'step':step,'phase':'prefill' if step==0 else 'decode',
                     'input_positions':current.shape[1], 'context_positions':ids.shape[1]-1,
                     'cache_before':before,'cache_after':0 if cache is None else cache[0][0].shape[2],
                     'cache_bytes':cache_bytes(cache),'seconds':seconds})
    return ids,torch.stack(logits_history,dim=1),rows


def main(use_cache):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--prompt-length',type=int,default=32)
    parser.add_argument('--new-tokens',type=int,default=16)
    parser.add_argument('--device',choices=['cpu','cuda','mps','auto'],default='cpu')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,default=ROOT/'.cache/kv-cache-comparison'/('cached' if use_cache else 'reference'))
    args=parser.parse_args()
    try:
        if min(args.prompt_length,args.new_tokens,args.threads,args.repeats)<1:
            raise ValueError('长度、生成数、线程数和重复次数须为正整数。')
        output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
        if any(output.iterdir()):raise ValueError('输出目录非空，请用 --output 指定新目录。')
        torch.set_num_threads(args.threads);torch.set_float32_matmul_precision('highest')
        if torch.cuda.is_available():torch.backends.cuda.matmul.allow_tf32=False
        device=resolve_device(args.device)
        checkpoint_hash=None
        if args.checkpoint:
            checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
            if checkpoint['variant']!='swiglu':raise ValueError('本课只接受前课 SwiGLU 检查点。')
            spec=checkpoint['spec'];model=build_model(spec,'swiglu')
            model.load_state_dict(checkpoint['model'],strict=True)
            checkpoint_hash=sha(args.checkpoint)
        else:
            spec=load_spec('local')
            spec['position_experiment']={'theta':10000.,'pairing':'adjacent','max_length':spec['model']['n_positions']}
            model=build_model(spec,'swiglu')
        if args.prompt_length+args.new_tokens-1 > model.config.n_positions:
            raise ValueError('生成期间要处理的前缀超出模型固定容量。')
        before=parameter_digest(model)
        decoder=CachedDecoder(model).to(device).eval()
        prompt=(torch.arange(args.prompt_length,device=device)%(model.config.vocab_size-1)+1)[None,:]
        print('相同 RoPE/SwiGLU 小模型；'+('前课检查点' if args.checkpoint else '随机权重')+'；合成 ID 输入。')
        print('模式：'+('KV Cache' if use_cache else '完整前缀重算')+f'；{device}/FP32；不反传、不更新。')
        generate(decoder,prompt,args.new_tokens,use_cache)  # 预热；缓存只在此调用内部存活。
        runs=[]
        for _ in range(args.repeats):
            ids,logits,rows=generate(decoder,prompt,args.new_tokens,use_cache)
            runs.append({'total_seconds':sum(r['seconds'] for r in rows),
                         'prefill_seconds':rows[0]['seconds'],
                         'decode_seconds':sum(r['seconds'] for r in rows[1:])})
        assert parameter_digest(model)==before
        assert all(p.grad is None for p in model.parameters())
        protocol={'spec':spec,'parameters_sha256':before,'checkpoint_sha256':checkpoint_hash,
                  'prompt_ids':prompt.cpu().tolist(),'new_tokens':args.new_tokens,'greedy':True,
                  'device':str(device),'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
                  'platform':platform.platform(),'torch':torch.__version__,'threads':args.threads,
                  'precision':'fp32','tf32':False,'repeats':args.repeats,'warmup_runs':1,
                  'sources':{n:sha(ROOT/n) for n in ('gpt2_model.py','architecture_variants.py','rope_model.py',
                      'swiglu_model.py','kv_cache_model.py','kv_cache_experiment.py','swiglu_training.py')}}
        # 分数在计时后移到 CPU；独立保存用于比较每轮全部词表 logits。
        torch.save(logits.cpu(),output/'logits.pt')
        report={'mode':'cached' if use_cache else 'reference','protocol':protocol,'runs':runs,
                'generated_ids':ids[:,args.prompt_length:].cpu().tolist(),'steps':rows,
                'processed_positions_per_layer':sum(r['input_positions'] for r in rows),
                'kv_tensor_bytes':rows[-1]['cache_bytes'], 'parameters_unchanged':True,
                'logits_sha256':sha(output/'logits.pt'),
                'median_seconds':statistics.median(r['total_seconds'] for r in runs),
                'median_prefill_seconds':statistics.median(r['prefill_seconds'] for r in runs),
                'median_decode_seconds':statistics.median(r['decode_seconds'] for r in runs),
                'timing_scope':'forward + greedy + ID append; per-step synchronized, excludes logging/CPU copies/save',
                'completed':True}
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        for r in rows:
            print(f"{r['phase']:7s} {r['step']:02d}：输入 {r['input_positions']} 个位置；上下文 {r['context_positions']}；缓存 {r['cache_before']} → {r['cache_after']}")
        print(f"每层累计处理 {report['processed_positions_per_layer']} 个位置；K/V 有效字节 {report['kv_tensor_bytes']}")
        print(f"{args.repeats} 次中位耗时 {report['median_seconds']*1000:.2f} ms；报告：{output/'report.json'}")
    except (ValueError,OSError,KeyError,RuntimeError) as error:
        parser.exit(1,str(error)+'\n')
