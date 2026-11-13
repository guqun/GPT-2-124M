"""第23块：随机初始化 → 莎剧短训练 → 验证 → 保存重载 → KV Cache 生成。

默认100次更新；--steps 3 可先检查流程。所有旧入口保留，非空输出目录拒绝覆盖。
当前只做单进程 FP32。训练数学与前课相同，复用已经核验的取批/累积/验证函数。
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

try:
    import torch
    import tiktoken
except ImportError as error:
    raise SystemExit(f'缺少课程依赖：{error}\n请使用 /Users/guqun/micrograd/venv/bin/python')
from modern_model import ModernConfig,ModernLM,generate
from prepare_architecture_baseline import ROOT,prepare_local_data,sha
from architecture_training import SequentialBatches,train_update,evaluate,learning_rate,resolve_device,synchronize


def load_config(path):
    # [T1] JSON 描述架构和实验，不包含已训练参数。
    spec=json.loads(path.read_text())
    c=ModernConfig(**spec['modern_model'])
    # 原数据准备函数仍读 model 小节，显式核对两份表示，避免无声漂移。
    for old,new in (('vocab_size','vocab_size'),('n_positions','max_length'),('n_embd','width'),('n_layer','layers'),('n_head','q_heads')):
        if spec['model'][old]!=getattr(c,new): raise ValueError('数据规格与现代模型配置不一致：'+old)
    t=spec['training']
    for key in ('batch_size','seq_len','accumulation_steps','planned_updates','validation_batches','validation_every'):
        if type(t[key]) is not int or t[key]<1: raise ValueError(key+' 必须为正整数')
    if t['seq_len']>c.max_length or t['world_size']!=1 or t['precision']!='fp32' or t['compile']:
        raise ValueError('本课限定单进程 FP32/eager，T 不得超过容量')
    if any(spec['model'][key]!=0 for key in ('embd_pdrop','attn_pdrop','resid_pdrop')):
        raise ValueError('整合模型当前限定 dropout=0')
    return spec,c


def make_optimizer(model,t):
    # [T2] 先建全部参数再建 AdamW；共享 E 在 parameters() 中只出现一次。
    params=list(model.parameters())
    return torch.optim.AdamW([
        {'params':[p for p in params if p.ndim>=2],'weight_decay':t['weight_decay']},
        {'params':[p for p in params if p.ndim<2],'weight_decay':0.0},
    ],lr=t['peak_lr'],betas=tuple(t['betas']),eps=t['adam_eps'])


def load_checkpoint(path,device):
    # 仅加载自己本项目生成、可信的检查点；不会下载官方权重。
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['format']!='modern-teaching-v1': raise ValueError('非本课检查点')
    model=ModernLM(ModernConfig(**saved['model_config']))
    model.load_state_dict(saved['model'],strict=True)
    return model.to(device).eval(),saved


def generation_check(model,prompt_ids,new_tokens):
    # [T6] 同一个训练后模型，检查缓存/重算逐轮 logits 与贪心 ID。
    cached,past=generate(model,prompt_ids,new_tokens,True)
    full,_=generate(model,prompt_ids,new_tokens,False)
    torch.testing.assert_close(cached,full,rtol=0,atol=0)
    maximum=0.0;cache=None
    was_training=model.training;model.eval()
    try:
        with torch.inference_mode():
            for step in range(new_tokens):
                prefix=cached[:,:prompt_ids.shape[1]+step]
                a,cache=model(prefix if step==0 else prefix[:,-1:],cache,use_cache=True,last_only=True)
                b=model(prefix,last_only=True)
                torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
                maximum=max(maximum,float((a-b).abs().max()))
    finally:
        model.train(was_training)
    size=sum(x.numel()*x.element_size() for pair in past for x in pair)
    return cached,dict(greedy_ids_equal=True,cache_logit_max_difference=maximum,
        kv_tensor_bytes=size,cache_shape_per_layer=list(past[0][0].shape),
        cache_length_explanation='prompt + new_tokens - 1; final selected ID not yet forwarded')


def run(args):
    spec,c=load_config(args.config)
    t=spec['training']
    if args.steps is not None: t['planned_updates']=args.steps
    steps=t['planned_updates']
    if steps<2: raise ValueError('steps 至少为2')
    t['lr_warmup_updates']=min(t['lr_warmup_updates'],max(1,steps//10))
    if args.threads<1: raise ValueError('threads 须为正数')
    torch.set_num_threads(args.threads);torch.set_float32_matmul_precision('highest')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=resolve_device(args.device)
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise ValueError(f'输出目录非空，请使用新的 --output：{out}')
    # [T3] CPU 读取/分词/90%—10%切分；每批 x,y 是错位1个 ID 的 [B,T]。
    train,val,data=prepare_local_data(spec)
    tokenizer=tiktoken.get_encoding('gpt2')
    prompt=torch.tensor([tokenizer.encode_ordinary(spec['generation']['prompt'])],dtype=torch.long,device=device)
    count=spec['generation']['new_tokens']
    if type(count) is not int or count<1 or prompt.shape[1]+count>c.max_length:
        raise ValueError('生成预算不合法')
    torch.manual_seed(spec['seed'])
    model=ModernLM(c).to(device).float()  # 全新随机参数；未导入旧检查点。
    optimizer=make_optimizer(model,t)
    loader=SequentialBatches(train,t['batch_size'],t['seq_len'])
    gamma_before={n:p.detach().clone() for n,p in model.named_parameters() if '.q_norm.' in n or '.k_norm.' in n}
    report=dict(spec=spec,data=data,device=str(device),torch_version=str(torch.__version__),threads=args.threads,
        parameters=sum(p.numel() for p in model.parameters()),training=[],validation=[],completed=False,
        scope='fresh random local integration run; not official Qwen3 or controlled architecture-quality comparison',
        sources={f:sha(ROOT/f) for f in ('modern_model.py','train_modern.py','architecture_training.py','prepare_architecture_baseline.py','gqa_model.py','qknorm_model.py','rope_model.py')})
    initial=evaluate(model,val,t,device)
    report['validation'].append(dict(update=0,**initial))
    print(f'现代模型整合：{report["parameters"]:,} 参数；随机初始化；{device}/FP32',flush=True)
    print(f'验证 000：loss={initial["loss"]:.6f}',flush=True)
    with (out/'metrics.jsonl').open('w') as log:
        for update in range(1,steps+1):
            lr=learning_rate(update,steps,t['lr_warmup_updates'],t['peak_lr'],t['min_lr'])
            for group in optimizer.param_groups: group['lr']=lr
            # [T4] K 次 loss/K.backward → 裁剪 → optimizer.step；不带 KV Cache。
            synchronize(device);start=time.perf_counter()
            record=train_update(model,optimizer,loader,t,device)
            synchronize(device);elapsed=time.perf_counter()-start
            record.update(update=update,lr=lr,seconds=elapsed,tokens=t['batch_size']*t['seq_len']*t['accumulation_steps'])
            report['training'].append(record);log.write(json.dumps(record)+'\n');log.flush()
            print(f'更新 {update:03d}：loss={record["loss"]:.6f} lr={lr:.7f}',flush=True)
            if update%t['validation_every']==0 or update==steps:
                result=evaluate(model,val,t,device)
                report['validation'].append(dict(update=update,**result))
                print(f'验证 {update:03d}：loss={result["loss"]:.6f}',flush=True)
    report['gamma_updates']={n:float((p-before).abs().max()) for n,before in gamma_before.items() for name,p in model.named_parameters() if name==n}
    # [T5] 保存配置+权重+AdamW状态；不保存 KV Cache（更新参数后旧缓存失效）。
    torch.save(dict(format='modern-teaching-v1',model_config=asdict(c),model=model.state_dict(),
        optimizer=optimizer.state_dict(),spec=spec,updates=steps,loader_cursor=loader.cursor),out/'checkpoint.pt')
    reloaded,_=load_checkpoint(out/'checkpoint.pt',device)
    model.eval()
    with torch.inference_mode(): torch.testing.assert_close(model(prompt),reloaded(prompt),rtol=0,atol=0)
    ids,check=generation_check(reloaded,prompt,count)
    report['generation']=dict(**check,text=tokenizer.decode(ids[0].tolist()),ids=ids[0].tolist())
    report['checkpoint_reload_exact']=True
    report['summary']=dict(updates=steps,training_tokens=sum(r['tokens'] for r in report['training']),initial_val_loss=initial['loss'],final_val_loss=report['validation'][-1]['loss'])
    report['checkpoint_sha256']=sha(out/'checkpoint.pt');report['completed']=True
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('生成：'+repr(report['generation']['text']),flush=True)
    print('缓存检查：'+json.dumps(check,ensure_ascii=False),flush=True)
    print('报告：'+str(out/'report.json'),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/modern-integrated-local.json')
    parser.add_argument('--steps',type=int)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--device',choices=['cpu','mps','cuda','auto'],default='cpu')
    parser.add_argument('--output',type=Path,default=ROOT/'.cache/modern-integrated/default')
    args=parser.parse_args()
    try: run(args)
    except (ValueError,OSError) as error: parser.exit(1,str(error)+'\n')


if __name__=='__main__': main()
