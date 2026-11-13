"""单元 3 / 3.1：固定短训练基线的模型、数据和预算；保留所有旧入口。

默认仅检查配置、编码本地语料、隔离 train/val 并展示首批，不创建模型。
--profile formal 只列出 124M/FineWeb 的方案与缺项，不悄悄使用 demo。
--smoke-step 在本机小模型上完整累积一次梯度并更新，检查数据可送入模型。
这一次更新不属于未来正式短训练，不保存 checkpoint，也不是测速。
阅读路线：load_spec → prepare_local_data → first_batch → smoke_step（可选）。
【本项目】选择实验条件及数据边界；【PyTorch】张量、层计算、求导和 AdamW。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIGS = {
    'local': ROOT / 'configs/architecture-baseline-local.json',
    'formal': ROOT / 'configs/architecture-baseline-124m.json',
}
# 固定 GPT-2 原始分词资产：仅使用已存在的项目缓存，不触发隐式下载。
TOKENIZER_ASSETS = {
    '6d1cbeee0f20b3d9449abfede4726ed8212e3aee':
        '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
    '6c7ea1a7e38e3a7f062df639a5b80947f075ffe6':
        '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_spec(profile):
    """[C1] JSON 是规格，不是模型权重；先检查预算与张量尺寸是否自洽。"""
    spec = json.loads(CONFIGS[profile].read_text())
    m, t = spec['model'], spec['training']
    positive = [m[k] for k in ('vocab_size', 'n_positions', 'n_embd', 'n_layer', 'n_head')]
    positive += [t[k] for k in ('batch_size', 'seq_len', 'world_size',
                               'accumulation_steps', 'planned_updates', 'validation_batches')]
    if any(type(v) is not int or v < 1 for v in positive):
        raise ValueError('模型尺寸与训练计数必须是正整数。')
    if m['n_embd'] % m['n_head'] or t['seq_len'] > m['n_positions']:
        raise ValueError('头数必须整除隐藏维度，T 不能超过位置表容量。')
    if not 0 < t['lr_warmup_updates'] < t['planned_updates']:
        raise ValueError('学习率预热更新次数必须在训练预算内。')
    return spec


def describe(spec):
    """[C2] 不分配 124M 权重，先用结构公式计算参数量与 Token 预算。"""
    m, t = spec['model'], spec['training']
    d, layers = m['n_embd'], m['n_layer']
    # 一个 Block：注意力 4D²+4D，MLP 8D²+5D，两组 LN 共 4D。
    # wte/lm_head 共享，只计一次 V*D；位置表 P*D；最终 LN 为 2D。
    parameters = m['vocab_size']*d + m['n_positions']*d + layers*(12*d*d+13*d) + 2*d
    micro = t['batch_size']*t['seq_len']
    effective = micro*t['world_size']*t['accumulation_steps']
    return dict(parameters=parameters, head_dim=d//m['n_head'], mlp_width=4*d,
                tokens_per_microbatch=micro, tokens_per_update=effective,
                planned_training_tokens=effective*t['planned_updates'],
                validation_tokens=micro*t['world_size']*t['validation_batches'],
                initialization='random; GPT(..., tie_embeddings=True, gpt2_init=True)',
                pretrained_weights_loaded=False, weight_tying=True,
                attention='explicit QK-softmax-V; causal MHA',
                position='learned absolute table', normalization='LayerNorm',
                activation='GELU tanh', qk_norm=False)


def split_tokens(ids):
    """[D1] 先按原 Token 位置分开，再分别错位，训练标签不会跨入验证区。"""
    cut = len(ids)*9//10
    return ids[:cut], ids[cut:], cut


def first_batch(ids, batch_size, seq_len, start=0):
    """[D2] N=B*T 个预测位置需要 N+1 个 ID；保留整数，模型随后查表。"""
    n = batch_size*seq_len
    if start < 0 or start+n+1 > len(ids):
        raise ValueError('当前 split 不足一批输入及最后一个标签；不会从另一 split 补齐。')
    import torch
    buf = torch.tensor(ids[start:start+n+1], dtype=torch.long)
    return buf[:-1].view(batch_size, seq_len), buf[1:].view(batch_size, seq_len)


def prepare_local_data(spec):
    """[D3] 校验真实莎剧原文及 GPT-2 缓存；输出两段独立的 ID 列表与数据指纹。"""
    import tiktoken
    data = spec['data']
    path = ROOT / data['path']
    if sha(path) != data['sha256']:
        raise ValueError('莎剧原文校验失败；不能静默换语料后仍称同一基线。')
    cache = ROOT / '.cache/tiktoken'
    for filename, expected in TOKENIZER_ASSETS.items():
        p = cache / filename
        if not p.is_file() or sha(p) != expected:
            raise ValueError(f'缺少或不匹配的 GPT-2 分词缓存：{p}；本入口不会下载。')
    # 即使用户环境有其他缓存配置，本实验也固定用项目内已核对的两份资产。
    os.environ['TIKTOKEN_CACHE_DIR'] = str(cache)
    tokenizer = tiktoken.get_encoding('gpt2')
    if tokenizer.n_vocab != spec['model']['vocab_size']:
        raise ValueError('词表与模型不一致。')
    # 与前课保持一致：连续莎剧，不额外插 EOT；FineWeb 的文档前缀规则另列。
    ids = tokenizer.encode_ordinary(path.read_text(encoding='utf-8'))
    train, val, cut = split_tokens(ids)
    t = spec['training']; n=t['batch_size']*t['seq_len']
    if len(train) < n*t['accumulation_steps']+1:
        raise ValueError('训练区不足一次完整梯度累积。')
    if len(val) < n*t['validation_batches']+1:
        raise ValueError('固定验证窗口不足，不能循环重复凑验证量。')
    # 指纹固定 ID 顺序，采用可跨平台比较的十进制序列，不依赖本机整数端序。
    digest = lambda values: hashlib.sha256(','.join(map(str, values)).encode()).hexdigest()
    report = dict(status='ready_for_local_flow_check',kind=data['kind'],
        raw_sha256=sha(path), tokenizer=tiktoken.__version__, tokenizer_assets=TOKENIZER_ASSETS,
        total_tokens=len(ids), train_range=[0,cut], val_range=[cut,len(ids)],
        train_token_sha256=digest(train), val_token_sha256=digest(val),
        train_full_batches=(len(train)-1)//n, val_full_batches=(len(val)-1)//n,
        fixed_validation_range=[cut,cut+n*t['validation_batches']+1],
        split_boundary='position-isolated; not document-level deduplication',
        wrap_policy='future training: skip insufficient tail; restart same split',
        corpus_pass_fraction=describe(spec)['planned_training_tokens']/len(train))
    return train, val, report


def smoke_step(spec, train, val):
    """[S1—S4] 仅验证一次完整参数更新；不把它纳入之后的训练或测速记录。"""
    import torch
    from torch.nn import functional as F
    from gpt2_model import GPT, GPTConfig
    t = spec['training']
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(spec['seed'])
    # [S1] 本项目 GPT 组织结构；PyTorch 分配 FP32 参数。没有 from_pretrained。
    model = GPT(GPTConfig(**spec['model']), tie_embeddings=True, gpt2_init=True).cpu().float()
    assert sum(p.numel() for p in model.parameters()) == describe(spec)['parameters']
    assert model.lm_head.weight is model.transformer.wte.weight
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW([
        {'params':[p for p in parameters if p.ndim>=2], 'weight_decay':t['weight_decay']},
        {'params':[p for p in parameters if p.ndim<2], 'weight_decay':0.0},
    ], lr=t['peak_lr']/t['lr_warmup_updates'], betas=tuple(t['betas']), eps=t['adam_eps'])
    vx,vy = first_batch(val,t['batch_size'],t['seq_len'])
    def validation_probe():
        model.eval()
        with torch.no_grad():
            return F.cross_entropy(model(vx).flatten(0,1),vy.flatten()).item()
    before = validation_probe()
    model.train(); optimizer.zero_grad(set_to_none=True)
    # [S2] x/y 换成下一批。loss/K 后 backward 累加到同一套 .grad；此时不更新参数。
    losses=[]; n=t['batch_size']*t['seq_len']; k=t['accumulation_steps']
    for micro in range(k):
        x,y = first_batch(train,t['batch_size'],t['seq_len'],start=micro*n)
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0,1),y.flatten())
        if not torch.isfinite(loss): raise ValueError('非有限 loss，停止流程检查。')
        (loss/k).backward()
        losses.append(loss.item())
    # [S3] 所有微批次反传完才裁剪、更新。error_if_nonfinite 防止写入无效梯度。
    norm=torch.nn.utils.clip_grad_norm_(parameters,t['clip_norm'],error_if_nonfinite=True)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    # [S4] 在同一验证首批观察更新后 loss；只是单批探针，不是完整验证评测。
    return dict(optimizer_updates=1,microbatches=k,training_tokens=n*k,
        train_loss=sum(losses)/k,grad_norm_before_clip=float(norm),
        validation_probe_before=before,validation_probe_after=validation_probe(),
        validation_probe_tokens=n,device='cpu',precision='fp32',torch_version=torch.__version__,
        checkpoint_saved=False,benchmark=False,scope='one-update pipeline check only')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=CONFIGS,default='local')
    parser.add_argument('--smoke-step',action='store_true',help='仅本机小模型，运行一次完整更新')
    parser.add_argument('--report',type=Path,help='默认在项目 .cache/runtime-check 下输出配置核对报告')
    args=parser.parse_args()
    if args.smoke_step and args.profile!='local':
        parser.error('完整 124M/FineWeb 尚待准备和测速；流程检查使用 --profile local。')
    try:
        spec=load_spec(args.profile); summary=describe(spec)
        report=dict(profile=args.profile,config=spec,config_sha256=sha(CONFIGS[args.profile]),
            summary=summary,model_source_sha256=sha(ROOT/'gpt2_model.py'),
            setup_source_sha256=sha(Path(__file__)),
            training_run=False,baseline_completed=False)
        if args.profile=='local':
            train,val,report['data']=prepare_local_data(spec)
            x,y=first_batch(train,spec['training']['batch_size'],spec['training']['seq_len'])
            report['first_batch']=dict(x_shape=list(x.shape),y_shape=list(y.shape),
                dtype=str(x.dtype),device=str(x.device),x_first_8=x[0,:8].tolist(),
                y_first_8=y[0,:8].tolist())
            if args.smoke_step:
                report['smoke']=smoke_step(spec,train,val);report['training_run']=True
        else:
            report['data']={'status':'pending', 'blockers':[
                '固定 FineWeb-Edu 数据版本与文档清单；按文档隔离 train/val 后再分片。',
                '记录每片 SHA-256、文档 ID 与 Token 数；确认实际可用批数覆盖预算。',
                '教学 demo 分片不能代替正式数据；不会自动回退到莎剧。']}
        print('单元 3 / 短训练基线：'+spec['purpose'])
        print(f"模型：{spec['model']['n_layer']} 层 / {spec['model']['n_embd']} 维 / "
              f"{spec['model']['n_head']} 头；{summary['parameters']:,} 参数；随机初始化、共享词表权重。")
        print(f"每微批次 {summary['tokens_per_microbatch']} 个预测位置；每更新 "
              f"{summary['tokens_per_update']:,}；预算 {summary['planned_training_tokens']:,}（测速后锁定）。")
        print('数据状态：'+report['data']['status'])
        if args.profile=='local':
            print(f"Token 区间 train=[0,{report['data']['train_range'][1]})，"
                  f"val=[{report['data']['val_range'][0]},{report['data']['val_range'][1]})；先切分再构造 x/y。")
            print('首批：'+json.dumps(report['first_batch'],ensure_ascii=False))
        if args.profile=='formal':
            for issue in report['data']['blockers']: print('待准备：'+issue)
        if 'smoke' in report: print('一次更新检查：'+json.dumps(report['smoke'],ensure_ascii=False))
        else: print('本次没有创建模型、反向传播或更新参数。')
        destination=args.report or ROOT/'.cache/runtime-check'/f'architecture-baseline-{args.profile}.json'
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print('报告：'+str(destination.resolve()))
    except (ImportError,OSError,ValueError) as error:
        parser.exit(1,f'检查失败：{error}\n课程解释器：/Users/guqun/micrograd/venv/bin/python\n')


if __name__=='__main__':
    main()
