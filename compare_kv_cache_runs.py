"""比较同一权重和输入的缓存/无缓存推理；先验 logits，再看计时。"""
import argparse
import json
from pathlib import Path
import torch
from prepare_architecture_baseline import ROOT,sha


def compare(reference,cached):
    a=json.loads(reference.read_text());b=json.loads(cached.read_text())
    if not a['completed'] or not b['completed'] or (a['mode'],b['mode'])!=('reference','cached'):
        raise ValueError('需要已完成的 reference 与 cached 报告。')
    if a['protocol']!=b['protocol']:raise ValueError('模型权重、配置、输入、设备或代码协议不一致。')
    tensors=[]
    for path,r in ((reference,a),(cached,b)):
        f=path.parent/'logits.pt'
        if sha(f)!=r['logits_sha256']:raise ValueError('保存的 logits 校验失败。')
        tensors.append(torch.load(f,map_location='cpu',weights_only=True))
    if a['generated_ids']!=b['generated_ids']:
        raise ValueError('两路贪心 Token 不同，后续前缀已分岔，不能直接比较后续分数；需检查是否近似并列。')
    torch.testing.assert_close(*tensors,rtol=1e-4,atol=1e-5)
    return {'max_abs_logit_difference':float((tensors[0]-tensors[1]).abs().max()),
            'same_greedy_tokens':True,'processed_positions':[a['processed_positions_per_layer'],b['processed_positions_per_layer']],
            'kv_tensor_bytes':b['kv_tensor_bytes'],
            'median_seconds':[a['median_seconds'],b['median_seconds']],
            'speedup_reference_over_cached':a['median_seconds']/b['median_seconds']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    base=ROOT/'.cache/kv-cache-comparison'
    parser.add_argument('--reference',type=Path,default=base/'reference/report.json')
    parser.add_argument('--cached',type=Path,default=base/'cached/report.json')
    args=parser.parse_args()
    try:print(json.dumps(compare(args.reference,args.cached),ensure_ascii=False,indent=2))
    except (ValueError,OSError,KeyError,AssertionError) as e:parser.exit(1,str(e)+'\n')
