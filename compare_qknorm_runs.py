"""重验保存的缓存/重算 logits；比较新增参数、相同缓存容量和 gamma 更新。"""
import argparse
import json
from pathlib import Path
import torch
from prepare_architecture_baseline import ROOT,sha


def compare(reference,qknorm):
    a,b=[json.loads(p.read_text()) for p in (reference,qknorm)]
    if not all(r['completed'] for r in (a,b)) or (a['variant'],b['variant'])!=('reference','qknorm'):
        raise ValueError('需要 reference 与 qknorm 的完整报告。')
    if a['protocol']!=b['protocol']: raise ValueError('共同初值、配置、输入、设备或代码不一致。')
    for path,r in ((reference,a),(qknorm,b)):
        f=path.parent/'logits.pt'
        if sha(f)!=r['logits_sha256']: raise ValueError('logits 文件校验失败。')
        tensors=torch.load(f,map_location='cpu',weights_only=True)
        torch.testing.assert_close(tensors['full'],tensors['cached'],rtol=1e-4,atol=1e-5)
        if tensors['full'].argmax(-1).tolist()!=r['generated_ids'] or tensors['cached'].argmax(-1).tolist()!=r['generated_ids']:
            raise ValueError('同模型生成 ID 与分数不一致。')
    cfg=b['protocol']['spec']['model'];d=cfg['n_embd']//cfg['n_head']
    added=2*d*cfg['n_layer']
    if b['parameters']-a['parameters']!=added or b['gamma_parameters']!=added or a['gamma_parameters']!=0:
        raise ValueError('新增参数量不匹配。')
    shape=[1,2,len(b['protocol']['prompt_ids'][0])+b['protocol']['new_tokens']-1,d]
    expected=2*cfg['n_layer']*4
    for dim in shape: expected*=dim
    if any(r['cache_shape']!=shape or r['kv_tensor_bytes']!=expected for r in (a,b)):
        raise ValueError('QK-Norm 不应改变 KV 头数或缓存字节。')
    gammas=b['gradient_probe']['gamma']
    if len(gammas)!=2*cfg['n_layer'] or not all(v['gradient_l2']>0 and v['max_abs_update']>0 for v in gammas.values()):
        raise ValueError('gamma 梯度或更新未通过。')
    return {'same_existing_parameter_initialization':True,'each_model_cache_equivalent':True,
            'cross_model_logits_required_equal':False,'parameters_reference_qknorm':[a['parameters'],b['parameters']],
            'gamma_parameters_added':added,'kv_tensor_bytes_both':expected,
            'cache_logit_max_differences':[a['max_abs_cache_logit_difference'],b['max_abs_cache_logit_difference']],
            'gamma_groups_updated_on_copy':len(gammas),'scope':'CPU/local check if run on CPU; no model quality claim'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    root=ROOT/'.cache/qknorm-comparison'
    parser.add_argument('--reference',type=Path,default=root/'reference/report.json')
    parser.add_argument('--qknorm',type=Path,default=root/'qknorm/report.json')
    args=parser.parse_args()
    try: print(json.dumps(compare(args.reference,args.qknorm),ensure_ascii=False,indent=2))
    except (OSError,ValueError,KeyError,RuntimeError,AssertionError) as error:
        parser.exit(1,str(error)+'\n请先运行 run_qknorm_reference.py 和 run_qknorm.py。\n')
