"""重验各模型缓存等价性，然后比较 MHA/GQA 的缓存；不要求跨模型输出相同。"""
import argparse
import json
from pathlib import Path
import torch
from prepare_architecture_baseline import ROOT, sha


def compare(reference, gqa):
    a, b = [json.loads(p.read_text()) for p in (reference,gqa)]
    if not all(r['completed'] for r in (a,b)) or (a['variant'],b['variant']) != ('mha','gqa'):
        raise ValueError('需要完整的 MHA 与 GQA 报告。')
    if a['protocol'] != b['protocol']:
        raise ValueError('原始模型、共同参数、输入、设备或代码协议不一致。')
    for path, report in ((reference,a),(gqa,b)):
        logits_file = path.parent/'logits.pt'
        if sha(logits_file) != report['logits_sha256']:
            raise ValueError('logits 文件校验失败。')
        saved = torch.load(logits_file,map_location='cpu',weights_only=True)
        torch.testing.assert_close(saved['full'],saved['cached'],rtol=1e-4,atol=1e-5)
        if not torch.equal(saved['full'].argmax(-1),saved['cached'].argmax(-1)):
            raise ValueError('同模型贪心 ID 不一致。')
        if saved['cached'].argmax(-1).tolist() != report['generated_ids']:
            raise ValueError('报告 ID 与保存分数不匹配。')
        dims = report['cache_shape_per_tensor']
        expected = 2*report['protocol']['spec']['model']['n_layer']*4
        for dimension in dims:
            expected *= dimension
        if expected != report['kv_tensor_bytes']:
            raise ValueError('缓存形状与字节数不匹配。')
    if a['q_heads'] != b['q_heads'] or a['kv_tensor_bytes']*b['kv_heads'] != b['kv_tensor_bytes']*a['kv_heads']:
        raise ValueError('缓存缩减比例与头数不符。')
    return {'each_model_cache_equivalent':True,
            'mha_gqa_logits_required_equal':False,
            'kv_bytes_mha_gqa':[a['kv_tensor_bytes'],b['kv_tensor_bytes']],
            'cache_reduction_fraction':1-b['kv_tensor_bytes']/a['kv_tensor_bytes'],
            'parameters_mha_gqa':[a['parameter_count'],b['parameter_count']],
            'cached_median_seconds_mha_gqa':[a['timings']['cached']['total_seconds'],b['timings']['cached']['total_seconds']],
            'local_cached_time_ratio_mha_over_gqa':a['timings']['cached']['total_seconds']/b['timings']['cached']['total_seconds'],
            'note':'只比较本机执行与存活 KV 张量；不代表峰值显存或语言质量。'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    root = ROOT/'.cache/gqa-comparison'
    parser.add_argument('--reference',type=Path,default=root/'mha/report.json')
    parser.add_argument('--gqa',type=Path,default=root/'gqa/report.json')
    args = parser.parse_args()
    try:
        print(json.dumps(compare(args.reference,args.gqa),ensure_ascii=False,indent=2))
    except (OSError,ValueError,KeyError,AssertionError,RuntimeError) as error:
        parser.exit(1,str(error)+'\n请先运行 generate_gqa_reference.py 与 generate_gqa.py。\n')
