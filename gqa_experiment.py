"""MHA/GQA 缓存实践：两份入口，共同协议；各自先验证缓存等价，再比较缓存字节。

默认随机权重、合成 ID、CPU/FP32，无下载、无训练，不评价语言质量。
K/V 由相同随机 MHA 起点按连续头分组取均值；不声称是已适配的预训练 GQA。
"""
import argparse
import hashlib
import json
import platform
import statistics
from pathlib import Path
import torch
from prepare_architecture_baseline import ROOT, load_spec, sha
from swiglu_model import build_model
from swiglu_training import resolve_device
from kv_cache_experiment import generate, parameter_digest
from gqa_model import GroupedDecoder, convert_attention

SOURCES = ('gqa_model.py', 'gqa_experiment.py', 'kv_cache_experiment.py', 'kv_cache_model.py',
           'swiglu_model.py', 'rope_model.py', 'architecture_variants.py', 'gpt2_model.py',
           'prepare_architecture_baseline.py', 'swiglu_training.py')


def shared_digest(model):
    """比较除 K/V 投影外共同参数；不将不同形状的 K/V 伪称为相同权重。"""
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if '.k_proj.' in name or '.v_proj.' in name:
            continue
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main(n_kv_heads):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu','cuda','mps','auto'], default='cpu')
    parser.add_argument('--prompt-length', type=int, default=32)
    parser.add_argument('--new-tokens', type=int, default=16)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    name = 'mha' if n_kv_heads == 4 else 'gqa'
    parser.add_argument('--output', type=Path, default=ROOT/'.cache/gqa-comparison'/name)
    args = parser.parse_args()
    try:
        if min(args.prompt_length,args.new_tokens,args.threads,args.repeats) < 1:
            raise ValueError('长度、次数和线程数必须为正。')
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError('输出目录非空，请用 --output 指定新目录。')
        torch.set_num_threads(args.threads)
        torch.set_float32_matmul_precision('highest')
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
        device = resolve_device(args.device)
        # [E1] 两入口从同一个 seed/spec 构建原 MHA；不加载语言模型权重。
        spec = load_spec('local')
        spec['position_experiment'] = {'theta':10000., 'pairing':'adjacent',
                                      'max_length':spec['model']['n_positions']}
        base = build_model(spec, 'swiglu')
        if base.config.n_head != 4:
            raise ValueError('本对照固定四个 Q 头，请勿单独修改配置头数。')
        if args.prompt_length + args.new_tokens - 1 > base.config.n_positions:
            raise ValueError('要处理的前缀超出固定上下文容量。')
        initial_hash = parameter_digest(base)
        base = convert_attention(base, n_kv_heads)
        common_hash = shared_digest(base)
        before = parameter_digest(base)
        decoder = GroupedDecoder(base).to(device).eval()
        prompt = (torch.arange(args.prompt_length, device=device) % (base.config.vocab_size-1)+1)[None,:]
        print(f'{name.upper()}：Q=4，K/V各={n_kv_heads}；{device}/FP32；随机起点，无训练。')
        # [E2] 比较的是同一模型的两条路径，不是 MHA 与 GQA 的输出等价性。
        full_ids, full_logits, full_rows = generate(decoder, prompt, args.new_tokens, False)
        cache_ids, cache_logits, cache_rows = generate(decoder, prompt, args.new_tokens, True)
        if not torch.equal(full_ids, cache_ids):
            raise ValueError('缓存/重算贪心路径分岔，先检查误差及近似并列，不继续跨前缀比较。')
        torch.testing.assert_close(full_logits, cache_logits, rtol=1e-4, atol=1e-5)
        error = float((full_logits-cache_logits).abs().max())
        # [E3] 上面的各一次完整运行用于预热；计时重复单独进行。
        timings = {}
        for use_cache, key in ((False,'full'), (True,'cached')):
            runs = []
            for _ in range(args.repeats):
                _, _, rows = generate(decoder, prompt, args.new_tokens, use_cache)
                runs.append({'total_seconds':sum(r['seconds'] for r in rows),
                             'prefill_seconds':rows[0]['seconds'],
                             'decode_seconds':sum(r['seconds'] for r in rows[1:])})
            timings[key] = {'runs':runs, **{k:statistics.median(r[k] for r in runs)
                                        for k in runs[0]}}
        assert parameter_digest(base) == before and all(p.grad is None for p in base.parameters())
        dim = base.config.n_embd // base.config.n_head
        end_length = args.prompt_length + args.new_tokens - 1
        expected_bytes = 2*base.config.n_layer*prompt.shape[0]*n_kv_heads*end_length*dim*4
        assert cache_rows[-1]['cache_bytes'] == expected_bytes
        protocol = {'spec':spec, 'source_mha_sha256':initial_hash, 'shared_parameters_sha256':common_hash,
                    'prompt_ids':prompt.cpu().tolist(), 'new_tokens':args.new_tokens,
                    'device':str(device), 'platform':platform.platform(), 'torch':str(torch.__version__),
                    'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
                    'precision':'fp32','tf32':False, 'threads':args.threads, 'repeats':args.repeats,
                    'sources':{n:sha(ROOT/n) for n in SOURCES}}
        torch.save({'full':full_logits.cpu(), 'cached':cache_logits.cpu()}, output/'logits.pt')
        report = {'completed':True, 'variant':name, 'q_heads':4, 'kv_heads':n_kv_heads,
                  'protocol':protocol, 'parameters_sha256':before,
                  'parameter_count':sum(p.numel() for p in base.parameters()),
                  'conversion':'contiguous_KV_weight_mean_pool_once; Q and remaining weights copied',
                  'cache_check':{'same_greedy_ids':True, 'max_abs_logit_difference':error},
                  'generated_ids':cache_ids[:,args.prompt_length:].cpu().tolist(),
                  'cache_shape_per_tensor':[1,n_kv_heads,end_length,dim],
                  'kv_tensor_bytes':expected_bytes, 'cache_steps':cache_rows,
                  'full_processed_positions':sum(r['input_positions'] for r in full_rows),
                  'cache_processed_positions':sum(r['input_positions'] for r in cache_rows),
                  'timings':timings, 'parameters_unchanged':True, 'logits_sha256':sha(output/'logits.pt'),
                  'timing_scope':'forward + greedy + append; synchronized; no CPU copy/logging/save; one warmup per path',
                  'limitations':'random weights; dynamic cat; explicit grouped matmul; no GPU validation implied; not quality benchmark'}
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(f'同模型缓存/重算核验通过：最大 logits 差 {error:.3g}；贪心 ID 一致。')
        print(f'每层 K、V 各 {report["cache_shape_per_tensor"]}；全部层有效缓存 {expected_bytes:,} 字节。')
        print(f'参数 {report["parameter_count"]:,}；缓存路径中位耗时 {timings["cached"]["total_seconds"]*1000:.2f} ms。')
        print(f'报告：{output/"report.json"}')
    except (ValueError,OSError,KeyError,RuntimeError,AssertionError) as error:
        parser.exit(1,str(error)+'\n')
