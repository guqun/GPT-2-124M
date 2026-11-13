"""读取两次独立运行的报告；只在数据/初值/训练/设备协议相同时展示对照。"""
import argparse
import json
from pathlib import Path
from prepare_architecture_baseline import ROOT


def compare(a, b):
    if not a['completed'] or not b['completed']:
        raise ValueError('有实验未完成。')
    if (a['variant'], b['variant']) != ('gpt2', 'rmsnorm_no_bias'):
        raise ValueError('需要依次传入 GPT-2 和 RMSNorm/no-bias 报告。')
    if a['protocol'] != b['protocol']:
        keys = sorted(k for k in set(a['protocol']) | set(b['protocol'])
                      if a['protocol'].get(k) != b['protocol'].get(k))
        raise ValueError('比较条件不一致：'+', '.join(keys)+'。请使用相同设置分别重跑。')
    # 数据顺序也核对实际读取起点，不能只凭配置相同就断言运行相同。
    if [r['microbatch_starts'] for r in a['training']] != [r['microbatch_starts'] for r in b['training']]:
        raise ValueError('实际微批次顺序不一致。')
    sa, sb = a['summary'], b['summary']
    return {'gpt2': sa, 'rmsnorm_no_bias': sb,
            'val_loss_delta_b_minus_a': sb['final_val_loss']-sa['final_val_loss'],
            'throughput_ratio_b_over_a': sb['measured_tokens_per_second']/sa['measured_tokens_per_second']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT/'.cache/architecture-comparison'
    parser.add_argument('--gpt2', type=Path, default=base/'gpt2/report.json')
    parser.add_argument('--modified', type=Path, default=base/'rmsnorm_no_bias/report.json')
    args = parser.parse_args()
    try:
        a = json.loads(args.gpt2.read_text()); b = json.loads(args.modified.read_text())
        result = compare(a, b)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, str(error)+'\n')
    print('相同协议、初始共同权重和取批顺序已核对。')
    print('指标                         GPT-2           RMSNorm/no-bias')
    for label, av, bv in (
        ('参数量', a['parameters'], b['parameters']),
        ('训练预测位置', a['summary']['training_tokens'], b['summary']['training_tokens']),
        ('初始验证 loss', a['summary']['initial_val_loss'], b['summary']['initial_val_loss']),
        ('最终验证 loss', a['summary']['final_val_loss'], b['summary']['final_val_loss']),
        ('测量吞吐 Token/s', a['summary']['measured_tokens_per_second'], b['summary']['measured_tokens_per_second'])):
        print(f'{label:20s} {av:14.6f} {bv:18.6f}')
    print(f'验证 loss 差（B−A）：{result["val_loss_delta_b_minus_a"]:+.6f}；负值表示 B 更低。')
    print(f'吞吐比（B/A）：{result["throughput_ratio_b_over_a"]:.3f}；大于 1 表示本次 B 更快。')
    print('组合改动、单种子短跑；不可单独归因于 RMSNorm，也不保证普遍更快/更好。')


if __name__ == '__main__':
    main()
