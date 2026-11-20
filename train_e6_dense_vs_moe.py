"""E6：同一批 Token、相同的前向线性投影 MAC 预算，比较稠密 MLP 与 8 选 2 + 共享专家。
默认 CPU/FP32、100 步。本文件独立运行，旧训练入口与检查点均保留。
"""
import argparse
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile
import time

import torch
from torch.nn import functional as F

from architecture_training import SequentialBatches, evaluate, learning_rate
from modern_model import ModernLM, ModernMLP
from prepare_architecture_baseline import ROOT, prepare_local_data
from run_small_moe import SmallMoE
from train_modern import load_config, make_optimizer

DENSE_WIDTHS = (341, 342, 342)
EXPERT_WIDTH = 113
AUX_COEF = 0.01


class E6LM(ModernLM):
    """复用原现代语言模型的参数和模块。E6 只训练/验证完整序列。"""

    def forward(self, ids, *, with_aux=False):
        x = self.transformer.wte(ids)
        balances, counts = [], []
        for block in self.transformer.h:
            if isinstance(block.mlp, SmallMoE):
                # [E3] 注意力、两次归一化和残差沿用 ModernBlock 的顺序。
                attn, _ = block.attn(block.ln_1(x))
                residual = x + attn
                mix = block.mlp(block.ln_2(residual))
                x = residual + mix.output
                balances.append(mix.balance_loss)
                counts.append(mix.counts)
            else:
                x, _ = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        if with_aux:
            # 三层各算一次负载损失，再取平均，系数不会随层数放大。
            aux = torch.stack(balances).mean() if balances else logits.new_zeros(())
            return logits, aux, counts
        return logits


def build_pair(config, seed):
    # [E1] 每 Token、每层：SwiGLU 的三张矩阵各做 D×F 次乘加（MAC）。
    # 路由器对所有 8 位打分。每行只执行 2 位路由专家与 1 位共享专家。
    d = config.width
    dense_costs = [3*d*f for f in DENSE_WIDTHS]
    route_cost = d * SmallMoE.num_experts
    expert_cost = 3*d*EXPERT_WIDTH*(SmallMoE.top_k+1)
    moe_costs = [expert_cost + route_cost] * config.layers
    if len(dense_costs) != config.layers or sum(dense_costs) != sum(moe_costs):
        raise ValueError('两边的逐 Token 投影预算不匹配')
    torch.manual_seed(seed)
    dense = E6LM(config)
    moe = copy.deepcopy(dense)  # 公共主干初值逐元素相同，两套模型各自训练。
    for i, f in enumerate(DENSE_WIDTHS):
        for model, module in (
            (dense, ModernMLP(replace(config, intermediate=f))),
            (moe, SmallMoE(replace(config, intermediate=EXPERT_WIDTH))),
        ):
            # 两侧所有前馈投影沿用 ModernLM 的初始化规则。Wr 随机初始化并学习。
            for name, weight in module.named_parameters():
                std = 0.02 / math.sqrt(2*config.layers) if name.endswith('down_proj.weight') else 0.02
                torch.nn.init.normal_(weight, std=std)
            model.transformer.h[i].mlp = module
    costs = dict(dense_by_layer=dense_costs, moe_by_layer=moe_costs,
                 router_per_layer=route_cost, active_experts_per_layer=expert_cost,
                 matched_projection_mac_per_token=sum(dense_costs))
    return {'dense': dense, 'moe': moe}, costs


def update(model, optimizer, batches, lr, clip):
    # [E4] 同一组 K 个微批次与学习率。CE 单独统计，便于直接比较两边。
    for group in optimizer.param_groups:
        group['lr'] = lr
    model.train()
    optimizer.zero_grad(set_to_none=True)
    ce_sum = aux_sum = 0.
    count_sum = None
    for x, y in batches:
        logits, aux, counts = model(x, with_aux=True)
        ce = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        ((ce + AUX_COEF*aux) / len(batches)).backward()
        ce_sum += ce.detach().item()
        aux_sum += aux.detach().item()
        if counts:
            per_layer = torch.stack(counts)
            count_sum = per_layer if count_sum is None else count_sum + per_layer
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    return dict(train_ce=ce_sum/len(batches), balance=aux_sum/len(batches),
                grad_norm=float(norm), routed_counts=count_sum.tolist() if count_sum is not None else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100, help='每组的更新次数，至少 2')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.steps < 2 or args.threads < 1:
        parser.error('steps 至少 2，threads 至少 1')
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest')
    spec, base = load_config(ROOT/'configs/modern-integrated-local.json')
    config = replace(base, layers=len(DENSE_WIDTHS))
    t = spec['training']
    t.update(planned_updates=args.steps, lr_warmup_updates=min(t['lr_warmup_updates'], max(1, args.steps//10)))
    train, val, data = prepare_local_data(spec)
    loader = SequentialBatches(train, t['batch_size'], t['seq_len'])
    models, costs = build_pair(config, args.seed)
    opts = {name: make_optimizer(m, t) for name, m in models.items()}
    tokens = t['batch_size'] * t['seq_len'] * t['accumulation_steps']
    parent = ROOT/'.cache/e6'
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='run-', dir=parent))/'report.json'
    report = dict(config=asdict(config), dense_widths=DENSE_WIDTHS, expert_width=EXPERT_WIDTH,
                  routed_experts=SmallMoE.num_experts, top_k=SmallMoE.top_k, shared_experts=1,
                  costs=costs, alpha=AUX_COEF, data=data, training=t, seed=args.seed, device='cpu',
                  torch_version=str(torch.__version__), threads=args.threads, tokens_per_update=tokens,
                  mac_scope='forward FFN and router linear projections, all blocks, per token',
                  timing_scope='CPU training updates excluding update 1 and validation; shared batch assembly excluded',
                  parameters={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()},
                  history=[], validation=[], summary={})
    print(f"E6 CPU/FP32：前馈与路由投影 {sum(costs['dense_by_layer']):,} MAC/Token；每组 {args.steps} 次更新。", flush=True)

    def validate(step):
        # [E5] 两侧读相同验证窗口，只用词表交叉熵比较语言建模质量。
        result = {name:evaluate(m, val, t, torch.device('cpu'))['loss'] for name,m in models.items()}
        report['validation'].append(dict(step=step, **result))
        print(f"验证 {step:03d}：dense CE={result['dense']:.6f}，MoE CE={result['moe']:.6f}", flush=True)

    validate(0)
    for step in range(1, args.steps+1):
        # [E2] 只取一遍数据，两个模型接收完全相同的 x、y；计时顺序逐步轮换。
        triplets = [loader.next_batch() for _ in range(t['accumulation_steps'])]
        batches = [(x,y) for x,y,_ in triplets]
        lr = learning_rate(step, args.steps, t['lr_warmup_updates'], t['peak_lr'], t['min_lr'])
        record = dict(step=step, lr=lr, starts=[start for _,_,start in triplets])
        for name in (('dense','moe') if step%2 else ('moe','dense')):
            start = time.perf_counter()
            values = update(models[name], opts[name], batches, lr, t['clip_norm'])
            values['seconds'] = time.perf_counter()-start
            record[name] = values
        report['history'].append(record)
        if step == 1 or step%10 == 0 or step == args.steps:
            print(f"训练 {step:03d}：dense CE={record['dense']['train_ce']:.4f}，"
                  f"MoE CE={record['moe']['train_ce']:.4f}，Lbal={record['moe']['balance']:.4f}", flush=True)
        if step % t['validation_every'] == 0 or step == args.steps:
            validate(step)
    for name in models:
        seconds = sum(r[name]['seconds'] for r in report['history'][1:])
        speed = (args.steps-1)*tokens/seconds
        report['summary'][name] = dict(final_val_ce=report['validation'][-1][name],
                                       training_tokens=args.steps*tokens, tokens_per_second=speed)
        print(f"{name}：{report['parameters'][name]:,} 参数，训练吞吐 {speed:.1f} Token/s")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('报告：', output)


if __name__ == '__main__':
    main()
