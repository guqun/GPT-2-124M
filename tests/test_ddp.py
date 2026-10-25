"""验证真实 CPU/Gloo 多进程、累积同步和单进程全局 batch 的等价关系。"""

from contextlib import redirect_stdout
from datetime import timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt2_model import GPT, GPTConfig
from train_dataloader import DataLoaderLite
from train_ddp import (
    DistributedDataLoaderLite, Runtime, accumulation_steps, configure_optimizer,
    learning_rate_at, train_updates,
)
from train_initialization import initialize_like_course
from train_weight_tying import tie_weights_like_course


ROOT = Path(__file__).resolve().parents[1]


def make_model():
    torch.manual_seed(42)
    config = GPTConfig(vocab_size=31, n_positions=8, n_embd=16, n_layer=2,
                       n_head=2, embd_pdrop=0, attn_pdrop=0, resid_pdrop=0)
    model = GPT(config, tie_embeddings=False, gpt2_init=False).double()
    tie_weights_like_course(model)
    initialize_like_course(model)
    return model


def distributed_worker(rank, rendezvous, output_dir):
    """每个 worker 都独立计算同一全局 batch 的参考梯度，避免只比较 loss。"""
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}',
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        raw_model, reference = make_model(), make_model()
        model = DDP(raw_model)
        optimizer = configure_optimizer(raw_model, torch.device('cpu'), 6e-4)
        reference_optimizer = configure_optimizer(reference, torch.device('cpu'), 6e-4)
        token_ids = [i % 31 for i in range(257)]
        loader = DistributedDataLoaderLite(token_ids, 2, 4, rank, 2)
        # R=2、K=2、B=2、T=4，总共 32 个位置；单进程一次喂 B=8、T=4。
        reference_loader = DataLoaderLite(token_ids, 8, 4)
        expected = []
        clip_original = torch.nn.utils.clip_grad_norm_
        for step in range(3):
            x, y = reference_loader.next_batch()
            reference_optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(reference(x).reshape(-1, 31), y.reshape(-1))
            loss.backward()
            gradients = [p.grad.clone() for p in reference.parameters()]
            norm = clip_original(reference.parameters(), 0.5, error_if_nonfinite=True)
            # 明确写出该用例的调度结果，不复用待测函数产生参考答案。
            lr = [6e-4, 6e-4, 3.3e-4][step]
            for group in reference_optimizer.param_groups:
                group['lr'] = lr
            reference_optimizer.step()
            expected.append({'loss': loss.item(), 'gradients': gradients,
                             'norm': norm.item(), 'lr': lr,
                             'parameters': [p.clone().detach() for p in reference.parameters()]})

        # 使用默认 DDP 归约运行完整训练，截取裁剪之前的梯度和 step 之后的参数。
        observed_gradients, observed_parameters = [], []

        def capture_then_clip(parameters, *args, **kwargs):
            parameters = list(parameters)
            observed_gradients.append([p.grad.detach().clone() for p in parameters])
            return clip_original(parameters, *args, **kwargs)

        step_original = optimizer.step

        def capture_after_step(*args, **kwargs):
            result = step_original(*args, **kwargs)
            observed_parameters.append([p.detach().clone() for p in raw_model.parameters()])
            return result

        runtime = Runtime(True, rank, rank, 2, torch.device('cpu'), 'gloo')
        with patch('train_ddp.torch.nn.utils.clip_grad_norm_', side_effect=capture_then_clip), \
                patch.object(optimizer, 'step', side_effect=capture_after_step), \
                redirect_stdout(io.StringIO()):
            records = train_updates(model, raw_model, optimizer, loader, runtime, steps=3,
                                    accum_steps=2, precision='fp32', max_lr=6e-4, min_lr=6e-5,
                                    warmup_steps=1, decay_steps=3, clip_norm=0.5)
        max_grad_error, max_parameter_error = 0.0, 0.0
        for step, (actual, target) in enumerate(zip(records, expected)):
            torch.testing.assert_close(actual['global_mean_loss'], target['loss'], rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(actual['gradient_norm_before_clip'], target['norm'],
                                       rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(actual['learning_rate'], target['lr'], rtol=1e-12, atol=1e-12)
            for actual_grad, target_grad in zip(observed_gradients[step], target['gradients']):
                torch.testing.assert_close(actual_grad, target_grad, rtol=1e-8, atol=1e-10)
                max_grad_error = max(max_grad_error, (actual_grad - target_grad).abs().max().item())
            for actual_param, target_param in zip(observed_parameters[step], target['parameters']):
                torch.testing.assert_close(actual_param, target_param, rtol=1e-8, atol=1e-10)
                max_parameter_error = max(max_parameter_error,
                                          (actual_param - target_param).abs().max().item())
            if actual['local_batch_starts'] != [step * 32 + rank * 8,
                                                step * 32 + rank * 8 + 16]:
                raise AssertionError('各进程取得的数据位置不正确。')
        for actual_p, reference_p in zip(raw_model.parameters(), reference.parameters()):
            for name in ('exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(optimizer.state[actual_p][name],
                                           reference_optimizer.state[reference_p][name],
                                           rtol=1e-8, atol=1e-10)

        # 第二轮单独观察通信时机。hook 完成正常的 SUM/world_size，并记录触发的前向号。
        sync_raw = make_model()
        sync_model = DDP(sync_raw)
        hook_state = {'forward': 0, 'sync_at_forward': []}

        def count_forward(module, inputs):
            hook_state['forward'] += 1

        def average_hook(state, bucket):
            state['sync_at_forward'].append(state['forward'])
            future = dist.all_reduce(bucket.buffer(), async_op=True).get_future()
            return future.then(lambda done: done.value()[0].div_(2))

        sync_model.register_forward_pre_hook(count_forward)
        sync_model.register_comm_hook(hook_state, average_hook)
        with redirect_stdout(io.StringIO()):
            train_updates(sync_model, sync_raw,
                          configure_optimizer(sync_raw, torch.device('cpu'), 6e-4),
                          DistributedDataLoaderLite(token_ids, 2, 4, rank, 2), runtime,
                          steps=2, accum_steps=2, precision='fp32', max_lr=6e-4, min_lr=6e-5,
                          warmup_steps=1, decay_steps=3, clip_norm=0.5)
        if set(hook_state['sync_at_forward']) != {2, 4}:
            raise AssertionError(f'梯度同步出现在错误的微批次：{hook_state}')
        result = {'rank': rank, 'steps': 3, 'world_size': 2, 'micro_batches_per_update': 2,
                  'tokens_per_update': 32, 'dtype': 'float64', 'backend': 'gloo',
                  'max_gradient_error_before_clip': max_grad_error,
                  'max_parameter_error_after_step': max_parameter_error,
                  'default_ddp_used_for_equivalence': True, 'optimizer_moments_match': True,
                  'separate_hook_probe_sync_at_forward': hook_state['sync_at_forward'],
                  'records': records}
        (Path(output_dir) / f'rank-{rank}.json').write_text(json.dumps(result, indent=2) + '\n')
    finally:
        dist.destroy_process_group()


class DistributedLoaderTests(unittest.TestCase):
    def test_ranks_read_disjoint_positions_and_wrap_together(self):
        loaders = [DistributedDataLoaderLite(list(range(25)), 2, 3, r, 2) for r in range(2)]
        actual_starts = []
        for _ in range(5):
            starts = [loader.current_position for loader in loaders]
            actual_starts.append(starts)
            outputs = [loader.next_batch() for loader in loaders]
            for start, (x, y) in zip(starts, outputs):
                self.assertEqual(x.flatten().tolist(), list(range(start, start + 6)))
                self.assertEqual(y.flatten().tolist(), list(range(start + 1, start + 7)))
            self.assertFalse(set(outputs[0][0].flatten().tolist()) & set(outputs[1][0].flatten().tolist()))
        self.assertEqual(actual_starts, [[0, 6], [12, 18], [0, 6], [12, 18], [0, 6]])

    def test_exact_global_fit_and_incomplete_tail(self):
        for size in (13, 24, 25, 26):
            loaders = [DistributedDataLoaderLite(list(range(size)), 2, 3, rank, 2) for rank in range(2)]
            rounds = (size - 1) // 12
            for step in range(rounds * 2 + 1):
                self.assertEqual([loader.current_position for loader in loaders],
                                 [(step % rounds) * 12, (step % rounds) * 12 + 6])
                for loader in loaders:
                    x, y = loader.next_batch()
                    self.assertEqual(tuple(x.shape), (2, 3))
                    self.assertEqual(tuple(y.shape), (2, 3))
        with self.assertRaises(ValueError):
            DistributedDataLoaderLite(list(range(12)), 2, 3, 0, 2)
        with self.assertRaises(ValueError):
            DistributedDataLoaderLite(list(range(25)), 2, 3, 2, 2)

    def test_single_rank_preserves_previous_loader(self):
        for size in (7, 12, 13, 25):
            previous = DataLoaderLite(list(range(size)), 2, 3)
            current = DistributedDataLoaderLite(list(range(size)), 2, 3)
            for _ in range(12):
                self.assertEqual(previous.current_position, current.current_position)
                for a, b in zip(previous.next_batch(), current.next_batch()):
                    torch.testing.assert_close(a, b)


class OptimizationTests(unittest.TestCase):
    def test_global_token_budget_and_schedule_boundaries(self):
        self.assertEqual(accumulation_steps(512, 2, 32, 1), 8)
        self.assertEqual(accumulation_steps(512, 2, 32, 2), 4)
        with self.assertRaises(ValueError):
            accumulation_steps(513, 2, 32, 2)
        with self.assertRaises(ValueError):
            accumulation_steps(0, 2, 32, 2)
        for step, expected in [(0, 0.5), (1, 1.0), (2, 1.0), (3, 0.55), (4, 0.1), (5, 0.1)]:
            self.assertAlmostEqual(learning_rate_at(step, 1.0, 0.1, 2, 4), expected)

    def test_optimizer_preserves_sharing_and_decay_groups(self):
        model = make_model()
        optimizer = configure_optimizer(model, torch.device('cpu'), 6e-4)
        params = [p for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(params), len({id(p) for p in params}))
        self.assertEqual({id(p) for p in params}, {id(p) for p in model.parameters()})
        self.assertEqual(sum(p is model.lm_head.weight for p in params), 1)
        for group in optimizer.param_groups:
            self.assertEqual(group['betas'], (0.9, 0.95))
            for parameter in group['params']:
                self.assertEqual(group['weight_decay'], 0.1 if parameter.ndim >= 2 else 0.0)


class RealDistributedTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), '需要 CPU/Gloo')
    def test_two_process_accumulation_matches_global_batch(self):
        cache = ROOT / '.cache/runtime-check'
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='ddp-check-', dir=cache) as directory:
            mp.spawn(distributed_worker, args=(str(Path(directory) / 'rendezvous'), directory),
                     nprocs=2, join=True)
            ranks = [json.loads((Path(directory) / f'rank-{rank}.json').read_text()) for rank in range(2)]
        result = {'scope': 'Small GPT, real CPU/Gloo processes; not a CUDA/NCCL test.', 'ranks': ranks}
        (cache / 'lesson11-ddp-equivalence.json').write_text(json.dumps(result, indent=2) + '\n')
        self.assertEqual(len(ranks), 2)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
