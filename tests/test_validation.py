"""验证数据隔离、冻结前向、恢复训练及真实双进程指标归约。"""

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
from train_ddp import DistributedDataLoaderLite, Runtime, configure_optimizer, train_updates
from train_initialization import initialize_like_course
from train_weight_tying import tie_weights_like_course
from train_validation import (
    ResettableDataLoaderLite, evaluate_validation, make_train_val_loaders, train_with_validation,
)


ROOT = Path(__file__).resolve().parents[1]


def make_model(dropout=0.0):
    torch.manual_seed(42)
    config = GPTConfig(vocab_size=31, n_positions=8, n_embd=16, n_layer=2, n_head=2,
                       embd_pdrop=dropout, attn_pdrop=dropout, resid_pdrop=dropout)
    model = GPT(config, tie_embeddings=False, gpt2_init=False).double()
    tie_weights_like_course(model)
    initialize_like_course(model)
    return model


def runtime_cpu():
    return Runtime(False, 0, 0, 1, torch.device('cpu'), 'none')


def update_options():
    return dict(steps=3, accum_steps=2, precision='fp32', max_lr=6e-4,
                min_lr=6e-5, warmup_steps=1, decay_steps=3, clip_norm=0.5)


def distributed_validation_worker(rank, rendezvous, output_dir):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=60))
    try:
        runtime = Runtime(True, rank, rank, 2, torch.device('cpu'), 'gloo')
        raw_model, reference = make_model(), make_model()
        model = DDP(raw_model)
        val_tokens = [(7 * i + 3) % 31 for i in range(97)]
        val_loader = ResettableDataLoaderLite(val_tokens, 2, 4, rank, 2)
        # 三个全局轮次：每轮两卡各 B=2,T=4；直接联合为 B=12,T=4 作参考。
        reference.eval()
        with torch.no_grad():
            joined = torch.tensor(val_tokens[:49])
            expected = F.cross_entropy(reference(joined[:-1].view(12, 4)).reshape(-1, 31),
                                       joined[1:].reshape(-1)).item()
        for parameter in raw_model.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)
        snapshot = [p.detach().clone() for p in raw_model.parameters()]
        observed = evaluate_validation(model, val_loader, runtime,
                                       validation_batches=3, precision='fp32')
        torch.testing.assert_close(observed['global_mean_loss'], expected, atol=1e-12, rtol=1e-12)
        for parameter, before in zip(raw_model.parameters(), snapshot):
            torch.testing.assert_close(parameter, before, atol=0, rtol=0)
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 0.25), atol=0, rtol=0)
        if observed['local_batch_starts'] != [rank * 8 + i * 16 for i in range(3)]:
            raise AssertionError('验证读取了错误的 rank 位置。')
        if not model.training:
            raise AssertionError('验证结束未恢复 train 模式。')

        # 检查验证前向不会干扰 DDP 后续的反向同步和参数更新。
        train_tokens = [i % 31 for i in range(257)]
        optimizer = configure_optimizer(raw_model, torch.device('cpu'), 6e-4)
        reference_optimizer = configure_optimizer(reference, torch.device('cpu'), 6e-4)
        options = update_options()
        options['steps'] = 2
        with redirect_stdout(io.StringIO()):
            actual_rows = train_with_validation(
                model, raw_model, optimizer,
                DistributedDataLoaderLite(train_tokens, 2, 4, rank, 2), val_loader, runtime,
                validation_every=1, validation_batches=2, **options)
            # 双卡 R2/B2/T4/K2 = 单卡 R1/B8/T4/K1，每次更新都处理 32 个位置。
            options['accum_steps'] = 1
            reference_rows = train_updates(
                reference, reference, reference_optimizer,
                DistributedDataLoaderLite(train_tokens, 8, 4), runtime_cpu(), **options)
        max_parameter_error, max_moment_error = 0.0, 0.0
        for actual, target in zip(raw_model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, target, rtol=1e-8, atol=1e-10)
            max_parameter_error = max(max_parameter_error, (actual - target).abs().max().item())
            for field in ('step', 'exp_avg', 'exp_avg_sq'):
                a, b = optimizer.state[actual][field], reference_optimizer.state[target][field]
                torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-10)
                max_moment_error = max(max_moment_error, (a - b).abs().max().item())
        for actual, target in zip(actual_rows['training'], reference_rows):
            torch.testing.assert_close(actual['global_mean_loss'], target['global_mean_loss'],
                                       rtol=1e-10, atol=1e-10)
        result = {'rank': rank, 'world_size': 2, 'backend': 'gloo', 'dtype': 'float64',
                  'validation_loss': observed['global_mean_loss'],
                  'single_process_joint_batch_loss': expected,
                  'validation_loss_error': abs(observed['global_mean_loss'] - expected),
                  'local_validation_starts': observed['local_batch_starts'],
                  'parameters_and_existing_gradients_unchanged_during_validation': True,
                  'following_training_updates': 2,
                  'max_parameter_error_after_training': max_parameter_error,
                  'max_optimizer_state_error_after_training': max_moment_error}
        (Path(output_dir) / f'rank-{rank}.json').write_text(json.dumps(result, indent=2) + '\n')
    finally:
        dist.destroy_process_group()


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_train_val_positions_and_shifted_labels_do_not_overlap(self):
        train, val, split = make_train_val_loaders(list(range(200)), 2, 4, runtime_cpu())
        self.assertEqual(split, 180)
        for _ in range(30):
            x, y = train.next_batch()
            self.assertLess(max(x.max().item(), y.max().item()), split)
        for _ in range(5):
            x, y = val.next_batch()
            self.assertGreaterEqual(min(x.min().item(), y.min().item()), split)
        previous_train_position = train.current_position
        val.reset()
        self.assertEqual(val.current_position, 0)
        self.assertEqual(train.current_position, previous_train_position)
        with self.assertRaisesRegex(ValueError, 'val-batches'):
            evaluate_validation(make_model(), val, runtime_cpu(), validation_batches=3, precision='fp32')

    def test_validation_matches_joint_batch_and_preserves_state_and_rng(self):
        model = make_model(dropout=0.2)
        optimizer = configure_optimizer(model, torch.device('cpu'), 6e-4)
        train = DistributedDataLoaderLite([i % 31 for i in range(257)], 2, 4)
        options = update_options()
        options['steps'] = 1
        with redirect_stdout(io.StringIO()):
            train_updates(model, model, optimizer, train, runtime_cpu(), **options)
        parameters = [p.detach().clone() for p in model.parameters()]
        gradients = [p.grad.clone() for p in model.parameters()]
        state = {p: {k: v.clone() for k, v in s.items()} for p, s in optimizer.state.items()}
        tokens = [(i * 7 + 3) % 31 for i in range(100)]
        val = ResettableDataLoaderLite(tokens, 2, 4)
        reference = make_model(dropout=0.2)
        reference.load_state_dict(model.state_dict())
        reference.eval()
        with torch.no_grad():
            data = torch.tensor(tokens[:25])
            expected = F.cross_entropy(reference(data[:-1].view(6, 4)).reshape(-1, 31),
                                       data[1:].reshape(-1)).item()
        rng = torch.random.get_rng_state().clone()
        old_start = train.current_position
        observed_forward = []
        handle = model.register_forward_hook(
            lambda module, inputs, output: observed_forward.append(
                (module.training, torch.is_grad_enabled(), output.requires_grad)))
        try:
            first = evaluate_validation(model, val, runtime_cpu(), validation_batches=3, precision='fp32')
            second = evaluate_validation(model, val, runtime_cpu(), validation_batches=3, precision='fp32')
        finally:
            handle.remove()
        self.assertEqual(first, second)
        self.assertAlmostEqual(first['global_mean_loss'], expected, places=12)
        self.assertEqual(first['global_predicted_positions'], 24)
        self.assertEqual(first['local_batch_starts'], [0, 8, 16])
        self.assertEqual(observed_forward, [(False, False, False)] * 6)
        self.assertTrue(all(module.training for module in model.modules()))
        self.assertEqual(train.current_position, old_start)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, atol=0, rtol=0)
        for p, before, grad in zip(model.parameters(), parameters, gradients):
            torch.testing.assert_close(p, before, atol=0, rtol=0)
            torch.testing.assert_close(p.grad, grad, atol=0, rtol=0)
            for field, value in state[p].items():
                torch.testing.assert_close(optimizer.state[p][field], value, atol=0, rtol=0)

    def test_mode_restored_on_error_and_eval_caller_stays_in_eval(self):
        model = make_model()
        val = ResettableDataLoaderLite(list(range(20)), 1, 4)
        with patch('train_validation.forward_and_loss', side_effect=RuntimeError('probe')):
            with self.assertRaisesRegex(RuntimeError, 'probe'):
                evaluate_validation(model, val, runtime_cpu(), validation_batches=2, precision='fp32')
        self.assertTrue(model.training)
        model.eval()
        evaluate_validation(model, val, runtime_cpu(), validation_batches=2, precision='fp32')
        self.assertFalse(model.training)

    def test_adding_validation_preserves_training_trajectory_including_dropout(self):
        baseline, observed = make_model(dropout=0.2), make_model(dropout=0.2)
        opt_base = configure_optimizer(baseline, torch.device('cpu'), 6e-4)
        opt_observed = configure_optimizer(observed, torch.device('cpu'), 6e-4)
        tokens = [i % 31 for i in range(257)]
        base_loader = DistributedDataLoaderLite(tokens, 2, 4)
        observed_loader = DistributedDataLoaderLite(tokens, 2, 4)
        val = ResettableDataLoaderLite([(i * 7) % 31 for i in range(100)], 2, 4)
        with redirect_stdout(io.StringIO()):
            torch.manual_seed(123)
            expected_rows = train_updates(baseline, baseline, opt_base, base_loader,
                                          runtime_cpu(), **update_options())
            torch.manual_seed(123)
            actual_rows = train_with_validation(
                observed, observed, opt_observed, observed_loader, val, runtime_cpu(),
                validation_every=2, validation_batches=3, **update_options())
        self.assertEqual([r['after_updates'] for r in actual_rows['validation']], [0, 2, 3])
        for actual, expected in zip(actual_rows['training'], expected_rows):
            for field in ('local_batch_starts', 'next_start', 'learning_rate', 'global_mean_loss'):
                self.assertEqual(actual[field], expected[field])
        for a, b in zip(observed.parameters(), baseline.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            for field in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(opt_observed.state[a][field], opt_base.state[b][field],
                                           atol=0, rtol=0)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), '需要 CPU/Gloo')
    def test_real_two_process_validation_and_following_training_match_joint_batches(self):
        cache = ROOT / '.cache/runtime-check'
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='validation-check-', dir=cache) as directory:
            mp.spawn(distributed_validation_worker,
                     args=(str(Path(directory) / 'rendezvous'), directory), nprocs=2, join=True)
            ranks = [json.loads((Path(directory) / f'rank-{r}.json').read_text()) for r in range(2)]
        report = {'scope': 'Real CPU/Gloo processes with small GPT; not CUDA/NCCL or FineWeb-Edu.',
                  'ranks': ranks}
        (ROOT / 'artifacts/lesson13-validation-equivalence.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    unittest.main()
