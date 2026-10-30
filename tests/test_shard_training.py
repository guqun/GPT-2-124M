"""分片接入后的行为检查：跨片验证、训练状态隔离，以及和纯训练的数值对照。"""

import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gpt2_model import GPT, GPTConfig
from train_ddp import Runtime, configure_optimizer, train_updates
from train_shards import (evaluate_shards, loader_cursor, make_shard_loaders,
                          shard_inventory, train_from_shards)


class TokenizerStub:
    def decode(self, ids):
        return ' '.join(map(str, ids))


class ShardTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        # 小词表的数值测试数据；不是 GPT-2 的真实文本编码。
        for split, index, values in [
            ('train', 1, list(range(1, 11))),     # B1/T3：3 批
            ('train', 2, list(range(12, 19))),    # 2 批
            ('train', 3, [1, 2, 3]),             # 整片太短，跳过
            ('val', 0, list(range(20, 27))),      # 2 批
            ('val', 4, list(range(5, 15))),       # 3 批
        ]:
            np.save(self.directory / f'edufineweb_{split}_{index:06d}.npy',
                    np.asarray(values, dtype=np.uint16), allow_pickle=False)
        self.runtime = Runtime(False, 0, 0, 1, torch.device('cpu'), 'none')

    def tearDown(self):
        self.temp.cleanup()

    def loaders(self):
        return make_shard_loaders(self.directory, 1, 3, self.runtime)

    def model(self):
        torch.manual_seed(42)
        return GPT(GPTConfig(vocab_size=31, n_positions=8, n_embd=16, n_layer=2, n_head=2,
                             embd_pdrop=0.2, attn_pdrop=0.2, resid_pdrop=0.2)).double()

    def test_validation_counts_all_shards_and_preserves_training_state(self):
        train, val, inventory = self.loaders()
        self.assertEqual(inventory['val']['batches_per_rank_per_pass'], 5)
        self.assertEqual(inventory['train']['batches_per_rank_per_pass'], 5)
        train.next_batch()
        before_cursor = loader_cursor(train)
        model = self.model()
        model(torch.tensor([[1, 2, 3]])).square().mean().backward()
        params = [p.detach().clone() for p in model.parameters()]
        grads = [p.grad.clone() for p in model.parameters()]
        rng = torch.random.get_rng_state().clone()
        observed = []
        handle = model.register_forward_hook(lambda m, inputs, outputs: observed.append(
            (m.training, torch.is_grad_enabled(), outputs.requires_grad)))
        try:
            first = evaluate_shards(model, val, self.runtime, validation_batches=5, precision='fp32')
            second = evaluate_shards(model, val, self.runtime, validation_batches=5, precision='fp32')
        finally:
            handle.remove()
        self.assertEqual(first, second)
        self.assertEqual(first['local_batches'], [
            {'file': 'edufineweb_val_000000.npy', 'start': i} for i in [0, 3]
        ] + [{'file': 'edufineweb_val_000004.npy', 'start': i} for i in [0, 3, 6]])
        self.assertEqual(first['global_predicted_positions'], 15)
        self.assertEqual(loader_cursor(train), before_cursor)
        self.assertTrue(model.training)
        self.assertEqual(observed, [(False, False, False)] * 10)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, atol=0, rtol=0)
        for p, before, grad in zip(model.parameters(), params, grads):
            torch.testing.assert_close(p, before, atol=0, rtol=0)
            torch.testing.assert_close(p.grad, grad, atol=0, rtol=0)

    def test_training_matches_no_observation_baseline_across_shard_switch(self):
        baseline = self.model()
        observed = copy.deepcopy(baseline)
        opt_base = configure_optimizer(baseline, torch.device('cpu'), 6e-4)
        opt_observed = configure_optimizer(observed, torch.device('cpu'), 6e-4)
        base_loader, _, _ = self.loaders()
        train_loader, val_loader, _ = self.loaders()
        options = dict(steps=3, accum_steps=2, precision='fp32', max_lr=6e-4,
                       min_lr=6e-5, warmup_steps=1, decay_steps=3, clip_norm=0.5)
        with redirect_stdout(io.StringIO()):
            torch.manual_seed(123)
            expected = train_updates(baseline, baseline, opt_base, base_loader,
                                     self.runtime, **options)
            expected_rng = torch.random.get_rng_state().clone()
            torch.manual_seed(123)
            actual = train_from_shards(
                observed, observed, opt_observed, train_loader, val_loader, self.runtime,
                validation_every=2, validation_batches=5, tokenizer=TokenizerStub(),
                prompt_ids=[1, 2], generation_every=2, max_new_tokens=2,
                num_sequences=2, top_k=5, **options)
        self.assertEqual([r['after_updates'] for r in actual['validation']], [0, 2, 3])
        self.assertEqual([r['after_updates'] for r in actual['generation']], [2, 3])
        self.assertEqual(actual['training'][1]['local_batches'], [
            {'file': 'edufineweb_train_000001.npy', 'start': 6},
            {'file': 'edufineweb_train_000002.npy', 'start': 0}])
        self.assertEqual(actual['training'][2]['local_batches'], [
            {'file': 'edufineweb_train_000002.npy', 'start': 3},
            {'file': 'edufineweb_train_000001.npy', 'start': 0}])
        for a, b in zip(actual['training'], expected):
            for field in ('global_mean_loss', 'learning_rate', 'gradient_norm_before_clip'):
                self.assertEqual(a[field], b[field])
        for record in actual['validation']:
            self.assertEqual(record['train_cursor_before'], record['train_cursor_after'])
        for record in actual['generation']:
            self.assertEqual(record['cursors_before'], record['cursors_after'])
        self.assertEqual(loader_cursor(train_loader), loader_cursor(base_loader))
        torch.testing.assert_close(torch.random.get_rng_state(), expected_rng, atol=0, rtol=0)
        for a, b in zip(observed.parameters(), baseline.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
            for field in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(opt_observed.state[a][field], opt_base.state[b][field],
                                           atol=0, rtol=0)
            self.assertEqual(opt_observed.state[a]['step'].item(), 3)

    def test_validation_rejects_duplicate_pass_and_restores_mode_on_failure(self):
        _, val, _ = self.loaders()
        model = self.model()
        for batches in (0, 6):
            with self.assertRaisesRegex(ValueError, 'val-batches'):
                evaluate_shards(model, val, self.runtime, validation_batches=batches, precision='fp32')
        with patch.object(model, 'forward', side_effect=RuntimeError('probe')):
            with self.assertRaisesRegex(RuntimeError, 'probe'):
                evaluate_shards(model, val, self.runtime, validation_batches=2, precision='fp32')
        self.assertTrue(model.training)
        model.eval()
        evaluate_shards(model, val, self.runtime, validation_batches=2, precision='fp32')
        self.assertFalse(model.training)

    def test_capacity_is_sum_of_per_file_batches_and_split_files_cannot_alias(self):
        # B2/T3 每轮需 7 个 ID。N=10 和 N=7 各一批，N=3 为零；不能当成20个ID连起来。
        train, _, _ = make_shard_loaders(self.directory, 2, 3, self.runtime)
        self.assertEqual(shard_inventory(train)['batches_per_rank_per_pass'], 2)
        alias = self.directory / 'edufineweb_val_000005.npy'
        alias.symlink_to(self.directory / 'edufineweb_train_000001.npy')
        with self.assertRaisesRegex(ValueError, '同一个文件'):
            self.loaders()


if __name__ == '__main__':
    unittest.main()
