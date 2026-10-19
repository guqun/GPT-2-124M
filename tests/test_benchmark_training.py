"""验证计时包装不改变训练、预热统计范围与设备边界。"""

import contextlib
import io
import unittest
from unittest.mock import patch

import torch

from benchmark_training import (
    benchmark_steps, configure_precision, select_device, summarize_timings,
)
from gpt2_model import GPT, GPTConfig
from train_dataloader import DataLoaderLite
from train_initialization import initialize_like_course
from train_step import next_token_loss
from train_weight_tying import tie_weights_like_course


class BenchmarkTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(42)
        config = GPTConfig(vocab_size=31, n_positions=8, n_embd=16, n_layer=2,
                           n_head=2, embd_pdrop=0, attn_pdrop=0, resid_pdrop=0)
        model = GPT(config, tie_embeddings=False, gpt2_init=False)
        tie_weights_like_course(model)
        initialize_like_course(model)
        return model

    def test_timed_training_matches_plain_training_including_warmup(self):
        actual, expected = self.make_model(), self.make_model()
        actual_opt = torch.optim.AdamW(actual.parameters(), lr=3e-4)
        expected_opt = torch.optim.AdamW(expected.parameters(), lr=3e-4)
        ids = [i % 31 for i in range(81)]
        actual_loader = DataLoaderLite(ids, 2, 4)
        expected_loader = DataLoaderLite(ids, 2, 4)
        expected_losses = []
        for _ in range(5):
            x, y = expected_loader.next_batch()
            expected_opt.zero_grad(set_to_none=True)
            loss = next_token_loss(expected(x), y)
            loss.backward()
            expected_opt.step()
            expected_losses.append(loss.item())
        events = []
        times = iter(range(10))

        def clock():
            events.append('clock')
            return next(times)

        with contextlib.redirect_stdout(io.StringIO()), \
                patch('benchmark_training.time.perf_counter', side_effect=clock), \
                patch('benchmark_training.synchronize', side_effect=lambda _: events.append('sync')):
            records, summary = benchmark_steps(actual, actual_opt, actual_loader,
                                               torch.device('cpu'), 2, 3)
        # 两端都等待；结束时间必须在第二次同步之后读取。
        self.assertEqual(events, ['sync', 'clock', 'sync', 'clock'] * 5)
        self.assertEqual([r['loss_before_update'] for r in records], expected_losses)
        self.assertEqual([r['batch_start'] for r in records], [0, 8, 16, 24, 32])
        self.assertEqual([r['phase'] for r in records], ['warmup'] * 2 + ['measure'] * 3)
        self.assertEqual(summary['measured_tokens'], 24)
        self.assertEqual(summary['tokens_per_second'], 8)
        for left, right in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertIs(actual.transformer.wte.weight, actual.lm_head.weight)

    def test_throughput_uses_total_time_and_excludes_warmup(self):
        records = [{'phase': 'warmup', 'seconds': 100},
                   {'phase': 'measure', 'seconds': 1},
                   {'phase': 'measure', 'seconds': 3}]
        summary = summarize_timings(records, 128)
        self.assertEqual(summary['measured_tokens'], 256)
        self.assertEqual(summary['mean_step_ms'], 2000)
        self.assertEqual(summary['tokens_per_second'], 64)
        with self.assertRaises(ValueError):
            summarize_timings([{'phase': 'warmup', 'seconds': 1}], 128)

    def test_missing_accelerator_and_cpu_tf32_fail_explicitly(self):
        with patch('torch.cuda.is_available', return_value=False), \
                patch('torch.backends.mps.is_available', return_value=False):
            self.assertEqual(select_device('auto').type, 'cpu')
            with self.assertRaises(ValueError):
                select_device('cuda')
            with self.assertRaises(ValueError):
                select_device('mps')
        with patch('torch.set_float32_matmul_precision') as set_precision:
            with self.assertRaises(ValueError):
                configure_precision(torch.device('cpu'), 'high')
            set_precision.assert_not_called()


if __name__ == '__main__':
    unittest.main()
