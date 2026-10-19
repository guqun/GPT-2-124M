"""检查真实 autocast 的类型、梯度/更新和观察模式；小模型只用于行为验证。"""

import contextlib
import io
import unittest
from unittest.mock import patch

import torch

from benchmark_training import benchmark_steps
from gpt2_model import GPT, GPTConfig
from train_dataloader import DataLoaderLite
from train_initialization import initialize_like_course
from train_mixed_precision import inspect_forward_dtypes, train_amp_steps, validate_amp
from train_weight_tying import tie_weights_like_course


class MixedPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.old_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision('highest')
        self.device = torch.device('cpu')

    def tearDown(self):
        torch.set_float32_matmul_precision(self.old_precision)

    def make_model(self):
        torch.manual_seed(42)
        config = GPTConfig(vocab_size=31, n_positions=8, n_embd=16, n_layer=2,
                           n_head=2, embd_pdrop=0, attn_pdrop=0, resid_pdrop=0)
        model = GPT(config, tie_embeddings=False, gpt2_init=False).float()
        tie_weights_like_course(model)
        initialize_like_course(model)
        return model

    def make_loader(self):
        return DataLoaderLite([i % 31 for i in range(81)], 2, 4)

    def test_fp32_mode_preserves_previous_training_trajectory(self):
        actual, expected = self.make_model(), self.make_model()
        actual_opt = torch.optim.AdamW(actual.parameters(), lr=3e-4)
        expected_opt = torch.optim.AdamW(expected.parameters(), lr=3e-4)
        with contextlib.redirect_stdout(io.StringIO()):
            old_records, _ = benchmark_steps(expected, expected_opt, self.make_loader(),
                                             self.device, 1, 3)
            new_records, summary = train_amp_steps(actual, actual_opt, self.make_loader(),
                                                   self.device, 'fp32', 1, 3)
        self.assertEqual([r['loss_before_update'] for r in new_records],
                         [r['loss_before_update'] for r in old_records])
        self.assertEqual([r['batch_start'] for r in new_records], [0, 8, 16, 24])
        self.assertEqual(summary['measured_tokens'], 24)
        for left, right in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_real_bf16_forward_has_fp32_gradients_and_updates_parameters(self):
        model = self.make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        before = model.lm_head.weight.detach().clone()
        forward_modes, backward_modes, optimizer_modes = [], [], []
        forward_handle = model.register_forward_pre_hook(
            lambda module, args: forward_modes.append(torch.is_autocast_cpu_enabled()))
        grad_handle = model.lm_head.weight.register_hook(
            lambda grad: backward_modes.append(torch.is_autocast_cpu_enabled()))
        original_step = optimizer.step

        def observed_step(*args, **kwargs):
            optimizer_modes.append(torch.is_autocast_cpu_enabled())
            return original_step(*args, **kwargs)

        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(optimizer, 'step', side_effect=observed_step):
                records, summary = train_amp_steps(model, optimizer, self.make_loader(),
                                                   self.device, 'bf16', 1, 2)
        finally:
            forward_handle.remove()
            grad_handle.remove()
        self.assertEqual(forward_modes, [True] * 3)
        self.assertEqual(backward_modes, [False] * 3)
        self.assertEqual(optimizer_modes, [False] * 3)
        self.assertEqual(summary['measured_tokens'], 16)
        self.assertFalse(torch.equal(before, model.lm_head.weight))
        self.assertIs(model.lm_head.weight, model.transformer.wte.weight)
        for record in records:
            self.assertEqual(record['input_dtype'], 'torch.int64')
            self.assertEqual(record['logits']['dtype'], 'torch.bfloat16')
            self.assertEqual(record['logits']['element_bytes'], 2)
            for name in ('parameter_dtype', 'loss_dtype', 'parameter_grad_dtype',
                         'adam_m_dtype', 'adam_v_dtype'):
                self.assertEqual(record[name], 'torch.float32')
        for parameter in model.parameters():
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(parameter.grad.dtype, torch.float32)
            self.assertTrue(torch.isfinite(parameter).all())
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_probe_observes_selective_casting_without_changing_model(self):
        model = self.make_model()
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        x, y = self.make_loader().next_batch()
        fp32, reference = inspect_forward_dtypes(model, x, y, self.device, 'fp32')
        bf16, actual = inspect_forward_dtypes(model, x, y, self.device, 'bf16')
        self.assertTrue(model.training)
        observations = {row['name']: row for row in bf16['observations']}
        self.assertEqual(observations['transformer.wte']['dtype'], 'torch.float32')
        self.assertEqual(observations['transformer.h.0.ln_1']['dtype'], 'torch.float32')
        self.assertEqual(observations['transformer.h.0.attn.c_attn']['dtype'], 'torch.bfloat16')
        self.assertEqual(observations['transformer.h.0.mlp.c_fc']['dtype'], 'torch.bfloat16')
        self.assertEqual(observations['transformer.h.0']['dtype'], 'torch.float32')
        self.assertEqual(observations['cross_entropy']['dtype'], 'torch.float32')
        fp32_head = next(row for row in fp32['observations'] if row['name'] == 'lm_head')
        self.assertEqual(fp32_head['logical_bytes'], 2 * observations['lm_head']['logical_bytes'])
        self.assertEqual(reference.shape, actual.shape)
        self.assertTrue(torch.isfinite(actual).all())
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            self.assertIsNone(parameter.grad)
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))
        model.eval()
        inspect_forward_dtypes(model, x, y, self.device, 'bf16')
        self.assertFalse(model.training)

    def test_unsupported_modes_do_not_silently_fall_back(self):
        with self.assertRaises(ValueError):
            validate_amp(torch.device('mps'), 'bf16')
        with patch('torch.cuda.is_bf16_supported', return_value=False):
            with self.assertRaises(ValueError):
                validate_amp(torch.device('cuda'), 'bf16')
        with self.assertRaises(ValueError):
            validate_amp(self.device, 'fp16')
        model = self.make_model().bfloat16()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        with self.assertRaises(ValueError):
            train_amp_steps(model, optimizer, self.make_loader(), self.device, 'bf16', 0, 1)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
