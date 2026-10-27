"""生成作为观察分支：使用当前参数，保护 RNG，并允许训练继续。"""

import copy
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gpt2_model import GPT, GPTConfig
from train_ddp import Runtime, configure_optimizer
from train_generation import sample_current_model, train_with_generation
from train_validation import ResettableDataLoaderLite, train_with_validation


def small_model():
    torch.manual_seed(42)
    return GPT(GPTConfig(vocab_size=31, n_positions=8, n_embd=16, n_layer=2, n_head=2,
                         embd_pdrop=0.2, attn_pdrop=0.2, resid_pdrop=0.2)).double()


class TokenizerStub:
    def decode(self, ids):
        return ' '.join(map(str, ids))


class PeriodicGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sampling_preserves_parameters_gradients_adamw_rng_and_mode(self):
        model = small_model()
        optimizer = configure_optimizer(model, torch.device('cpu'), 6e-4)
        model(torch.tensor([[1, 2, 3]])).square().mean().backward()
        optimizer.step()
        params = [p.detach().clone() for p in model.parameters()]
        grads = [p.grad.clone() for p in model.parameters()]
        states = copy.deepcopy(optimizer.state_dict())
        prompt = torch.tensor([1, 2, 3])
        rng = torch.random.get_rng_state().clone()
        observed = []
        handle = model.register_forward_hook(lambda module, inputs, output: observed.append(
            (module.training, torch.is_grad_enabled(), output.requires_grad, list(inputs[0].shape))))
        try:
            first = sample_current_model(model, prompt, max_new_tokens=3,
                                         num_sequences=2, top_k=5)
            second = sample_current_model(model, prompt, max_new_tokens=3,
                                          num_sequences=2, top_k=5)
        finally:
            handle.remove()
        self.assertEqual(first, second)
        self.assertEqual(observed, [(False, False, False, [2, t]) for t in (3, 4, 5)] * 2)
        self.assertTrue(all(m.training for m in model.modules()))
        self.assertEqual(prompt.tolist(), [1, 2, 3])
        for ids in first['token_ids']:
            self.assertEqual(ids[:3], prompt.tolist())
        torch.testing.assert_close(torch.random.get_rng_state(), rng, atol=0, rtol=0)
        for p, before, grad in zip(model.parameters(), params, grads):
            torch.testing.assert_close(p, before, atol=0, rtol=0)
            torch.testing.assert_close(p.grad, grad, atol=0, rtol=0)
        current = optimizer.state_dict()
        self.assertEqual(current['param_groups'], states['param_groups'])
        for index, fields in states['state'].items():
            for key, value in fields.items():
                torch.testing.assert_close(current['state'][index][key], value, atol=0, rtol=0)

    def test_inserting_generation_does_not_change_training_trajectory(self):
        baseline = small_model()
        sampled = copy.deepcopy(baseline)
        opt_base = configure_optimizer(baseline, torch.device('cpu'), 6e-4)
        opt_sample = configure_optimizer(sampled, torch.device('cpu'), 6e-4)
        runtime = Runtime(False, 0, 0, 1, torch.device('cpu'), 'none')
        loaders = [ResettableDataLoaderLite([i % 31 for i in range(257)], 2, 4) for _ in range(2)]
        vals = [ResettableDataLoaderLite([(i * 7) % 31 for i in range(100)], 2, 4) for _ in range(2)]
        options = dict(steps=3, accum_steps=2, precision='fp32', max_lr=6e-4, min_lr=6e-5,
                       warmup_steps=1, decay_steps=3, clip_norm=0.5,
                       validation_every=2, validation_batches=2)
        with redirect_stdout(io.StringIO()):
            torch.manual_seed(123)
            expected = train_with_validation(baseline, baseline, opt_base, loaders[0], vals[0],
                                             runtime, **options)
            expected_rng = torch.random.get_rng_state().clone()
            torch.manual_seed(123)
            actual = train_with_generation(sampled, sampled, opt_sample, loaders[1], vals[1],
                                           runtime, tokenizer=TokenizerStub(), prompt_ids=[1, 2],
                                           generation_every=2, max_new_tokens=3,
                                           num_sequences=2, top_k=5, **options)
        self.assertEqual(actual['training'], expected['training'])
        self.assertEqual(actual['validation'], expected['validation'])
        self.assertEqual([r['after_updates'] for r in actual['generation']], [2, 3])
        self.assertEqual(loaders[0].current_position, loaders[1].current_position)
        torch.testing.assert_close(torch.random.get_rng_state(), expected_rng, atol=0, rtol=0)
        for a, b in zip(sampled.parameters(), baseline.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
            for field in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(opt_sample.state[a][field], opt_base.state[b][field],
                                           atol=0, rtol=0)

    def test_uses_current_weights_and_last_position_and_appends_each_token(self):
        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor([0., 10., 0., 0., 0., 0., 0.]))
                self.config = SimpleNamespace(vocab_size=7, n_positions=8)
                self.inputs = []

            def forward(self, ids):
                self.inputs.append(ids.clone())
                logits = self.bias.expand(ids.size(0), ids.size(1), 7).clone()
                logits[:, :-1, 6] = 100.  # 非最后位置故意给出不同的最高分。
                return logits

        model = Probe()
        first = sample_current_model(model, [2, 3], max_new_tokens=2, num_sequences=2, top_k=1)
        self.assertEqual(first['token_ids'], [[2, 3, 1, 1]] * 2)
        self.assertEqual(model.inputs[1].tolist(), [[2, 3, 1]] * 2)
        with torch.no_grad():
            model.bias[4] = 20.
        second = sample_current_model(model, [2, 3], max_new_tokens=2, num_sequences=2, top_k=1)
        self.assertEqual(second['token_ids'], [[2, 3, 4, 4]] * 2)

    def test_mode_restoration_and_context_limit(self):
        model = small_model()
        with patch.object(model, 'forward', side_effect=RuntimeError('probe')):
            with self.assertRaisesRegex(RuntimeError, 'probe'):
                sample_current_model(model, [1], top_k=5)
        self.assertTrue(model.training)
        model.eval()
        result = sample_current_model(model, [1] * 7, max_new_tokens=3, top_k=5)
        self.assertFalse(model.training)
        self.assertEqual(result['new_tokens_per_sequence'], 1)
        self.assertEqual(result['stop_reason'], 'context_limit')
        with self.assertRaisesRegex(ValueError, '提示词'):
            sample_current_model(model, [], top_k=5)
        with self.assertRaisesRegex(ValueError, 'top-k'):
            sample_current_model(model, [1], top_k=32)


if __name__ == '__main__':
    unittest.main()
