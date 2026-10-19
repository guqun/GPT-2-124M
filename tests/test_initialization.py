"""初始化行为：初值尺度、共享引用、可复现性与训练连续性。"""

import math
import unittest

import torch
from torch import nn

from gpt2_model import GPT, GPTConfig
from train_initialization import initialize_like_course, residual_accumulation_demo
from train_step import make_batch, next_token_loss
from train_weight_tying import tie_weights_like_course


class InitializationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = GPTConfig(vocab_size=257, n_positions=128, n_embd=128,
                                n_layer=2, n_head=2, embd_pdrop=0,
                                attn_pdrop=0, resid_pdrop=0)

    def make_model(self, seed=42):
        torch.manual_seed(seed)
        model = GPT(self.config, tie_embeddings=False, gpt2_init=False)
        tie_weights_like_course(model)
        initialize_like_course(model)
        return model

    def test_scales_biases_layernorm_and_shared_identity(self):
        model = self.make_model()
        self.assertIs(model.transformer.wte.weight, model.lm_head.weight)
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                expected = 0.02 / math.sqrt(4) if name.endswith('c_proj') else 0.02
                self.assertAlmostEqual(module.weight.std(unbiased=False).item(),
                                       expected, delta=expected * 0.06, msg=name)
                self.assertLess(abs(module.weight.mean().item()), expected * 0.04, name)
            if isinstance(module, nn.Linear) and module.bias is not None:
                self.assertEqual(torch.count_nonzero(module.bias).item(), 0, name)
            if isinstance(module, nn.LayerNorm):
                torch.testing.assert_close(module.weight, torch.ones_like(module.weight), rtol=0, atol=0)
                torch.testing.assert_close(module.bias, torch.zeros_like(module.bias), rtol=0, atol=0)

    def test_same_seed_reproduces_initial_values_but_training_changes_them(self):
        model, repeated = self.make_model(), self.make_model()
        for actual, expected in zip(model.parameters(), repeated.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        x, y = make_batch(list(range(1, 10)), 2, 4)
        logits = model(x)
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        loss = next_token_loss(logits, y)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        optimizer.step()
        trained = model.transformer.wte.weight.detach().clone()
        self.assertFalse(torch.equal(trained, before['transformer.wte.weight']))
        model(x)  # 下一次前向不能把训练后的参数重新初始化。
        torch.testing.assert_close(model.transformer.wte.weight, trained, rtol=0, atol=0)
        self.assertIs(model.transformer.wte.weight, model.lm_head.weight)

    def test_residual_demo_controls_growth_without_consuming_model_rng(self):
        torch.manual_seed(123)
        before = torch.random.get_rng_state().clone()
        result = residual_accumulation_demo()
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
        self.assertAlmostEqual(result['unscaled_observed_std'], 10, delta=1.5)
        self.assertAlmostEqual(result['scaled_observed_std'], 1, delta=0.15)
        self.assertAlmostEqual(result['unscaled_observed_std'] / result['scaled_observed_std'],
                               10, places=5)


if __name__ == '__main__':
    unittest.main()
