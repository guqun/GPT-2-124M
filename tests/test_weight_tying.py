"""权重共享的行为检查：引用方向、去重、双路径梯度和一次更新。"""

import unittest

import torch

from gpt2_model import GPT, GPTConfig
from train_step import make_batch, next_token_loss
from train_weight_tying import tie_weights_like_course


class WeightTyingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.config = GPTConfig(vocab_size=17, n_positions=8, n_embd=16,
                                n_layer=1, n_head=2, embd_pdrop=0,
                                attn_pdrop=0, resid_pdrop=0)

    def untied_model(self):
        return GPT(self.config, tie_embeddings=False, gpt2_init=False).double()

    def test_retains_output_parameter_and_optimizer_sees_it_once(self):
        model = self.untied_model()
        old_input = model.transformer.wte.weight
        old_output = model.lm_head.weight
        output_values = old_output.detach().clone()
        before_count = sum(p.numel() for p in model.parameters())
        tie_weights_like_course(model)
        self.assertIs(model.transformer.wte.weight, old_output)
        self.assertIs(model.lm_head.weight, old_output)
        self.assertFalse(any(p is old_input for p in model.parameters()))
        torch.testing.assert_close(old_output, output_values, rtol=0, atol=0)
        self.assertEqual(before_count - sum(p.numel() for p in model.parameters()),
                         self.config.vocab_size * self.config.n_embd)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        self.assertEqual(sum(p is old_output for group in optimizer.param_groups
                             for p in group['params']), 1)

    def test_shared_gradient_sums_both_paths_and_is_updated_once(self):
        shared = self.untied_model()
        tie_weights_like_course(shared)
        separate = self.untied_model()
        # 相同数值、两份 Parameter：它是分开计算两条梯度贡献的对照模型。
        separate.load_state_dict(shared.state_dict())
        self.assertIsNot(separate.transformer.wte.weight, separate.lm_head.weight)
        x, y = make_batch(list(range(1, 8)), batch_size=2, seq_len=3)
        shared_logits, separate_logits = shared(x), separate(x)
        torch.testing.assert_close(shared_logits, separate_logits, rtol=0, atol=0)
        next_token_loss(shared_logits, y).backward()
        next_token_loss(separate_logits, y).backward()
        expected_gradient = (separate.transformer.wte.weight.grad
                             + separate.lm_head.weight.grad)
        self.assertGreater(separate.transformer.wte.weight.grad.abs().sum().item(), 0)
        self.assertGreater(separate.lm_head.weight.grad.abs().sum().item(), 0)
        torch.testing.assert_close(shared.transformer.wte.weight.grad, expected_gradient,
                                   rtol=1e-10, atol=1e-12)
        before = shared.transformer.wte.weight.detach().clone()
        # 用 SGD 的显式公式核对更新次数；课程主脚本仍使用 AdamW。
        optimizer = torch.optim.SGD(shared.parameters(), lr=0.01)
        optimizer.step()
        torch.testing.assert_close(shared.transformer.wte.weight,
                                   before - 0.01 * expected_gradient, rtol=1e-10, atol=1e-12)
        self.assertIs(shared.transformer.wte.weight, shared.lm_head.weight)


if __name__ == '__main__':
    unittest.main()
