"""训练行为检查：使用同一 GPT 类的小配置，不读取权重或训练语料。"""
import copy
import math
import unittest

import torch

from gpt2_model import GPT, GPTConfig
from train_step import make_batch, next_token_loss, training_step


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def small_model(self):
        return GPT(GPTConfig(vocab_size=17, n_positions=8, n_embd=16,
                             n_layer=1, n_head=2, embd_pdrop=0,
                             attn_pdrop=0, resid_pdrop=0))

    def test_targets_are_next_tokens_across_batch_boundary(self):
        x, y = make_batch(list(range(10, 17)), batch_size=2, seq_len=3)
        self.assertEqual(x.tolist(), [[10, 11, 12], [13, 14, 15]])
        self.assertEqual(y.tolist(), [[11, 12, 13], [14, 15, 16]])
        self.assertEqual((x.dtype, y.dtype), (torch.long, torch.long))
        with self.assertRaises(ValueError):
            make_batch(list(range(6)), batch_size=2, seq_len=3)

    def test_loss_uses_every_position_and_raw_logits(self):
        logits = torch.zeros(2, 2, 2, dtype=torch.float64)
        logits[0, 0, 0] = math.log(3)  # 第一题正确类概率 3/4，其余三题为 1/2。
        logits.requires_grad_()
        targets = torch.tensor([[0, 0], [1, 0]]).t()  # 非连续存储的 [B,T]
        loss = next_token_loss(logits, targets)
        expected = (math.log(4 / 3) + 3 * math.log(2)) / 4
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertAlmostEqual(loss.item(), expected, places=12)
        loss.backward()
        self.assertAlmostEqual(logits.grad[0, 0, 0].item(), -0.25 / 4, places=12)
        self.assertAlmostEqual(logits.grad[0, 1, 1].item(), -0.5 / 4, places=12)
        self.assertAlmostEqual(logits.grad[1, 0, 0].item(), -0.5 / 4, places=12)
        self.assertAlmostEqual(logits.grad[1, 1, 0].item(), -0.5 / 4, places=12)

    def test_backward_and_optimizer_update_have_distinct_effects(self):
        model = self.small_model()
        x, y = make_batch(list(range(1, 8)), 2, 3)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        # 用无动量 SGD 的明确数学规则，独立核对 training_step 是否执行一次更新。
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        record = training_step(model, optimizer, x, y)
        self.assertTrue(record['probe']['unchanged_after_backward'])
        self.assertGreater(record['probe']['max_abs_update'], 0)
        self.assertGreater(record['gradient_l2_norm'], 0)
        self.assertEqual(record['logits_shape'], [2, 3, 17])
        self.assertIs(model.transformer.wte.weight, model.lm_head.weight)
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertEqual(p.shape, p.grad.shape, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            torch.testing.assert_close(p, before[name] - 0.01 * p.grad)

    def test_old_gradients_do_not_leak_into_new_step(self):
        clean = self.small_model()
        dirty = copy.deepcopy(clean)
        for p in dirty.parameters():
            p.grad = torch.full_like(p, 100.0)
        x, y = make_batch(list(range(1, 6)), 1, 4)
        for model in (clean, dirty):
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
            training_step(model, optimizer, x, y)
        for actual, expected in zip(dirty.parameters(), clean.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_video_stage_has_separate_tables_and_both_are_trained(self):
        config = self.small_model().config
        model = GPT(config, tie_embeddings=False, gpt2_init=False)
        self.assertIsNot(model.transformer.wte.weight, model.lm_head.weight)
        tied = GPT(config)
        self.assertEqual(sum(p.numel() for p in model.parameters())
                         - sum(p.numel() for p in tied.parameters()),
                         config.vocab_size * config.n_embd)
        x, y = make_batch(list(range(1, 8)), 2, 3)
        embedding_before = model.transformer.wte.weight.detach().clone()
        head_before = model.lm_head.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        training_step(model, optimizer, x, y)
        self.assertFalse(torch.equal(embedding_before, model.transformer.wte.weight))
        self.assertFalse(torch.equal(head_before, model.lm_head.weight))

    def test_fixed_batch_is_unchanged_while_model_learns(self):
        model = GPT(self.small_model().config, tie_embeddings=False, gpt2_init=False)
        x, y = make_batch(list(range(1, 8)), 2, 3)
        x_before, y_before = x.clone(), y.clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        records = [training_step(model, optimizer, x, y) for _ in range(30)]
        with torch.no_grad():
            final_logits = model(x)
            final_loss = next_token_loss(final_logits, y).item()
        # 检查真实学习行为，不要求每一步单调下降，也不依赖某个浮点结果。
        self.assertLess(final_loss, records[0]['loss_before_update'] / 4)
        self.assertEqual(int((final_logits.argmax(-1) == y).sum()), y.numel())
        torch.testing.assert_close(x, x_before, rtol=0, atol=0)
        torch.testing.assert_close(y, y_before, rtol=0, atol=0)

    def test_reject_invalid_targets_and_disabled_autograd(self):
        logits = torch.zeros(1, 2, 3)
        for targets in (torch.tensor([[0]]), torch.tensor([[0., 1.]]),
                        torch.tensor([[-1, 1]]), torch.tensor([[0, 3]])):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                next_token_loss(logits, targets)
        model = self.small_model()
        x, y = make_batch([1, 2, 3], 1, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with torch.no_grad(), self.assertRaises(ValueError):
            training_step(model, optimizer, x, y)


if __name__ == '__main__':
    unittest.main()
