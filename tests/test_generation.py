"""小型行为测试：不下载权重，独立核对生成循环和采样规则。"""
from types import SimpleNamespace
import unittest

import torch

from generate_text import generate, select_next_token


class PrefixModel(torch.nn.Module):
    """按输入长度预测已知 ID，记录实际收到的前缀；与 GPT 的计算实现无关。"""
    def __init__(self, next_ids=(2, 3, 5), capacity=8):
        super().__init__()
        self.config = SimpleNamespace(n_positions=capacity, vocab_size=6)
        self.next_ids = next_ids
        self.seen = []
        self.eval()

    def forward(self, ids):
        self.seen.append(ids.tolist()[0])
        logits = torch.zeros(1, ids.shape[1], 6)
        # 更早的位置故意强烈偏向 ID 4，帮助检测错误地使用第一位置的实现。
        logits[:, :, 4] = 100
        logits[:, -1, :] = 0
        logits[:, -1, self.next_ids[len(self.seen) - 1]] = 100
        return logits


class GenerationTests(unittest.TestCase):
    def run_model(self, model, ids=None, **kwargs):
        if ids is None:
            ids = torch.tensor([[1]])
        return generate(model, ids, eos_token_id=5, top_k=0, **kwargs)

    def test_autoregressive_feedback_and_eos(self):
        model = PrefixModel()
        original = torch.tensor([[1]])
        result, steps, reason = self.run_model(model, original, max_new_tokens=6)
        self.assertEqual(result.tolist(), [[1, 2, 3, 5]])
        self.assertEqual(model.seen, [[1], [1, 2], [1, 2, 3]])
        self.assertEqual(original.tolist(), [[1]])
        self.assertEqual([s['logits_shape'] for s in steps], [[1, 1, 6], [1, 2, 6], [1, 3, 6]])
        self.assertEqual(reason, 'eos')

    def test_budget_and_context_limits(self):
        model = PrefixModel()
        result, _, reason = self.run_model(model, max_new_tokens=1)
        self.assertEqual(result.tolist(), [[1, 2]])
        self.assertEqual(reason, 'max_new_tokens')
        model = PrefixModel(capacity=2)
        result, _, reason = self.run_model(model, max_new_tokens=8)
        self.assertEqual(result.tolist(), [[1, 2]])
        self.assertEqual(reason, 'context_limit')
        self.assertEqual(len(model.seen), 1)

    def test_zero_budget_does_not_call_model(self):
        model = PrefixModel()
        result, steps, reason = self.run_model(model, max_new_tokens=0)
        self.assertEqual(result.tolist(), [[1]])
        self.assertEqual((steps, reason), ([], 'max_new_tokens'))
        self.assertEqual(model.seen, [])

    def test_sampling_support_temperature_and_reproducibility(self):
        scores = torch.tensor([[0., 1., 2., 3., 4., 5.]])
        rng = torch.Generator().manual_seed(42)
        selected, probs = select_next_token(scores, 'sample', 1., 2, rng)
        self.assertIn(selected.item(), [4, 5])
        self.assertEqual(probs[0, :4].tolist(), [0., 0., 0., 0.])
        self.assertAlmostEqual(probs[0, 5].item(), 0.7310586, places=6)
        _, colder = select_next_token(scores, 'sample', .5, 2, rng)
        self.assertGreater(colder[0, 5].item(), probs[0, 5].item())

        def sample_sequence(seed):
            rng = torch.Generator().manual_seed(seed)
            return [select_next_token(scores, 'sample', 1., 2, rng)[0].item() for _ in range(20)]
        sequence = sample_sequence(42)
        self.assertEqual(sequence, sample_sequence(42))
        self.assertTrue(set(sequence).issubset({4, 5}))
        self.assertEqual(select_next_token(scores, 'sample', 1., 1, rng)[0].item(), 5)

    def test_reject_invalid_inputs(self):
        model = PrefixModel()
        for ids in [torch.tensor([1]), torch.empty((1, 0), dtype=torch.long),
                    torch.tensor([[6]]), torch.tensor([[1.0]])]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.run_model(model, ids)
        for kwargs in [{'temperature': 0}, {'temperature': float('nan')},
                       {'max_new_tokens': -1}, {'strategy': 'unknown'}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_model(model, **kwargs)
        model.train()
        with self.assertRaises(ValueError):
            self.run_model(model)


if __name__ == '__main__':
    unittest.main()
