"""结构替换、因果性、累积等价、验证隔离及比较协议的行为检查。"""
import copy
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from architecture_variants import RMSNorm, build_model, common_parameter_digest
from architecture_training import SequentialBatches, evaluate, train_update, learning_rate
from compare_architecture_runs import compare
from prepare_architecture_baseline import load_spec


def tiny_spec():
    spec = copy.deepcopy(load_spec('local'))
    spec['model'].update(vocab_size=31, n_positions=8, n_embd=8, n_layer=2, n_head=2)
    spec['training'].update(batch_size=1, seq_len=3, accumulation_steps=2,
                            validation_batches=2, clip_norm=1e6)
    return spec


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rms_formula_gradient_and_no_centering(self):
        norm = RMSNorm(2, 1e-5).double()
        x = torch.tensor([[[3., 4.]]], dtype=torch.double, requires_grad=True)
        expected = x / torch.sqrt((x*x).sum(-1, keepdim=True)/2 + 1e-5)
        torch.testing.assert_close(norm(x), expected)
        self.assertTrue(torch.autograd.gradcheck(norm, (x,)))
        self.assertGreater(norm(x).mean().item(), 0)
        self.assertIsNone(dict(norm.named_parameters()).get('bias'))

    def test_only_intended_changes_and_identical_shared_start(self):
        spec = tiny_spec()
        a, b = build_model(spec, 'gpt2'), build_model(spec, 'rmsnorm_no_bias')
        self.assertEqual(common_parameter_digest(a), common_parameter_digest(b))
        pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
        self.assertTrue(set(pa)-set(pb))
        self.assertTrue(all(name.endswith('.bias') for name in set(pa)-set(pb)))
        for name in pb:
            torch.testing.assert_close(pa[name], pb[name], rtol=0, atol=0)
        self.assertEqual(sum(isinstance(m, RMSNorm) for m in b.modules()), 5)
        self.assertFalse(any(isinstance(m, nn.LayerNorm) for m in b.modules()))
        self.assertTrue(all(m.bias is None for m in b.modules() if isinstance(m, nn.Linear)))
        self.assertIs(b.lm_head.weight, b.transformer.wte.weight)
        self.assertEqual(a.transformer.wpe.weight.shape, b.transformer.wpe.weight.shape)

    def test_both_variants_remain_causal(self):
        for variant in ('gpt2', 'rmsnorm_no_bias'):
            model = build_model(tiny_spec(), variant).eval()
            with torch.no_grad():
                a = model(torch.tensor([[1, 2, 3, 4]]))
                b = model(torch.tensor([[1, 2, 9, 8]]))
            torch.testing.assert_close(a[:, :2], b[:, :2], rtol=0, atol=0)

    def test_accumulation_matches_large_batch_parameter_update(self):
        for variant in ('gpt2', 'rmsnorm_no_bias'):
            spec = tiny_spec(); t = spec['training']
            a = build_model(spec, variant).double(); b = copy.deepcopy(a)
            oa = torch.optim.SGD(a.parameters(), lr=.03)
            ob = torch.optim.SGD(b.parameters(), lr=.03)
            ids = list(range(20))
            loader = SequentialBatches(ids, 1, 3)
            result = train_update(a, oa, loader, t, torch.device('cpu'))
            ob.zero_grad()
            x = torch.tensor([[0, 1, 2], [3, 4, 5]])
            y = torch.tensor([[1, 2, 3], [4, 5, 6]])
            loss = F.cross_entropy(b(x).flatten(0, 1), y.flatten())
            loss.backward(); ob.step()
            self.assertEqual(result['microbatch_starts'], [0, 3])
            for p, q in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(p, q, rtol=1e-10, atol=1e-12)
            self.assertTrue(all(p.grad is None for p in a.parameters()))

    def test_validation_preserves_parameters_gradients_rng_and_mode(self):
        spec = tiny_spec(); model = build_model(spec, 'rmsnorm_no_bias').train()
        for p in model.parameters(): p.grad = torch.ones_like(p)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        rng = torch.get_rng_state().clone()
        result = evaluate(model, list(range(20)), spec['training'], torch.device('cpu'))
        self.assertEqual(result['prediction_positions'], 6)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for n, p in model.named_parameters():
            self.assertTrue(torch.equal(p, before[n]))
            self.assertTrue(torch.equal(p.grad, torch.ones_like(p)))

    def test_loader_tail_and_learning_rate_boundaries(self):
        loader = SequentialBatches(list(range(8)), 1, 3)
        self.assertEqual([loader.next_batch()[2] for _ in range(3)], [0, 3, 0])
        self.assertEqual(loader.wraps, 1)
        self.assertAlmostEqual(learning_rate(1, 100, 10, .0006, .00006), .00006)
        self.assertAlmostEqual(learning_rate(10, 100, 10, .0006, .00006), .0006)
        self.assertAlmostEqual(learning_rate(100, 100, 10, .0006, .00006), .00006)

    def test_comparison_rejects_incompatible_experiments(self):
        a = {'completed': True, 'variant': 'gpt2', 'protocol': {'seed': 42},
             'training': [{'microbatch_starts': [0, 3]}],
             'summary': {'final_val_loss': 3., 'measured_tokens_per_second': 10.}}
        b = copy.deepcopy(a); b['variant'] = 'rmsnorm_no_bias'
        self.assertEqual(compare(a, b)['val_loss_delta_b_minus_a'], 0.)
        b['protocol']['seed'] = 43
        with self.assertRaises(ValueError): compare(a, b)
        b['protocol']['seed'] = 42; b['training'][0]['microbatch_starts'] = [3, 6]
        with self.assertRaises(ValueError): compare(a, b)


if __name__ == '__main__':
    unittest.main()
