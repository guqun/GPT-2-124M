"""基线数据隔离、预算与真实模型参数计数；不要求训练 loss 每步下降。"""
import unittest
import torch
from gpt2_model import GPT, GPTConfig
from prepare_architecture_baseline import load_spec, describe, split_tokens, first_batch


class BaselineTests(unittest.TestCase):
    def test_split_before_targets_prevents_validation_leak(self):
        train, val, cut = split_tokens(list(range(100)))
        self.assertEqual(cut,90)
        x,y=first_batch(train,2,4,start=81)
        self.assertEqual(y[-1,-1].item(),89)
        self.assertNotIn(90,y.flatten().tolist())
        with self.assertRaises(ValueError): first_batch(train,2,4,start=82)
        vx,vy=first_batch(val,2,4)
        self.assertEqual(vx[0,0].item(),90)
        self.assertTrue(torch.equal(vx.flatten()[1:],vy.flatten()[:-1]))

    def test_local_budget_covers_all_accumulated_positions(self):
        spec=load_spec('local'); report=describe(spec)
        self.assertEqual(report['tokens_per_microbatch'],256)
        self.assertEqual(report['tokens_per_update'],1024)
        self.assertEqual(report['planned_training_tokens'],102400)
        self.assertEqual(report['validation_tokens'],2048)

    def test_parameter_formula_matches_registered_shared_parameters(self):
        spec=load_spec('local')
        model=GPT(GPTConfig(**spec['model']))
        self.assertIs(model.lm_head.weight,model.transformer.wte.weight)
        self.assertEqual(sum(p.numel() for p in model.parameters()),describe(spec)['parameters'])
        self.assertEqual(describe(load_spec('formal'))['parameters'],124439808)

    def test_formal_cannot_mislabel_demo_as_ready(self):
        spec=load_spec('formal')
        self.assertEqual(spec['data']['kind'],'fineweb_edu_shards_pending')
        self.assertIsNone(spec['data']['manifest'])
        self.assertIsNone(spec['data']['revision'])
        self.assertEqual(describe(spec)['planned_training_tokens'],32768000)


if __name__=='__main__':
    unittest.main()
