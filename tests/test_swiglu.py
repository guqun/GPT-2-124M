"""检验门控数值/梯度、替换范围和训练协议；小模型避免训练时间噪声。"""
import ast
import copy
from pathlib import Path
import unittest
import torch
from gpt2_model import GPTConfig
from prepare_architecture_baseline import load_spec
from rope_model import build_model as build_rope
from swiglu_model import SwiGLU, build_model, common_parameter_digest, experiment_settings
from swiglu_training import SequentialBatches, train_update
from compare_swiglu_runs import compare


def spec():
    result = copy.deepcopy(load_spec('local'))
    result['model'].update(vocab_size=31, n_positions=8, n_embd=8, n_layer=2, n_head=2)
    result['training'].update(batch_size=1, seq_len=3, accumulation_steps=2, clip_norm=1e6)
    result['position_experiment'] = {'theta':10000., 'pairing':'adjacent', 'max_length':8}
    return result


class SwiGLUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_numeric_example_and_parameter_gradients(self):
        m = SwiGLU(GPTConfig(n_embd=2, n_layer=2, resid_pdrop=0), 2).double()
        with torch.no_grad():
            m.gate_proj.weight.copy_(torch.tensor([[1.,0.],[0.,-1.]]))
            m.up_proj.weight.copy_(torch.tensor([[2.,0.],[0.,3.]]))
            m.down_proj.weight.copy_(torch.tensor([[1.,1.],[1.,-1.]]))
        z = torch.tensor([[[1.,2.]]], dtype=torch.double, requires_grad=True)
        # 独立标量表达式，而不是再调用被测模块。
        gate0 = 1/(1+__import__('math').exp(-1))
        gate1 = -2/(1+__import__('math').exp(2))
        expected = torch.tensor([[[gate0*2+gate1*6, gate0*2-gate1*6]]], dtype=torch.double)
        torch.testing.assert_close(m(z), expected)
        self.assertTrue(torch.autograd.gradcheck(m, (z,)))
        m(z).sum().backward()
        # 两个输出求和时第二门控分量抵消；第一分量的梯度为 2。
        s = gate0
        torch.testing.assert_close(m.up_proj.weight.grad,
            torch.tensor([[2*s,4*s],[0.,0.]], dtype=torch.double))
        torch.testing.assert_close(m.gate_proj.weight.grad,
            torch.tensor([[4*(s+s*(1-s)),8*(s+s*(1-s))],[0.,0.]], dtype=torch.double))
        torch.testing.assert_close(m.down_proj.weight.grad,
            torch.tensor([[gate0*2,gate1*6],[gate0*2,gate1*6]], dtype=torch.double))

    def test_parameters_and_unchanged_backbone(self):
        s=spec(); a=build_model(s,'gelu'); b=build_model(s,'swiglu')
        self.assertEqual(common_parameter_digest(a), common_parameter_digest(b))
        self.assertIs(b.lm_head.weight, b.transformer.wte.weight)
        self.assertNotIn('wpe', b.transformer)
        self.assertFalse(any(n.endswith('.bias') for n,_ in b.named_parameters()))
        settings=experiment_settings(128)
        self.assertEqual(settings['swiglu_hidden'],341)
        self.assertEqual(settings['swiglu_parameters_per_block'],130944)
        self.assertEqual(settings['gelu_parameters_per_block'],131072)
        z=torch.tensor([[1,2,3]])
        torch.testing.assert_close(a(z), build_rope(s,'rope')(z), rtol=0,atol=0)
        # 把旧 MLP 放回 B，应恢复 A 的完整前向：证明没有暗改其它环节。
        for left,right in zip(a.transformer.h,b.transformer.h):
            right.mlp=copy.deepcopy(left.mlp)
        torch.testing.assert_close(a(z),b(z),rtol=0,atol=0)

    def test_causality_checkpoint_and_accumulation(self):
        s=spec(); a=build_model(s,'swiglu').double(); b=copy.deepcopy(a)
        x=torch.tensor([[0,1,2],[3,4,5]])
        torch.testing.assert_close(a(x)[:,:2],a(torch.tensor([[0,1,9],[3,4,9]]))[:,:2],rtol=0,atol=0)
        oa=torch.optim.SGD(a.parameters(),lr=.03); ob=torch.optim.SGD(b.parameters(),lr=.03)
        train_update(a,oa,SequentialBatches(list(range(20)),1,3),s['training'],torch.device('cpu'))
        torch.nn.functional.cross_entropy(b(x).flatten(0,1),torch.tensor([1,2,3,4,5,6])).backward()
        ob.step()
        for p,q in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(p,q,rtol=1e-10,atol=1e-12)
        restored=build_model(s,'swiglu').double()
        restored.load_state_dict(a.state_dict(),strict=True)
        torch.testing.assert_close(restored(x),a(x),rtol=0,atol=0)

    def test_training_helpers_unchanged(self):
        root=Path(__file__).resolve().parents[1]
        trees=[ast.parse((root/n).read_text()) for n in ('rope_training.py','swiglu_training.py')]
        for name in ('SequentialBatches','learning_rate','synchronize','evaluate','train_update','resolve_device'):
            nodes=[next(n for n in tree.body if getattr(n,'name',None)==name) for tree in trees]
            self.assertEqual(ast.dump(nodes[0]),ast.dump(nodes[1]),name)

    def test_protocol_rejection(self):
        a={'completed':True,'variant':'gelu','protocol':{'seed':42},'training':[],
           'summary':{'final_val_loss':2.,'measured_tokens_per_second':10.}}
        b=copy.deepcopy(a); b['variant']='swiglu'
        self.assertEqual(compare(a,b)['val_loss_delta_b_minus_a'],0.)
        b['protocol']['seed']=43
        with self.assertRaises(ValueError): compare(a,b)


if __name__=='__main__': unittest.main()
