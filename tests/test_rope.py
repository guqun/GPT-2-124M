"""位置处理数值、数据流与训练对照的行为检查。"""
import ast
import copy
from pathlib import Path
import unittest
import torch
from architecture_variants import build_model as old_build
from prepare_architecture_baseline import load_spec
from rope_model import apply_rotary, RotaryEmbedding, build_model, common_parameter_digest
from rope_training import train_update, SequentialBatches
from compare_rope_runs import compare


def spec():
    result=copy.deepcopy(load_spec('local'))
    result['model'].update(vocab_size=31,n_positions=8,n_embd=8,n_layer=2,n_head=2)
    result['training'].update(batch_size=1,seq_len=3,accumulation_steps=2,clip_norm=1e6)
    result['position_experiment']={'theta':10000.,'pairing':'adjacent','max_length':8}
    return result


class RopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_formula_matches_independent_complex_rotation_and_gradcheck(self):
        torch.manual_seed(4)
        x=torch.randn(2,3,5,4,dtype=torch.double,requires_grad=True)
        angles=torch.arange(5,dtype=torch.double)[:,None]*torch.tensor([1.,.01],dtype=torch.double)
        actual=apply_rotary(x,angles.cos(),angles.sin())
        reference=torch.view_as_real(torch.view_as_complex(x.reshape(2,3,5,2,2))*torch.polar(torch.ones_like(angles),angles)).flatten(-2)
        torch.testing.assert_close(actual,reference)
        self.assertTrue(torch.autograd.gradcheck(lambda a:apply_rotary(a,angles.cos(),angles.sin()),(x,)))
        torch.testing.assert_close(actual[:,:,0],x[:,:,0],rtol=0,atol=0)
        torch.testing.assert_close(actual.square().sum(-1),x.square().sum(-1))

    def test_relative_position_invariance(self):
        q=torch.tensor([1.,2.,3.,4.],dtype=torch.double)
        k=torch.tensor([2.,1.,0.,1.],dtype=torch.double)
        frequencies=torch.tensor([1.,.01],dtype=torch.double)
        def rotate(x,t):
            angle=t*frequencies
            return apply_rotary(x,angle.cos(),angle.sin())
        torch.testing.assert_close(rotate(q,1)@rotate(k,0),rotate(q,6)@rotate(k,5))

    def test_topology_and_paired_initialization(self):
        a,b=build_model(spec(),'absolute'),build_model(spec(),'rope')
        self.assertEqual(common_parameter_digest(a),common_parameter_digest(b))
        self.assertNotIn('wpe',b.transformer)
        self.assertIs(b.lm_head.weight,b.transformer.wte.weight)
        self.assertEqual(sum(p.numel() for p in a.parameters())-sum(p.numel() for p in b.parameters()),64)
        self.assertFalse(any('rope' in n for n,_ in b.named_parameters()))
        self.assertFalse(any('rope' in n for n in b.state_dict()))
        x=torch.tensor([[1,2,3]])
        torch.testing.assert_close(a(x),old_build(spec(),'rmsnorm_no_bias')(x),rtol=0,atol=0)

    def test_only_position_path_changed(self):
        a,b=build_model(spec(),'absolute'),build_model(spec(),'rope')
        # 禁用双方位置处理后，两边完整前向必须相同，包括 V/MLP/残差等。
        with torch.no_grad():
            a.transformer.wpe.weight.zero_()
            for block in b.transformer.h:
                block.attn.rope.cos.fill_(1);block.attn.rope.sin.zero_()
            x=torch.tensor([[1,2,3,4],[4,3,2,1]])
            torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)

    def test_causal_mask_and_length_limits(self):
        model=build_model(spec(),'rope').eval()
        with torch.no_grad():
            a=model(torch.tensor([[1,2,3,4]]));b=model(torch.tensor([[1,2,9,8]]))
        torch.testing.assert_close(a[:,:2],b[:,:2],rtol=0,atol=0)
        with self.assertRaises(ValueError): model(torch.ones(1,9,dtype=torch.long))
        with self.assertRaises(ValueError): RotaryEmbedding(3,8)

    def test_accumulation_matches_large_batch(self):
        s=spec(); a=build_model(s,'rope').double();b=copy.deepcopy(a)
        oa=torch.optim.SGD(a.parameters(),lr=.03);ob=torch.optim.SGD(b.parameters(),lr=.03)
        train_update(a,oa,SequentialBatches(list(range(20)),1,3),s['training'],torch.device('cpu'))
        loss=torch.nn.functional.cross_entropy(b(torch.tensor([[0,1,2],[3,4,5]])).flatten(0,1),torch.tensor([1,2,3,4,5,6]))
        loss.backward();ob.step()
        for p,q in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(p,q,rtol=1e-10,atol=1e-12)

    def test_training_helpers_identical_to_previous_lesson(self):
        root=Path(__file__).resolve().parents[1]
        trees=[ast.parse((root/name).read_text()) for name in ('architecture_training.py','rope_training.py')]
        for name in ('SequentialBatches','learning_rate','synchronize','evaluate','train_update','resolve_device'):
            nodes=[next(n for n in tree.body if getattr(n,'name',None)==name) for tree in trees]
            self.assertEqual(ast.dump(nodes[0]),ast.dump(nodes[1]),name)

    def test_comparator_rejects_other_protocol(self):
        a={'completed':True,'variant':'absolute','protocol':{'theta':10000},'training':[],
           'summary':{'final_val_loss':2.,'measured_tokens_per_second':10.}}
        b=copy.deepcopy(a);b['variant']='rope'
        self.assertEqual(compare(a,b)['val_loss_delta_b_minus_a'],0.)
        b['protocol']['theta']=500000
        with self.assertRaises(ValueError):compare(a,b)


if __name__=='__main__':unittest.main()
