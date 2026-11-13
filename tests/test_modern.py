"""整合行为：与逐课模型对齐、梯度累积、因果性、分块缓存、检查点。"""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
import torch
from modern_model import ModernConfig,ModernLM,generate
from qknorm_model import build_model
from prepare_architecture_baseline import load_spec
from train_modern import make_optimizer,load_checkpoint,generation_check


def config(**kw):
    args=dict(vocab_size=31,max_length=16,width=16,layers=2,q_heads=4,kv_heads=2,intermediate=43)
    args.update(kw)
    return ModernConfig(**args)


class ModernTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_previous_model_forward_and_all_gradients(self):
        s=copy.deepcopy(load_spec('local'))
        s['model'].update(vocab_size=31,n_positions=16,n_embd=16,n_layer=2,n_head=4)
        s['position_experiment']={'theta':10000.}
        old=build_model(s,True);new=ModernLM(config())
        new.load_state_dict(old.state_dict(),strict=True)
        ids=torch.tensor([[1,2,3,4],[4,3,2,1]])
        a,b=old(ids),new(ids)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        a.square().mean().backward();b.square().mean().backward()
        oldp=dict(old.named_parameters());newp=dict(new.named_parameters())
        self.assertEqual(set(oldp),set(newp))
        for name in oldp:
            torch.testing.assert_close(oldp[name].grad,newp[name].grad,rtol=1e-4,atol=1e-6)

    def test_cache_chunking_causality_and_greedy(self):
        m=ModernLM(config()).eval();ids=torch.tensor([[1,2,3,4,5,6],[6,5,4,3,2,1]])
        with torch.inference_mode():
            full=m(ids)
            for sizes in ([6],[1]*6,[3,2,1]):
                cache=None;offset=0;pieces=[]
                for n in sizes:
                    z,cache=m(ids[:,offset:offset+n],cache,use_cache=True)
                    pieces.append(z);offset+=n
                torch.testing.assert_close(full,torch.cat(pieces,1),rtol=1e-4,atol=1e-6)
                self.assertEqual(cache[0][0].shape,(2,2,6,4))
            changed=ids.clone();changed[:,-1]=15
            torch.testing.assert_close(m(changed)[:,:-1],full[:,:-1],rtol=0,atol=0)
        _,r=generation_check(m,ids[:,:3],4)
        self.assertTrue(r['greedy_ids_equal'])
        self.assertEqual(r['cache_shape_per_layer'],[2,2,6,4])

    def test_accumulation_equals_full_batch_and_gamma_updates(self):
        torch.manual_seed(3);a=ModernLM(config());b=copy.deepcopy(a)
        ids=torch.tensor([[1,2,3],[4,5,6]]);y=ids+1
        t=dict(weight_decay=.1,peak_lr=.001,betas=[.9,.95],adam_eps=1e-8)
        opt=make_optimizer(a,t)
        before={n:p.detach().clone() for n,p in a.named_parameters() if '_norm.' in n}
        for i in range(2):
            (torch.nn.functional.cross_entropy(a(ids[i:i+1]).flatten(0,1),y[i:i+1].flatten())/2).backward()
        torch.nn.functional.cross_entropy(b(ids).flatten(0,1),y.flatten()).backward()
        for pa,pb in zip(a.parameters(),b.parameters()): torch.testing.assert_close(pa.grad,pb.grad,rtol=1e-4,atol=1e-6)
        opt.step()
        self.assertTrue(all(not torch.equal(before[n],p) for n,p in a.named_parameters() if n in before))
        self.assertIs(a.lm_head.weight,a.transformer.wte.weight)
        params=[p for g in opt.param_groups for p in g['params']]
        self.assertEqual(len(params),len(set(id(p) for p in params)))

    def test_checkpoint_reload_and_untied_config(self):
        for tie in (True,False):
            c=config(tie_embeddings=tie);m=ModernLM(c).eval();ids=torch.tensor([[1,2]])
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'model.pt'
                torch.save(dict(format='modern-teaching-v1',model_config=asdict(c),model=m.state_dict()),path)
                loaded,_=load_checkpoint(path,torch.device('cpu'))
                torch.testing.assert_close(m(ids),loaded(ids),rtol=0,atol=0)
                self.assertEqual(loaded.lm_head.weight is loaded.transformer.wte.weight,tie)

    def test_guards(self):
        with self.assertRaises(ValueError): config(kv_heads=3)
        with self.assertRaises(ValueError): config(width=12)
        m=ModernLM(config());ids=torch.ones(1,2,dtype=torch.long)
        with self.assertRaises(ValueError):m(ids,use_cache=True)
        m.eval()
        with self.assertRaises(ValueError):m(ids,use_cache=True)
        with torch.inference_mode():
            _,cache=m(ids,use_cache=True)
            for bad in (cache[:1],((cache[0][0][:,:1],cache[0][1]),cache[1])):
                with self.assertRaises(ValueError):m(ids,bad,use_cache=True)
            with self.assertRaises(ValueError):m(torch.ones(1,15,dtype=torch.long),cache,use_cache=True)
            with self.assertRaises(ValueError):m(ids,cache)
        m.train();generate(m,ids,2);self.assertTrue(m.training)


if __name__=='__main__': unittest.main()
