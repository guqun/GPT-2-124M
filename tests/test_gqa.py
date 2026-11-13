"""GQA 独立重复 KV 对照、梯度、缓存分块与旧 MHA 回归。"""
import copy
import math
import unittest
import torch
from prepare_architecture_baseline import load_spec
from swiglu_model import build_model
from gqa_model import convert_attention, GroupedDecoder, grouped_attention
from kv_cache_model import cache_bytes
from kv_cache_experiment import generate, parameter_digest


def model(kv=2):
    spec = copy.deepcopy(load_spec('local'))
    spec['model'].update(vocab_size=31,n_positions=12,n_embd=16,n_layer=2,n_head=4)
    spec['position_experiment'] = {'theta':10000.}
    base = build_model(spec,'swiglu').eval()
    return base if kv is None else convert_attention(base,kv).eval()


class GQATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_four_kv_heads_preserve_original_mha(self):
        base = model(None)
        ids = torch.tensor([[1,2,3,4],[4,3,2,1]])
        with torch.inference_mode():
            expected = base(ids)
            actual = convert_attention(base,4)(ids)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-6)

    def test_grouped_kernel_matches_repeated_kv_and_gradients(self):
        # 测试独立使用 repeat_interleave；实现使用五维广播，防止同一种轴错误自证。
        torch.manual_seed(7)
        for hkv in (1,2,4):
            q = torch.randn(2,4,2,4,dtype=torch.float64,requires_grad=True)
            k = torch.randn(2,hkv,5,4,dtype=torch.float64,requires_grad=True)
            v = torch.randn_like(k,requires_grad=True)
            q2,k2,v2 = [x.detach().clone().requires_grad_() for x in (q,k,v)]
            actual = grouped_attention(q,k,v,3)
            kr,vr = [x.repeat_interleave(4//hkv,dim=1) for x in (k2,v2)]
            scores = q2 @ kr.transpose(-2,-1)/2
            mask = torch.tensor([[True,True,True,True,False],[True]*5])
            expected = scores.masked_fill(~mask,float('-inf')).softmax(-1) @ vr
            torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
            probe = torch.randn_like(actual)
            ga = torch.autograd.grad((actual*probe).sum(),(q,k,v))
            gb = torch.autograd.grad((expected*probe).sum(),(q2,k2,v2))
            for a,b in zip(ga,gb):
                torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)

    def test_weight_pooling_and_full_input_projection(self):
        base = model(None)
        old = base.transformer.h[0].attn.c_attn.weight.detach().clone()
        convert_attention(base,2)
        a = base.transformer.h[0].attn
        torch.testing.assert_close(a.q_proj.weight,old[:16],rtol=0,atol=0)
        for layer,source in ((a.k_proj,old[16:32]),(a.v_proj,old[32:])):
            expected = torch.stack(((source[0:4]+source[4:8])/2,(source[8:12]+source[12:16])/2)).reshape(8,16)
            torch.testing.assert_close(layer.weight,expected,rtol=0,atol=0)
        self.assertEqual(a.k_proj.in_features,16)

    def test_chunks_rope_causality_and_cache_storage(self):
        ids = torch.tensor([[1,2,3,4,5,6],[6,5,4,3,2,1]])
        for kv in (1,2,4):
            base = model(kv); dec = GroupedDecoder(base).eval()
            with torch.inference_mode(): expected = base(ids)
            for sizes in ([6],[1]*6,[3,2,1]):
                cache=None;start=0;pieces=[]
                for n in sizes:
                    old=cache
                    piece,cache=dec(ids[:,start:start+n],cache,last_only=False)
                    pieces.append(piece); start+=n
                    self.assertEqual(cache[0][0].shape,(2,kv,start,4))
                    if old is not None:
                        for op,np in zip(old,cache):
                            for a,b in zip(op,np):
                                torch.testing.assert_close(b[:,:,:a.shape[2]],a,rtol=0,atol=0)
                torch.testing.assert_close(torch.cat(pieces,1),expected,rtol=1e-4,atol=1e-6)
                self.assertEqual(cache_bytes(cache),2*2*2*kv*6*4*4)
                self.assertEqual(sum(a.untyped_storage().nbytes() for p in cache for a in p),cache_bytes(cache))
            _,past=dec(ids[:,:3])
            a,_=dec(ids[:,3:5],past,last_only=False)
            changed=ids[:,3:5].clone();changed[:,1]=9
            b,_=dec(changed,past,last_only=False)
            torch.testing.assert_close(a[:,:1],b[:,:1],rtol=0,atol=0)

    def test_generation_reset_and_no_weight_updates(self):
        dec=GroupedDecoder(model()).eval();ids=torch.tensor([[1,2,3]])
        before=parameter_digest(dec.model)
        a,za,_=generate(dec,ids,4,False)
        b,zb,rows=generate(dec,ids,4,True)
        self.assertTrue(torch.equal(a,b))
        torch.testing.assert_close(za,zb,rtol=1e-4,atol=1e-6)
        self.assertEqual([r['input_positions'] for r in rows],[3,1,1,1])
        self.assertEqual(rows[-1]['cache_after'],6)
        c,zc,_=generate(dec,ids,4,True)
        self.assertTrue(torch.equal(b,c));torch.testing.assert_close(zb,zc,rtol=0,atol=0)
        self.assertEqual(before,parameter_digest(dec.model))
        self.assertTrue(all(p.grad is None for p in dec.parameters()))

    def test_bad_cache_and_capacity(self):
        for invalid in (0,3,5):
            with self.assertRaises(ValueError): model(invalid)
        dec=GroupedDecoder(model()).eval()
        _,cache=dec(torch.ones(1,11,dtype=torch.long))
        dec(torch.ones(1,1,dtype=torch.long),cache)
        for ids,old in ((torch.ones(1,2,dtype=torch.long),cache),
                        (torch.ones(2,1,dtype=torch.long),cache),
                        (torch.ones(1,1,dtype=torch.long),cache[:1]),
                        (torch.ones(1,1,dtype=torch.long),tuple((k.repeat_interleave(2,1),v.repeat_interleave(2,1)) for k,v in cache))):
            with self.assertRaises(ValueError): dec(ids,old)
        with self.assertRaises(ValueError): dec(torch.ones(1,1,dtype=torch.long),cache,use_cache=False)
        dec.train()
        with self.assertRaises(ValueError): dec(torch.ones(1,1,dtype=torch.long))


if __name__=='__main__': unittest.main()
