"""QK-Norm 的统计轴、参数共享、顺序、缓存与梯度行为。"""
import copy
import unittest
import torch
from prepare_architecture_baseline import load_spec
from qknorm_model import HeadRMSNorm,build_model,add_qk_norm
from swiglu_model import build_model as build_swiglu
from gqa_model import convert_attention,GroupedDecoder
from kv_cache_model import cache_bytes
from kv_cache_experiment import parameter_digest
from qknorm_experiment import gradient_probe,common_digest


def spec():
    s=copy.deepcopy(load_spec('local'))
    s['model'].update(vocab_size=31,n_positions=12,n_embd=16,n_layer=2,n_head=4)
    s['position_experiment']={'theta':10000.}
    return s


class QKNormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_last_axis_manual_values_and_gamma_gradient(self):
        norm=HeadRMSNorm(2,1e-5)
        with torch.no_grad(): norm.weight.copy_(torch.tensor([2.,.5]))
        x=torch.tensor([[[[3.,4.],[0.,0.]],[[6.,8.],[-3.,4.]]]],requires_grad=True)
        y=norm(x)
        expected=torch.empty_like(x)
        normalized=torch.empty_like(x)
        # 独立逐行手算；每个头/位置分别统计，不能跨头/位置求 mean。
        for h in range(2):
            for t in range(2):
                r=((float(x[0,h,t,0])**2+float(x[0,h,t,1])**2)/2+1e-5)**.5
                normalized[0,h,t]=x.detach()[0,h,t]/r
                expected[0,h,t]=normalized[0,h,t]*torch.tensor([2.,.5])
        torch.testing.assert_close(y,expected)
        y.sum().backward()
        torch.testing.assert_close(norm.weight.grad,normalized.sum((0,1,2)))
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(list(norm.weight.shape),[2])
        for eps in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError): HeadRMSNorm(2,eps)

    def test_reference_is_old_gqa_and_common_weights_match(self):
        s=spec();original=convert_attention(build_swiglu(s,'swiglu'),2).eval()
        ids=torch.tensor([[1,2,3,4]])
        with torch.inference_mode(): expected=original(ids)
        reference=build_model(s,False).eval();changed=build_model(s,True).eval()
        with torch.inference_mode(): actual=reference(ids)
        torch.testing.assert_close(expected,actual,rtol=0,atol=0)
        self.assertEqual(common_digest(reference),common_digest(changed))
        self.assertEqual(sum(p.numel() for p in changed.parameters())-sum(p.numel() for p in reference.parameters()),16)
        a,b=changed.transformer.h
        self.assertIsNot(a.attn.q_norm.weight,a.attn.k_norm.weight)
        self.assertIsNot(a.attn.q_norm.weight,b.attn.q_norm.weight)
        self.assertTrue(torch.equal(a.attn.q_norm.weight,torch.ones(4)))

    def test_nonuniform_gamma_before_rope_and_cached_v_untouched(self):
        model=build_model(spec(),True).eval();attn=model.transformer.h[0].attn
        with torch.no_grad():
            attn.q_norm.weight.copy_(torch.tensor([1.,2.,.5,3.]))
            attn.k_norm.weight.copy_(torch.tensor([2.,.5,3.,1.]))
        x=torch.randn(2,4,16)
        with torch.inference_mode():
            _,past=attn.step(x[:,:2])
            branch,cache=attn.step(x[:,2:],past)
            q=attn.q_proj(x[:,2:]).reshape(2,2,4,4).transpose(1,2)
            k=attn.k_proj(x[:,2:]).reshape(2,2,2,4).transpose(1,2)
            v=attn.v_proj(x[:,2:]).reshape(2,2,2,4).transpose(1,2)
            # 独立 RMS 公式 + 复数 RoPE，避免只把被测方法再调用一次。
            norm=lambda a,g:a/(a.square().sum(-1,keepdim=True)/4+1e-5).sqrt()*g
            def rotate(a):
                angles=torch.tensor([2.,3.])[:,None]*torch.tensor([1.,.01])[None,:]
                return torch.view_as_real(torch.view_as_complex(a.contiguous().reshape(*a.shape[:-1],2,2))*torch.polar(torch.ones_like(angles),angles)).flatten(-2)
            q=rotate(norm(q,attn.q_norm.weight));k=rotate(norm(k,attn.k_norm.weight))
            torch.testing.assert_close(cache[0][:,:,2:],k)
            torch.testing.assert_close(cache[1][:,:,2:],v,rtol=0,atol=0)
            torch.testing.assert_close(cache[0][:,:,:2],past[0],rtol=0,atol=0)
            fullk=torch.cat((past[0],k),2).repeat_interleave(2,1)
            fullv=torch.cat((past[1],v),2).repeat_interleave(2,1)
            mask=torch.tensor([[True,True,True,False],[True]*4])
            scores=(q@fullk.transpose(-2,-1)/2).masked_fill(~mask,float('-inf'))
            h=scores.softmax(-1)@fullv
            expected=attn.c_proj(h.transpose(1,2).contiguous().reshape(2,2,16))
            torch.testing.assert_close(branch,expected,rtol=1e-4,atol=1e-6)

    def test_chunked_cache_matches_full_with_learned_scales(self):
        m=build_model(spec(),True).eval()
        with torch.no_grad():
            for block in m.transformer.h:
                block.attn.q_norm.weight.copy_(torch.tensor([.7,1.3,.9,2.]))
                block.attn.k_norm.weight.copy_(torch.tensor([1.2,.8,1.5,.5]))
        dec=GroupedDecoder(m).eval();ids=torch.tensor([[1,2,3,4,5,6],[6,5,4,3,2,1]])
        with torch.inference_mode(): expected=m(ids)
        for lengths in ([6],[1]*6,[3,2,1]):
            cache=None;start=0;pieces=[]
            for n in lengths:
                z,cache=dec(ids[:,start:start+n],cache,last_only=False)
                pieces.append(z);start+=n
            torch.testing.assert_close(torch.cat(pieces,1),expected,rtol=1e-4,atol=1e-6)
            self.assertEqual(cache[0][0].shape,(2,2,6,4))
            self.assertEqual(cache_bytes(cache),2*2*2*2*6*4*4)
        _,past=dec(ids[:,:3]);z,_=dec(ids[:,3:5],past,last_only=False)
        change=ids[:,3:5].clone();change[:,1]=12
        z2,_=dec(change,past,last_only=False)
        torch.testing.assert_close(z[:,:1],z2[:,:1],rtol=0,atol=0)

    def test_gamma_updates_only_on_training_copy(self):
        m=build_model(spec(),True).eval();before=parameter_digest(m)
        result=gradient_probe(m)
        self.assertEqual(len(result['gamma']),4)
        self.assertTrue(all(g['gradient_l2']>0 and g['max_abs_update']>0 for g in result['gamma'].values()))
        self.assertEqual(before,parameter_digest(m))
        self.assertTrue(all(p.grad is None for p in m.parameters()))
        self.assertFalse(m.training)

    def test_cache_limits_and_mode_guard_still_apply(self):
        dec=GroupedDecoder(build_model(spec(),True)).eval()
        _,cache=dec(torch.ones(1,11,dtype=torch.long))
        dec(torch.ones(1,1,dtype=torch.long),cache)
        with self.assertRaises(ValueError): dec(torch.ones(1,2,dtype=torch.long),cache)
        with self.assertRaises(ValueError): dec(torch.ones(1,1,dtype=torch.long),cache[:1])
        dec.train()
        with self.assertRaises(ValueError): dec(torch.ones(1,1,dtype=torch.long))


if __name__=='__main__': unittest.main()
