"""检验缓存是否保持原模型计算，以及 RoPE 偏移/长方形遮罩/请求重置。"""
import copy
import unittest
import torch
from prepare_architecture_baseline import load_spec
from swiglu_model import build_model
from kv_cache_model import CachedDecoder,cache_bytes
from kv_cache_experiment import generate,parameter_digest


def model():
    spec=copy.deepcopy(load_spec('local'))
    spec['model'].update(vocab_size=31,n_positions=12,n_embd=8,n_layer=2,n_head=2)
    spec['position_experiment']={'theta':10000.,'pairing':'adjacent','max_length':12}
    return build_model(spec,'swiglu').eval()


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_full_forward_equals_original_model(self):
        base=model();dec=CachedDecoder(base).eval()
        ids=torch.tensor([[1,2,3,4],[4,3,2,1]])
        with torch.inference_mode(): expected=base(ids)
        for use in (False,True):
            result,cache=dec(ids,use_cache=use,last_only=False)
            torch.testing.assert_close(result,expected,rtol=0,atol=0)
            self.assertEqual(cache is not None,use)

    def test_prefill_single_and_multi_token_chunks(self):
        base=model();dec=CachedDecoder(base).eval()
        ids=torch.tensor([[1,2,3,4,5,6],[6,5,4,3,2,1]])
        with torch.inference_mode():expected=base(ids)
        # 全序列、全单步，以及先 3 后 2 再 1：覆盖带缓存的矩形因果遮罩。
        for lengths in ([6],[1,1,1,1,1,1],[3,2,1]):
            cache=None;start=0;pieces=[]
            for n in lengths:
                old=None if cache is None else tuple((k.clone(),v.clone()) for k,v in cache)
                piece,cache=dec(ids[:,start:start+n],cache,last_only=False)
                pieces.append(piece);start+=n
                self.assertEqual(cache[0][0].shape,(2,2,start,4))
                if old is not None:
                    for (ok,ov),(k,v) in zip(old,cache):
                        torch.testing.assert_close(k[:,:,:ok.shape[2]],ok,rtol=0,atol=0)
                        torch.testing.assert_close(v[:,:,:ov.shape[2]],ov,rtol=0,atol=0)
            torch.testing.assert_close(torch.cat(pieces,dim=1),expected,rtol=1e-4,atol=1e-6)
            self.assertEqual(cache_bytes(cache),2*2*2*2*6*4*4)

    def test_rope_offset_and_rectangular_mask(self):
        # 直接手算第二次调用的角度与遮罩，核对捕获到的缓存 K/V。
        base=model();dec=CachedDecoder(base).eval();ids=torch.tensor([[1,2,3,4]])
        _,past=dec(ids[:,:2]);_,cache=dec(ids[:,2:],past)
        a=base.transformer.h[0].attn
        with torch.inference_mode():
            z=base.transformer.h[0].ln_1(base.transformer.wte(ids[:,2:]))
            _,k,v=a.c_attn(z).split(8,dim=-1)
            k=k.view(1,2,2,4).transpose(1,2);v=v.view(1,2,2,4).transpose(1,2)
            angles=torch.tensor([2.,3.])[:,None]*torch.tensor([1.,.01])[None,:]
            rotated=torch.view_as_real(torch.view_as_complex(k.contiguous().reshape(1,2,2,2,2))*
                torch.polar(torch.ones_like(angles),angles)).flatten(-2)
            torch.testing.assert_close(cache[0][0][:,:,2:],rotated)
            torch.testing.assert_close(cache[0][1][:,:,2:],v,rtol=0,atol=0)
        a,_=dec(ids[:,2:],past,last_only=False)
        b,_=dec(torch.tensor([[3,9]]),past,last_only=False)
        torch.testing.assert_close(a[:,:1],b[:,:1],rtol=0,atol=0)

    def test_generation_counts_reset_and_parameter_isolation(self):
        dec=CachedDecoder(model()).eval();x=torch.tensor([[1,2,3]])
        before=parameter_digest(dec.model)
        a,za,ra=generate(dec,x,4,False);b,zb,rb=generate(dec,x,4,True)
        self.assertTrue(torch.equal(a,b));torch.testing.assert_close(za,zb,rtol=1e-4,atol=1e-6)
        self.assertEqual([r['input_positions'] for r in ra],[3,4,5,6])
        self.assertEqual([r['input_positions'] for r in rb],[3,1,1,1])
        self.assertEqual(rb[-1]['cache_after'],6)  # 最后生成的第 7 个位置尚未输入。
        c,zc,_=generate(dec,x,4,True)
        self.assertTrue(torch.equal(c,b));torch.testing.assert_close(zc,zb,rtol=0,atol=0)
        self.assertEqual(before,parameter_digest(dec.model))
        self.assertTrue(all(p.grad is None for p in dec.parameters()))

    def test_boundaries_and_bad_caches(self):
        dec=CachedDecoder(model()).eval()
        _,cache=dec(torch.ones(1,11,dtype=torch.long))
        dec(torch.ones(1,1,dtype=torch.long),cache)  # 总长刚好 12。
        with self.assertRaises(ValueError): dec(torch.ones(1,2,dtype=torch.long),cache)
        with self.assertRaises(ValueError): dec(torch.ones(2,1,dtype=torch.long),cache)
        with self.assertRaises(ValueError): dec(torch.ones(1,1,dtype=torch.long),cache[:1])
        with self.assertRaises(ValueError): dec(torch.ones(1,1,dtype=torch.long),cache,use_cache=False)
        dec.train()
        with self.assertRaises(ValueError): dec(torch.ones(1,1,dtype=torch.long))


if __name__=='__main__':unittest.main()
