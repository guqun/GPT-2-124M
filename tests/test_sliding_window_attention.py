"""窗口语义、局部可见性、独立逐行参考及反向传播检查。"""
import unittest

import torch

from compare_attention_windows import project_once, output_projection, worked_example
from gqa_model import grouped_attention
from modern_model import ModernAttention, ModernConfig
from sliding_window_attention import attention_with_window


class WindowAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def inputs(self, t=5, kv=2):
        return (torch.randn(2, 4, t, 4, dtype=torch.float64, requires_grad=True),
                torch.randn(2, kv, t, 4, dtype=torch.float64, requires_grad=True),
                torch.randn(2, kv, t, 4, dtype=torch.float64, requires_grad=True))

    def test_global_and_covering_window_match_previous_kernel(self):
        for kv in (1, 2, 4):
            q, k, v = self.inputs(kv=kv)
            expected = grouped_attention(q, k, v, 0)
            for w in (None, 5, 8):
                torch.testing.assert_close(attention_with_window(q, k, v, w).heads,
                                           expected, rtol=0, atol=0)

    def test_row_slice_reference_and_gradients(self):
        q, k, v = self.inputs()
        actual = attention_with_window(q, k, v, 3)
        # 独立参考：逐头逐位置，只取可见 K/V 的切片，不构造全局遮罩。
        rows = []
        for h in range(4):
            head = []
            for t in range(5):
                lo = max(0, t-2)
                scores = (q[:, h, t:t+1] @ k[:, h//2, lo:t+1].transpose(-2, -1)) / 2
                head.append(scores.softmax(-1) @ v[:, h//2, lo:t+1])
            rows.append(torch.cat(head, dim=1))
        expected = torch.stack(rows, dim=1)
        torch.testing.assert_close(actual.heads, expected, rtol=1e-12, atol=1e-12)
        probe = torch.randn_like(expected)
        ga = torch.autograd.grad((actual.heads*probe).sum(), (q,k,v), retain_graph=True)
        gb = torch.autograd.grad((expected*probe).sum(), (q,k,v))
        for a,b in zip(ga,gb):
            torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)

    def test_masked_positions_have_no_direct_effect_or_gradient(self):
        q,k,v = self.inputs()
        trace = attention_with_window(q,k,v,3)
        self.assertEqual(trace.allowed[3].tolist(), [False,True,True,True,False])
        # 查询D不能读A，也不能读未来E；改变它们的K/V不影响本层D输出。
        k2,v2 = k.detach().clone(),v.detach().clone()
        k2[:,:, [0,4]] += 100
        v2[:,:, [0,4]] -= 100
        changed = attention_with_window(q,k2,v2,3)
        torch.testing.assert_close(trace.heads[:,:,3],changed.heads[:,:,3],rtol=0,atol=0)
        gs,gk,gv = torch.autograd.grad(trace.heads[:,:,3].sum(),(trace.scores,k,v))
        for gradient in (gs[:,:,3,[0,4]],gk[:,:,[0,4]],gv[:,:,[0,4]]):
            self.assertEqual(torch.count_nonzero(gradient).item(),0)
        torch.testing.assert_close(trace.weights.sum(-1),torch.ones_like(trace.weights.sum(-1)))

    def test_single_token_window_one_and_invalid_arguments(self):
        for t in (1,2,5):
            q,k,v = self.inputs(t)
            trace = attention_with_window(q,k,v,1)
            torch.testing.assert_close(trace.heads,v.repeat_interleave(2,dim=1),rtol=0,atol=0)
            if t <= 2:
                torch.testing.assert_close(attention_with_window(q,k,v,3).heads,
                                           grouped_attention(q,k,v,0),rtol=0,atol=0)
        for bad in (0,-1,True,1.5):
            with self.assertRaises(ValueError): attention_with_window(q,k,v,bad)

    def test_shared_projection_backward_and_worked_example(self):
        c = ModernConfig(width=16,q_heads=4,kv_heads=2,max_length=16)
        layer = ModernAttention(c).double().eval()
        u = torch.randn(1,5,16,dtype=torch.float64,requires_grad=True)
        q,k,v = project_once(layer,u)
        global_out = output_projection(layer,attention_with_window(q,k,v).heads)
        torch.testing.assert_close(global_out,layer(u)[0],rtol=1e-12,atol=1e-12)
        local_out = output_projection(layer,attention_with_window(q,k,v,3).heads)
        local_out.square().sum().backward()
        for p in layer.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        example = worked_example(3)
        self.assertAlmostEqual(example['global_D'],5.6,places=5)
        self.assertAlmostEqual(example['window_D'],4.5,places=5)


if __name__ == '__main__':
    unittest.main()
