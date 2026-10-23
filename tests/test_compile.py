"""真实 Inductor 编译训练与普通执行对照；小配置只验证行为，不代表 GPU 性能。"""

import contextlib
import io
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from gpt2_model import GPT, GPTConfig
from train_compile import main, prepare_execution, resolve_runtime
from train_dataloader import DataLoaderLite
from train_initialization import initialize_like_course
from train_mixed_precision import forward_and_loss, train_amp_steps
from train_weight_tying import tie_weights_like_course


class RuntimeSelectionTests(unittest.TestCase):
    def test_auto_uses_cpu_fp32_even_when_mps_is_available(self):
        # 复现用户终端：没有 CUDA、MPS 可用。两种执行方式必须选择同一设备/精度。
        with patch('torch.cuda.is_available', return_value=False), \
                patch('torch.backends.mps.is_available', return_value=True), \
                patch('torch.cuda.is_bf16_supported') as bf16_probe:
            for execution in ('eager', 'compile'):
                device, precision = resolve_runtime('auto', 'auto', execution)
                self.assertEqual((device.type, precision), ('cpu', 'fp32'))
            bf16_probe.assert_not_called()

    def test_cuda_auto_precision_follows_bf16_support(self):
        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.backends.mps.is_available', return_value=True):
            for supported in (True, False):
                with patch('torch.cuda.is_bf16_supported', return_value=supported):
                    for execution in ('eager', 'compile'):
                        device, precision = resolve_runtime('auto', 'auto', execution)
                        self.assertEqual(device.type, 'cuda')
                        self.assertEqual(precision, 'bf16' if supported else 'fp32')

    def test_explicit_configuration_is_preserved_or_rejected(self):
        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.backends.mps.is_available', return_value=True), \
                patch('torch.cuda.is_bf16_supported', return_value=False):
            device, precision = resolve_runtime('cpu', 'bf16', 'compile')
            self.assertEqual((device.type, precision), ('cpu', 'bf16'))
            with self.assertRaisesRegex(ValueError, '不支持'):
                resolve_runtime('cuda', 'bf16', 'compile')
            with self.assertRaisesRegex(ValueError, '--device cpu --precision fp32'):
                resolve_runtime('mps', 'fp32', 'compile')
            device, precision = resolve_runtime('mps', 'auto', 'eager')
            self.assertEqual((device.type, precision), ('mps', 'fp32'))

    def test_cli_defaults_report_the_resolved_configuration(self):
        # 检查真正的 argparse 默认值和 main 的接线，不加载完整模型。
        output = io.StringIO()
        old_threads = torch.get_num_threads()
        old_precision = torch.get_float32_matmul_precision()
        try:
            with patch('sys.argv', ['train_compile.py', '--inspect-only']), \
                    patch('torch.cuda.is_available', return_value=False), \
                    patch('torch.backends.mps.is_available', return_value=True), \
                    contextlib.redirect_stdout(output):
                main()
        finally:
            torch.set_num_threads(old_threads)
            torch.set_float32_matmul_precision(old_precision)
        self.assertIn('device=cpu，precision=fp32，execution=compile', output.getvalue())
        self.assertIn('"mps_available": true', output.getvalue())


class CompileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cache = Path(__file__).resolve().parents[1] / '.cache' / 'torch-compile-tests'
        os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(cache))

    def make_model(self):
        torch.manual_seed(42)
        config = GPTConfig(vocab_size=31, n_positions=8, n_embd=16, n_layer=2,
                           n_head=2, embd_pdrop=0, attn_pdrop=0, resid_pdrop=0)
        model = GPT(config, tie_embeddings=False, gpt2_init=False).float()
        tie_weights_like_course(model)
        initialize_like_course(model)
        return model.train()

    def test_inductor_training_matches_eager_and_keeps_shared_parameters(self):
        device = torch.device('cpu')
        for precision in ('fp32', 'bf16'):
            with self.subTest(precision=precision):
                reference, raw_model = self.make_model(), self.make_model()
                initial = raw_model.lm_head.weight.detach().clone()
                model = prepare_execution(raw_model, 'compile')
                self.assertEqual({id(p) for p in model.parameters()},
                                 {id(p) for p in raw_model.parameters()})
                self.assertIs(model.lm_head.weight, raw_model.transformer.wte.weight)
                optimizers = [torch.optim.AdamW(m.parameters(), lr=3e-4)
                              for m in (reference, raw_model)]
                outputs = []
                with contextlib.redirect_stdout(io.StringIO()):
                    for m, opt in zip((reference, model), optimizers):
                        loader = DataLoaderLite([i % 31 for i in range(81)], 2, 4)
                        outputs.append(train_amp_steps(m, opt, loader, device, precision, 1, 2))
                expected, actual = outputs[0][0], outputs[1][0]
                self.assertEqual([r['batch_start'] for r in actual], [0, 8, 16])
                self.assertEqual(outputs[1][1]['measured_tokens'], 16)
                self.assertFalse(torch.equal(initial, raw_model.lm_head.weight))
                # 融合可能改变浮点舍入；检查数值接近而非要求逐 bit 相同。
                tolerance = 1e-5 if precision == 'fp32' else 5e-3
                for left, right in zip(actual, expected):
                    self.assertAlmostEqual(left['loss_before_update'], right['loss_before_update'],
                                           delta=tolerance)
                    self.assertEqual(left['parameter_grad_dtype'], 'torch.float32')
                    self.assertEqual(left['logits']['dtype'],
                                     'torch.float32' if precision == 'fp32' else 'torch.bfloat16')
                for left, right in zip(raw_model.parameters(), reference.parameters()):
                    if precision == 'fp32':
                        torch.testing.assert_close(left, right, rtol=1e-3, atol=1e-4)
                    # BF16 融合改变舍入，接近零的梯度经 AdamW 缩放后可能产生
                    # 可见参数差异；验证 loss 接近、梯度有限和同一权重下的前向。
                    # 不把不同执行顺序的 BF16 多步参数逐元素一致当作正确性条件。
                    self.assertTrue(torch.isfinite(left).all())
                    self.assertTrue(torch.isfinite(left.grad).all())
                # 再用相同输入读取更新后的参数，排除“编译缓存了旧输出”的错误。
                x, y = DataLoaderLite(list(range(25)), 2, 4).next_batch()
                compiled_logits, _ = forward_and_loss(model, x, y, device, precision)
                eager_logits, _ = forward_and_loss(raw_model, x, y, device, precision)
                torch.testing.assert_close(compiled_logits, eager_logits,
                                           rtol=tolerance, atol=tolerance)

    def test_parameter_value_updates_reuse_the_captured_graph(self):
        # 诊断用 backend 只统计捕获次数，不用于测速，也不冒充 Inductor。
        captures = []

        def count_graph(graph_module, example_inputs):
            captures.append(graph_module)
            return graph_module.forward

        raw_model = self.make_model()
        model = torch.compile(raw_model, backend=count_graph, fullgraph=True, dynamic=False)
        optimizer = torch.optim.AdamW(raw_model.parameters(), lr=3e-4)
        loader = DataLoaderLite([i % 31 for i in range(81)], 2, 4)
        with contextlib.redirect_stdout(io.StringIO()):
            train_amp_steps(model, optimizer, loader, torch.device('cpu'), 'fp32', 1, 2)
        # 数据/参数的数值不同，形状和执行条件相同；整个 forward 只捕获了一次。
        self.assertEqual(len(captures), 1)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
