"""评分对齐、右补齐隔离、评测不干扰训练、真实 Gloo 不等题数归约。"""

import copy
from contextlib import redirect_stdout
from datetime import timedelta
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt2_model import GPT, GPTConfig
from hellaswag_eval import render_example, load_questions, score_candidates, evaluate_hellaswag
from train_ddp import Runtime, configure_optimizer
from train_shards import make_shard_loaders, train_from_shards
from train_hellaswag import train_with_hellaswag


ROOT = Path(__file__).resolve().parents[1]


class NumberTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [int(x) for x in text.split()]

    def decode(self, ids):
        return ' '.join(map(str, ids))


def questions():
    return [render_example({'ctx': '1 2', 'endings': ['3 4', '5', '6 0 7', '8 9'],
                            'label': i % 4}, NumberTokenizer(), index=i,
                           max_length=8, vocab_size=31) for i in range(3)]


def model(dropout=0.0):
    torch.manual_seed(42)
    return GPT(GPTConfig(vocab_size=31, n_positions=8, n_embd=16, n_layer=2, n_head=2,
                         embd_pdrop=dropout, attn_pdrop=dropout, resid_pdrop=dropout)).double()


def runtime():
    return Runtime(False, 0, 0, 1, torch.device('cpu'), 'none')


def distributed_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        rt = Runtime(True, rank, rank, 2, torch.device('cpu'), 'gloo')
        raw, reference = model(), model()
        wrapped = DDP(raw)
        records = []
        # 3 题分成 2+1；1 题分成 1+0。题数不等和空 rank 都必须安全归约。
        for count in (3, 1):
            result = evaluate_hellaswag(raw, questions()[:count], rt)
            expected = evaluate_hellaswag(reference, questions()[:count], runtime())
            assert result['global_total'] == count
            assert result['accuracy'] == expected['accuracy']
            assert result['local_question_indices'] == list(range(rank, count, 2))
            records.append(result)
        # 评测后仍能用 DDP 反向更新，并与单进程联合 batch 得到同一参数。
        x = torch.tensor([[rank + 1, 2, 3]])
        y = torch.tensor([[2, 3, rank + 4]])
        opt = torch.optim.SGD(raw.parameters(), lr=0.01)
        opt.zero_grad()
        F.cross_entropy(wrapped(x).reshape(-1, 31), y.reshape(-1)).backward()
        opt.step()
        ref_opt = torch.optim.SGD(reference.parameters(), lr=0.01)
        ref_opt.zero_grad()
        joint_x = torch.tensor([[1, 2, 3], [2, 2, 3]])
        joint_y = torch.tensor([[2, 3, 4], [2, 3, 5]])
        F.cross_entropy(reference(joint_x).reshape(-1, 31), joint_y.reshape(-1)).backward()
        ref_opt.step()
        for actual, expected in zip(raw.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        Path(output, f'rank-{rank}.json').write_text(json.dumps(records))
    finally:
        dist.destroy_process_group()


class HellaSwagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_numeric_score_includes_first_ending_and_excludes_context_and_padding(self):
        tokens = torch.tensor([[1, 2, 3, 4, 5, 0]] * 4)
        mask = torch.tensor([[0, 0, 1, 1, 1, 0]] * 4)
        logits = torch.zeros((4, 6, 8), dtype=torch.double)
        for position, token, p in [(1, 3, .8), (2, 4, .6), (3, 5, .9)]:
            # 其他七项 exp=1，因此 exp(target)=7p/(1-p) 时 target 概率恰为 p。
            logits[:, position, token] = math.log(7 * p / (1 - p))
        result = score_candidates(tokens, mask, logits)
        expected = sum(-math.log(p) for p in (.8, .6, .9)) / 3
        torch.testing.assert_close(result['mean_loss'], torch.full((4,), expected, dtype=torch.double))
        self.assertEqual(result['token_count'].tolist(), [3] * 4)
        self.assertEqual(result['prediction'], 0)  # 并列时取第一项。
        logits[:, 0, :] = torch.arange(8) * 50   # 上下文损失变了，不影响得分。
        logits[:, 4:, :] = torch.arange(8) * -50 # 补齐及末尾输出不影响得分。
        torch.testing.assert_close(score_candidates(tokens, mask, logits)['mean_loss'],
                                   result['mean_loss'], atol=0, rtol=0)

    def test_batched_score_equals_separate_prefix_predictions_and_padding_is_causal(self):
        q, net = questions()[0], model().eval()
        with torch.no_grad():
            observed = score_candidates(q.tokens, q.mask, net(q.tokens))
            expected = []
            for row in range(4):
                terms = []
                for target in torch.where(q.mask[row] == 1)[0].tolist():
                    z = net(q.tokens[row:row + 1, :target])[0, -1]
                    terms.append(-z.log_softmax(-1)[q.tokens[row, target]])
                expected.append(torch.stack(terms).mean())
            torch.testing.assert_close(observed['mean_loss'], torch.stack(expected),
                                       atol=1e-12, rtol=1e-12)
            self.assertEqual(observed['token_count'].tolist(), [2, 1, 3, 2])
            # C 的真实 ID 0 必须计分；只改右侧 padding，不改上下文和真实候选。
            padded = q.tokens.clone()
            for row, length in enumerate([4, 3, 5, 4]):
                padded[row, length:] = 30
            actual = score_candidates(padded, q.mask, net(padded))
            torch.testing.assert_close(actual['mean_loss'], observed['mean_loss'],
                                       atol=1e-12, rtol=1e-12)

    def test_input_limits_and_invalid_data_fail_without_silent_truncation(self):
        row = {'ctx': '1 2', 'endings': ['3', '4 5', '6', '7'], 'label': '2'}
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            path = Path(tmp, 'questions.jsonl')
            path.write_text('\n'.join(json.dumps(row) for _ in range(3)))
            qs, meta = load_questions(path, NumberTokenizer(), limit=2, max_length=8, vocab_size=31)
            self.assertEqual([q.index for q in qs], [0, 1])
            self.assertEqual(qs[0].label, 2)
            self.assertEqual(meta['examples'], 2)
            all_qs, _ = load_questions(path, NumberTokenizer(), limit=0, max_length=8, vocab_size=31)
            self.assertEqual(len(all_qs), 3)
            for changed in ({'label': -1}, {'label': True}, {'endings': ['3'] * 3},
                            {'ctx': ''}, {'endings': [''] * 4}):
                with self.assertRaises(ValueError):
                    render_example({**row, **changed}, NumberTokenizer(), index=0,
                                   max_length=8, vocab_size=31)
            with self.assertRaisesRegex(ValueError, '超过'):
                load_questions(path, NumberTokenizer(), limit=0, max_length=2, vocab_size=31)
            path.write_text('')
            with self.assertRaisesRegex(ValueError, '为空'):
                load_questions(path, NumberTokenizer(), limit=0, max_length=8, vocab_size=31)
        q = questions()[0]
        with self.assertRaisesRegex(ValueError, '至少'):
            score_candidates(q.tokens, torch.zeros_like(q.mask), model()(q.tokens))

    def test_eval_preserves_parameters_grads_rng_and_restores_mode_on_failure(self):
        net = model(.2)
        net(torch.tensor([[1, 2, 3]])).square().mean().backward()
        params = [p.detach().clone() for p in net.parameters()]
        grads = [p.grad.clone() for p in net.parameters()]
        rng = torch.random.get_rng_state().clone()
        self.assertEqual(evaluate_hellaswag(net, questions(), runtime()),
                         evaluate_hellaswag(net, questions(), runtime()))
        self.assertTrue(net.training)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, atol=0, rtol=0)
        for p, before, grad in zip(net.parameters(), params, grads):
            torch.testing.assert_close(p, before, atol=0, rtol=0)
            torch.testing.assert_close(p.grad, grad, atol=0, rtol=0)
        with patch.object(net, 'forward', side_effect=RuntimeError('failure')):
            with self.assertRaisesRegex(RuntimeError, 'failure'):
                evaluate_hellaswag(net, questions(), runtime())
        self.assertTrue(net.training)
        net.eval()
        evaluate_hellaswag(net, questions(), runtime())
        self.assertFalse(net.training)

    def test_training_trajectory_matches_previous_entry_including_adam_rng_and_cursors(self):
        baseline, observed = model(.2), model(.2)
        opts = dict(steps=3, accum_steps=2, precision='fp32', max_lr=6e-4,
                    min_lr=6e-5, warmup_steps=1, decay_steps=3, clip_norm=.5,
                    validation_every=2, validation_batches=2, tokenizer=NumberTokenizer(),
                    prompt_ids=[1, 2], generation_every=2, max_new_tokens=2,
                    num_sequences=2, top_k=5)
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            for split, idx, values in [('train', 1, list(range(1, 17))),
                                       ('train', 2, list(range(1, 9))),
                                       ('val', 0, list(range(10, 26)))]:
                np.save(Path(tmp, f'edufineweb_{split}_{idx:06d}.npy'),
                        np.array(values, dtype=np.uint16))
            train1, val1, _ = make_shard_loaders(tmp, 1, 3, runtime())
            train2, val2, _ = make_shard_loaders(tmp, 1, 3, runtime())
            opt1 = configure_optimizer(baseline, torch.device('cpu'), 6e-4)
            opt2 = configure_optimizer(observed, torch.device('cpu'), 6e-4)
            with redirect_stdout(io.StringIO()):
                torch.manual_seed(123)
                expected = train_from_shards(baseline, baseline, opt1, train1, val1, runtime(), **opts)
                rng = torch.random.get_rng_state().clone()
                torch.manual_seed(123)
                actual = train_with_hellaswag(observed, observed, opt2, train2, val2, runtime(),
                                              questions=questions(), hellaswag_every=2, **opts)
            for key in ('training', 'validation', 'generation'):
                self.assertEqual(actual[key], expected[key])
            self.assertEqual([r['after_updates'] for r in actual['hellaswag']], [0, 2, 3])
            for row in actual['hellaswag']:
                self.assertEqual(row['cursors_before'], row['cursors_after'])
            torch.testing.assert_close(torch.random.get_rng_state(), rng, atol=0, rtol=0)
            for a, b in zip(baseline.parameters(), observed.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
                torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
                for key in ('step', 'exp_avg', 'exp_avg_sq'):
                    torch.testing.assert_close(opt1.state[a][key], opt2.state[b][key], atol=0, rtol=0)

    def test_real_gloo_uneven_questions_empty_rank_and_resume_backward(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as tmp:
            mp.spawn(distributed_worker, args=(str(Path(tmp, 'rendezvous')), tmp),
                     nprocs=2, join=True)
            records = [json.loads(Path(tmp, f'rank-{i}.json').read_text()) for i in range(2)]
            self.assertEqual(records[0][0]['local_question_indices'], [0, 2])
            self.assertEqual(records[1][0]['local_question_indices'], [1])
            self.assertEqual(records[1][1]['local_total'], 0)


if __name__ == '__main__':
    unittest.main()
