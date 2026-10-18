"""检查跨批次数据覆盖、尾部边界与重复读取行为。"""

import unittest

import torch

from train_dataloader import DataLoaderLite


class DataLoaderTests(unittest.TestCase):
    def test_new_batches_cover_consecutive_positions_with_shifted_targets(self):
        loader = DataLoaderLite(list(range(14)), batch_size=2, seq_len=3)
        x0, y0 = loader.next_batch()
        x1, y1 = loader.next_batch()
        self.assertEqual(x0.tolist(), [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(y0.tolist(), [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(x1.tolist(), [[6, 7, 8], [9, 10, 11]])
        self.assertEqual(y1.tolist(), [[7, 8, 9], [10, 11, 12]])
        # 本批最后的目标成为下批第一个输入：不能前进 B*T+1。
        self.assertEqual(y0[-1, -1].item(), x1[0, 0].item())
        self.assertEqual(x0.dtype, torch.long)
        self.assertEqual(x0.device.type, "cpu")

    def test_exact_fit_is_kept_and_incomplete_tail_is_skipped(self):
        # 12 个 Token 不够做两个各有 6 个输入位置的 batch；第 2 批还缺一个答案。
        # 13 个恰好够；14 个多一个不能单独组成 batch 的尾部 Token。
        for size, expected_starts in [(7, [0, 0, 0]), (12, [0, 0, 0]),
                                      (13, [0, 6, 0]), (14, [0, 6, 0])]:
            with self.subTest(size=size):
                loader = DataLoaderLite(list(range(size)), 2, 3)
                starts = []
                for _ in range(3):
                    starts.append(loader.current_position)
                    x, y = loader.next_batch()
                    self.assertEqual(tuple(x.shape), (2, 3))
                    self.assertEqual(tuple(y.shape), (2, 3))
                self.assertEqual(starts, expected_starts)

    def test_rewind_repeats_the_first_batch_without_mutating_prior_outputs(self):
        loader = DataLoaderLite(list(range(13)), 2, 3)
        x0, y0 = loader.next_batch()
        saved_x, saved_y = x0.clone(), y0.clone()
        loader.next_batch()
        x_again, y_again = loader.next_batch()
        torch.testing.assert_close(x_again, saved_x, rtol=0, atol=0)
        torch.testing.assert_close(y_again, saved_y, rtol=0, atol=0)
        torch.testing.assert_close(x0, saved_x, rtol=0, atol=0)
        torch.testing.assert_close(y0, saved_y, rtol=0, atol=0)

    def test_loading_does_not_consume_random_numbers(self):
        before = torch.get_rng_state().clone()
        loader = DataLoaderLite(list(range(13)), 2, 3)
        for _ in range(4):
            loader.next_batch()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_rejects_impossible_batch_sizes(self):
        for batch_size, seq_len, count in [(0, 3, 20), (2, 0, 20), (2, 3, 6)]:
            with self.subTest(B=batch_size, T=seq_len, count=count):
                with self.assertRaises(ValueError):
                    DataLoaderLite(list(range(count)), batch_size, seq_len)


if __name__ == "__main__":
    unittest.main()
