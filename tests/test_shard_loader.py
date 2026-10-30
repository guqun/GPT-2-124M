"""核对文件隔离、错位标签、完整批次边界与各 rank 同轮换片。"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from load_shards import ShardDataLoaderLite, load_tokens


def save(directory, split, index, values, dtype=np.uint16):
    path = Path(directory) / f"edufineweb_{split}_{index:06d}.npy"
    np.save(path, np.array(values, dtype=dtype), allow_pickle=False)
    return path


class ShardLoaderTests(unittest.TestCase):
    def test_shift_stride_switch_wrap_and_preserve_returned_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            save(directory, 'train', 1, range(100, 116))
            save(directory, 'train', 2, range(200, 208))
            save(directory, 'train', 3, range(300, 303))  # 整个尾片也不够 7 个 ID。
            loader = ShardDataLoaderLite(directory, 2, 3)
            first_x, first_y = loader.next_batch()
            self.assertEqual(first_x.tolist(), [[100, 101, 102], [103, 104, 105]])
            self.assertEqual(first_y.tolist(), [[101, 102, 103], [104, 105, 106]])
            second_x, second_y = loader.next_batch()
            self.assertEqual(second_x.tolist(), [[106, 107, 108], [109, 110, 111]])
            self.assertEqual(second_y.tolist(), [[107, 108, 109], [110, 111, 112]])
            self.assertEqual(loader.current_file.name, 'edufineweb_train_000002.npy')
            third_x, third_y = loader.next_batch()
            self.assertEqual(third_x.tolist(), [[200, 201, 202], [203, 204, 205]])
            self.assertEqual(third_y.tolist(), [[201, 202, 203], [204, 205, 206]])
            again_x, again_y = loader.next_batch()
            torch.testing.assert_close(again_x, first_x, rtol=0, atol=0)
            torch.testing.assert_close(again_y, first_y, rtol=0, atol=0)
            self.assertEqual(first_x.tolist(), [[100, 101, 102], [103, 104, 105]])

    def test_exact_fit_uses_last_batch(self):
        for length, expected_starts in [(7, [0, 0, 0]), (12, [0, 0, 0]), (13, [0, 6, 0])]:
            with self.subTest(length=length), tempfile.TemporaryDirectory() as directory:
                save(directory, 'train', 1, range(length))
                loader = ShardDataLoaderLite(directory, 2, 3)
                starts = []
                for _ in range(3):
                    starts.append(loader.current_position)
                    loader.next_batch()
                self.assertEqual(starts, expected_starts)

    def test_split_sort_reset_and_rng_are_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            save(directory, 'train', 9, range(100, 113))
            save(directory, 'train', 2, range(13))
            save(directory, 'val', 0, range(500, 513))
            rng = torch.get_rng_state().clone()
            train = ShardDataLoaderLite(directory, 2, 3)
            val = ShardDataLoaderLite(directory, 2, 3, 'val')
            self.assertEqual(train.current_file.name, 'edufineweb_train_000002.npy')
            train.next_batch()
            train.next_batch()
            self.assertEqual(train.current_file.name, 'edufineweb_train_000009.npy')
            val.next_batch()
            val.reset()
            self.assertEqual(train.current_file.name, 'edufineweb_train_000009.npy')
            self.assertEqual(val.next_batch()[0][0, 0].item(), 500)
            train.reset()
            self.assertEqual(train.next_batch()[0][0, 0].item(), 0)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_rank_batches_match_single_global_batch_across_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            save(directory, 'train', 1, range(13))
            save(directory, 'train', 2, range(100, 107))
            save(directory, 'train', 3, range(200, 205))  # 双 rank 都必须跳过。
            ranks = [ShardDataLoaderLite(directory, 1, 3, rank=r, world_size=2) for r in range(2)]
            global_loader = ShardDataLoaderLite(directory, 2, 3)
            for _ in range(8):
                batches = [loader.next_batch() for loader in ranks]
                expected_x, expected_y = global_loader.next_batch()
                torch.testing.assert_close(torch.cat([xy[0] for xy in batches]), expected_x, rtol=0, atol=0)
                torch.testing.assert_close(torch.cat([xy[1] for xy in batches]), expected_y, rtol=0, atol=0)
                self.assertEqual(ranks[0].current_shard, ranks[1].current_shard)
                self.assertEqual(ranks[0].global_position, ranks[1].global_position)

    def test_conversion_and_invalid_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save(directory, 'train', 1, [0, 123, 50256])
            tokens = load_tokens(path)
            self.assertEqual(tokens.tolist(), [0, 123, 50256])
            self.assertEqual(tokens.dtype, torch.long)
            self.assertEqual(tokens.device.type, 'cpu')
            with self.assertRaises(ValueError):
                ShardDataLoaderLite(directory, 2, 3)  # 全部太短，不能无限跳片。
            for values, dtype in [([[1, 2]], np.uint16), ([65535], np.uint16), ([1.5], np.float32)]:
                path = save(directory, 'train', 1, values, dtype)
                with self.assertRaises(ValueError):
                    load_tokens(path)


if __name__ == '__main__':
    unittest.main()
