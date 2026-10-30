"""检查分片边界是否丢失、重复或多写 Token；不需要网络与模型。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare_fineweb import encode_document, write_shards


class DataPreparationTests(unittest.TestCase):
    def test_reconstruct_short_exact_and_multiple_boundaries(self):
        # 含：单篇跨三片、文档中间换片、恰好写满、短尾片。
        for lengths in ([2], [5], [2, 3], [13, 4], [4, 4]):
            with self.subTest(lengths=lengths), tempfile.TemporaryDirectory() as directory:
                docs = []
                offset = 0
                for length in lengths:
                    docs.append(np.arange(offset, offset + length, dtype=np.uint16))
                    offset += length
                result = write_shards(docs, Path(directory), 5)
                loaded = [np.load(Path(directory) / s["file"], allow_pickle=False) for s in result["shards"]]
                np.testing.assert_array_equal(np.concatenate(loaded), np.concatenate(docs))
                self.assertEqual(len(loaded), (sum(lengths) + 4) // 5)
                self.assertTrue(all(a.dtype == np.uint16 and 0 < len(a) <= 5 for a in loaded))
                self.assertTrue(all(len(a) == 5 for a in loaded[:-1]))
                self.assertEqual([s["split"] for s in result["shards"]], ["val"] + ["train"] * (len(loaded) - 1))
                self.assertEqual(result["total_tokens"], sum(lengths))

    def test_eot_and_overflow_checked_before_cast(self):
        class Tokenizer:
            eot_token = 50256

            def encode_ordinary(self, text):
                return [1, 65536] if text == "invalid" else [1, 65535]

        np.testing.assert_array_equal(encode_document("valid", Tokenizer()), [50256, 1, 65535])
        with self.assertRaises(ValueError):
            encode_document("invalid", Tokenizer())

    def test_preserve_existing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "old.npy"
            marker.write_bytes(b"existing data")
            with self.assertRaises(ValueError):
                write_shards([np.array([1, 2], dtype=np.uint16)], directory, 5)
            self.assertEqual(marker.read_bytes(), b"existing data")


if __name__ == "__main__":
    unittest.main()
