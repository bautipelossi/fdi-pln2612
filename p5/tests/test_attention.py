import unittest

import torch

from src.attention import Attention


class AttentionTests(unittest.TestCase):
    def test_forward_preserves_shape(self):
        torch.manual_seed(0)
        attn = Attention(d_model=8, n_heads=2, max_seq_len=4, dropout=0.0)
        x = torch.randn(2, 4, 8)

        y = attn(x)

        self.assertEqual(y.shape, x.shape)

    def test_rejects_incompatible_heads(self):
        with self.assertRaises(ValueError):
            Attention(d_model=10, n_heads=4, max_seq_len=4, dropout=0.0)


if __name__ == "__main__":
    unittest.main()
