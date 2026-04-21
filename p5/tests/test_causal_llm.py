import unittest

import torch

from src.causal_llm import CausalLLM


class CausalLLMTests(unittest.TestCase):
    def test_forward_returns_logits_and_loss(self):
        torch.manual_seed(0)
        model = CausalLLM(
            vocab_size=12,
            max_seq_len=6,
            d_model=8,
            n_heads=2,
            n_layers=1,
            expansion=2,
            dropout=0.0,
        )
        x = torch.randint(0, 12, (2, 6))

        logits, loss = model(x, x)

        self.assertEqual(logits.shape, (2, 6, 12))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0)

    def test_generate_with_top_k_one_picks_best_token(self):
        model = CausalLLM(
            vocab_size=5,
            max_seq_len=4,
            d_model=8,
            n_heads=2,
            n_layers=1,
            expansion=2,
            dropout=0.0,
        )

        def fake_forward(idx, targets=None):
            logits = torch.full((idx.size(0), idx.size(1), 5), -10.0)
            logits[:, :, 2] = 10.0
            return logits, None

        model.forward = fake_forward

        generated = model.generate([0], max_tokens=3, temperature=1.0, top_k=1)

        self.assertEqual(generated, [2, 2, 2])

    def test_generate_validates_sampling_arguments(self):
        model = CausalLLM(
            vocab_size=5,
            max_seq_len=4,
            d_model=8,
            n_heads=2,
            n_layers=1,
            expansion=2,
            dropout=0.0,
        )

        with self.assertRaises(ValueError):
            model.generate([0], max_tokens=1, temperature=0)
        with self.assertRaises(ValueError):
            model.generate([0], max_tokens=1, top_k=0)


if __name__ == "__main__":
    unittest.main()
