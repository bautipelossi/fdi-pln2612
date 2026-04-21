import unittest

from src.tokenizer import BPETokenizer


class BPETokenizerTests(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        tokenizer = BPETokenizer("banana bandana", vocab_size=20)
        ids = tokenizer.encode("banana")

        self.assertEqual(tokenizer.decode(ids), "banana")

    def test_decode_rejects_unknown_id(self):
        tokenizer = BPETokenizer("abc", vocab_size=10)

        with self.assertRaises(ValueError):
            tokenizer.decode([len(tokenizer.vocab)])

    def test_repr_is_valid_on_python_311(self):
        tokenizer = BPETokenizer("a b", vocab_size=10)

        self.assertIn("tokens:", repr(tokenizer))


if __name__ == "__main__":
    unittest.main()
