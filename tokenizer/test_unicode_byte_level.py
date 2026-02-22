import unittest

from tokenizer.loader import TokenizerConfig
from tokenizer.tokenizer import ConfigurableTokenizer


class TestUnicodeByteLevelCoverage(unittest.TestCase):
    def test_byte_level_bpe_round_trips_polish_text(self) -> None:
        config = TokenizerConfig(
            name="unicode_test",
            tokenizer_type="bpe",
            byte_fallback=True,
            vocab_size=300,
            min_frequency=1,
            special_tokens=["<|endoftext|>", "<|pad|>"],
            pre_tokenizer="byte_level",
            decoder="byte_level",
        )
        tokenizer = ConfigurableTokenizer(config)
        tokenizer.tokenizer.train_from_iterator(
            ["Simple ascii training text to keep merges tiny."],
            tokenizer.trainer,
        )

        sample = "zażółć gęślą jaźń"
        encoding = tokenizer.tokenizer.encode(sample)
        decoded = tokenizer.tokenizer.decode(encoding.ids, skip_special_tokens=False)

        self.assertEqual(decoded.lstrip(" "), sample)
        self.assertEqual(encoding.offsets[-1][1], len(sample))
        covered = [False] * len(sample)
        for start, end in encoding.offsets:
            for index in range(start, min(end, len(sample))):
                covered[index] = True
        self.assertTrue(all(covered))


if __name__ == "__main__":
    unittest.main()
