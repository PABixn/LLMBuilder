from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer, Whitespace as WhitespacePreTokenizer, Metaspace as MetaspacePreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder, WordPiece as WordpieceDecoder, Metaspace as MetaspaceDecoder
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer

from tokenizer_loader import TokenizerConfig
from tokenizers import Tokenizer


class ConfigurableTokenizer:
    def __init__(self, config: TokenizerConfig):
        self.config = config

        self.tokenizer = self.get_tokenizer()
        self.trainer = self.get_trainer()

    def train_from_file(self):
        self.tokenizer.train(["datasets/shake.txt"], self.trainer)

        self.tokenizer.save(self.config.name + ".json")

    def get_tokenizer(self):
        tok = None

        if self.config.tokenizer_type == "bpe":
            if self.config.byte_fallback:
                tok = Tokenizer(BPE(byte_fallback=True))
            else:
                tok = Tokenizer(BPE(byte_fallback=False, unk_token=self.config.unk_token))

        elif self.config.tokenizer_type == "wordpiece":
            tok = Tokenizer(WordPiece(unk_token=self.config.unk_token))

        elif self.config.tokenizer_type == "unigram":
            tok = Tokenizer(Unigram())
        else:
            raise ValueError(f"Unknown tokenizer type {self.config.tokenizer_type}")

        tok.pre_tokenizer = self._get_pre_tokenizer()
        tok.decoder = self._get_decoder()
        tok.add_special_tokens(self.config.special_tokens)

        return tok

    def get_trainer(self):
        if self.config.tokenizer_type == "bpe":
            return BpeTrainer(
                vocab_size=self.config.vocab_size,
                min_frequency=self.config.min_frequency,
                special_tokens=self.config.special_tokens,
            )
        elif self.config.tokenizer_type == "wordpiece":
            return WordPieceTrainer(
                vocab_size=self.config.vocab_size,
                min_frequency=self.config.min_frequency,
                special_tokens=self.config.special_tokens,
            )
        elif self.config.tokenizer_type == "unigram":
            return UnigramTrainer(
                vocab_size=self.config.vocab_size,
                special_tokens=self.config.special_tokens,
            )
        else:
            raise ValueError(f"Unknown trainer type {self.config.tokenizer_type}")

    def _get_pre_tokenizer(self):
        if self.config.pre_tokenizer == "byte_level":
            return ByteLevelPreTokenizer()
        elif self.config.pre_tokenizer == "whitespace":
            return WhitespacePreTokenizer()
        elif self.config.pre_tokenizer == "metaspace":
            return MetaspacePreTokenizer()
        else:
            raise ValueError(f"Unknown pre_tokenizer type {self.config.pre_tokenizer}")

    def _get_decoder(self):
        if self.config.decoder == "byte_level":
            return ByteLevelDecoder()
        elif self.config.decoder == "wordpiece":
            return WordpieceDecoder()
        elif self.config.decoder == "metaspace":
            return MetaspaceDecoder()
        else:
            raise ValueError(f"Unknown decoder type {self.config.decoder}")
