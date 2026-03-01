export const defaultTokenizerConfig: Record<string, unknown> = {
  name: "shake_bpe_bytelevel",
  tokenizer_type: "bpe",
  byte_fallback: true,
  vocab_size: 1000,
  min_frequency: 2,
  special_tokens: ["<|endoftext|>", "<|pad|>"],
  pre_tokenizer: "byte_level",
  decoder: "byte_level",
};

export const defaultDataloaderConfig: Record<string, unknown> = {
  datasets: [
    {
      name: "text",
      data_files: {
        train: "datasets/shake.txt",
      },
      split: "train",
      text_columns: ["text"],
      weight: 1.0,
    },
  ],
  budget: {
    limit: 250000,
    unit: "chars",
    behavior: "truncate",
  },
  mixing: {
    seed: 42,
    exhausted_policy: "stop",
  },
  shuffle: {
    buffer_size: 1000,
    seed: 123,
  },
  normalization: {
    normalize_newlines: true,
    collapse_whitespace: false,
    strip: true,
    lowercase: false,
    drop_empty: true,
  },
  record_separator: "",
};
