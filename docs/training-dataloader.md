# Training Dataloader

This dataloader streams or maps Hugging Face datasets, pretokenizes with a provided `tokenizers.Tokenizer`,
interleaves multiple sources by weight, and packs token IDs into fixed-length training sequences. It yields
`(inputs, targets)` where targets are shifted by one token.

## Key features
- Accepts an initialized `tokenizers.Tokenizer` (no tokenizer loading inside the dataloader).
- Works with streaming datasets (`streaming: true`) and map-style datasets (`streaming: false`).
- Optional pretokenize-to-disk for map-style datasets via `cache_dir`.
- Weighted mixing, per-dataset shuffle buffers, and worker sharding.
- Token packing with optional BOS/EOS insertion and padding/drop_last behavior.
- Optional node-level split for distributed training.

## Config format
Config is JSON and validated with `pydantic` in `training/dataloader_config.py`.

### Top-level fields
- `datasets`: list of dataset specs.
- `add_bos`: prepend BOS token to each record.
- `add_eos`: append EOS token to each record.
- `bos_token`/`eos_token`: token strings to look up in the tokenizer (required if `add_bos`/`add_eos`).
- `pad_token`: token string for padding when `drop_last` is false.
- `drop_last`: drop final partial sequence instead of padding.
- `token_dtype`: one of `int64`, `int32`, `int16`, `uint8`.
- `pretokenize_batch_size`: batch size for `IterableDataset.map(batched=True)`.
- `cache_dir`: optional cache directory for pretokenized map-style datasets.
- `mixing`: mixing configuration.
- `normalization`: global normalization defaults.
- `record_separator`: string appended to each record before tokenization.
- `shuffle`: optional shuffle configuration (applied per dataset unless overridden).
- `node_split`: enable node-level split using `RANK`/`WORLD_SIZE` or overrides below.
- `node_rank`/`node_world_size`: explicit distributed overrides.

### Dataset spec fields
- `name`: dataset name (or builder like "json" or "text").
- `config`: optional HF dataset config name.
- `split`: dataset split (default `"train"`).
- `streaming`: whether to stream the dataset (`true` by default).
- `text_columns`: list of columns to join into a text record.
- `text_joiner`: string used when joining multiple text columns (default `"\n"`).
- `weight`: mixing weight.
- `filters`: optional HF filters (`[column, op, value]` triplets).
- `columns`: optional column subset for streaming (e.g., Parquet).
- `data_files`: optional local file spec for local data.
- `normalization`: optional per-dataset normalization override.
- `record_separator`: optional per-dataset record separator override.
- `shuffle`: optional per-dataset shuffle override (`false` disables shuffle for that dataset).

### Mixing config fields
- `seed`: optional seed for deterministic interleaving.
- `exhausted_policy`: `"stop"`, `"repeat"`, or `"drop"`.

### Shuffle config fields
- `buffer_size`: shuffle buffer size (default `1000`).
- `seed`: optional RNG seed for deterministic shuffle.

## Example: streaming training
```json
{
  "datasets": [
    {
      "name": "text",
      "data_files": { "train": "datasets/shake.txt" },
      "split": "train",
      "text_columns": ["text"],
      "weight": 1.0,
      "streaming": true
    }
  ],
  "add_eos": true,
  "eos_token": "<|endoftext|>",
  "drop_last": true,
  "token_dtype": "int64",
  "pretokenize_batch_size": 256,
  "record_separator": "\n",
  "shuffle": { "buffer_size": 10000, "seed": 123 },
  "mixing": { "seed": 42, "exhausted_policy": "stop" }
}
```

## Example: map-style pretokenize cache
```json
{
  "datasets": [
    {
      "name": "json",
      "data_files": { "train": "datasets/my_data/*.jsonl" },
      "split": "train",
      "text_columns": ["text"],
      "weight": 1.0,
      "streaming": false
    }
  ],
  "add_bos": true,
  "bos_token": "<|bos|>",
  "add_eos": true,
  "eos_token": "<|eos|>",
  "drop_last": false,
  "pad_token": "<|pad|>",
  "token_dtype": "int64",
  "pretokenize_batch_size": 512,
  "cache_dir": "datasets/cache"
}
```

## Python usage

```python
from tokenizer.tokenizer import ConfigurableTokenizer
from tokenizer.loader import load_tokenizer_config
from training.dataloader_config import load_training_dataloader_config
from training.training_config import load_training_config
from training.dataloader import TrainingDataLoader

# Build tokenizer
config = load_tokenizer_config("tokenizer/tok_config.json")
ctok = ConfigurableTokenizer(config)
ctok.train_from_file(["datasets/shake.txt"])

# Load configs
loader_config = load_training_dataloader_config("my_training_dataloader_config.json")
train_config = load_training_config("training/training_config.json")

# Create loader
loader = TrainingDataLoader(
    loader_config,
    ctok.tokenizer,
    batch_size=8,
    seq_len=train_config.seq_len,
    num_workers=2,
)
inputs, targets = loader.next_batch()
print(inputs.shape, targets.shape)
```

## Notes
- `drop_last=false` requires `pad_token` (or `eos_token`) to pad the final sequence.
- `add_bos`/`add_eos` require the corresponding token strings to exist in the tokenizer vocabulary.
- For distributed training, set `node_split=true` and provide `RANK`/`WORLD_SIZE` env vars (or config overrides).
- When `num_workers > 0`, each worker shards the underlying dataset.
- `TrainingDataLoader.reset()` advances the epoch and reshuffles (if `shuffle.seed` is set).
