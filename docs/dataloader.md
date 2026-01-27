# Streaming Text Dataloader

This dataloader streams Hugging Face datasets, interleaves multiple sources by weight, and enforces a fixed text
budget (characters or bytes) without tokenization. It yields raw text strings, optionally chunked into fixed-size
blocks for better throughput.

## Key features
- Streaming-only (uses `load_dataset(..., streaming=True)`).
- Dataset mixing by weight using smooth weighted round-robin scheduling.
- Fixed text budget with `stop` or `truncate` behavior.
- Optional lightweight normalization per dataset or globally.
- Optional chunking into fixed-size text blocks.
- Optional streaming shuffle with a configurable buffer.

## Config format
Config is JSON and validated with `pydantic` in `dataloader_config.py`.

### Top-level fields
- `datasets`: list of dataset specs.
- `budget`: text budget configuration.
- `chunking`: optional chunking configuration.
- `mixing`: mixing configuration.
- `normalization`: global normalization defaults.
- `record_separator`: string appended to each record (defaults to `""` for raw records).
- `shuffle`: optional shuffle configuration (applied per dataset unless overridden).

### Dataset spec fields
- `name`: dataset name (or a builder like `"json"` or `"text"`).
- `config`: optional HF dataset config name.
- `split`: dataset split (default `"train"`).
- `text_columns`: list of columns to join into a text record.
- `text_joiner`: string used when joining multiple text columns (default `"\n"`).
- `weight`: mixing weight.
- `filters`: optional HF streaming filters (`[column, op, value]` triplets).
- `columns`: optional column subset for streaming (e.g., Parquet).
- `data_files`: optional local file spec for streaming local data.
- `normalization`: optional per-dataset normalization override.
- `record_separator`: optional per-dataset record separator override.
- `shuffle`: optional per-dataset shuffle override (`false` disables shuffle for that dataset).

### Budget config fields
- `limit`: maximum total text budget.
- `unit`: `"chars"` or `"bytes"`.
- `behavior`: `"stop"` (drop record that would exceed) or `"truncate"` (truncate final record).

### Chunking config fields
- `chunk_size`: size of each output chunk (same unit as budget).
- `drop_last`: drop final partial chunk if `True`.

### Mixing config fields
- `seed`: optional seed for deterministic interleaving.
- `exhausted_policy`: `"stop"`, `"repeat"`, or `"drop"`.

### Shuffle config fields
- `buffer_size`: shuffle buffer size (default `1000`); larger buffers improve randomness but use more memory.
- `seed`: optional base seed for deterministic shuffle. If omitted, a per-run random seed
  is generated and shared across workers.

## Example: 30% natural language + 70% code
```json
{
  "datasets": [
    {
      "name": "HuggingFaceFW/fineweb",
      "split": "train",
      "text_columns": ["text"],
      "weight": 0.3
    },
    {
      "name": "codeparrot/github-code",
      "split": "train",
      "text_columns": ["code"],
      "weight": 0.7
    }
  ],
  "budget": {
    "limit": 25000000,
    "unit": "chars",
    "behavior": "truncate"
  },
  "mixing": {
    "seed": 42,
    "exhausted_policy": "stop"
  },
  "shuffle": {
    "buffer_size": 10000,
    "seed": 123
  },
  "chunking": {
    "chunk_size": 2048,
    "drop_last": false
  },
  "record_separator": "\n"
}
```

## Example: local text streaming + byte budget
```json
{
  "datasets": [
    {
      "name": "text",
      "data_files": { "train": "datasets/shake.txt" },
      "split": "train",
      "text_columns": ["text"],
      "weight": 1.0
    }
  ],
  "budget": {
    "limit": 1048576,
    "unit": "bytes",
    "behavior": "stop"
  },
  "mixing": {
    "seed": 7,
    "exhausted_policy": "stop"
  },
  "record_separator": ""
}
```

## Python usage

```python
from tokenizer.dataloader_config import load_dataloader_config
from tokenizer.dataloader import create_dataloader, build_dataset

config = load_dataloader_config("my_dataloader_config.json")

# Use the raw iterable dataset
dataset = build_dataset(config)
for sample in dataset:
  print(sample)
  break

# Or wrap in a PyTorch DataLoader
dataloader = create_dataloader(config, batch_size=4, num_workers=2, prefetch_factor=4)
for batch in dataloader:
  print(batch)
  break
```

## Notes
- When `num_workers > 0`, each worker shards the underlying streaming dataset.
- Shuffle is applied per dataset before worker sharding, then the shuffled datasets are interleaved.
- Call `dataset.set_epoch(epoch)` between epochs to reshuffle with `seed + epoch` (works with persistent workers).
- If you set `record_separator` to `"\n"`, each record is suffixed with a newline before chunking.
- For strict reproducibility, keep fixed `mixing.seed` and `shuffle.seed` values and avoid changing dataset ordering.
