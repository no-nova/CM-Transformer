# Checkpoints

The five checkpoints this study consumes are **third-party** and are not
redistributed here. Every number in `experiments/cm-transformer/results/` and in
the paper was produced from the exact files listed below: each weight file is
identified by size **and** SHA-256, and `scripts/fetch_models.py --check`
re-verifies those digests against whatever is on disk.

| # | Target directory | Hugging Face repository | File in repository | Bytes | SHA-256 |
|---|---|---|---|---|---|
| 1 | `models/Qwen3.5-0.8B/` | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) | `model.safetensors-00001-of-00001.safetensors` → saved as `model.safetensors` | 1,746,942,600 | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| 2 | `models/Qwen3-0.6B/` | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | `model.safetensors` | 1,503,300,328 | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| 3 | `models/Qwen3.5-4B-IQ4_NL.gguf` | [`unsloth/Qwen3.5-4B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) | `Qwen3.5-4B-IQ4_NL.gguf` | 2,579,944,608 | `ff5c3e9740a5aa53f04fdf3b0b8cc75da556bf8948cdb19d61c512d3a43465d9` |
| 4 | `models/SmolVLM2-256M-Video-Instruct/` | [`HuggingFaceTB/SmolVLM2-256M-Video-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct) | `model.safetensors` | 1,025,998,224 | `dcc8f4c00d30d70bb3028a1bfc373e1b1972e25919f8788183cae4ae72431ce2` |
| 5 | `models/whisper-base/` | [`openai/whisper-base`](https://huggingface.co/openai/whisper-base) | `model.safetensors` | 290,403,936 | `07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b` |

Total on disk: **≈ 7.1 GB** of weights. Nothing under `models/` is written by
any script — `common.Weights` memory-maps the safetensors read-only and the GGUF
is opened with `"rb"`.

## Fetching

```bash
python3 scripts/fetch_models.py            # all five (~7.1 GB)
python3 scripts/fetch_models.py --only qwen35_08b,whisper_base
python3 scripts/fetch_models.py --check    # verify digests only, no network
```

The script uses `huggingface_hub` when installed and falls back to plain HTTPS
downloads from `huggingface.co/<repo>/resolve/main/<file>` otherwise. Set
`HF_ENDPOINT` to use a mirror (e.g. `https://hf-mirror.com`). Downloads are
verified against the digests above before being moved into place; a mismatch
aborts with a non-zero exit code and the partial file is left for inspection.
A copy populated from a different precision or from an ONNX-only mirror fails
this check, because both the size and the digest differ.

### Tokenizer / config files

`tests/test_model_ref.py` reads `models/Qwen3.5-0.8B/config.json` and
`common.build_tokenizer` reads the tokenizer files, so each directory also
needs its metadata files. The script fetches the ones used here:

| Directory | Metadata files |
|---|---|
| `Qwen3.5-0.8B` | `config.json`, `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `preprocessor_config.json`, `video_preprocessor_config.json` |
| `Qwen3-0.6B` | `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` |
| `SmolVLM2-256M-Video-Instruct` | `config.json`, `generation_config.json`, `chat_template.json`, `processor_config.json`, `preprocessor_config.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `vocab`, `merges.txt` |
| `whisper-base` | `config.json`, `generation_config.json`, `preprocessor_config.json`, `normalizer.json`, `added_tokens.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` |
| `Qwen3.5-4B-IQ4_NL.gguf` | (single file) |

Only the weight digests are contractual. Upstream may re-upload tokenizer
metadata; if a tokenizer file has changed, the tests still pass because no result
depends on byte-identical tokenizer JSON — the prompts are embedded in
`data/benchmarks/*.jsonl`.

## Disk and time

| Step | Cost |
|---|---|
| `pytest tests/ -q` with all weights present | ~13 s on CPU |
| `python3 verify_official.py` (reference vs official `transformers` forward) | a few minutes on CPU |
| full P1/P2/P8 sweeps | hours on GPU (CUDA 13.5, sm_120) |

A clone without weights still runs the checks that only read the committed result
JSONs and the delivery manifest (`33 passed, 10 skipped`); the model-dependent modules
are skipped with a message naming the missing file and pointing here.
