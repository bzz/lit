# Token Saliency CLIs (standalone, no lit dependency)

This repo now includes two standalone saliency CLIs that extract the core saliency logic from LIT's Hugging Face saliency path.

## Provenance in this repo

- `lit_nlp/components/gradient_maps.py`
  - `GradientNorm._interpret()` computes L2 norm over token gradients.
  - `GradientDotInput._interpret()` computes gradient-dot-input.
- `lit_nlp/examples/prompt_debugging/transformers_lms.py`
  - `HFSalienceModel._pred_pt()` and `_pred_tf()` show how to compute token losses, backprop to embeddings, and derive per-token saliency scores.

## Scripts

- `tools_hf_token_saliency_cli.py`
  - PyTorch + Hugging Face transformers version (cross-platform).
- `tools_mlx_token_saliency_cli.py`
  - MLX version for macOS / Apple Silicon.

## How it works

Both CLIs follow LIT's prompt-debugging saliency pattern:

1. **Prompt + target**: The prompt (`--text`) and target (`--target`) are concatenated, then tokenized as a single sequence.
2. **Generation**: If `--target` is omitted, the model generates a continuation first.
3. **Masked loss**: Cross-entropy loss is computed only on the target portion (not the prompt), then backpropagated through the input embeddings.
4. **Saliency scores**: Per-token scores are derived from the gradients — showing which tokens (especially in the prompt) most influenced the target output.

## Usage (PyTorch + HF)

```bash
# Generate target automatically:
python tools_hf_token_saliency_cli.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --text "Explain why the sky appears blue." \
  --method grad-norm \
  --device cpu \
  --raw

# Provide target explicitly:
python tools_hf_token_saliency_cli.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --text "Explain why the sky appears blue." \
  --target "The sky appears blue due to Rayleigh scattering." \
  --method grad-dot-input \
  --raw

# Compute saliency for specific target tokens only:
python tools_hf_token_saliency_cli.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --text "Explain why the sky appears blue." \
  --target "The sky appears blue due to Rayleigh scattering." \
  --target-ids 0-2,5 \
  --raw
```

## Usage (MLX on macOS)

```bash
# Generate target automatically:
python tools_mlx_token_saliency_cli.py \
  --model Qwen/Qwen3-0.6B-MLX-4bit \
  --text "Summarize gradient saliency in one sentence." \
  --method grad-dot-input \
  --raw

# Provide target explicitly:
python tools_mlx_token_saliency_cli.py \
  --model Qwen/Qwen3-0.6B-MLX-4bit \
  --text "Summarize gradient saliency in one sentence." \
  --target "Gradient saliency measures token importance via loss gradients." \
  --method grad-norm \
  --raw

# Compute saliency for specific target tokens only:
python tools_mlx_token_saliency_cli.py \
  --model Qwen/Qwen3-0.6B-MLX-4bit \
  --text "Summarize gradient saliency in one sentence." \
  --target "Gradient saliency measures token importance via loss gradients." \
  --target-ids 0-2,5 \
  --raw
```

If the tokenizer requires remote code:

```bash
python tools_mlx_token_saliency_cli.py \
  --model Qwen/Qwen3-0.6B-MLX-4bit \
  --text "Hello" \
  --trust-remote-code
```

## `--target-ids` format

Accepts comma-separated indices or ranges relative to the target portion (0 = first target token):

- `0,2,5` — select tokens at indices 0, 2, and 5
- `0-3,7` — select tokens at indices 0, 1, 2, 3, and 7

When `--target-ids` is specified, loss is computed only for the selected tokens. The full output is still displayed, with the selected tokens highlighted (bold + underline in the heatmap, `target*` in the raw table).

## Notes

- Both scripts use the next-token language-modeling objective (shifted labels).
- Both scripts support `grad-norm` and `grad-dot-input`.
- Both scripts use Rich for terminal token highlighting.
- To skip the HuggingFace Hub online check for cached models, set `HF_HUB_OFFLINE=1`.
