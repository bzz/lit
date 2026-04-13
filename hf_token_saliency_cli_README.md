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

## Usage (PyTorch + HF)

```bash
python tools_hf_token_saliency_cli.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --text "Explain why the sky appears blue." \
  --method grad-norm \
  --device cpu \
  --raw
```

## Usage (MLX on macOS)

```bash
python tools_mlx_token_saliency_cli.py \
  --model Qwen/Qwen3-0.6B-MLX-4bit \
  --text "Summarize gradient saliency in one sentence." \
  --method grad-dot-input \
  --raw
```

If the tokenizer requires remote code:

```bash
python tools_mlx_token_saliency_cli.py \
  --model Qwen/Qwen3-0.6B-MLX-4bit \
  --text "Hello" \
  --trust-remote-code
```

## Notes

- Both scripts use the next-token language-modeling objective (shifted labels).
- Both scripts support `grad-norm` and `grad-dot-input`.
- Both scripts use Rich for terminal token highlighting.
