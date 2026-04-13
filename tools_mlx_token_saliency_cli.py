#!/usr/bin/env python3
"""Standalone MLX token saliency CLI for macOS.

Computes token-level saliency matching the LIT prompt-debugging pattern:
- Accepts a prompt (--text) and optionally a target (--target).
- If no target is provided, generates one using mlx_lm.generate.
- Concatenates prompt + target, computes loss on the target portion only.
- Shows which tokens in the prompt influenced the generated output.

Designed for MLX-converted causal LMs (e.g. Qwen/Qwen3-0.6B-MLX-4bit).
Supports:
- grad-norm
- grad-dot-input
"""

from __future__ import annotations

import argparse
from typing import Any

from rich.console import Console

from tools_hf_token_saliency_cli import (  # noqa: E402
    Method,
    SaliencyResult,
    parse_target_ids,
    render_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute token-level saliency for an MLX causal LM on macOS."
    )
    parser.add_argument("--model", required=True, help="MLX model repo or local path")
    parser.add_argument("--text", required=True, help="Prompt text")
    parser.add_argument(
        "--target",
        default=None,
        help="Target text (model output). If omitted, the model generates it.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate when --target is not provided (default: 256)",
    )
    parser.add_argument(
        "--target-ids",
        default=None,
        help="Absolute token indices to compute loss for, matching the 'i' column in --raw (e.g. '11,13-15,23'). Default: all target tokens.",
    )
    parser.add_argument(
        "--method",
        default="grad-norm",
        choices=["grad-norm", "grad-dot-input"],
        help="Saliency method",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="Truncate prompt+target to this many tokens (0 = no limit)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True into tokenizer config for some models.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Apply the model's chat template (wraps --text as a user message)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Color all tokens with saliency (not just the prompt)",
    )
    parser.add_argument("--raw", action="store_true", help="Print raw token saliency table")
    return parser.parse_args()


def _resolve_embed_tokens(model: Any) -> Any:
    paths = [
        ("model", "embed_tokens"),
        ("embed_tokens",),
        ("model", "model", "embed_tokens"),
    ]
    for path in paths:
        cur = model
        ok = True
        for name in path:
            if not hasattr(cur, name):
                ok = False
                break
            cur = getattr(cur, name)
        if ok:
            return cur
    raise AttributeError(
        "Could not locate embedding layer. Tried model.embed_tokens variants."
    )


def _to_py_list(x: Any) -> list[float]:
    try:
        return x.tolist()
    except AttributeError:
        import numpy as np

        return np.asarray(x).tolist()


def compute_saliency(
    model_name: str,
    text: str,
    target: str | None,
    method: Method,
    max_length: int,
    max_tokens: int,
    trust_remote_code: bool,
    target_ids: set[int] | None,
    chat: bool = False,
    full: bool = False,
) -> SaliencyResult:
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import generate, load
    except ImportError as exc:
        raise RuntimeError(
            "This script requires `mlx` and `mlx-lm` (macOS / Apple Silicon). "
            "Install with: pip install mlx-lm"
        ) from exc

    tokenizer_config = {"trust_remote_code": True} if trust_remote_code else None
    model, tokenizer = load(model_name, tokenizer_config=tokenizer_config)

    # Apply chat template if requested.
    prompt_text = text
    if chat:
        messages = [{"role": "user", "content": text}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    # Generate target if not provided.
    if target is None:
        target = generate(model, tokenizer, prompt_text, max_tokens=max_tokens)
    assert target is not None

    # Tokenize prompt and target separately, then concatenate IDs.
    # This avoids BPE merging tokens across the boundary.
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=not chat)
    target_id_list = tokenizer.encode(target, add_special_tokens=False)
    full_ids = prompt_ids + target_id_list
    prompt_len = len(prompt_ids)

    if max_length > 0 and len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        prompt_len = min(prompt_len, len(full_ids))

    num_target = len(full_ids) - prompt_len
    if num_target < 1:
        raise ValueError(
            "No target tokens after concatenation. "
            "Provide a --target or increase --max-tokens."
        )
    if len(full_ids) < 2:
        raise ValueError("Need at least 2 tokens to compute next-token loss.")

    # Build loss mask (length = len(full_ids) - 1, matching shifted logits).
    # Position i in shifted logits predicts token i+1.
    # --target-ids uses absolute indices matching the 'i' column in --raw output.
    # With --full: loss on all tokens (or up to --target-ids).
    # Without --full: loss only on target tokens (abs index >= prompt_len).
    shifted_len = len(full_ids) - 1
    loss_mask = [0.0] * shifted_len
    start = 0 if full else prompt_len - 1
    for i in range(start, shifted_len):
        abs_idx = i + 1  # absolute index of the predicted token
        if target_ids is None or abs_idx in target_ids:
            loss_mask[i] = 1.0

    resolved_target_indices = sorted(target_ids) if target_ids is not None else None

    input_ids = mx.array([full_ids])
    loss_mask_mx = mx.array([loss_mask], dtype=mx.float32)
    embed_tokens = _resolve_embed_tokens(model)
    input_embs = embed_tokens(input_ids)

    def loss_fn(embs):
        logits = model(input_ids, input_embeddings=embs)
        logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        per_tok_ce = nn.losses.cross_entropy(logits, targets)
        masked_ce = per_tok_ce * loss_mask_mx
        return masked_ce.astype(mx.float32).sum()

    loss_and_grad = mx.value_and_grad(loss_fn)
    loss, grads = loss_and_grad(input_embs)
    mx.eval(loss, grads)

    if method == "grad-norm":
        scores = mx.linalg.norm(grads, axis=-1)[0]
        scores = scores / (scores.sum() + 1e-12)
    elif method == "grad-dot-input":
        scores = mx.sum(grads * input_embs, axis=-1)[0]
    else:
        raise ValueError(f"Unsupported method: {method}")

    score_list = _to_py_list(scores)
    score_list[0] = 0.0
    tokens = tokenizer.convert_ids_to_tokens(full_ids)

    return SaliencyResult(
        tokens=tokens,
        token_ids=full_ids,
        scores=score_list,
        method=method,
        model_name=model_name,
        prompt_len=prompt_len,
        target_text=target,
        target_id_indices=resolved_target_indices,
    )


def main() -> None:
    args = parse_args()
    console = Console()
    console.print(f"Loading MLX model: [bold]{args.model}[/bold]")

    target_ids = parse_target_ids(args.target_ids) if args.target_ids else None

    if args.target is None:
        console.print(f"No --target provided; generating up to {args.max_tokens} tokens...")

    result = compute_saliency(
        model_name=args.model,
        text=args.text,
        target=args.target,
        method=args.method,
        max_length=args.max_length,
        max_tokens=args.max_tokens,
        trust_remote_code=args.trust_remote_code,
        target_ids=target_ids,
        chat=args.chat,
        full=args.full,
    )
    render_result(console, result, show_raw=args.raw, full=args.full)


if __name__ == "__main__":
    main()
