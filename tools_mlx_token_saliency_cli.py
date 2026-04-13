#!/usr/bin/env python3
"""Standalone MLX token saliency CLI for macOS.

Designed for MLX-converted causal LMs (e.g. Qwen/Qwen3-0.6B-MLX-4bit).
Supports:
- grad-norm
- grad-dot-input
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

Method = Literal["grad-norm", "grad-dot-input"]


@dataclass
class SaliencyResult:
    tokens: list[str]
    token_ids: list[int]
    scores: list[float]
    method: Method
    model_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute token-level saliency for an MLX causal LM on macOS."
    )
    parser.add_argument("--model", required=True, help="MLX model repo or local path")
    parser.add_argument("--text", required=True, help="Input text to analyze")
    parser.add_argument(
        "--method",
        default="grad-norm",
        choices=["grad-norm", "grad-dot-input"],
        help="Saliency method",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Tokenizer truncation length (hard cut on token ids)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True into tokenizer config for some models.",
    )
    parser.add_argument("--raw", action="store_true", help="Print raw token saliency table")
    return parser.parse_args()


def score_to_style(score: float, min_s: float, max_s: float, signed: bool) -> str:
    eps = 1e-12
    if signed:
        bound = max(abs(min_s), abs(max_s), eps)
        norm = max(-1.0, min(1.0, score / bound))
        if norm >= 0:
            intensity = int(40 + 180 * norm)
            return f"black on rgb({intensity},80,80)"
        intensity = int(40 + 180 * abs(norm))
        return f"white on rgb(70,70,{intensity})"

    denom = max(max_s - min_s, eps)
    norm = max(0.0, min(1.0, (score - min_s) / denom))
    intensity = int(35 + 200 * norm)
    return f"black on rgb({intensity},110,110)"


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
    # MLX arrays generally expose .tolist(); np fallback for safety.
    try:
        return x.tolist()
    except AttributeError:
        import numpy as np

        return np.asarray(x).tolist()


def compute_saliency(
    model_name: str,
    text: str,
    method: Method,
    max_length: int,
    trust_remote_code: bool,
) -> SaliencyResult:
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load
    except ImportError as exc:
        raise RuntimeError(
            "This script requires `mlx` and `mlx-lm` (macOS / Apple Silicon). "
            "Install with: pip install mlx-lm"
        ) from exc

    tokenizer_config = {"trust_remote_code": True} if trust_remote_code else None
    model, tokenizer = load(model_name, tokenizer_config=tokenizer_config)

    token_ids = tokenizer.encode(text, add_special_tokens=True)
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    if len(token_ids) < 2:
        raise ValueError("Need at least 2 tokens after tokenization to compute next-token loss.")

    input_ids = mx.array([token_ids])
    embed_tokens = _resolve_embed_tokens(model)
    input_embs = embed_tokens(input_ids)

    def loss_fn(embs):
        logits = model(input_ids, input_embeddings=embs)
        logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        per_tok_ce = nn.losses.cross_entropy(logits, targets)
        return per_tok_ce.astype(mx.float32).sum()

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
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    return SaliencyResult(
        tokens=tokens,
        token_ids=token_ids,
        scores=score_list,
        method=method,
        model_name=model_name,
    )


def render_result(console: Console, result: SaliencyResult, show_raw: bool = False) -> None:
    signed = result.method == "grad-dot-input"
    min_s = min(result.scores) if result.scores else 0.0
    max_s = max(result.scores) if result.scores else 0.0

    title = f"MLX Token Saliency — {result.model_name} ({result.method})"
    text = Text()
    for tok, score in zip(result.tokens, result.scores):
        display_tok = tok.replace("Ġ", " ").replace("▁", " ")
        if display_tok == "":
            display_tok = "␠"
        text.append(display_tok, style=score_to_style(score, min_s, max_s, signed=signed))

    console.print(Panel(text, title=title, subtitle="higher intensity = larger |saliency|"))

    if show_raw:
        table = Table(title="Raw token saliency")
        table.add_column("i", justify="right")
        table.add_column("token")
        table.add_column("token_id", justify="right")
        table.add_column("score", justify="right")
        for i, (tok, tid, score) in enumerate(
            zip(result.tokens, result.token_ids, result.scores)
        ):
            table.add_row(str(i), repr(tok), str(tid), f"{score:.6f}")
        console.print(table)


def main() -> None:
    args = parse_args()
    console = Console()
    console.print(f"Loading MLX model: [bold]{args.model}[/bold]")
    result = compute_saliency(
        model_name=args.model,
        text=args.text,
        method=args.method,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
    )
    render_result(console, result, show_raw=args.raw)


if __name__ == "__main__":
    main()
