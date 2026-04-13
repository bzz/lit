#!/usr/bin/env python3
"""Self-contained Hugging Face token saliency CLI.

Computes token-level saliency for causal LMs with two methods:
- grad-norm: L2 norm of d(loss)/d(embedding) per token.
- grad-dot-input: dot product between gradients and input embeddings.

No dependency on lit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        description="Compute token-level sequence saliency for a Hugging Face causal LM."
    )
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--text", required=True, help="Input text to analyze")
    parser.add_argument(
        "--method",
        default="grad-norm",
        choices=["grad-norm", "grad-dot-input"],
        help="Saliency method",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (cpu, cuda, cuda:0, ...)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Tokenizer truncation length",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Also print raw token/saliency table",
    )
    return parser.parse_args()


def score_to_style(score: float, min_s: float, max_s: float, signed: bool) -> str:
    eps = 1e-12
    if signed:
        bound = max(abs(min_s), abs(max_s), eps)
        norm = max(-1.0, min(1.0, score / bound))
        # Positive => red; negative => blue.
        if norm >= 0:
            intensity = int(40 + 180 * norm)
            return f"black on rgb({intensity},80,80)"
        intensity = int(40 + 180 * abs(norm))
        return f"white on rgb(70,70,{intensity})"

    denom = max(max_s - min_s, eps)
    norm = max(0.0, min(1.0, (score - min_s) / denom))
    intensity = int(35 + 200 * norm)
    return f"black on rgb({intensity},110,110)"


def compute_saliency(
    model_name: str,
    text: str,
    method: Method,
    device: str,
    max_length: int,
) -> SaliencyResult:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(device)

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)

    embeddings = model.get_input_embeddings()(input_ids)
    embeddings.retain_grad()

    outputs = model(
        input_ids=None,
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
    )

    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    label_mask = attention_mask[:, 1:].float()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    per_token_loss = per_token_loss.view_as(labels)
    masked_loss = (per_token_loss * label_mask).sum()

    model.zero_grad(set_to_none=True)
    if embeddings.grad is not None:
        embeddings.grad.zero_()
    masked_loss.backward()

    grads = embeddings.grad.detach()[0]
    embs = embeddings.detach()[0]
    attn = attention_mask[0].bool()

    if method == "grad-norm":
        scores = torch.linalg.norm(grads, dim=-1)
        scores = scores / (scores.sum() + 1e-12)
    elif method == "grad-dot-input":
        scores = torch.sum(grads * embs, dim=-1)
    else:
        raise ValueError(f"Unsupported method: {method}")

    token_ids = input_ids[0][attn].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    scores = scores[attn].detach().cpu().tolist()

    # First token typically has no next-token loss target.
    if scores:
        scores[0] = 0.0

    return SaliencyResult(
        tokens=tokens,
        token_ids=token_ids,
        scores=scores,
        method=method,
        model_name=model_name,
    )


def render_result(console: Console, result: SaliencyResult, show_raw: bool = False) -> None:
    signed = result.method == "grad-dot-input"
    min_s = min(result.scores) if result.scores else 0.0
    max_s = max(result.scores) if result.scores else 0.0

    title = f"Token Saliency — {result.model_name} ({result.method})"
    text = Text()
    for tok, score in zip(result.tokens, result.scores):
        display_tok = tok.replace("Ġ", " ").replace("▁", " ")
        if display_tok == "":
            display_tok = "␠"
        style = score_to_style(score, min_s, max_s, signed=signed)
        text.append(display_tok, style=style)

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
    console.print(f"Loading model: [bold]{args.model}[/bold] on {args.device}")
    result = compute_saliency(
        model_name=args.model,
        text=args.text,
        method=args.method,
        device=args.device,
        max_length=args.max_length,
    )
    render_result(console, result, show_raw=args.raw)


if __name__ == "__main__":
    main()
