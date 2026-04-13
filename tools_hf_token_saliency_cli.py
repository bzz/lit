#!/usr/bin/env python3
"""Self-contained Hugging Face token saliency CLI.

Computes token-level saliency matching the LIT prompt-debugging pattern:
- Accepts a prompt (--text) and optionally a target (--target).
- If no target is provided, generates one using model.generate().
- Concatenates prompt + target, computes loss on the target portion only.
- Shows which tokens in the prompt influenced the generated output.

Supports:
- grad-norm: L2 norm of d(loss)/d(embedding) per token.
- grad-dot-input: dot product between gradients and input embeddings.

No dependency on lit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

from rich.console import Console
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
    prompt_len: int
    target_text: str
    target_id_indices: list[int] | None


def parse_target_ids(spec: str) -> set[int]:
    """Parse '0,2-5,8' into {0, 2, 3, 4, 5, 8}."""
    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.update(range(int(lo), int(hi) + 1))
        else:
            indices.add(int(part))
    return indices


def parse_args() -> argparse.Namespace:
    import torch

    parser = argparse.ArgumentParser(
        description="Compute token-level sequence saliency for a Hugging Face causal LM."
    )
    parser.add_argument("--model", required=True, help="HF model id or local path")
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
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (cpu, cuda, cuda:0, ...)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Tokenizer truncation length for the full prompt+target sequence",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Apply the model's chat template (wraps --text as a user message)",
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
        # Diverging: blue (negative) → grey (zero) → red (positive).
        bound = max(abs(min_s), abs(max_s), eps)
        norm = max(-1.0, min(1.0, score / bound))
        if norm >= 0:
            r = int(80 + 170 * norm)
            g = int(80 - 50 * norm)
            b = int(80 - 50 * norm)
            return f"white on rgb({r},{g},{b})"
        a = abs(norm)
        r = int(80 - 50 * a)
        g = int(80 - 50 * a)
        b = int(80 + 170 * a)
        return f"white on rgb({r},{g},{b})"

    # Unsigned: grey (low) → bright yellow (high).
    denom = max(max_s - min_s, eps)
    norm = max(0.0, min(1.0, (score - min_s) / denom))
    r = int(70 + 180 * norm)
    g = int(70 + 160 * norm)
    b = int(70 - 50 * norm)
    return f"black on rgb({r},{g},{b})"


def _render_legend(signed: bool) -> Text:
    """Build a color-scale legend."""
    legend = Text()
    steps = 7
    if signed:
        legend.append("negative ")
        for i in range(steps):
            norm = -1.0 + 2.0 * (i / (steps - 1))
            if norm >= 0:
                r = int(80 + 170 * norm)
                g = int(80 - 50 * norm)
                b = int(80 - 50 * norm)
            else:
                a = abs(norm)
                r = int(80 - 50 * a)
                g = int(80 - 50 * a)
                b = int(80 + 170 * a)
            legend.append("  ", style=f"on rgb({r},{g},{b})")
        legend.append(" positive")
    else:
        legend.append("low ")
        for i in range(steps):
            norm = i / (steps - 1)
            r = int(70 + 180 * norm)
            g = int(70 + 160 * norm)
            b = int(70 - 50 * norm)
            legend.append("  ", style=f"on rgb({r},{g},{b})")
        legend.append(" high")
    return legend


def compute_saliency(
    model_name: str,
    text: str,
    target: str | None,
    method: Method,
    device: str,
    max_length: int,
    max_tokens: int,
    target_ids: set[int] | None,
    chat: bool = False,
) -> SaliencyResult:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(device)

    # Apply chat template if requested.
    prompt_text = text
    if chat:
        messages = [{"role": "user", "content": text}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    # Generate target if not provided.
    if target is None:
        prompt_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=not chat)
        prompt_input_ids = prompt_enc["input_ids"].to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                prompt_input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        new_ids = gen_ids[0][prompt_input_ids.shape[1]:]
        target = tokenizer.decode(new_ids, skip_special_tokens=True)
    assert target is not None

    # Tokenize prompt and target separately, then concatenate IDs.
    # This avoids BPE merging tokens across the boundary.
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=not chat)
    target_id_list = tokenizer.encode(target, add_special_tokens=False)
    full_ids = prompt_ids + target_id_list
    prompt_len = len(prompt_ids)

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
    prompt_len = min(prompt_len, len(full_ids))

    num_target = len(full_ids) - prompt_len
    if num_target < 1:
        raise ValueError(
            "No target tokens after concatenation. "
            "Provide a --target or increase --max-tokens."
        )

    input_ids = torch.tensor([full_ids], device=device)
    attention_mask = torch.ones_like(input_ids)

    # Build loss mask (length = seq_len - 1, matching shifted logits).
    # Position i in shifted logits predicts token i+1.
    # We want loss where predicted token (i+1) is a target token (abs index >= prompt_len).
    # --target-ids uses absolute indices matching the 'i' column in --raw output.
    seq_len = input_ids.shape[1]
    shifted_len = seq_len - 1
    loss_mask = torch.zeros(1, shifted_len, device=device)
    for i in range(prompt_len - 1, shifted_len):
        abs_idx = i + 1  # absolute index of the predicted token
        if target_ids is None or abs_idx in target_ids:
            loss_mask[0, i] = 1.0

    resolved_target_indices = sorted(target_ids) if target_ids is not None else None

    embeddings = model.get_input_embeddings()(input_ids)
    embeddings.retain_grad()

    outputs = model(
        input_ids=None,
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
    )

    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    per_token_loss = per_token_loss.view_as(labels)
    masked_loss = (per_token_loss * loss_mask).sum()

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

    token_ids_list = input_ids[0][attn].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids_list)
    scores_list = scores[attn].detach().cpu().tolist()

    # First token has no next-token loss target.
    if scores_list:
        scores_list[0] = 0.0

    return SaliencyResult(
        tokens=tokens,
        token_ids=token_ids_list,
        scores=scores_list,
        method=method,
        model_name=model_name,
        prompt_len=prompt_len,
        target_text=target,
        target_id_indices=resolved_target_indices,
    )


def _display_tok(tok: str) -> str:
    display = tok.replace("Ġ", " ").replace("▁", " ")
    return display if display else "␠"


def render_result(console: Console, result: SaliencyResult, show_raw: bool = False) -> None:
    # Target text for easy copying.
    console.print(f"\n[bold]Target:[/bold] {result.target_text}\n")

    signed = result.method == "grad-dot-input"
    prompt_scores = result.scores[:result.prompt_len]
    min_s = min(prompt_scores) if prompt_scores else 0.0
    max_s = max(prompt_scores) if prompt_scores else 0.0

    has_selection = result.target_id_indices is not None
    selected: set[int] = set(result.target_id_indices) if result.target_id_indices is not None else set()

    console.print(_render_legend(signed))
    console.print(f"[bold]Prompt saliency[/bold] ({result.method}):")
    prompt_text = Text()
    for i in range(result.prompt_len):
        tok, score = result.tokens[i], result.scores[i]
        prompt_text.append(_display_tok(tok), style=score_to_style(score, min_s, max_s, signed=signed))
    console.print(prompt_text)

    console.print(f"\n[bold]Target:[/bold]")
    target_text = Text()
    for i in range(result.prompt_len, len(result.tokens)):
        tok = result.tokens[i]
        if has_selection and i in selected:
            target_text.append(_display_tok(tok), style="underline")
        else:
            target_text.append(_display_tok(tok), style="dim")
    console.print(target_text)

    if show_raw:
        table = Table(title="Raw token saliency")
        table.add_column("i", justify="right")
        table.add_column("token")
        table.add_column("token_id", justify="right")
        table.add_column("region", justify="center")
        table.add_column("score", justify="right")
        for i, (tok, tid, score) in enumerate(
            zip(result.tokens, result.token_ids, result.scores)
        ):
            if i < result.prompt_len:
                region = "prompt"
                region_style = ""
            else:
                if has_selection and i in selected:
                    region = "target*"
                    region_style = "bold cyan"
                else:
                    region = "target"
                    region_style = "cyan"
            table.add_row(
                str(i), repr(_display_tok(tok)), str(tid),
                Text(region, style=region_style),
                f"{score:.6f}",
            )
        console.print(table)


def main() -> None:
    args = parse_args()
    console = Console()
    console.print(f"Loading model: [bold]{args.model}[/bold] on {args.device}")

    target_ids = parse_target_ids(args.target_ids) if args.target_ids else None

    if args.target is None:
        console.print(f"No --target provided; generating up to {args.max_tokens} tokens...")

    result = compute_saliency(
        model_name=args.model,
        text=args.text,
        target=args.target,
        method=args.method,
        device=args.device,
        max_length=args.max_length,
        max_tokens=args.max_tokens,
        target_ids=target_ids,
        chat=args.chat,
    )
    render_result(console, result, show_raw=args.raw)


if __name__ == "__main__":
    main()
