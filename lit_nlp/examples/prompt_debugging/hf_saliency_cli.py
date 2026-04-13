"""Super-simple CLI for token-level saliency on Hugging Face causal LMs.

Example:

  python -m lit_nlp.examples.prompt_debugging.hf_saliency_cli \
    --model Qwen/Qwen3-0.6B \
    --prompt "Translate to French: I love science." \
    --max-new-tokens 8
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


METHODS = ("grad_l2", "grad_dot_input")
DTYPES = ("auto", "float32", "bfloat16", "float16")
PROMPT_STYLE = "bold cyan"
RESPONSE_STYLE = "bold green"
TARGET_STYLE = "bold black on yellow"


def clean_subword_token(token: str) -> str:
  token = token.replace("Ċ", "\n")
  token = token.replace("Ġ", "▁")
  token = token.replace("<0x0A>", "\n")
  return token


def parse_token_range(spec: str, upper_bound: int) -> tuple[int, int]:
  if ":" in spec:
    start_str, end_str = spec.split(":", 1)
    start = int(start_str)
    end = int(end_str)
  else:
    start = int(spec)
    end = start + 1
  if start < 0 or end <= start or end > upper_bound:
    raise ValueError(
        f"Invalid token range {spec!r}; expected 0 <= start < end <= {upper_bound}."
    )
  return start, end


def pick_device(device_flag: str) -> str:
  if device_flag != "auto":
    return device_flag
  return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(dtype_flag: str) -> torch.dtype | str:
  if dtype_flag not in DTYPES:
    raise ValueError(f"Unsupported dtype {dtype_flag!r}; expected one of {DTYPES}.")
  if dtype_flag == "auto":
    return "auto"
  return {
      "float32": torch.float32,
      "bfloat16": torch.bfloat16,
      "float16": torch.float16,
  }[dtype_flag]


def load_model(model_name: str, device: str, dtype: torch.dtype | str):
  tokenizer = AutoTokenizer.from_pretrained(
      model_name,
      padding_side="left",
      use_fast=False,
  )
  if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is None:
      raise ValueError("Tokenizer must define either a pad_token or eos_token.")
    tokenizer.pad_token = tokenizer.eos_token

  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      torch_dtype=dtype,
      low_cpu_mem_usage=True,
  )
  model.eval()
  model.to(device)
  return tokenizer, model


def tokenize_text(tokenizer, text: str, device: str) -> dict[str, torch.Tensor]:
  encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
  return {name: tensor.to(device) for name, tensor in encoded.items()}


def maybe_generate_response(
    *,
    model,
    tokenizer,
    prompt: str,
    response: str | None,
    max_new_tokens: int,
    device: str,
) -> tuple[str, torch.Tensor, int]:
  if response is not None:
    encoded_prompt = tokenize_text(tokenizer, prompt, device)
    encoded_full = tokenize_text(tokenizer, prompt + response, device)
    prompt_length = encoded_prompt["input_ids"].shape[1]
    return response, encoded_full["input_ids"], prompt_length

  prompt_inputs = tokenize_text(tokenizer, prompt, device)
  generated = model.generate(
      **prompt_inputs,
      max_new_tokens=max_new_tokens,
      do_sample=False,
      pad_token_id=tokenizer.pad_token_id,
  )
  prompt_length = prompt_inputs["input_ids"].shape[1]
  response_ids = generated[0, prompt_length:]
  response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
  return response_text, generated, prompt_length


def display_sequence(
    console: Console,
    *,
    tokens: Sequence[str],
    prompt_length: int,
) -> None:
  table = Table(show_lines=False, header_style="bold magenta")
  table.add_column("idx", justify="right", no_wrap=True)
  table.add_column("part", no_wrap=True)
  table.add_column("token", overflow="fold")
  for index, token in enumerate(tokens):
    region = "prompt" if index < prompt_length else "response"
    style = PROMPT_STYLE if region == "prompt" else RESPONSE_STYLE
    table.add_row(str(index), region, Text(token or "∅", style=style))
  console.print(Panel(table, title="Tokenized sequence"))


def build_target_mask(length: int, selected_range: tuple[int, int]) -> torch.Tensor:
  mask = torch.zeros((1, length), dtype=torch.bool)
  start, end = selected_range
  mask[:, start:end] = True
  mask[:, 0] = False
  return mask


def compute_saliency(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_mask: torch.Tensor,
    method: str,
) -> torch.Tensor:
  target_ids = torch.roll(input_ids, shifts=-1, dims=1)
  loss_mask = torch.roll(target_mask, shifts=-1, dims=1).to(input_ids.device)

  embedding_table = model.get_input_embeddings()
  embeddings = embedding_table(input_ids)
  embeddings.requires_grad_(True)
  outputs = model(
      input_ids=None,
      inputs_embeds=embeddings,
      attention_mask=attention_mask,
  )
  per_token_loss = torch.nn.functional.cross_entropy(
      outputs.logits.permute(0, 2, 1), target_ids, reduction="none"
  )
  masked_loss = per_token_loss * loss_mask
  grads = torch.autograd.grad(
      masked_loss, embeddings, grad_outputs=torch.ones_like(masked_loss)
  )[0]
  embeddings = embeddings.detach()

  if method == "grad_dot_input":
    scores = torch.sum(grads * embeddings, dim=2)
  elif method == "grad_l2":
    scores = torch.norm(grads, dim=2)
  else:
    raise ValueError(f"Unsupported saliency method {method!r}.")

  scores = (scores * attention_mask).detach().cpu()[0]
  scores[0] = 0
  return scores


def score_style(score: float, max_abs: float, method: str) -> str:
  if max_abs == 0:
    return ""
  intensity = max(0, min(5, int(abs(score) / max_abs * 5)))
  if method == "grad_dot_input":
    palette = (
        ("white on blue", "white on bright_blue"),
        ("white on dark_red", "white on red"),
    )
    negative, positive = palette
    return positive[intensity >= 4] if score >= 0 else negative[intensity >= 4]
  return "black on yellow" if intensity >= 4 else "black on khaki1"


def render_saliency_text(
    *,
    tokens: Sequence[str],
    scores: Sequence[float],
    prompt_length: int,
    target_range: tuple[int, int],
    method: str,
) -> Text:
  max_abs = max((abs(score) for score in scores), default=0.0)
  rendered = Text()
  start, end = target_range
  for index, (token, score) in enumerate(zip(tokens, scores)):
    prefix = f"{index}:"
    if start <= index < end:
      rendered.append(prefix, style=TARGET_STYLE)
      rendered.append(token, style=TARGET_STYLE)
    else:
      rendered.append(
          prefix,
          style=PROMPT_STYLE if index < prompt_length else RESPONSE_STYLE,
      )
      rendered.append(token, style=score_style(score, max_abs, method))
    rendered.append(" ")
  return rendered


def display_results(
    console: Console,
    *,
    tokens: Sequence[str],
    scores: torch.Tensor,
    prompt_length: int,
    target_range: tuple[int, int],
    method: str,
    top_k: int,
) -> None:
  score_values = [float(value) for value in scores.tolist()]
  rendered = render_saliency_text(
      tokens=tokens,
      scores=score_values,
      prompt_length=prompt_length,
      target_range=target_range,
      method=method,
  )
  console.print(
      Panel(
          rendered,
          title=f"Token saliency ({method})",
          subtitle="Prompt tokens are cyan, response tokens are green, targets are yellow.",
      )
  )

  ranked = sorted(
      [
          (index, token, score_values[index])
          for index, token in enumerate(tokens)
          if not (target_range[0] <= index < target_range[1])
      ],
      key=lambda item: abs(item[2]),
      reverse=True,
  )[:top_k]

  table = Table(header_style="bold magenta")
  table.add_column("rank", justify="right")
  table.add_column("idx", justify="right")
  table.add_column("part")
  table.add_column("token")
  table.add_column("score", justify="right")
  for rank, (index, token, score) in enumerate(ranked, start=1):
    table.add_row(
        str(rank),
        str(index),
        "prompt" if index < prompt_length else "response",
        token or "∅",
        f"{score:+.4f}",
    )
  console.print(Panel(table, title=f"Top {len(ranked)} influential tokens"))


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", required=True, help="Hugging Face model name.")
  parser.add_argument("--prompt", required=True, help="Prompt text.")
  parser.add_argument(
      "--response",
      help="Optional response text. If omitted, the model generates one.",
  )
  parser.add_argument(
      "--max-new-tokens",
      type=int,
      default=16,
      help="Number of tokens to generate when --response is omitted.",
  )
  parser.add_argument(
      "--target",
      help="Target token range as start:end (end-exclusive) over the displayed token indices.",
  )
  parser.add_argument(
      "--method",
      choices=METHODS,
      default="grad_l2",
      help="Saliency method to use.",
  )
  parser.add_argument(
      "--dtype",
      choices=DTYPES,
      default="auto",
      help="Torch dtype for model loading.",
  )
  parser.add_argument(
      "--device",
      choices=("auto", "cpu", "cuda"),
      default="auto",
      help="Device to run on.",
  )
  parser.add_argument(
      "--top-k",
      type=int,
      default=12,
      help="How many influential tokens to show in the ranking table.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  console = Console()
  device = pick_device(args.device)
  dtype = pick_dtype(args.dtype)

  console.print(
      Panel.fit(
          f"[bold]Loading[/bold] {args.model}\nmethod={args.method}  device={device}  dtype={args.dtype}"
      )
  )
  try:
    tokenizer, model = load_model(args.model, device, dtype)
  except OSError as exc:
    console.print(
        Panel(
            str(exc),
            title="Model load failed",
            style="bold red",
        )
    )
    raise SystemExit(1) from exc
  response_text, input_ids, prompt_length = maybe_generate_response(
      model=model,
      tokenizer=tokenizer,
      prompt=args.prompt,
      response=args.response,
      max_new_tokens=args.max_new_tokens,
      device=device,
  )
  attention_mask = torch.ones_like(input_ids, device=device)
  tokens = [
      clean_subword_token(token)
      for token in tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
  ]

  console.print(Panel(args.prompt, title="Prompt", style=PROMPT_STYLE))
  console.print(Panel(response_text or "(empty response)", title="Response", style=RESPONSE_STYLE))
  display_sequence(console, tokens=tokens, prompt_length=prompt_length)

  safe_target_start = prompt_length
  if safe_target_start >= input_ids.shape[1]:
    safe_target_start = max(1, input_ids.shape[1] - 1)
  target_spec = args.target or Prompt.ask(
      "Target token range",
      default=f"{safe_target_start}:{input_ids.shape[1]}",
  )
  target_range = parse_token_range(target_spec, input_ids.shape[1])
  target_mask = build_target_mask(input_ids.shape[1], target_range).to(device)

  model.zero_grad(set_to_none=True)
  scores = compute_saliency(
      model=model,
      input_ids=input_ids,
      attention_mask=attention_mask,
      target_mask=target_mask,
      method=args.method,
  )
  display_results(
      console,
      tokens=tokens,
      scores=scores,
      prompt_length=prompt_length,
      target_range=target_range,
      method=args.method,
      top_k=args.top_k,
  )


if __name__ == "__main__":
  main()
