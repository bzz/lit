"""Super-simple standalone CLI for HF sequence saliency (PyTorch only).

This script intentionally has no lit_nlp imports.
"""

from __future__ import annotations

import argparse
import json

import torch
import transformers


def _pad1d(arr: list[int], min_len: int, pad_val: int, pad_left: bool, max_len: int) -> list[int]:
  if pad_left:
    padded = [pad_val] * max(0, min_len - len(arr)) + arr
    return padded[-max_len:]
  padded = arr + [pad_val] * max(0, min_len - len(arr))
  return padded[:max_len]


def _left_pad_target_mask(seq_length: int, target_mask: list[int], pad_left: bool) -> torch.Tensor:
  # Match LIT logic: first token position (index 0) is never predicted.
  modified = [0] + list(target_mask[1:])
  padded = _pad1d(
      modified, min_len=seq_length, max_len=seq_length, pad_val=0, pad_left=pad_left
  )
  return torch.tensor([padded], dtype=torch.bool)


def _clean_token(token: str) -> str:
  return token.replace("Ċ", "\n").replace("Ġ", "▁").replace("<0x0A>", "\n")


def _parse_target_mask(raw_mask: str | None, seq_length: int) -> list[int]:
  if not raw_mask:
    return [1] * seq_length
  vals = [int(x.strip()) for x in raw_mask.split(",") if x.strip()]
  if any(v not in (0, 1) for v in vals):
    raise ValueError("--target-mask must contain only 0/1 values.")
  return vals


def run(model_name: str, prompt: str, target: str, target_mask: str | None) -> dict:
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  tokenizer = transformers.AutoTokenizer.from_pretrained(
      model_name, use_fast=False, padding_side="left"
  )
  # Align with LIT behavior for decoder-style models that lack a pad token.
  if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
  model = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(device)
  model.eval()

  text = prompt + target
  encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
  input_ids = encoded["input_ids"]
  attention_mask = encoded["attention_mask"]
  # Next-token targets: token t predicts token t+1.
  # `roll` wraps the last position, matching LIT's implementation; the shifted
  # loss mask below zeroes out that wrapped position.
  target_ids = torch.roll(input_ids, shifts=-1, dims=1)

  user_mask = _parse_target_mask(target_mask, seq_length=target_ids.shape[1])
  padded_target_mask = _left_pad_target_mask(
      seq_length=target_ids.shape[1],
      target_mask=user_mask,
      pad_left=tokenizer.padding_side == "left",
  )
  # Shift the loss mask so it aligns with target_ids above.
  loss_mask = torch.roll(padded_target_mask, shifts=-1, dims=1).to(device)

  embedding_table = model.get_input_embeddings()
  embs = embedding_table(input_ids)
  embs.requires_grad_()
  outs = model(input_ids=None, inputs_embeds=embs, attention_mask=attention_mask)

  loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
  # CrossEntropyLoss expects class dim at index 1: [batch, vocab, seq].
  per_token_loss = loss_fn(outs.logits.permute(0, 2, 1), target_ids)
  masked_loss = per_token_loss * loss_mask

  grads = torch.autograd.grad(
      masked_loss, embs, grad_outputs=torch.ones_like(masked_loss)
  )[0]
  # Match LIT behavior: use detached inputs in grad dot input scoring.
  embs_detached = embs.detach()
  grad_l2 = torch.norm(grads, dim=2)
  grad_dot_input = torch.sum(grads * embs_detached, dim=2)

  keep = attention_mask[0].bool()
  kept_ids = input_ids[0][keep].detach().cpu().tolist()
  tokens = [_clean_token(t) for t in tokenizer.convert_ids_to_tokens(kept_ids)]

  return {
      "model": model_name,
      "prompt": prompt,
      "target": target,
      "tokens": tokens,
      "grad_l2": grad_l2[0][keep].detach().cpu().to(torch.float).tolist(),
      "grad_dot_input": grad_dot_input[0][keep].detach().cpu().to(torch.float).tolist(),
  }


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Compute sequence saliency (grad_l2 + grad_dot_input) for an HF causal LM."
  )
  parser.add_argument("--model", required=True, help="HF model name or local path.")
  parser.add_argument("--prompt", required=True, help="Prompt text.")
  parser.add_argument("--target", default="", help="Optional target continuation.")
  parser.add_argument(
      "--target-mask",
      default=None,
      help="Optional comma-separated 0/1 mask over tokenized prompt+target.",
  )
  parser.add_argument("--indent", type=int, default=2, help="JSON output indent.")
  args = parser.parse_args()

  out = run(
      model_name=args.model,
      prompt=args.prompt,
      target=args.target,
      target_mask=args.target_mask,
  )
  print(json.dumps(out, indent=args.indent, ensure_ascii=False))


if __name__ == "__main__":
  main()
