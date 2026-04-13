import torch
from lit_nlp.examples.prompt_debugging import hf_saliency_cli
from transformers import GPT2Config
from transformers import GPT2LMHeadModel


def test_parse_token_range():
  assert hf_saliency_cli.parse_token_range("2:4", 6) == (2, 4)
  assert hf_saliency_cli.parse_token_range("3", 6) == (3, 4)


def test_compute_saliency_for_supported_methods():
  model = GPT2LMHeadModel(
      GPT2Config(vocab_size=32, n_positions=16, n_embd=8, n_layer=1, n_head=1)
  )
  model.eval()
  input_ids = torch.tensor([[1, 2, 3, 4, 5]])
  attention_mask = torch.ones_like(input_ids)
  target_mask = hf_saliency_cli.build_target_mask(input_ids.shape[1], (3, 5))

  for method in hf_saliency_cli.METHODS:
    scores = hf_saliency_cli.compute_saliency(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_mask=target_mask,
        method=method,
    )
    assert scores.shape == (input_ids.shape[1],)
    assert torch.isfinite(scores).all()
    assert scores[0].item() == 0
