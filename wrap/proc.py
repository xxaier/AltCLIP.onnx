#!/usr/bin/env python

from flagai.model.mm.AltCLIP import AltCLIPProcess
from .config import MODEL_FP

proc = AltCLIPProcess.from_pretrained(MODEL_FP)

_tokenizer = proc.tokenizer
transform = proc.feature_extractor


def tokenizer(li):
  tokenizer_out = _tokenizer(li,
                             padding=True,
                             truncation=True,
                             max_length=77,
                             return_tensors='pt')
  text = tokenizer_out["input_ids"]
  attention_mask = tokenizer_out["attention_mask"]
  return text, attention_mask
