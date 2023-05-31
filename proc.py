#!/usr/bin/env python

from flagai.model.mm.AltCLIP import AltCLIPProcess
from config import MODEL_FP

proc = AltCLIPProcess.from_pretrained(MODEL_FP)

tokenizer = proc.tokenizer
transform = proc.feature_extractor
