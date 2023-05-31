#!/usr/bin/env python

from flagai.model.mm.AltCLIP import AltCLIPProcess
from config import FP_MODEL

proc = AltCLIPProcess.from_pretrained(FP_MODEL)

tokenizer = proc.tokenizer
transform = proc.feature_extractor
