#!/usr/bin/env python

from flagai.model.mm.AltCLIP import AltCLIPProcess
from config import M18

proc = AltCLIPProcess.from_pretrained(M18)

print(dir(proc))
print(proc)
tokenizer = proc.tokenizer
transform = proc.feature_extractor
