#!/usr/bin/env python

from flagai.model.mm.AltCLIP import AltCLIPProcess
from wrap.config import MODEL_FP

proc = AltCLIPProcess.from_pretrained(MODEL_FP)
