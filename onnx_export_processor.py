#!/usr/bin/env python

from flagai.model.mm.AltCLIP import AltCLIPProcess
from wrap.config import ONNX_DIR, MODEL_FP
from os import makedirs
from os.path import join

makedirs(ONNX_DIR, exist_ok=True)
proc = AltCLIPProcess.from_pretrained(MODEL_FP)
save_fp = join(ONNX_DIR, "AltCLIPProcess")
proc.save_pretrained(save_fp)
