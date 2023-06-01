#!/usr/bin/env python

import onnxruntime
from wrap.config import MODEL_NAME
from os.path import join, dirname, abspath

ROOT = dirname(abspath(__file__))
session = onnxruntime.SessionOptions()
option = onnxruntime.RunOptions()
option.log_severity_level = 2


def onnx_load(kind):
  fp = join(ROOT, f'onnx/{MODEL_NAME}/{kind}.onnx')

  sess = onnxruntime.InferenceSession(
      fp,
      sess_options=session,
      providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
  return sess
