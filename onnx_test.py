#!/usr/bin/env python

import onnxruntime
from os.path import join, dirname, abspath
from export.config import MODEL_NAME

ROOT = dirname(abspath(__file__))
session = onnxruntime.SessionOptions()
option = onnxruntime.RunOptions()
option.log_severity_level = 2

kind = 'txt'
fp = join(ROOT, f'onnx/{MODEL_NAME}.{kind}.onnx')

img_session = onnxruntime.InferenceSession(
    fp,
    sess_options=session,
)

print(fp)
