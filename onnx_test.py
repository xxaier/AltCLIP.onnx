#!/usr/bin/env python

import onnxruntime
from config import ONNX_FP, MODEL_NAME
from os.path import join

session = onnxruntime.SessionOptions()
option = onnxruntime.RunOptions()
option.log_severity_level = 2

kind = 'txt'
fp = join(ONNX_FP, f'{MODEL_NAME}.{kind}.onnx')

img_session = onnxruntime.InferenceSession(fp,
                                           sess_options=session,
                                           providers=["CoreML"])

print(fp)
