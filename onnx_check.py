#!/usr/bin/env python

from config import ONNX_FP, MODEL_NAME
from os.path import join
import onnx


def check(name):
  fp = join(ONNX_FP, f'{name}/{MODEL_NAME}.{name}.onnx')
  print(fp, onnx.checker.check_model(fp))
  onnx_model = onnx.load(fp)
  return onnx_model


print(check('img'))
