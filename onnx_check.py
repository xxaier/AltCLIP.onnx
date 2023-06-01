#!/usr/bin/env python

from wrap.config import ONNX_FP
from os.path import join
import onnx


def check(name):
  fp = join(ONNX_FP, f'{name}.onnx')
  print(fp, onnx.checker.check_model(fp) is None)
  # onnx_model = onnx.load(fp)
  # return onnx_model


check('img')
check('txt')
