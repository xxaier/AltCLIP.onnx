#!/usr/bin/env python

from config import ONNX_FP, MODEL_NAME
from os.path import join
import onnx

fp = join(ONNX_FP, f'img/{MODEL_NAME}.img.onnx')
onnx.checker.check_model(fp)
