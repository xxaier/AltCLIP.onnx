#!/usr/bin/env python

from config import ONNX_FP, MODEL_NAME
from os.path import join
import onnx

onnx_model = onnx.load(join(ONNX_FP, f'img/{MODEL_NAME}.img.onnx'))
onnx.checker.check_model(onnx_model)
