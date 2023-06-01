#!/usr/bin/env python

from os.path import abspath, dirname, join

ROOT = dirname(abspath(__file__))
MODEL_DIR = join(ROOT, 'model')
MODEL_NAME = 'AltCLIP-XLMR-L-m18'
MODEL_FP = join(MODEL_DIR, MODEL_NAME)
ONNX_DIR = join(dirname(ROOT), 'onnx')
ONNX_FP = join(ONNX_DIR, MODEL_NAME)

opset_version = 17
