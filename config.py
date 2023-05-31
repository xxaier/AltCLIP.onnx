#!/usr/bin/env python

import torch
from os.path import abspath, dirname, join

ROOT = dirname(abspath(__file__))
MODEL_DIR = join(ROOT, 'model')
MODEL_NAME = 'AltCLIP-XLMR-L-m18'
MODEL_FP = join(MODEL_DIR, MODEL_NAME)

if torch.cuda.is_available():
  device = 'cuda'
elif torch.backends.mps.is_available():
  device = 'mps'
else:
  device = 'cpu'

DEVICE = torch.device(device)
