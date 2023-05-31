#!/usr/bin/env python

import torch
from os.path import abspath, dirname, join

ROOT = dirname(abspath(__file__))
DIR_MODEL = join(ROOT, 'model')
NAME_MODEL = 'AltCLIP-XLMR-L-m18'
FP_MODEL = join(DIR_MODEL, NAME_MODEL)

if torch.cuda.is_available():
  device = 'cuda'
elif torch.backends.mps.is_available():
  device = 'mps'
else:
  device = 'cpu'

DEVICE = torch.device(device)
