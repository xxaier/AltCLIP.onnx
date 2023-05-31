#!/usr/bin/env python

import torch
from config import DIR_MODEL
from flagai.auto_model.auto_loader import AutoLoader

if torch.cuda.is_available():
  device = 'cuda'
elif torch.backends.mps.is_available():
  device = 'mps'
else:
  device = 'cpu'

DEVICE = torch.device(device)

loader = AutoLoader(task_name="txt_img_matching",
                    model_name="AltCLIP-XLMR-L-m18",
                    model_dir=DIR_MODEL)

MODEL = loader.get_model()

MODEL.eval()
MODEL.to(DEVICE)
MODEL = torch.compile(MODEL)
