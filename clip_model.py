#!/usr/bin/env python

import torch
import torch.nn as nn
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


class ImgModel(nn.Module):

  def __init__(self):
    super(ImgModel, self).__init__()
    self.model = MODEL

  def forward(self, image):
    return self.model.get_image_features(image)


class TxtModel(nn.Module):

  def __init__(self):
    super(TxtModel, self).__init__()
    self.model = MODEL

  def forward(self, tmpl, kind_li, image):
    tokenizer_out = tokenizer([tmpl % i for i in kind_li],
                              padding=True,
                              truncation=True,
                              max_length=77,
                              return_tensors='pt')
    text = tokenizer_out["input_ids"].to(DEVICE)
    attention_mask = tokenizer_out["attention_mask"].to(DEVICE)
    return self.model.get_text_features(text, attention_mask=attention_mask)


IMG = ImgModel()
TXT = TxtModel()
