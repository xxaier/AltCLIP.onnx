#!/usr/bin/env python

import torch
import torch.nn as nn
from .device import DEVICE
from .config import MODEL_FP
from flagai.model.mm.AltCLIP import CLIPHF

MODEL = CLIPHF.from_pretrained(MODEL_FP)

# loader = AutoLoader(task_name="txt_img_matching",
#                     model_name=MODEL_NAME,
#                     model_dir=MODEL_DIR)
#
# MODEL = loader.get_model()

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

  def forward(self, text, attention_mask):
    return self.model.get_text_features(text, attention_mask=attention_mask)


IMG = ImgModel()
TXT = TxtModel()
