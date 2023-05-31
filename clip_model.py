#!/usr/bin/env python

import torch
import torch.nn as nn
from config import MODEL_DIR, MODEL_NAME, DEVICE
from flagai.auto_model.auto_loader import AutoLoader
from proc import tokenizer

loader = AutoLoader(task_name="txt_img_matching",
                    model_name=MODEL_NAME,
                    model_dir=MODEL_DIR)

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

  def forward(self, li):
    tokenizer_out = tokenizer(li,
                              padding=True,
                              truncation=True,
                              max_length=77,
                              return_tensors='pt')
    text = tokenizer_out["input_ids"].to(DEVICE)
    attention_mask = tokenizer_out["attention_mask"].to(DEVICE)
    return self.model.get_text_features(text, attention_mask=attention_mask)


IMG = ImgModel()
TXT = TxtModel()
