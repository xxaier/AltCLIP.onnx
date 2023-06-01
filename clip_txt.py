#!/usr/bin/env python

import torch
from PIL import Image
from os.path import join
from time import time
from glob import glob
from wrap.config import ROOT
from wrap.proc import transform, tokenizer
from wrap.device import DEVICE
from wrap.clip_model import IMG, TXT

COST = None


def inference(jpg, tmpl_kind_li):
  global COST
  image = Image.open(jpg)
  image = transform(image)
  image = torch.tensor(image["pixel_values"]).to(DEVICE)
  print('image.size', image.size())
  with torch.no_grad():
    begin = time()
    image_features = IMG.forward(image)
    if COST is not None:
      COST += (time() - begin)

  for tmpl, kind_li in tmpl_kind_li:
    begin = time()
    with torch.no_grad():
      li = []
      for i in kind_li:
        li.append(tmpl % i)
      text_features = TXT.forward(*tokenizer(li))
      text_probs = (image_features @ text_features.T).softmax(dim=-1)

    if COST is not None:
      COST += (time() - begin)

      for kind, p in zip(kind_li, text_probs.cpu().numpy()[0].tolist()):
        p = round(p * 10000)
        if p:
          print("  %s %.1f%%" % (kind, p / 100))
  return


if __name__ == "__main__":
  li = glob(join(ROOT, 'jpg/*.jpg'))
  # 预热，py.compile 要第一次运行才编译
  txt2vec((
      'a photo of dog',
      'a photo for chinese woman',
  ))
