#!/usr/bin/env python

import torch
from PIL import Image
from config import ROOT
from os.path import basename, join
from time import time
from proc import transform,tokenizer
from glob import glob
from device import DEVICE
from clip_model import IMG, TXT

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
  inference(li[0],
            (('a photo of %s', ('cat', 'rat', 'dog', 'man', 'woman')), ))
  COST = 0
  for i in li:
    print("\n* " + basename(i))
    inference(i, (('a photo of %s', ('cat', 'rat', 'dog', 'man', 'woman')),
                  ('一张%s的图片', ('猫', '老鼠', '狗', '男人', '女人'))))
    break
  print('\ncost %2.fms' % (1000 * COST))
