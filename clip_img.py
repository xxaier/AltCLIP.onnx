#!/usr/bin/env python

from wrap.proc import transform
from wrap.clip_model import IMG
import torch


def img2vec(img):
  img = transform(img)
  img = torch.tensor(img)
  return IMG.forward(img)


if __name__ == "__main__":
  from wrap.config import IMG_DIR
  from os.path import join
  fp = join(IMG_DIR, 'cat.jpg')
  from PIL import Image
  img = Image.open(fp)
  print(img2vec(img))
