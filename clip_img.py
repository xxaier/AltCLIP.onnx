#!/usr/bin/env python

from wrap.proc import transform
from wrap.clip_model import IMG


def img2vec(img):
  return IMG.forward(*transform(img))


if __name__ == "__main__":
  from wrap.config import IMG_DIR
  from os.path import join
  img = join(IMG_DIR, 'cat.jpg')
