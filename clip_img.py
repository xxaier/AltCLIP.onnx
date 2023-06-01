#!/usr/bin/env python

from wrap.proc import transform
from wrap.clip_model import IMG


def txt2vec(img):
  return IMG.forward(*transform(img))


if __name__ == "__main__":
  from wrap.config import ROOT
  li = glob(join(ROOT, 'jpg/*.jpg'))
  from test_txt import TEST_TXT
  for li in TEST_TXT:
    r = txt2vec(li)
    for txt, i in zip(li, r):
      print(txt)
      print(i)
      print('\n')
