#!/usr/bin/env python

from wrap.proc import tokenizer
from wrap.clip_model import TXT


def txt2vec(li):
  return TXT.forward(*tokenizer(li))


if __name__ == "__main__":
  from os.path import join
  from glob import glob
  from wrap.config import ROOT
  from test_txt import TEST_TXT

  li = glob(join(ROOT, 'jpg/*.jpg'))
  for li in TEST_TXT:
    r = txt2vec(li)
    for txt, i in zip(li, r):
      print(txt)
      print(i)
      print('\n')
