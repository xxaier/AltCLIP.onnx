#!/usr/bin/env python

from wrap.proc import tokenizer
from onnx_load import onnx_load


class TxtVec:

  def __init__(self):
    self.sess = onnx_load('txt')

  def __call__(self, li):
    text, attention_mask = tokenizer(li)
    text = text.numpy()
    attention_mask = attention_mask.numpy()
    output = self.sess.run(None, {
        'input': text,
        'attention_mask': attention_mask
    })
    return output[0]


if __name__ == '__main__':
  txt2vec = TxtVec()
  from test_txt import TEST_TXT
  for li in TEST_TXT:
    r = txt2vec(li)
    for txt, i in zip(li, r):
      print(txt)
      print(i)
      print('\n')
