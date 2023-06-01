#!/usr/bin/env python

import onnxruntime
from os.path import join, dirname, abspath
from wrap.config import MODEL_NAME
from wrap.proc import tokenizer

ROOT = dirname(abspath(__file__))
session = onnxruntime.SessionOptions()
option = onnxruntime.RunOptions()
option.log_severity_level = 2


def load(kind):
  fp = join(ROOT, f'onnx/{MODEL_NAME}/{kind}.onnx')

  sess = onnxruntime.InferenceSession(
      fp,
      sess_options=session,
      providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
  return sess


class TxtVec:

  def __init__(self):
    self.sess = load('txt')

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
  r = txt2vec(('a photo of dog', 'a photo for chinese woman'))
  print(len(r[0]))
  for i in r:
    print(i)
