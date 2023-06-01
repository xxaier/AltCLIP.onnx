#!/usr/bin/env python

import onnxruntime
from os.path import join, dirname, abspath
from wrap.config import MODEL_NAME
from wrap.proc import tokenizer

ROOT = dirname(abspath(__file__))
session = onnxruntime.SessionOptions()
option = onnxruntime.RunOptions()
option.log_severity_level = 2

kind = 'txt'
fp = join(ROOT, f'onnx/{MODEL_NAME}/{kind}.onnx')

sess = onnxruntime.InferenceSession(fp,
                                    sess_options=session,
                                    providers=['CPUExecutionProvider'])
# 'CoreMLExecutionProvider'

# for pos, i in enumerate(sess.get_inputs()):
#   print('input', pos, i.name)
# input_name = sess.get_inputs()[1].name
# for pos, i in enumerate(sess.get_outputs()):
#   print('output', pos, i.name)

# attention_mask 在处理多个序列时的作用 https://zhuanlan.zhihu.com/p/414511434


def txt2vec(li):
  text, attention_mask = tokenizer(li)
  text = text.numpy()
  attention_mask = attention_mask.numpy()
  output = sess.run(None, {'input': text, 'attention_mask': attention_mask})
  return output


if __name__ == '__main__':
  r = txt2vec(('a photo of dog', 'a photo for chinese woman'))
  for i in r:
    print(i)
