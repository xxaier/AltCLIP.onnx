#!/usr/bin/env python

from wrap.proc import transform
from PIL import Image
from onnx_load import onnx_load


class ImgVec:

  def __init__(self):
    self.sess = onnx_load('img')

  def __call__(self, img):
    output = self.sess.run(None, {'input': transform(img)})
    return output


if __name__ == '__main__':
  from wrap.config import IMG_DIR
  from os.path import join
  img = Image.open(join(IMG_DIR, 'cat.jpg'))
  img2vec = ImgVec()
  vec = img2vec(img)
  print(vec)
