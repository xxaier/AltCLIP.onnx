#!/usr/bin/env python

from wrap.proc import transform
from PIL import Image
from onnx_load import onnx_load


class ImgVec:

  def __init__(self):
    self.sess = onnx_load('img')

  def __call__(self, image):
    output = self.sess.run(None, {'input': transform(image).numpy()})
    return output


if __name__ == '__main__':
  from wrap.config import ROOT
  from os.path import join
  img = Image.open(join(ROOT, 'cat.jpg'))
  img2vec = ImgVec()
  vec = img2vec(img)
  print(vec)
