#!/usr/bin/env python

from PIL import Image
from os import makedirs
from os.path import join
from wrap.clip_model import TXT, IMG
from wrap.config import ONNX_FP, ROOT, opset_version
from wrap.device import DEVICE
from wrap.proc import transform, tokenizer
import torch

JPG = join(ROOT, 'jpg/cat.jpg')

image = Image.open(JPG)
image = transform(image)
image = torch.tensor(image["pixel_values"]).to(DEVICE)


def onnx_export(outdir, model, args, **kwds):
  makedirs(ONNX_FP, exist_ok=True)
  name = f'{outdir}.onnx'
  fp = join(ONNX_FP, name)
  torch.onnx.export(
      model,
      args,
      fp,
      export_params=True,
      # verbose=True,
      opset_version=opset_version,
      do_constant_folding=False,
      input_names=['input'],
      output_names=['output'],
      **kwds)
  print(name, "DONE\n")
  # rename(fp, join(ONNX_DIR, name))


# 参考 https://github.com/OFA-Sys/Chinese-CLIP/blob/master/cn_clip/deploy/pytorch_to_onnx.py

onnx_export('txt',
            TXT,
            tokenizer(['a photo of cat', 'a image of cat'], ),
            dynamic_axes={'input': {
                0: 'batch',
                1: 'batch'
            }})

onnx_export('img', IMG, image, dynamic_axes={'input': {0: 'batch'}})
