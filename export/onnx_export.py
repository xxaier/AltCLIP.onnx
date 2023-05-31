#!/usr/bin/env python

from os import makedirs
from os.path import join
import torch
from config import ONNX_FP, MODEL_NAME, ROOT, DEVICE, opset_version
from clip_model import IMG, TXT
from PIL import Image
from proc import transform

JPG = join(ROOT, 'jpg/cat.jpg')

image = Image.open(JPG)
image = transform(image)
image = torch.tensor(image["pixel_values"]).to(DEVICE)


def onnx_export(outdir, model, args, **kwds):
  makedirs(ONNX_FP, exist_ok=True)
  name = f'{MODEL_NAME}.{outdir}.onnx'
  fp = join(ONNX_FP, name)
  torch.onnx.export(model,
                    args,
                    fp,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=False,
                    input_names=['input'],
                    output_names=['output'],
                    **kwds)
  # rename(fp, join(ONNX_DIR, name))


# 参考 https://github.com/OFA-Sys/Chinese-CLIP/blob/master/cn_clip/deploy/pytorch_to_onnx.py

onnx_export('img', IMG, image, dynamic_axes={'input': {0: 'batch'}})

onnx_export('txt', TXT, ['a photo of cat', 'a image of cat'])
