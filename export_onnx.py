#!/usr/bin/env python

from os import makedirs, rename
from os.path import join
import torch
from config import ONNX_FP, ONNX_DIR, MODEL_NAME, ROOT, DEVICE, opset_version
from clip_model import IMG, TXT
from PIL import Image
from proc import transform

JPG = join(ROOT, 'jpg/cat.jpg')

image = Image.open(JPG)
image = transform(image)
image = torch.tensor(image["pixel_values"]).to(DEVICE)


def onnx_export(outdir, model, args):
  onnx_txt = join(ONNX_FP, outdir)
  makedirs(onnx_txt, exist_ok=True)
  name = f'{MODEL_NAME}.{outdir}.onnx'
  fp = join(onnx_txt, name)
  torch.onnx.export(model,
                    args,
                    fp,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=False,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {
                        0: 'batch'
                    }})
  # rename(fp, join(ONNX_DIR, name))


onnx_export('img', IMG, image)

onnx_export('txt', TXT, ['a photo of cat', 'a image of cat'])
