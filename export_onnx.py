#!/usr/bin/env python

from os.path import join
import torch
from config import ONNX_FP, ROOT, DEVICE, opset_version
from clip_model import IMG
from PIL import Image
from proc import transform

JPG = join(ROOT, 'jpg/cat.jpg')

image = Image.open(JPG)
image = transform(image)
image = torch.tensor(image["pixel_values"]).to(DEVICE)

torch.onnx.export(
    IMG,  # model being run
    image,  # model input (or a tuple for multiple inputs)
    ONNX_FP +
    "img.onnx",  # where to save the model (can be a file or file-like object)
    export_params=
    True,  # store the trained parameter weights inside the model file
    opset_version=opset_version,  # the ONNX version to export the model to
    do_constant_folding=
    False,  # whether to execute constant folding for optimization
    input_names=['input'],  # the model's input names
    output_names=['output'],  # the model's output names
    dynamic_axes={'input': {
        0: 'batch'
    }})

# torch.onnx.export(
#     TXT,  # model being run
#     text,  # model input (or a tuple for multiple inputs)
#     ONNX_FP +
#     "txt.onnx",  # where to save the model (can be a file or file-like object)
#     export_params=
#     True,  # store the trained parameter weights inside the model file
#     opset_version=opset_version,  # the ONNX version to export the model to
#     do_constant_folding=
#     False,  # whether to execute constant folding for optimization
#     input_names=['input'],  # the model's input names
#     output_names=['output'],  # the model's output names
#     dynamic_axes={'input': {
#         0: 'batch'
#     }})
