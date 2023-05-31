#!/usr/bin/env python

import torch
from config import MODEL_NAME, ONNX_DIR, opset_version
from path import join
from clip_model import IMG, TXT

torch.onnx.export(
    IMG,  # model being run
    image,  # model input (or a tuple for multiple inputs)
    join(ONNX_DIR, MODEL_NAME + ".img.onnx"
         ),  # where to save the model (can be a file or file-like object)
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

torch.onnx.export(
    TXT,  # model being run
    text,  # model input (or a tuple for multiple inputs)
    join(ONNX_DIR, MODEL_NAME + "txt.onnx"
         ),  # where to save the model (can be a file or file-like object)
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
