#!/usr/bin/env python

import torch.nn as nn
from clip_model import MODEL, DEVICE


class ImgModel(nn.Module):

  def __init__(self):
    super(ImgModel, self).__init__()
    self.model = MODEL

  def forward(self, image):
    return self.model.get_image_features(image)


class TxtModel(nn.Module):

  def __init__(self):
    super(TxtModel, self).__init__()
    self.model = MODEL

  def forward(self, tmpl, kind_li, image):
    tokenizer_out = tokenizer([tmpl % i for i in kind_li],
                              padding=True,
                              truncation=True,
                              max_length=77,
                              return_tensors='pt')
    text = tokenizer_out["input_ids"].to(DEVICE)
    attention_mask = tokenizer_out["attention_mask"].to(DEVICE)
    return self.model.get_text_features(text, attention_mask=attention_mask)


IMG = ImgModel()
TXT = TxtModel()

# torch.onnx.export(
#     img_model,  # model being run
#     image,  # model input (or a tuple for multiple inputs)
#     "openai_vit_img.onnx",  # where to save the model (can be a file or file-like object)
#     export_params=
#     True,  # store the trained parameter weights inside the model file
#     opset_version=12,  # the ONNX version to export the model to
#     do_constant_folding=
#     False,  # whether to execute constant folding for optimization
#     input_names=['input'],  # the model's input names
#     output_names=['output'],  # the model's output names
#     dynamic_axes={'input': {
#         0: 'batch'
#     }})
# torch.onnx.export(
#     txt_model,  # model being run
#     text,  # model input (or a tuple for multiple inputs)
#     "openai_vit_txt.onnx",  # where to save the model (can be a file or file-like object)
#     export_params=
#     True,  # store the trained parameter weights inside the model file
#     opset_version=12,  # the ONNX version to export the model to
#     do_constant_folding=
#     False,  # whether to execute constant folding for optimization
#     input_names=['input'],  # the model's input names
#     output_names=['output'],  # the model's output names
#     dynamic_axes={'input': {
#         0: 'batch'
#     }})
