#!/usr/bin/env python

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
