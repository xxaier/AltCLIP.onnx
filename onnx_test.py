#!/usr/bin/env python

import onnxruntime
from os.path import join, dirname, abspath
from export.config import MODEL_NAME

ROOT = dirname(abspath(__file__))
session = onnxruntime.SessionOptions()
option = onnxruntime.RunOptions()
option.log_severity_level = 2

kind = 'txt'
fp = join(ROOT, f'onnx/{MODEL_NAME}/{kind}.onnx')

sess = onnxruntime.InferenceSession(
    fp,
    sess_options=session,
    providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])

input_name = sess.get_inputs()
output_name = sess.get_outputs()[0].name

print(input_name)
print(output_name)
output = sess.run([output_name],
                  {"input": ["a photo of dog", 'a photo of cat']})
# prob = np.squeeze(output[0])

print(">", output)
