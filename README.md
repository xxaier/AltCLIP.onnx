## 脱机使用

对 [./jpg](./jpg) 目录下面五张图跑中文和英文提示词的分类

运行输出 :

* dog.jpg
  dog 100.00%
  狗 100.00%

* rat.jpg
  rat 100.00%
  老鼠 100.00%

* man.jpg
  man 100.00%
  男人 100.00%

* cat.jpg
  cat 100.00%
  猫 100.00%

* woman.jpg
  woman 100.00%
  女人 100.00%

## 测试环境

### 苹果笔记本

* torch 2.1.0.dev20230531
* Python 3.11.3
* MacOS 13.3.1
* Apple M2 Max 38 核心 GPU (简称 M2)

### 没加 pytorch.compile

M2 MPS 638ms
M2 CPU 5785ms

### 加了 pytorch.compile

M2 MPS 626ms
M2 CPU 5351ms

## 遇到的问题

### flagai 和 onnx 的 protobuf 版本冲突导致报错

```
Traceback (most recent call last):
  File "/Users/z/art/clip/.direnv/python-3.11.3/lib/python3.11/site-packages/torch/onnx/_internal/onnx_proto_utils.py", line 221, in _add_onnxscript_fn
    import onnx
  File "/Users/z/art/clip/.direnv/python-3.11.3/lib/python3.11/site-packages/onnx/__init__.py", line 13, in <module>
    from onnx.external_data_helper import (
  File "/Users/z/art/clip/.direnv/python-3.11.3/lib/python3.11/site-packages/onnx/external_data_helper.py", line 11, in <module>
    from onnx.onnx_pb import AttributeProto, GraphProto, ModelProto, TensorProto
  File "/Users/z/art/clip/.direnv/python-3.11.3/lib/python3.11/site-packages/onnx/onnx_pb.py", line 4, in <module>
    from .onnx_ml_pb2 import *  # noqa
    ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/z/art/clip/.direnv/python-3.11.3/lib/python3.11/site-packages/onnx/onnx_ml_pb2.py", line 5, in <module>
    from google.protobuf.internal import builder as _builder
ImportError: cannot import name 'builder' from 'google.protobuf.internal' (/Users/z/art/clip/.direnv/python-3.11.3/lib/python3.11/site-packages/protobuf-3.19.6-py3.11.egg/google/protobuf/int
ernal/__init__.py)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/z/art/clip/./onnx_export.py", line 40, in <module>
    onnx_export('txt',
  File "/Users/z/art/clip/./onnx_export.py", line 23, in onnx_export
    torch.onnx.export(
  File "/Users/z/art/clip/.direnv/python-3.11.3/lib/python3.11/site-packages/torch/onnx/utils.py", line 507, in export
    _export(
  File "/Users/z/art/clip/.direnv/python-3.11.3/lib/python3.11/site-packages/torch/onnx/utils.py", line 1639, in _export
    proto = onnx_proto_utils._add_onnxscript_fn(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/z/art/clip/.direnv/python-3.11.3/lib/python3.11/site-packages/torch/onnx/_internal/onnx_proto_utils.py", line 223, in _add_onnxscript_fn
    raise errors.OnnxExporterError("Module onnx is not installed!") from e
torch.onnx.errors.OnnxExporterError: Module onnx is not installed!
```

解决方案
```
pip uninstall -y protobuf
pip install --upgrade protobuf
```
