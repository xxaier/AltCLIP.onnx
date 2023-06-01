# AltCLIP-XLMR-L-m18 模型学习 & 踩坑笔记

## 序言

虽然写了多年代码，但是从来没有接触过机器学习。

最近想做一个分享 stable diffusion 模型和图片的网站，需要实现图片搜索和推荐。

这就需要对图片向量化，并且我还想同时支持多国语言搜索，经过多番研究，感觉 [AltCLIP-XLMR-L-m18](https://model.baai.ac.cn/model-detail/100095) 我的需求。

这里先做一些概念科普，虽然很简单，但是向我这样的新人也是研究了许久才搞明白。

AltCLIP，Alt 是 alternative，也就是『替代物』，翻译成为中文就是 CLIP 的替代品。

CLIP, Connecting text and images，翻译成中文就是：图文对齐。

AltCLIP，简单理解就是『图文对齐的替代品』。

对齐，是机器学习中的一个概念，我理解就是把 A 映射到 C，把 B 映射到 C，如果 A 和 B 是一组，那么 A 和 B 应该是映射到同一个 C。

图文对齐，就是可以对狗的图片生成一个向量，对『一张狗的图片』这句话的文本也生成一个向量，然后这对文本和图片生成的向量应该尽可能的靠近（也就是对齐），向量距离有很多种，可以参见[距离计算方式](https://www.bookstack.cn/read/milvus-1.1-zh/milvus_basics-metric.md)。

## 下载模型

## 脱机使用

## 零样本分类测试

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

M2 MPS 691ms
M2 CPU 3301ms

## 遇到的问题

### 服务器上安装报错

```
/root/art/clip_test/.direnv/python-3.11.3/lib/python3.11/site-packages/setuptools/command/install.py:34: SetuptoolsDeprecationWarning: setup.py install is deprecated. Use build and pip and o
ther standards-based tools.
  warnings.warn(
aiohttp/_websocket.c:196:12: fatal error: longintrepr.h: No such file or directory
  196 |   #include "longintrepr.h"
      |            ^~~~~~~~~~~~~~~
```

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
