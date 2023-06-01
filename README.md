[‼️]: ✏️README.mdt

# AltCLIP-XLMR-L-m18 模型学习 & 踩坑笔记 (未完版)

## 序言

虽然写了多年代码，但是从来没有接触过机器学习。

最近想做一个分享 stable diffusion 模型和图片的网站，需要实现图片搜索和推荐。

这就需要选一种算法对图片和文本向量化，并且我还想能用多国语言搜索，经过多番研究， [AltCLIP-XLMR-L-m18](https://model.baai.ac.cn/model-detail/100095)  挺符合我的需求。

先做一些概念科普，虽然很简单，但是向我这样的新人也是研究了许久才搞明白。

AltCLIP，Alt 是 alternative，也就是『替代物』，翻译成为中文就是 CLIP 的替代品。

CLIP, Connecting text and images，翻译成中文就是：图文对齐。

AltCLIP，简单理解就是『图文对齐的替代品』。

对齐，是机器学习中的一个概念，我理解就是把 A 映射到 C，把 B 映射到 C，如果 A 和 B 是一组，那么 A 和 B 应该是映射到同一个 C。

图文对齐，就是可以对狗的图片生成一个向量，对『一张狗的图片』这句话的文本也生成一个向量，然后这对文本和图片生成的向量应该尽可能的靠近（也就是对齐），向量距离有很多种，可以参见[距离计算方式](https://www.bookstack.cn/read/milvus-1.1-zh/milvus_basics-metric.md)。

XLM-R: State-of-the-art cross-lingual understanding through self-supervision , 通过自我监督实现最先进的跨语言理解。

简单的说，就是对多语言进行对齐，比如把『一张狗的图片』和 "a image of dog" 映射到一起。

L 是大模型（我看模型参数有 4556MB）。

m18 中的 m 是 Multilingual，代表多语言，18 是代表 18 种语言。

AltCLIP-XLMR-L-m18 支持英语、中文、日语、泰语、韩语、印地语、乌克兰语、阿拉伯语、土耳其语、越南语、波兰语、荷兰语、葡萄牙语、意大利语、西班牙语、德语、法语和俄语

### 智源的八卦

AltCLIP-XLMR-L-m18 是 智源出品的模型，

『智源』这是哪家机构，怎么之前好像没听说过？

事实上，这家机构确实很年轻 —— 2018 年创建，诞生至今也不过五年。

智源人工智能研究院依托了北京大学、清华大学、中国科学院、百度、小米、字节跳动、美团点评、旷视科技等北京人工智能领域优势单位共建，实行理事会领导下的院长负责制，张宏江担任理事长，北京大学信息科学技术学院教授黄铁军担任智源研究院院长。

ChatGPT 开发商 OpenAI 的最大投资者， 微软总裁布拉德·史密斯（Brad Smith） 2023 年 4 月表示，中国的研究机构和公司将成为 ChatGPT 的主要竞争对手。

史密斯在东京接受日经亚洲 (Nikkei Asia) 采访时说：

> 我们认为有三家公司处于绝对的前沿，一是与微软合作的 Open AI，二是谷歌，三是北京智源人工智能研究院（Beijing Academy of Artificial Intelligence，BAAI）。

> 谁领先谁落后可能会在一年中的某个时期段发生一些变化，但有一件事是绝对不变的：差距几乎总是以月而不是以年来衡量。

## 第一个坑：下载模型

我一开始去官网下，发现选下载方式中的 zip 或者 pipeline 都不行。

然后才发现用 [AutoLoader](https://github.com/FlagAI-Open/FlagAI/blob/master/examples/AltCLIP-m18/README.md#%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86-inference) 加载模型可以自动下载。

但是 AutoLoader，每次启动都会联网检查（会卡上一会儿），而且没法在无网的地铁上写代码（地铁上只要有座位我就会写代码）。

于是，我拆分了一下，用 [./down.py](./down.py) 来下载模型。

```python
#!/usr/bin/env python

from wrap.config import MODEL_NAME, MODEL_DIR
from flagai.auto_model.auto_loader import AutoLoader

loader = AutoLoader(task_name="txt_img_matching",
                    model_name=MODEL_NAME,
                    model_dir=MODEL_DIR)

loader.get_model()
```

平时开发就直接读取本地模型，参见 [./wrap/clip_model.py](./wrap/clip_model.py)

```
from flagai.model.mm.AltCLIP import CLIPHF

MODEL = CLIPHF.from_pretrained(MODEL_FP)

MODEL.eval()
MODEL.to(DEVICE)
```

## 第二个坑：FlagAI 版本

截止 2023-05-31 日， FlagAI 在 pypi 上发布的版本还不支持 AltCLIP-XLMR-L-m18，需要从代码仓库安装最新版。

[./wrap/setup.sh](./wrap/setup.sh)

我写了一个脚本来搞定这些。

```bash
#!/usr/bin/env bash

DIR=$(realpath $0) && DIR=${DIR%/*}
cd $DIR
set -ex

direnv allow

direnv exec . pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu

direnv exec . pip install -r requirements.txt

if [ ! -d "FlagAI" ]; then
  git clone --depth=1 git@github.com:FlagAI-Open/FlagAI.git
fi

cd FlagAI
git pull
direnv exec . python setup.py install
```

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
