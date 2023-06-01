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

## 零样本分类测试

对 [./jpg](./jpg) 目录下面五张图分别跑中文和英文提示词的分类

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

## 性能测试

穷人，没有线上 GPU 服务器，跑了一下苹果笔记本和服务器 CPU 的推理耗时。

基于零样本分类测试，跑了一下耗时，包含了 5 张图片的向量生成和每张图片中英各 5 个问题（合计 50 个问题）的文本向量生成。

* torch 2.1.0.dev20230531
* Python 3.11.3

### 苹果笔记本

* MacOS 13.3.1
* Apple M2 Max 38 核心 GPU (简称 M2)

M2 MPS 691ms
M2 CPU 3301ms

### [contabo.com](https://contabo.com) 服务器 （每月 10.49 欧元，大概 80 人民币）

* Ubuntu 22.04.2 LTS
* 16G 内存 6 核 AuthenticAMD

CPU 推理耗时 13161ms

CPU 信息如下
```
Vendor ID:               AuthenticAMD
  Model name:            AMD EPYC 7282 16-Core Processor
    CPU family:          23
    Model:               49
    Thread(s) per core:  1
    Core(s) per socket:  6
    Socket(s):           1
    Stepping:            0
    BogoMIPS:            5599.99
    Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse
                         36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rd
                         tscp lm rep_good nopl cpuid extd_apicid tsc_known_freq pni pclmulqdq
                         ssse3 fma cx16 sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer a
                         es xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy cr8_legacy abm
                          sse4a misalignsse 3dnowprefetch osvw perfctr_core ssbd ibpb stibp vm
                         mcall fsgsbase tsc_adjust bmi1 avx2 smep bmi2 rdseed adx smap clflush
                         opt clwb sha_ni xsaveopt xsavec xgetbv1 wbnoinvd arat umip arch_capab
                         ilities
```

## 转为 ONNX 提速

服务器上的性能有点慢，考虑转为 onnx 提速。

ONNX(Open Neural Network Exchange)，开放神经网络交换，用于在各种深度学习训练和推理框架转换的一个中间表示格式。

在实际业务中，可以使用 Pytorch 或者 TensorFlow 训练模型，导出成 ONNX 格式，然后在转换成目标设备上支撑的模型格式，比如 TensorRT Engine、NCNN、MNN 等格式。

ONNX 可以用 onnx runtime 运行，减少部署的依赖。

[./onnx_export.py](./onnx_export.py) 中我实现了把 AltCLIP-XLMR-L-m18 转为 ONNX。

```python
#!/usr/bin/env python

from PIL import Image
from os import makedirs
from os.path import join
from wrap.clip_model import TXT, IMG
from wrap.config import ONNX_FP, opset_version, IMG_DIR
from wrap.proc import transform, tokenizer
import torch

JPG = join(IMG_DIR, 'cat.jpg')

image = Image.open(JPG)
image = transform(image)
image = torch.tensor(image)


def onnx_export(outdir, model, args, **kwds):
  makedirs(ONNX_FP, exist_ok=True)
  name = f'{outdir}.onnx'
  fp = join(ONNX_FP, name)
  torch.onnx.export(
      model,
      args,
      fp,
      export_params=True,
      # verbose=True,
      opset_version=opset_version,
      do_constant_folding=False,
      output_names=['output'],
      **kwds)
  print(name, "DONE\n")
  # rename(fp, join(ONNX_DIR, name))


# 参考 https://github.com/OFA-Sys/Chinese-CLIP/blob/master/cn_clip/deploy/pytorch_to_onnx.py

onnx_export('txt',
            TXT,
            tokenizer(['a photo of cat', 'a image of cat'], ),
            input_names=['input', 'attention_mask'],
            dynamic_axes={
                'input': {
                    0: 'batch',
                    1: 'batch',
                },
                'attention_mask': {
                    0: 'batch',
                    1: 'batch',
                }
            })

onnx_export('img',
            IMG,
            image,
            input_names=['input'],
            dynamic_axes={'input': {
                0: 'batch'
            }})
```

运行 [./onnx_txt.py](onnx_txt.py) 可以输出文本向量

```python
#!/usr/bin/env python

from wrap.proc import tokenizer
from onnx_load import onnx_load

MODEL = onnx_load('txt')


def txt2vec(li):
  text, attention_mask = tokenizer(li)
  text = text.numpy()
  attention_mask = attention_mask.numpy()
  output = MODEL.run(None, {'input': text, 'attention_mask': attention_mask})
  return output[0]


if __name__ == '__main__':
  from test_txt import TEST_TXT
  for li in TEST_TXT:
    r = txt2vec(li)
    for txt, i in zip(li, r):
      print(txt)
      print(i)
      print('\n')
```

运行 [./onnx_img.py](onnx_img.py) 可以输出图片向量

```python
#!/usr/bin/env python

from wrap.proc import transform
from PIL import Image
from onnx_load import onnx_load

MODEL = onnx_load('img')


def img2vec(img):
  return MODEL.run(None, {'input': transform(img)})[0]


if __name__ == '__main__':
  from wrap.config import IMG_DIR
  from os.path import join
  img = Image.open(join(IMG_DIR, 'cat.jpg'))
  vec = img2vec(img)
  print(vec)
```

如下图所示，onnx 生成的向量和用 pytorch 生成的向量一致。

![](https://pub-b8db533c86124200a9d799bf3ba88099.r2.dev/2023/06/NwoXqac.webp)

服务器上 CPU 推理速度提升到 8296ms , 速度提升 13161/8296 - 1 = 58%

参考 [实操专栏 | 记一次 onnx 加速模型推理尝试](https://zhuanlan.zhihu.com/p/466804404)中，onnx 提速为 0.058/0.033-1 = 75%，效果类似。

### 后续的待做

参考 [onnx simplifier 和 onnx optimizer](https://mp.weixin.qq.com/s/q0Aa2LRpeCPCnIzRJbMmaQ)

* 用 onnx-simplifier 简化 onnx
* 用 onnx optimizer 优化 onnx
* 把 onnx 转为 [MNN](https://www.mnn.zone) ，评测下性能

## 没搞明白的事情

### 为什么分类给的结果概率是只有 0 和 100%？

我测试了一张女人抱着狗的图，给出来的分类还是『狗』100%

```python
with torch.no_grad():
  text_probs = (image_features @ text_features.T).softmax(dim=-1)

for kind, p in zip(kind_li, text_probs.cpu().numpy()[0].tolist()):
  p = round(p * 10000)
  if p:
    print("  %s %.1f%%" % (kind, p / 100))
```

### 是否需要归一化向量后再录入向量数据库才能做搜索匹配 ?

我看[Chinese-CLIP 中这么写](https://github.com/OFA-Sys/Chinese-CLIP/blob/master/README.md)

```
# 对特征进行归一化，请使用归一化后的图文特征用于下游任务
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
```

那么 AltCLIP-XLMR-L-m18 中得到图片和文本的向量

```
image_features = model.get_image_features(image)
text_features = model.get_text_features(text,attention_mask=attention_mask)
```
是不是也需要归一化之后才能放到向量数据库（用来做后续的搜索）；还是说，不需要这个操作？

## 零碎的问题

### 服务器上 aiohttp 安装报错

研究了下，aiohttp 不支持 python3.11, 不管就了是，不影响使用。

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
