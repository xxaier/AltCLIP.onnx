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
