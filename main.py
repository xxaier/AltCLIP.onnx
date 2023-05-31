#!/usr/bin/env python

import torch
from PIL import Image
from config import MODEL, ROOT
from os.path import basename, join
from flagai.auto_model.auto_loader import AutoLoader
from time import time
from glob import glob

if torch.cuda.is_available():
  DEVICE = 'cuda'
elif torch.backends.mps.is_available():
  DEVICE = 'mps'
else:
  DEVICE = 'cpu'

device = torch.device(DEVICE)

print(DEVICE, MODEL)

loader = AutoLoader(task_name="txt_img_matching",
                    model_name="AltCLIP-XLMR-L-m18",
                    model_dir=MODEL)

model = loader.get_model()
tokenizer = loader.get_tokenizer()
transform = loader.get_transform()

model.eval()
model.to(device)
#model = torch.compile(model)
tokenizer = loader.get_tokenizer()

COST = None


def inference(jpg, tmpl, kind_li):
  image = Image.open(jpg)
  image = transform(image)
  image = torch.tensor(image["pixel_values"]).to(device)
  begin = time()
  tokenizer_out = tokenizer([tmpl % i for i in kind_li],
                            padding=True,
                            truncation=True,
                            max_length=77,
                            return_tensors='pt')
  text = tokenizer_out["input_ids"].to(device)
  attention_mask = tokenizer_out["attention_mask"].to(device)
  with torch.no_grad():
    image_features = model.get_image_features(image)
    text_features = model.get_text_features(text,
                                            attention_mask=attention_mask)
    text_probs = (image_features @ text_features.T).softmax(dim=-1)

  global COST
  if COST is not None:
    COST += (time() - begin)
    for kind, p in zip(kind_li, text_probs.cpu().numpy()[0].tolist()):
      p = round(p * 10000)
      if p:
        print("  %s %.2f%%" % (kind, p / 100))
  return


if __name__ == "__main__":
  li = glob(join(ROOT, 'jpg/*.jpg'))
  # 预热，py.compile 要第一次运行才编译
  inference(li[0], 'a photo of %s', ['cat', 'rat', 'dog', 'man', 'woman'])
  COST = 0
  for i in li:
    print("\n* " + basename(i))
    inference(i, 'a photo of %s', ['cat', 'rat', 'dog', 'man', 'woman'])
    inference(i, '一张%s的图片', ['猫', '老鼠', '狗', '男人', '女人'])
  print('\ncost %2.fms' % (1000 * COST))
