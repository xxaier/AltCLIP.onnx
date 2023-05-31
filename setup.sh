#!/usr/bin/env bash

DIR=$(realpath $0) && DIR=${DIR%/*}
cd $DIR
set -ex

if [[ uname == *"Darwin"* ]]; then
  pkg=onnxruntime-silicon
else
  pkg=onnxruntime
fi
pip install onnxruntime
