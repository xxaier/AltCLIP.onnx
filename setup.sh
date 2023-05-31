#!/usr/bin/env bash

DIR=$(realpath $0) && DIR=${DIR%/*}
cd $DIR
set -ex

direnv allow

direnv exec . pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu

direnv exec . pip install -r requirements.txt

if [[ $(uname) != *"Darwin"* ]]; then
  direnv exec . pip install onnxruntime-silicon
else
  pip install onnxruntime
fi

if [ ! -d "FlagAI" ]; then
  git clone --depth=1 git@github.com:FlagAI-Open/FlagAI.git
fi

cd FlagAI
git pull
direnv exec . python setup.py install
