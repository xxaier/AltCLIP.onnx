#!/usr/bin/env bash

DIR=$(realpath $0) && DIR=${DIR%/*}
cd $DIR
set -ex

case $(uname -s) in
Darwin*)
  pkg=onnxruntime-silicon
  ;;
*)
  pkg=onnxruntime
  ;;
esac

direnv allow
direnv exec . pip install $pkg
