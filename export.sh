#!/usr/bin/env bash

DIR=$(realpath $0) && DIR=${DIR%/*}
cd $DIR
set -ex

direnv exec . ./onnx_export_processor.py
direnv exec . ./onnx_export.py
