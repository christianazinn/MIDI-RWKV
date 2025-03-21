#!/bin/bash

ROOT_PATH="/home/christian/MIDI-RWKV"

python3 "$ROOT_PATH/flash-linear-attention/utils/convert_from_rwkv7.py" --rwkv7 "$ROOT_PATH/$1" --output "$ROOT_PATH/src/outputs/m2fla"
python3 "$ROOT_PATH/rwkv.cpp/python/convert_pytorch_to_ggml.py" "$ROOT_PATH/$1" "$ROOT_PATH/src/outputs/m2fla/rcpp.bin" "FP16"
python3 "$ROOT_PATH/rwkv.cpp/python/quantize.py" "$ROOT_PATH/src/outputs/m2fla/rcpp.bin" "$ROOT_PATH/src/outputs/m2fla/rcpq.bin" "Q4_0"