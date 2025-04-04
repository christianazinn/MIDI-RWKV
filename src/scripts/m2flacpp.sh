#!/bin/bash

SRC_PATH="/home/christian/MIDI-RWKV"
# MODEL_PATH="/home/christian/MIDI-RWKV/src/outputs"
MODEL_PATH="/home/christian/RWKV-LM/RWKV-v5/out/L8-D512-x070"

python3 "$SRC_PATH/flash-linear-attention/utils/convert_from_rwkv7.py" --rwkv7 "$MODEL_PATH/$1" --output "$SRC_PATH/src/outputs/m2fla"
python3 "$SRC_PATH/rwkv.cpp/python/convert_pytorch_to_ggml.py" "$MODEL_PATH/$1" "$SRC_PATH/src/outputs/m2fla/rcpp.bin" "FP16"
python3 "$SRC_PATH/rwkv.cpp/python/quantize.py" "$SRC_PATH/src/outputs/m2fla/rcpp.bin" "$SRC_PATH/src/outputs/m2fla/rcpq.bin" "Q4_0"
