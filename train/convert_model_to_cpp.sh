#!/bin/bash
SRC_PATH="$PROJECT_ROOT"
# can change for different models
MODEL_PATH="$PROJECT_ROOT"

# Check if a model filename was provided
if [ $# -lt 1 ]; then
  echo "Error: No model filename provided"
  echo "Usage: $0 MODEL_FILENAME"
  exit 1
fi

python3 "$SRC_PATH/rwkv.cpp/python/convert_pytorch_to_ggml.py" "$MODEL_PATH/$1" "$SRC_PATH/rwkv.cpp/python/rwkv_cpp/rcpp.bin" "FP16"
