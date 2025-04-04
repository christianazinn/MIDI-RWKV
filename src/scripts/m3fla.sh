#!/bin/bash
SRC_PATH="/home/christian/MIDI-RWKV"
DEFAULT_MODEL_PATH="/home/christian/RWKV-LM/RWKV-v5/out/L8-D512-x070"
ALT_MODEL_PATH="/home/christian/MIDI-RWKV/src/outputs"

# Parse command line options
USE_ALT_PATH=false
while getopts "a" opt; do
  case $opt in
    a) USE_ALT_PATH=true ;;
    *) echo "Usage: $0 [-a] MODEL_FILENAME" >&2
       echo "  -a  Use alternate model path ($ALT_MODEL_PATH)" >&2
       exit 1 ;;
  esac
done

# Shift the arguments to remove the options
shift $((OPTIND-1))

# Check if a model filename was provided
if [ $# -lt 1 ]; then
  echo "Error: No model filename provided"
  echo "Usage: $0 [-a] MODEL_FILENAME"
  exit 1
fi

# Set MODEL_PATH based on flag
if [ "$USE_ALT_PATH" = true ]; then
  MODEL_PATH="$ALT_MODEL_PATH"
  echo "Using alternate model path: $MODEL_PATH"
else
  MODEL_PATH="$DEFAULT_MODEL_PATH"
  echo "Using default model path: $MODEL_PATH"
fi

python3 "$SRC_PATH/flash-linear-attention/utils/convert_from_rwkv7.py" --rwkv7 "$MODEL_PATH/$1" --output "$SRC_PATH/src/outputs/m2fla"
python3 "$SRC_PATH/rwkv.cpp/python/convert_pytorch_to_ggml.py" "$MODEL_PATH/$1" "$SRC_PATH/src/outputs/m2fla/rcpp.bin" "FP16"
python3 "$SRC_PATH/rwkv.cpp/python/quantize.py" "$SRC_PATH/src/outputs/m2fla/rcpp.bin" "$SRC_PATH/src/outputs/m2fla/rcpq.bin" "Q4_0"
