#!/bin/bash

if [ $# -ne 4 ]; then
    echo "Usage: $0 model1 model2 model3 model4"
    exit 1
fi

models=("$@")

for i in {0..3}; do
    CUDA_VISIBLE_DEVICES=$i uv run main.py --model=${models[$i]} &
done

wait