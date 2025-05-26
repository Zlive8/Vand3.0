#!/bin/bash

DATASET_PATH="/data/public/dataset/mvtec_ad_2"

MODEL_DIR="/data/zhangxin/zhang/INP-Former++/result/0.5-fa-train1"

OUTPUT_DIR="/data/zhangxin/zhang/INP-Former++/result/0.5-fa-train1/result-erode"

# Train call
python train.py \
    --dataset_path $DATASET_PATH \
    --save_path $MODEL_DIR \

python test.py \
    --model_path $MODEL_DIR \
    --output_path $OUTPUT_DIR \
    --dataset_path $DATASET_PATH