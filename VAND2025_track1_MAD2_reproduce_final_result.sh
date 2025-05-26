#!/bin/bash

DATASET_PATH="" #Root directory of the MVTec AD 2 dataset

MODEL_DIR="" #Directory to save/load trained model weights

OUTPUT_DIR="" #Output directory for inference results

# Train call
python train.py \
    --dataset_path $DATASET_PATH \
    --save_path $MODEL_DIR \

python test.py \
    --model_path $MODEL_DIR \
    --output_path $OUTPUT_DIR \
    --dataset_path $DATASET_PATH
