#!/bin/bash

LABEL=0
MODEL="resnet50_custom"
OUTPUT_DIR="./results_resnet50_custom"
DATA_DIR="/blue/ruogu.fang/jhong.dongyou/neurflow/data/tanlikesmath/diabetic-retinopathy-resized/versions/7" 
TAU=16
KEEP_DATA=True
BATCH_SIZE=128
BATCH_SIZE_IG=4
DEVICE="cuda:0"

cd ..

python run.py --label "$LABEL" --model "$MODEL" --output_dir "$OUTPUT_DIR" --data_dir "$DATA_DIR" --tau $TAU --keep_data "$KEEP_DATA" --batch_size "$BATCH_SIZE" --batch_size_ig "$BATCH_SIZE_IG" --device "$DEVICE"

