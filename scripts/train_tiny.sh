#!/usr/bin/env bash

if [ ! -f "train.py" ]; then
    echo "Error: train.py not found!"
    exit 1
fi

if [ ! -d "./output/save-checkpoint/" ]; then
    mkdir -p ./output/save-checkpoint/
fi

"D:/software/coding/anaconda environment/tqsdk/python.exe" -m torchrun \
--nproc_per_node=8 \
train.py \
/d/sufe/研二/毕业论文/code/data/train/ \
--batch-size 128 \
--pin-mem \
--model transxnet_t \
--drop-path 0.1 \
--lr 2e-3 \
--warmup-epochs 5 \
--sync-bn \
--model-ema \
--model-ema-decay 0.9998 \
--val-start-epoch 250 \
--val-freq 50 \
--native-amp \
--output ./save-checkpoint/

echo "Starting script..."