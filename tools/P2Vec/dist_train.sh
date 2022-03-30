#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --arch PCL \
    --work-dir checkpoints/PCL/mocov2ep800-mocofcn-voc \
    --base-lr 0.02 --batch-size 256 --crop-size 512 --epochs 20 --print-freq 50 --checkpoint-freq 10 \
    --dist-url 'tcp://localhost:29500' --multiprocessing-distributed --world-size 1 --rank 0 \
    --pretrained /p300/checkpoint-resnet50/mmseg_mocov2_800ep.pth \
    --update-interval 16 \
    data/voc/train
