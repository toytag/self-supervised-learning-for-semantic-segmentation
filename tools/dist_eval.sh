#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python lincls.py \
    --lr 30. --wd 0. --batch-size 256 --epochs 100 -p 50 \
    --dist-url 'tcp://localhost:29500' --multiprocessing-distributed --world-size 1 --rank 0 \
    --pretrained checkpoints/checkpoint_100.pth \
    --resume checkpoint.pth.tar \
    datasets/imagenet/