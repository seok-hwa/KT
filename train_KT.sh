#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 \
    train_KT.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --dataset citys \
    --batch-size 16 \
    --workers 16 \
    --lr 0.02 \
    --lambda-kd 1.0 \
    --lambda-mask 1.0 \
    --lambda-custom 1.0 \
    --lambda-contrastive 1.0 \
    --crop-size 512 1024 \
    --max-iterations 40000 \
    --data [your dataset path]/cityscapes/ \
    --save-dir [your directory path to store checkpoint files] \
    --log-dir [your directory path to store log files] \
    --teacher-pretrained [your teacher weights path]/deeplabv3_resnet101_citys_best_model.pth \
    --student-pretrained-base [your pretrained-backbone path]/resnet18-imagenet.pth 