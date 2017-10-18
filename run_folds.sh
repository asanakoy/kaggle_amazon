#!/usr/bin/env bash

 for gpu_id in 1 2 3
 do
     echo "FOLD ${gpu_id}"
     CUDA_VISIBLE_DEVICES=${gpu_id} python main.py --model=resnet50 --fold=${gpu_id} --suf=_fixed &
 done

#CUDA_VISIBLE_DEVICES=1 python main.py --model=resnet50 --fold=0 --suf=_lr0.01 --lr=0.01 &
#CUDA_VISIBLE_DEVICES=5 python main.py --model=resnet50 --fold=4 --suf=_lr0.01 --lr=0.01 -b=32 &

wait
