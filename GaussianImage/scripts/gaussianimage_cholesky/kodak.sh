#!/bin/bash

data_path=$1

if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

# for num_points in 800 1000 3000 5000 7000 9000
# do
# CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
# --data_name kodak --model_name GaussianImage_Cholesky --num_points $num_points --iterations 50000
# done

CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name kodak --model_name GaussianImage_Cholesky --num_points 70000 --iterations 50000 --num_gabor 1 # --lr 2e-4 --reset_iter 3000

# data_path="./dataset/kodak"
# iterations=50000
# model_name="WIPESImage_Cholesky"
# data_name="kodak"


# if [ -z "$data_path" ]; then
#     echo "Error: No data_path provided."
#     echo "Usage: $0 <data_path>"
#     exit 1
# fi

# for num_points in 70000
# do
# CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
# --data_name $data_name --model_name $model_name --num_points $num_points --iterations $iterations --save_imgs
# done