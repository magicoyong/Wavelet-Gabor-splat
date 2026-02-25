#!/bin/bash

data_path=$1

if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

# for num_points in 500 600 800 1000 1200 1500
# do
# CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
# --data_name kodak --model_name GaussianImage_Cholesky --num_points $num_points --iterations 150000
# done

CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name kodak --model_name GaussianImage_Cholesky --num_points 2000 --init_iterations 50000 --mix_iterations 150000
