#!/bin/bash
# Ablation: random init vs obs_first init under random & elementwise masks.
# Runs a minimal comparison on a single image. Change IMAGE_PATH as needed.

IMAGE_PATH="datasets/Baboon.tiff"
NUM_POINTS=20000
NUM_GABOR=2
ITERATIONS=30000
LR=1e-2

echo "========================================="
echo "  Init Strategy Ablation Experiment"
echo "========================================="

for MASK_TYPE in random elementwise; do
  for MASK_RATIO in 0.5 0.9; do
    for INIT_STRATEGY in random obs_first; do
      echo ""
      echo "--- mask=${MASK_TYPE}, ratio=${MASK_RATIO}, init=${INIT_STRATEGY} ---"

      OBS_RATIO_FLAG=""
      if [ "$INIT_STRATEGY" = "obs_first" ]; then
        OBS_RATIO_FLAG="--obs_init_ratio 0.8"
      fi

      python inpainting_train.py \
        --image_path "$IMAGE_PATH" \
        --mask_type "$MASK_TYPE" \
        --mask_ratio "$MASK_RATIO" \
        --num_points "$NUM_POINTS" \
        --num_gabor "$NUM_GABOR" \
        --iterations "$ITERATIONS" \
        --lr "$LR" \
        --init_strategy "$INIT_STRATEGY" \
        $OBS_RATIO_FLAG \
        --save_imgs

    done
  done
done

echo ""
echo "Ablation complete. Check checkpoints_inpainting/ for results."
