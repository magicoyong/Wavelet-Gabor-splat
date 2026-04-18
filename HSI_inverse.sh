for num_points in 10000 20000 30000
do
  for img in Baboon.tiff feamle.tiff house.tiff Plane.tiff
  do
    for num_gabor in 2 3 
        do
            for lambda_dwt in 1e-1 5e-1 1e-2 5e-2 1e-3
            do
                python inpainting_train.py \
                    --image_path datasets/$img \
                    --mask_type elementwise \
                    --mask_ratio 0.9 \
                    --num_points $num_points \
                    --lr 1e-2 \
                    --iteration 30000
                    -- num_gabor $num_gabor
                    -- lambda_dwt $lambda_dwt    
            done
        done
  done
done
