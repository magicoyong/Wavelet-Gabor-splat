# for rank in 20 25 30 35 40 45
# do
# echo "NMF decomposition"
# python endmember.py --dataset paviau --rank $rank
# echo "Gaussian inversion solve"
# # HSI Gaussian inpainting
# python inpainting_train_hsi.py --dataset paviau --rank $rank \
#     --mask_type elementwise --mask_ratio 9e-1 --num_points 10000 --iterations 50000
# done




echo "NMF Decomposition"
#python endmember.py --dataset paviau --rank 15
echo "Gaussian Inversion Solve"
# HSI Gaussian inpainting
python inpainting_train_hsi.py --dataset paviau --rank 15 \
    --mask_type elementwise --mask_ratio 9e-1 --num_points 10000 --iterations 50000
