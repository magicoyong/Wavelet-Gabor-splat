## Gabor Inverse Problem



## Quick Started



### Requirements

```bash
cd gsplat
pip install .[dev]
BUILD_NO_CUDA=1 pip install -e .[dev]
cd ../
pip install -r requirements.txt
```

If you encounter errors while installing the packages listed in requirements.txt, you can try installing each Python package individually using the pip command.

Before training, you need to download the [kodak](https://r0k.us/graphics/kodak/) and [DIV2K-validation](https://data.vision.ee.ethz.ch/cvl/DIV2K/) datasets. The dataset folder is organized as follows.

```bash
├── dataset
│   | kodak 
│     ├── kodim01.png
│     ├── kodim02.png 
│     ├── ...
│   | DIV2K_valid_LR_bicubic
│     ├── X2
│        ├── 0801x2.png
│        ├── 0802x2.png
│        ├── ...
```

#### Representation

```bash
sh ./scripts/gaussianimage_cholesky/kodak.sh datasets/kodak/
sh ./scripts/gaussianimage_rs/kodak.sh /path/to/your/dataset
sh ./scripts/3dgs/kodak.sh /path/to/your/dataset

sh ./scripts/gaussianimage_cholesky/div2k.sh datasets/DIV2K_valid_LR_bicubic/X2
sh ./scripts/gaussianimage_rs/div2k.sh /path/to/your/dataset
sh ./scripts/3dgs/div2k.sh /path/to/your/dataset
```

#### Compression

After overfitting the image, we load the checkpoints from image representation and apply quantization-aware training technique to obtain the image compression results of GaussianImage models.

```bash
sh ./scripts/gaussianimage_cholesky/kodak_comp.sh /path/to/your/dataset
sh ./scripts/gaussianimage_rs/kodak_comp.sh /path/to/your/dataset

sh ./scripts/gaussianimage_cholesky/div2k_comp.sh /path/to/your/dataset
sh ./scripts/gaussianimage_rs/div2k_comp.sh /path/to/your/dataset
```

## Inpainting (Inverse Problem)

We provide an independent inpainting pipeline that solves the masked image reconstruction inverse problem by optimizing Gabor/Gaussian splatting parameters against partially observed images.

#### Quick Start

```bash
# Random pixel-wise mask (50% missing)
python inpainting_train.py \
    --image_path datasets/kodak/kodim01.png \
    --mask_type random --mask_ratio 0.5 \
    --num_points 50000 --num_gabor 2 \
    --iterations 30000 --lr 1e-3

# Block occlusion mask (4 blocks of 64x64)
python inpainting_train.py \
    --image_path datasets/kodak/kodim01.png \
    --mask_type block --block_size 64 --num_blocks 4 \
    --num_points 50000 --num_gabor 2 \
    --iterations 30000 --lr 1e-3

# With regularization
python inpainting_train.py \
    --image_path datasets/kodak/kodim01.png \
    --mask_type random --mask_ratio 0.5 \
    --num_points 50000 --num_gabor 2 \
    --iterations 30000 --lr 1e-3 \
    --lambda_gabor_l1 0.01 --lambda_cholesky_l2 0.001
```

#### Key Parameters

| Parameter | Description | Default |
|---|---|---|
| `--image_path` | Path to input image | (required) |
| `--mask_type` | `random` (pixel-wise) or `block` (square blocks) | `random` |
| `--mask_ratio` | Fraction of pixels to drop (random mask) | `0.5` |
| `--block_size` | Side length of each occluding block | `64` |
| `--num_blocks` | Number of blocks to drop | `4` |
| `--lambda_gabor_l1` | L1 sparsity on Gabor weights | `0.0` |
| `--lambda_position_l2` | L2 regularization on positions | `0.0` |
| `--lambda_cholesky_l2` | L2 regularization on covariance | `0.0` |
| `--loss_type` | `L2` (MSE) or `L1` for data fidelity | `L2` |

#### Output

Results are saved to `checkpoints_inpainting/<mask_type>_<mask_ratio>/<model_config>/<image_name>/`:
- `*_gt.png` — ground truth image
- `*_observed.png` — masked observation
- `*_reconstruction.png` — inpainted result
- `*_mask.png` — binary mask visualization
- `*_error.png` — per-pixel error map
- `config.yaml` — run configuration
- `gaussian_model.pth.tar` — trained model

#### Tests

```bash
# Run inpainting tests (CPU tests + CUDA tests if available)
pytest test_inpainting.py -v
```

## HSI Inpainting

HSI 训练入口使用统一的 --dataset 参数。它同时支持内置 HSI 数据集名称、四个多光谱场景短名，以及直接传入场景目录路径。

```bash
# Built-in HSI datasets
python inpainting_train_hsi.py --dataset Urban
python inpainting_train_hsi.py --dataset JasperRidge

# Multispectral scene short names under HSI/
python inpainting_train_hsi.py --dataset beads_ms
python inpainting_train_hsi.py --dataset chart_and_stuffed_toy_ms
python inpainting_train_hsi.py --dataset feathers_ms
python inpainting_train_hsi.py --dataset flowers_ms

# Directory path also works
python inpainting_train_hsi.py --dataset HSI/beads_ms
```

