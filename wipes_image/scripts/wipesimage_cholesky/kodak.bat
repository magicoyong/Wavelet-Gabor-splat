@echo off

set data_path=.\dataset\kodak
set iterations=50000
set model_name=WIPESImage_Cholesky
set data_name=kodak

if "%data_path%"=="" (
    echo Error: No data_path provided.
    echo Usage: train_kodak.bat ^<data_path^>
    exit /b
)

for %%i in (70000) do (
    set CUDA_VISIBLE_DEVICES=0
    python train.py -d %data_path% ^
        --data_name %data_name% ^
        --model_name %model_name% ^
        --num_points %%i ^
        --iterations %iterations% ^
        --save_imgs
)

pause