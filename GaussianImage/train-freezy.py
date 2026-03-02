import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms

class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        image_path: Path,
        num_points: int = 2000,
        model_name:str = "GaussianImage_Cholesky",
        iterations:int = 30000,
        num_gabor: int = 2,
        model_path = None,
        args = None,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = image_path_to_tensor(image_path).to(self.device)

        self.num_points = num_points
        image_path = Path(image_path)
        self.image_name = image_path.stem
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.init_iterations = iterations
        self.mix_iterations = args.mix_iterations
        self.num_gabor = num_gabor
        self.threshold = args.threshold
        self.wavelet = args.wavelet
        self.level = args.level
        self.save_imgs = args.save_imgs
        self.log_dir = Path(f"./dwt_checkpoints/{args.data_name}/{model_name}_{args.mix_iterations}_{num_points}/{self.image_name}")
        
        if model_name == "GaussianImage_Cholesky":
            ## gaussianimage_cholesky
            from gaussianimage_cholesky import GaussianImage_Cholesky
            self.gaussian_model = GaussianImage_Cholesky(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False).to(self.device)

        elif model_name == "GaussianImage_RS":
            from gaussianimage_rs import GaussianImage_RS
            self.gaussian_model = GaussianImage_RS(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False).to(self.device) 

        elif model_name == "3DGS":
            from gaussiansplatting_3d import Gaussian3D
            self.gaussian_model = Gaussian3D(loss_type="Fusion2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, sh_degree=args.sh_degree, lr=args.lr).to(self.device)

        self.logwriter = LogWriter(self.log_dir)

        if model_path is not None:
            print(f"loading model path:{model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)

    def train(self):     
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.init_iterations+1), desc="Training progress")
        best_psnr = 0
        self.gaussian_model.train()
        start_time = time.time()
        for iter in range(1, self.init_iterations+1):
            loss, psnr = self.gaussian_model.train_iter(self.gt_image)
            psnr_list.append(psnr)
            iter_list.append(iter)
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)
        end_time = time.time() - start_time
        progress_bar.close()
        psnr_value, ms_ssim_value = self.test()
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model()
            test_end_time = (time.time() - test_start_time)/100

        self.logwriter.write("Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, test_end_time, 1/test_end_time))
        torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        np.save(self.log_dir / "training.npy", {"iterations": iter_list, "training_psnr": psnr_list, "training_time": end_time, 
        "psnr": psnr_value, "ms-ssim": ms_ssim_value, "rendering_time": test_end_time, "rendering_fps": 1/test_end_time})
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time

    def train_gaussian(self):
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.init_iterations+1), desc="Training progress")
        best_psnr = 0
        self.gaussian_model.train()
        start_time = time.time()
        for iter in range(1, self.init_iterations+1):
            loss, psnr = self.gaussian_model.train_iter(self.gt_image)
            psnr_list.append(psnr)
            iter_list.append(iter)
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)
        end_time = time.time() - start_time
        progress_bar.close()
        psnr_value, ms_ssim_value = self.test_gaussian()
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model()
            test_end_time = (time.time() - test_start_time)/100

        self.logwriter.write("Initial training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, test_end_time, 1/test_end_time))
        torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        np.save(self.log_dir / "init_training.npy", {"iterations": iter_list, "training_psnr": psnr_list, "training_time": end_time, 
        "psnr": psnr_value, "ms-ssim": ms_ssim_value, "rendering_time": test_end_time, "rendering_fps": 1/test_end_time})
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time
    
    def train_mix(self):
        from dwt_cholesky import Mix_Cholesky
        ## TODO: iteration
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.mix_iterations+1), desc="Training progress")
        best_psnr = 0
        self.mix_model = Mix_Cholesky(loss_type="L2", opt_type="adan", H=self.H, W=self.W, BLOCK_H=self.gaussian_model.BLOCK_H, BLOCK_W=self.gaussian_model.BLOCK_W,
                                      device=self.device, lr=1e-3, threshold = self.threshold, wavelet = self.wavelet, level = self.level, xyz = self.gaussian_model._xyz, 
                                      conic = self.gaussian_model._cholesky, features = self.gaussian_model._features_dc, num_gabor = self.num_gabor, quantize=False).to(self.device)
        self.mix_model.train()
        start_time = time.time()
        for iter in range(1, self.mix_iterations+1):
            loss, psnr = self.mix_model.train_iter(self.gt_image)
            psnr_list.append(psnr)
            iter_list.append(iter)
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)
        end_time = time.time() - start_time
        progress_bar.close()
        psnr_value, ms_ssim_value = self.test_mix()
        with torch.no_grad():
            self.mix_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.mix_model()
            test_end_time = (time.time() - test_start_time)/100

        self.logwriter.write("Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, test_end_time, 1/test_end_time))
        torch.save(self.mix_model.state_dict(), self.log_dir / "mix_model.pth.tar")
        np.save(self.log_dir / "training.npy", {"iterations": iter_list, "training_psnr": psnr_list, "training_time": end_time, 
        "psnr": psnr_value, "ms-ssim": ms_ssim_value, "rendering_time": test_end_time, "rendering_fps": 1/test_end_time})
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time
    
    def train_all(self):
        from dwt_cholesky_all import All_Cholesky
        ## TODO: iteration
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.mix_iterations+1), desc="Training progress")
        best_psnr = 0
        # self.gaussian_model = GaussianImage_Cholesky(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
        #         device=self.device, lr=args.lr, quantize=False).to(self.device)
        self.all_model = All_Cholesky(loss_type="L2", opt_type="adan", H=self.H, W=self.W, BLOCK_H=self.gaussian_model.BLOCK_H, BLOCK_W=self.gaussian_model.BLOCK_W,
                                      device=self.device, lr=1e-3, threshold = self.threshold, wavelet = self.wavelet, level = self.level, xyz = self.gaussian_model._xyz, 
                                      conic = self.gaussian_model._cholesky, features = self.gaussian_model._features_dc, num_gabor = self.num_gabor, quantize=False).to(self.device)
        self.all_model.train()
        start_time = time.time()
        for iter in range(1, self.mix_iterations+1):
            loss, psnr = self.all_model.train_iter(self.gt_image)
            psnr_list.append(psnr)
            iter_list.append(iter)
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},", "threshold":f"{self.all_model.threshold.cpu().numpy():.{4}f}"})
                    progress_bar.update(10)
        end_time = time.time() - start_time
        progress_bar.close()
        psnr_value, ms_ssim_value = self.test_all()
        with torch.no_grad():
            self.all_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.all_model()
            test_end_time = (time.time() - test_start_time)/100

        self.logwriter.write("Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, test_end_time, 1/test_end_time))
        torch.save(self.all_model.state_dict(), self.log_dir / "all_model.pth.tar")
        np.save(self.log_dir / "training.npy", {"iterations": iter_list, "training_psnr": psnr_list, "training_time": end_time, 
        "psnr": psnr_value, "ms-ssim": ms_ssim_value, "rendering_time": test_end_time, "rendering_fps": 1/test_end_time})
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time

    def test_gaussian(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model()
        mse_loss = F.mse_loss(out["render"].float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()
        self.logwriter.write("Test PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))
        return psnr, ms_ssim_value
    
    def test_mix(self):
        self.mix_model.eval()
        with torch.no_grad():
            out = self.mix_model()
        mse_loss = F.mse_loss(out["render"].float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()
        self.logwriter.write("Test PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))
        if self.save_imgs:
            transform = transforms.ToPILImage()
            img = transform(out["render"].float().squeeze(0))
            name = self.image_name + "_fitting_freezy.png" 
            img.save(str(self.log_dir / name))
        return psnr, ms_ssim_value
    
    def test(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model()
        mse_loss = F.mse_loss(out["render"].float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()
        self.logwriter.write("Test PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))
        if self.save_imgs:
            transform = transforms.ToPILImage()
            img = transform(out["render"].float().squeeze(0))
            name = self.image_name + "_fitting.png" 
            img.save(str(self.log_dir / name))
        return psnr, ms_ssim_value
    
    def test_all(self):
        self.all_model.eval()
        with torch.no_grad():
            out = self.all_model()
        mse_loss = F.mse_loss(out["render"].float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()
        self.logwriter.write("Test PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))
        if self.save_imgs:
            transform = transforms.ToPILImage()
            img = transform(out["render"].float().squeeze(0))
            name = self.image_name + "_fitting_all.png" 
            img.save(str(self.log_dir / name))
        return psnr, ms_ssim_value

def image_path_to_tensor(image_path: Path):
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0) #[1, C, H, W]
    return img_tensor

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./datasets/kodak/', help="Training dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default='kodak', help="Training dataset"
    )
    parser.add_argument(
        "--mix_iterations", type=int, default=50000, help="number of training epochs in mix model (default: %(default)s)"
    )
    parser.add_argument(
        "--init_iterations", type=int, default=50000, help="number of training epochs in gaussian model(default: %(default)s)"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianImage_Cholesky", help="model selection: GaussianImage_Cholesky, GaussianImage_RS, 3DGS"
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument(
        "--num_gabor", type=int, default=2, help="The number of gabor frequency (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=50000,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold", type=float, default = 2.5, help="The threshold for splitting high freqs and low freqs"
    )
    parser.add_argument( 
        "--level", type=int, default = 1, help="The level for dwt"
    )
    parser.add_argument(
        "--wavelet", type=str, default="haar", help="The wavelet for dwt"
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", default = True, help="Save image")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    logwriter = LogWriter(Path(f"./freezy_checkpoints/{args.data_name}/{args.model_name}_{args.mix_iterations}_{args.num_points}"))
    gaussian_psnrs, gaussian_ms_ssims, gaussian_training_times, gaussian_eval_times, gaussian_eval_fpses = [], [], [], [], []
    mix_psnrs, mix_ms_ssims, mix_training_times, mix_eval_times, mix_eval_fpses = [], [], [], [], []
    image_h, image_w = 0, 0
    if args.data_name == "kodak":
        #image_length, start = 24, 0
        image_length, start = 1, 0
    elif args.data_name == "DIV2K_valid_LRX2":
        image_length, start = 100, 800
    for i in range(start, start+image_length):
        if args.data_name == "kodak":
            image_path = Path(args.dataset) / f'kodim{i+1:02}.png'
        elif args.data_name == "DIV2K_valid_LRX2":
            image_path = Path(args.dataset) /  f'{i+1:04}x2.png'

        trainer = SimpleTrainer2d(image_path=image_path, num_points=args.num_points, 
            iterations=args.init_iterations, model_name=args.model_name, num_gabor= args.num_gabor, args=args, model_path=args.model_path)
        psnr, ms_ssim, training_time, eval_time, eval_fps = trainer.train_gaussian()
        gaussian_psnrs.append(psnr)
        gaussian_ms_ssims.append(ms_ssim)
        gaussian_training_times.append(training_time) 
        gaussian_eval_times.append(eval_time)
        gaussian_eval_fpses.append(eval_fps)
        image_h += trainer.H
        image_w += trainer.W
        image_name = image_path.stem
        logwriter.write("{}: {}x{}, Gaussian PSNR:{:.4f}, Gaussian MS-SSIM:{:.4f}, Gaussian Training:{:.4f}s, Gaussian Eval:{:.8f}s, Gaussian FPS:{:.4f}".format(
            image_name, trainer.H, trainer.W, psnr, ms_ssim, training_time, eval_time, eval_fps))
        
        psnr, ms_ssim, training_time, eval_time, eval_fps = trainer.train_mix()
        mix_psnrs.append(psnr)
        mix_ms_ssims.append(ms_ssim)
        mix_training_times.append(training_time) 
        mix_eval_times.append(eval_time)
        mix_eval_fpses.append(eval_fps)
        image_name = image_path.stem
        logwriter.write("{}: {}x{}, Mix PSNR:{:.4f}, Mix MS-SSIM:{:.4f}, Mix Training:{:.4f}s, Mix Eval:{:.8f}s, Mix FPS:{:.4f}".format(
            image_name, trainer.H, trainer.W, psnr, ms_ssim, training_time, eval_time, eval_fps))

    avg_gaussian_psnr = torch.tensor(gaussian_psnrs).mean().item()
    avg_gaussian_ms_ssim = torch.tensor(gaussian_ms_ssims).mean().item()
    avg_gaussian_training_time = torch.tensor(gaussian_training_times).mean().item()
    avg_gaussian_eval_time = torch.tensor(gaussian_eval_times).mean().item()
    avg_gaussian_eval_fps = torch.tensor(gaussian_eval_fpses).mean().item()

    avg_mix_psnr = torch.tensor(mix_psnrs).mean().item()
    avg_mix_ms_ssim = torch.tensor(mix_ms_ssims).mean().item()
    avg_mix_training_time = torch.tensor(mix_training_times).mean().item()
    avg_mix_eval_time = torch.tensor(mix_eval_times).mean().item()
    avg_mix_eval_fps = torch.tensor(mix_eval_fpses).mean().item()

    avg_h = image_h//image_length
    avg_w = image_w//image_length

    logwriter.write("Average Gaussian: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
        avg_h, avg_w, avg_gaussian_psnr, avg_gaussian_ms_ssim, avg_gaussian_training_time, avg_gaussian_eval_time, avg_gaussian_eval_fps))
    logwriter.write("Average Mix_freezy: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
        avg_h, avg_w, avg_mix_psnr, avg_mix_ms_ssim, avg_mix_training_time, avg_mix_eval_time, avg_mix_eval_fps))
    
if __name__ == "__main__":
    main(sys.argv[1:])
