"""
HSI compression training module.

Provides SimpleTrainerHSI and train_nd() entry point used by main_lor.py.
Trains GaussianImage_Cholesky_EA to compress HSI via low-rank Gaussian splatting.
"""

import math
import time
from pathlib import Path
import copy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import LogWriter, loss_fn
from quantize import FakeQuantizationHalf


def compute_sam(gt, pred):
    """Spectral Angle Mapper (mean SAM in degrees).

    Args:
        gt, pred: (H, W, C) numpy arrays.
    """
    dot = np.sum(gt * pred, axis=-1)
    norm_gt = np.linalg.norm(gt, axis=-1)
    norm_pred = np.linalg.norm(pred, axis=-1)
    cos_angle = dot / (norm_gt * norm_pred + 1e-8)
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.mean(np.arccos(cos_angle) * 180 / np.pi)


class SimpleTrainerHSI:
    """Trains 2D Gaussian splatting to compress an HSI."""

    def __init__(
        self,
        ground_truth: torch.Tensor,
        endmember: np.ndarray,
        num_points: int = 2000,
        model_name: str = "GaussianImage_Cholesky_nd",
        iterations: int = 50000,
        model_path=None,
        data_name="HSI",
        image_name=None,
    ):
        self.device = torch.device("cuda:0")
        torch.cuda.synchronize()
        self.gt_image = ground_truth.to(self.device).half()
        self.endmember = endmember
        self.num_points = num_points
        BLOCK_H, BLOCK_W = 16, 16
        self.H = self.gt_image.shape[2]
        self.W = self.gt_image.shape[3]
        self.rank = endmember.shape[0]
        self.C = self.gt_image.shape[1]
        self.iterations = iterations
        self.log_dir = Path(
            f"./checkpoints/{data_name}/{model_name}_{iterations}_{num_points}_{self.rank}/{image_name}"
        )
        self.image_name = image_name

        torch.cuda.synchronize()
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"GPU memory GT: {gpu_memory:.2f} MB")

        if model_name == "GaussianImage_Cholesky_nd":
            from gaussianimage_cholesky_unknown import GaussianImage_Cholesky_EA
            self.gaussian_model = GaussianImage_Cholesky_EA(
                loss_type="L2",
                opt_type="adan",
                num_points=self.num_points,
                GT=self.gt_image,
                E=self.endmember,
                H=self.H, W=self.W, C=self.C, rank=self.rank,
                BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
                device=self.device,
                lr=5e-3,
                quantize=False,
            ).to(self.device)

        torch.cuda.synchronize()
        model_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"GPU memory after model initialization: {model_gpu_memory:.2f} MB "
              f"(Model size: {model_gpu_memory - gpu_memory:.2f} MB)")

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
        progress_bar = tqdm(range(1, self.iterations + 1), desc="Training progress")
        self.gaussian_model.train()
        start_time = time.time()
        best_psnr = 0

        for iter in range(1, self.iterations + 1):
            loss, psnr = self.gaussian_model.train_iter_quantize()
            psnr_list.append(psnr)
            iter_list.append(iter)
            if best_psnr < psnr:
                best_psnr = psnr
                best_model_dict = copy.deepcopy(self.gaussian_model.state_dict())
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({
                        "Loss": f"{loss.item():.{7}f}",
                        "Total PSNR": f"{psnr:.{4}f}",
                    })
                    progress_bar.update(10)

        end_time = time.time() - start_time
        progress_bar.close()
        psnr_value, ms_ssim_value, sam, bpppb = self.test()
        torch.save(
            self.gaussian_model.state_dict(),
            self.log_dir / "gaussian_model.pth.tar",
        )
        self.gaussian_model.load_state_dict(best_model_dict)
        best_psnr_value, best_ms_ssim_value, best_sam, best_bpppb = self.test(True)
        torch.save(best_model_dict, self.log_dir / "gaussian_model.best.pth.tar")

        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model.forward_quantize()
            test_end_time = (time.time() - test_start_time) / 100

        self.logwriter.write(
            "Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(
                end_time, test_end_time, 1 / test_end_time
            )
        )
        torch.save(
            self.gaussian_model.state_dict(),
            self.log_dir / "gaussian_model.pth.tar",
        )
        return (
            psnr_value, ms_ssim_value, end_time, test_end_time,
            1 / test_end_time, bpppb,
            best_psnr_value, best_ms_ssim_value, best_sam, best_bpppb,
        )

    def test(self, best=False):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model.forward_quantize()
            A = out["render"].float()
            E = FakeQuantizationHalf.apply(
                self.gaussian_model.endmember.to(torch.float32)
            )
            I = A @ E
            I = I.view(-1, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous()

            mse_per_channel = F.mse_loss(I, self.gt_image, reduction='none')
            mse_per_channel_avg = mse_per_channel.mean(dim=(0, 2, 3))
            psnr_per_channel = 10 * torch.log10(1.0 / mse_per_channel_avg)
            psnr = psnr_per_channel.mean().item()

        from pytorch_msssim import ms_ssim
        ms_ssim_value = ms_ssim(
            I, self.gt_image.float(), data_range=1,
            size_average=True, win_size=7,
        ).item()

        mean_sam = compute_sam(
            self.gt_image.squeeze(0).permute(1, 2, 0).cpu().numpy(),
            I.squeeze(0).permute(1, 2, 0).cpu().numpy(),
        )

        m_bit, s_bit, r_bit, c_bit = out["unit_bit"]
        e_bit = self.rank * self.C * 16
        bpppb = (m_bit + s_bit + r_bit + c_bit + e_bit) / self.H / self.W / self.C

        strings = "Best Test" if best else "Test"
        self.logwriter.write(
            "{} PSNR:{:.4f}, MS_SSIM:{:.6f}, bpppb:{:.4f}".format(
                strings, psnr, ms_ssim_value, bpppb
            )
        )
        return psnr, ms_ssim_value, mean_sam, bpppb


def train_nd(
    gt, endmember, image_name,
    iterations, num_points,
    model_name="Gaussian_Cholesky_nd",
):
    logwriter = LogWriter(
        Path(f"./checkpoints/compression/{model_name}_{iterations}_{num_points}")
    )
    trainer = SimpleTrainerHSI(
        ground_truth=gt,
        endmember=endmember,
        num_points=num_points,
        iterations=iterations,
        model_name=model_name,
        image_name=image_name,
    )
    (
        psnr, ms_ssim_val, training_time, eval_time, eval_fps,
        bpppb, best_psnr_value, best_ms_ssim_value, best_sam, best_bpppb,
    ) = trainer.train()
    logwriter.write(
        "{}: {}x{}x{}, Rank: {}, bpppb: {:.4f}, PSNR:{:.4f}, MS-SSIM:{:.4f}, "
        "Best bpppb: {:.4f}, Best PSNR:{:.4f}, Best MS-SSIM:{:.4f}, "
        "Best SAM: {:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
            image_name, trainer.H, trainer.W, trainer.C, trainer.rank,
            bpppb, psnr, ms_ssim_val,
            best_bpppb, best_psnr_value, best_ms_ssim_value, best_sam,
            training_time, eval_time, eval_fps,
        )
    )
