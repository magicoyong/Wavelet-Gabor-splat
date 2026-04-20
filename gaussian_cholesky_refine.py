import sys
from pathlib import Path

local_gsplat_root = Path(__file__).resolve().parent / "gsplat"
if str(local_gsplat_root) not in sys.path:
    sys.path.insert(0, str(local_gsplat_root))

from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from quantize import *
from optimizer import Adan
from torch.utils.data import DataLoader, Dataset


class CoefficientRefinementBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"coef_refine_kernel must be a positive odd integer, got {kernel_size}")

        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=True,
        )
        self.depthwise_dilated = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding * 2,
            dilation=2,
            groups=channels,
            bias=True,
        )
        self.act = nn.GELU()
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.refine_alpha = nn.Parameter(torch.tensor(0.0))
        self.max_refine_scale = 0.10
        nn.init.kaiming_uniform_(self.depthwise.weight, a=np.sqrt(5))
        if self.depthwise.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.depthwise.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.depthwise.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.depthwise_dilated.weight, a=np.sqrt(5))
        if self.depthwise_dilated.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.depthwise_dilated.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.depthwise_dilated.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.pointwise.weight, a=np.sqrt(5))
        if self.pointwise.bias is not None:
            self.pointwise.bias.data.zero_()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        dilated_weight_key = prefix + "depthwise_dilated.weight"
        dilated_bias_key = prefix + "depthwise_dilated.bias"
        if dilated_weight_key not in state_dict:
            state_dict[dilated_weight_key] = self.depthwise_dilated.weight.detach().clone()
        if self.depthwise_dilated.bias is not None and dilated_bias_key not in state_dict:
            state_dict[dilated_bias_key] = self.depthwise_dilated.bias.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, coef_map):
        local_refined = self.depthwise(coef_map)
        context_refined = self.depthwise_dilated(coef_map)
        refined = torch.tanh(self.pointwise(self.act(local_refined + context_refined)))
        alpha = self.max_refine_scale * torch.tanh(self.refine_alpha)
        return torch.clamp(coef_map + alpha * refined, 0, 1)


class GaussianImage_Cholesky_EA(nn.Module):
    def __init__(self, loss_type="L2", **kwargs): #L2 SSIM Fusion1 Fusion2
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W, self.rank, self.C = kwargs["H"], kwargs["W"], kwargs["rank"], kwargs["C"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) 
        #torch.seed(1234)
        torch.cuda.synchronize()
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2) # MB
        self.device = kwargs["device"]
        self.image = kwargs["GT"]#.to(torch.float32) # (1, C, H, W)
        
        self.endmember = torch.tensor(kwargs["E"]).to(self.device)#.half() # (rank, C)
        self.use_coef_refine = kwargs.get("use_coef_refine", True)
        self.coef_refine_kernel = kwargs.get("coef_refine_kernel", 3)
        self.use_endmember_calib = kwargs.get("use_endmember_calib", True)
        self.calib_rank = kwargs.get("calib_rank", 2)
        
        torch.cuda.synchronize()
        E_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2) # MB
        print(f"E0 GPU memory usage: {E_gpu_memory - gpu_memory} MB")

        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        #self._xyz = self._initialize_xyz_from_abundance(self.abundance, self.init_num_points)
        self._cholesky = nn.Parameter(torch.zeros(self.init_num_points, 3)) # 0.5 * torch.rand(self.init_num_points, 3)
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        
        self._features_dc = nn.Parameter(0.5 * torch.rand(self.init_num_points, self.rank))
        self.coef = nn.Parameter(torch.tensor(0.0))

        self.last_size = (self.H, self.W)
        self.quantize = kwargs["quantize"]
        self.register_buffer('background', torch.ones(self.rank))
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))

        if self.use_coef_refine:
            self.coef_refine = CoefficientRefinementBlock(self.rank, kernel_size=self.coef_refine_kernel)
        else:
            self.coef_refine = None

        if self.use_endmember_calib:
            if self.calib_rank < 1:
                raise ValueError(f"calib_rank must be >= 1, got {self.calib_rank}")
            self.endmember_u = nn.Parameter(torch.empty(self.rank, self.calib_rank))
            self.endmember_v = nn.Parameter(torch.empty(self.calib_rank, self.C))
            self.endmember_gamma = nn.Parameter(torch.tensor(0.0))
            self.max_calib_scale = 0.05
            nn.init.kaiming_uniform_(self.endmember_u, a=np.sqrt(5))
            nn.init.kaiming_uniform_(self.endmember_v, a=np.sqrt(5))
            self.endmember_v.data.mul_(1e-3)
        else:
            self.register_parameter('endmember_u', None)
            self.register_parameter('endmember_v', None)
            self.register_parameter('endmember_gamma', None)
            self.max_calib_scale = 0.0
        
        self.endmember_quantizer = FakeQuantizationHalf.apply #UniformQuantizer(signed=False, bits=6, learned=True, num_channels=self.rank)
        self.xyz_quantizer = FakeQuantizationHalf.apply 
        self.features_dc_quantizer = VectorQuantizer(codebook_dim=self.rank, codebook_size=72,num_quantizers=2, vector_type="vector", kmeans_iters=8) 
        self.cholesky_quantizer = UniformQuantizer(signed=False, bits=8, learned=True, num_channels=3)

        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3000, gamma=0.5)

    def _init_data(self):
        self.cholesky_quantizer._init_data(self._cholesky)

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)
    
    @property
    def get_features(self):
        return self._features_dc
    
    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_cholesky_elements(self):
        return self._cholesky+self.cholesky_bound

    def get_calibrated_endmember(self):
        if not self.use_endmember_calib:
            return self.endmember
        low_rank_delta = torch.tanh(self.endmember_u @ self.endmember_v)
        gamma = self.max_calib_scale * torch.tanh(self.endmember_gamma)
        return torch.clamp(self.endmember + gamma * low_rank_delta, min=0.0)

    def rasterize_coefficients(self, means, cholesky_elements, features):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(
            means,
            cholesky_elements,
            self.H,
            self.W,
            self.tile_bounds,
        )
        rendered_chunks = []
        for feature_chunk in torch.split(features, 4, dim=1):
            valid_channels = feature_chunk.shape[1]
            if valid_channels < 4:
                feature_chunk = torch.cat(
                    [
                        feature_chunk,
                        torch.zeros(
                            feature_chunk.shape[0],
                            4 - valid_channels,
                            device=feature_chunk.device,
                            dtype=feature_chunk.dtype,
                        ),
                    ],
                    dim=1,
                )
            rendered_chunks.append(
                rasterize_gaussians_sum(
                    self.xys,
                    depths,
                    self.radii,
                    conics,
                    num_tiles_hit,
                    feature_chunk,
                    self._opacity,
                    self.H,
                    self.W,
                    self.BLOCK_H,
                    self.BLOCK_W,
                    return_alpha=False,
                )
            )
            rendered_chunks[-1] = rendered_chunks[-1][..., :valid_channels]

        coef_map = torch.cat(rendered_chunks, dim=2)
        coef_map = torch.clamp(coef_map, 0, 1)
        if self.use_coef_refine:
            coef_map = coef_map.permute(2, 0, 1).unsqueeze(0).contiguous()
            coef_map = self.coef_refine(coef_map)
            coef_map = coef_map.squeeze(0).permute(1, 2, 0).contiguous()
        return coef_map.view(self.H * self.W, self.rank).contiguous()

    def forward(self):
        coefficient_map = self.rasterize_coefficients(self.get_xyz, self.get_cholesky_elements, self.get_features)
        return {"render": coefficient_map}

    def train_iter(self):
        render_pkg = self.forward()
        A = render_pkg["render"] # (H * W, rank)
        calibrated_endmember = self.get_calibrated_endmember()
        flatimage = A @ calibrated_endmember
        image = flatimage.view(-1, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous() #[1, C, H, W]
        
        # update abundance
        loss =   loss_fn(image, self.image, self.loss_type, lambda_value=0.7) 
        loss.backward()
        '''0.15 * loss_fn(A, self.abundance) + 0.85 *0.05 * loss_fn(A, self.abundance) + 0.95 *
        # update endmember: multiplication rule
        with torch.no_grad():
            A_new = self.forward()["render"]
            image_new = (A_new @ self.endmember)
            self.endmember *=  (A_new.T @ self.image.view(-1, self.C)) / (A_new.T @ image_new)
        '''
        with torch.no_grad():
            I = (A @ calibrated_endmember).view(-1, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous()
            mse_loss = F.mse_loss(I, self.image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
            
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        self.scheduler.step()
        return loss, psnr
    
    def forward_quantize(self):
        # quantize plan1: "GaussianImage"
        l_vqm, m_bit = 0, 16*self.init_num_points*2
        means = torch.tanh(FakeQuantizationHalf.apply(self._xyz))#FakeQuantizationHalf.apply(self._xyz)
        
        cholesky_elements, l_vqs, s_bit = self.cholesky_quantizer(self._cholesky)#self._cholesky, 0, 32*self.init_num_points*3 32*self.init_num_points*3#self.cholesky_quantizer(self._cholesky)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        l_vqr, r_bit = 0, 0
        
        features, l_vqc, c_bit = self.get_features, 0, 32*self.init_num_points*self.rank#self.features_dc_quantizer(self.get_features)#self.get_features, 0, 32*self.init_num_points*self.rank

        out_img = self.rasterize_coefficients(means, cholesky_elements, features) * torch.exp(self.coef)
        
        vq_loss = l_vqm + l_vqs + l_vqr + l_vqc
        return {"render": out_img, "vq_loss": vq_loss, "unit_bit":[m_bit, s_bit, r_bit, c_bit]}

    def train_iter_quantize(self):
        render_pkg = self.forward_quantize()
        A = render_pkg["render"] # (H * W, rank)
        E = FakeQuantizationHalf.apply(self.get_calibrated_endmember())
        flatimage = A @ E # image: (H * W, C)
        image = flatimage.view(-1, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous() #[1, C, H, W]
        
        # update abundance
        loss =  loss_fn(image, self.image, self.loss_type, lambda_value=0.7) + 0.05 * render_pkg['vq_loss']
        loss.backward()
        #0.05 * loss_fn(A, self.abundance) + 0.95 * 
        with torch.no_grad():
            I = (A @ self.get_calibrated_endmember()).view(-1, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous()
            # Compute the elementwise MSE between the predicted and target images.
            # This yields a tensor of shape [B, C, H, W].
            mse_per_channel = F.mse_loss(I, self.image, reduction='none')
            # Average over the batch, height, and width dimensions.
            # The resulting tensor has shape [C], where each element is the average MSE for that channel.
            mse_per_channel_avg = mse_per_channel.mean(dim=(0, 2, 3))
            # Compute PSNR for each channel using the formula:
            # PSNR = 10 * log10(1 / MSE)
            psnr_per_channel = 10 * torch.log10(1.0 / mse_per_channel_avg)
            # If desired, you can further average these values to get a single scalar PSNR:
            psnr = psnr_per_channel.mean().item()
        
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss, psnr
    