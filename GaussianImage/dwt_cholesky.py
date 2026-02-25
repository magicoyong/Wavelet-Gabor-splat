from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum, rasterize_gabor_sum
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from quantize import *
from optimizer import Adan
import pywt

class Mix_Cholesky(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        # CUDA accelerate
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) # 
        self.device = kwargs["device"]

        self.threshold = kwargs["threshold"]
        self.level = kwargs["level"]
        self.wavelet = kwargs["wavelet"]
        # xyz \in [-1, 1]
        xyz = kwargs["xyz"]
        cholesky = kwargs["conic"]
        features = kwargs["features"]
        high_mask, low_mask = self.split_premitive_by_frequency(xyz, self.threshold, self.wavelet, self.level)
        self.register_buffer('high_mask', high_mask)
        self.register_buffer('low_mask', low_mask)
        self.high_mu = nn.Parameter(xyz[self.high_mask].detach().clone())
        self.low_mu = nn.Parameter(xyz[self.low_mask].detach().clone())
        self.high_cholesky = nn.Parameter(cholesky[self.high_mask].detach().clone())
        self.low_cholesky = nn.Parameter(cholesky[self.low_mask].detach().clone())
        self.high_feature_dc = nn.Parameter(features[self.high_mask].detach().clone())
        self.low_features_dc = nn.Parameter(features[self.low_mask].detach().clone())

        self.num_gabor = kwargs["num_gabor"]
        n_low = self.low_mask.sum().item()
        n_high = self.high_mask.sum().item()
        self.register_buffer('low_opacity', torch.ones((n_low, 1)))
        self.register_buffer('high_opacity', torch.ones((n_high, 1)))
        
        ## small weight
        self.gabor_freqs = nn.Parameter(torch.ones(len(self.high_mu) * self.num_gabor, 2) * 0.1)
        self.gabor_weights = nn.Parameter(torch.ones(len(self.high_mu) * self.num_gabor, 1) * 0.01)
        
        self.last_size = (self.H, self.W)

        self.quantize = kwargs["quantize"] # T or F

        self.register_buffer('background', torch.ones(3))
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))

        if self.quantize:
            self.xyz_quantizer = FakeQuantizationHalf.apply 
            self.features_dc_quantizer = VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5) 
            self.cholesky_quantizer = UniformQuantizer(signed=False, bits=6, learned=True, num_channels=3)

        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

        
    def split_premitive_by_frequency(self, mu, threshold = 2.5, wavelet='haar', level=1):
        """
        根据高频信息的显著程度，将原始点云划分为低频点集和高频点集。
            
        参数：
        - point_cloud: numpy array, 原始点云 [N, 3]
        - high_ratio: float, 高频点云占总点云的比例 (0.0 ~ 1.0)，例如 0.2 表示 20% 是高频点
        - wavelet: str, 小波类型
        - level: int, 分解层数
            
        返回：
        - high_mask: boolean tensor
        - low_mask: boolean tensor
        """
        mu_np = mu.detach().cpu().numpy()
        high_freq_signal = np.zeros_like(mu_np)

        # 1. 仅计算每个点的高频偏移信号
        for i in range(2):
            coeffs = pywt.wavedec(mu_np[:, i], wavelet, level=level)
            # 重构高频部分
            high_freq_signal[:, i] = pywt.waverec([None] + coeffs[1:], wavelet)[:mu_np.shape[0]]

        # 2. 计算每个点的高频能量（向量的 L2 范数）
        # 能量越大，说明该点的高频特征越显著（处于边缘或细节处）
        hf_magnitude = np.linalg.norm(high_freq_signal, axis=1)

        # 3. 根据阈值生成布尔掩码 (Mask)
        # 能量大于阈值的点被划分为高频点，其余为低频点
        high_mask = hf_magnitude > threshold
        low_mask = ~high_mask  # 取反

        return torch.from_numpy(high_mask).to(mu.device), torch.from_numpy(low_mask).to(mu.device)

    def _init_data(self):
        self.cholesky_quantizer._init_data(self._cholesky)

    @property
    def get_low_mu(self):
        return torch.tanh(self.low_mu)
    
    @property
    def get_high_mu(self):
        return torch.tanh(self.high_mu)
    
    @property
    def get_low_features(self):
        return self.low_features_dc
    
    @property
    def get_high_features(self):
        return self.high_feature_dc
    
    @property
    def get_low_opacity(self):
        return self.low_opacity
    
    @property
    def get_high_opacity(self):
        return self.high_opacity

    @property
    def get_low_cholesky_elements(self):
        return self.low_cholesky + self.cholesky_bound
    
    @property
    def get_high_cholesky_elements(self):
        return self.high_cholesky + self.cholesky_bound
    
    @property
    def get_gabor_freqs(self):
        return self.gabor_freqs
    
    @property
    def get_gabor_weights(self):
        return self.gabor_weights
    
    @property
    def get_num_gabor(self):
        return self.num_gabor

    def forward(self): 
        # 合并低频和高频点，使用单次 rasterize_gabor_sum 调用
        # 低频点的 gabor 权重为 0，此时 H = 1.0，等价于普通高斯
        # 这避免了子集点数过少导致 gsplat tile_bins 越界的 CUDA 错误
        # 同时也避免了 background 被重复加两次的问题
        
        n_low = self.low_mu.size(0)
        n_high = self.high_mu.size(0)
        
        # 拼接均值、协方差、颜色和不透明度（低频在前，高频在后）
        all_mu = torch.cat([self.get_low_mu, self.get_high_mu], dim=0)
        all_cholesky = torch.cat([self.get_low_cholesky_elements, self.get_high_cholesky_elements], dim=0)
        all_features = torch.cat([self.get_low_features, self.get_high_features], dim=0)
        all_opacity = torch.cat([self.low_opacity, self.high_opacity], dim=0)
        
        # 构造完整的 gabor 参数：低频点填零，高频点使用实际参数
        # 内存布局: [point0_freq0, point0_freq1, ..., point1_freq0, ...]
        low_gabor_freqs = torch.zeros(n_low * self.num_gabor, 2, device=all_mu.device)
        low_gabor_weights = torch.zeros(n_low * self.num_gabor, 1, device=all_mu.device)
        all_gabor_freqs = torch.cat([low_gabor_freqs, self.gabor_freqs], dim=0)
        all_gabor_weights = torch.cat([low_gabor_weights, self.gabor_weights], dim=0)
        
        # 统一投影
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(
            all_mu, all_cholesky, self.H, self.W, self.tile_bounds)
        
        # 统一光栅化（gabor 权重为 0 的点等价于普通高斯）
        out_img = rasterize_gabor_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            all_features, all_opacity,
            all_gabor_freqs[:, 0], all_gabor_freqs[:, 1], all_gabor_weights.squeeze(-1),
            self.num_gabor,
            self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False)
        
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}
    
## training
    def train_iter(self, gt_image):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

        self.scheduler.step()
        return loss, psnr

    def forward_quantize(self):
        l_vqm, m_bit = 0, 16*self.init_num_points*2
        means = torch.tanh(self.xyz_quantizer(self._xyz))
        cholesky_elements, l_vqs, s_bit = self.cholesky_quantizer(self._cholesky)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        l_vqr, r_bit = 0, 0
        colors, l_vqc, c_bit = self.features_dc_quantizer(self.get_features)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        vq_loss = l_vqm + l_vqs + l_vqr + l_vqc
        return {"render": out_img, "vq_loss": vq_loss, "unit_bit":[m_bit, s_bit, r_bit, c_bit]}

    def train_iter_quantize(self, gt_image):
        render_pkg = self.forward_quantize()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7) + render_pkg["vq_loss"]
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss, psnr

    def compress_wo_ec(self):
        means = torch.tanh(self.xyz_quantizer(self._xyz))
        quant_cholesky_elements, cholesky_elements = self.cholesky_quantizer.compress(self._cholesky)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        colors, feature_dc_index = self.features_dc_quantizer.compress(self.get_features)
        return {"xyz":self._xyz.half(), "feature_dc_index": feature_dc_index, "quant_cholesky_elements": quant_cholesky_elements,}

    def decompress_wo_ec(self, encoding_dict):
        xyz, feature_dc_index, quant_cholesky_elements = encoding_dict["xyz"], encoding_dict["feature_dc_index"], encoding_dict["quant_cholesky_elements"]
        means = torch.tanh(xyz.float())
        cholesky_elements = self.cholesky_quantizer.decompress(quant_cholesky_elements)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        colors = self.features_dc_quantizer.decompress(feature_dc_index)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render":out_img}

    def analysis_wo_ec(self, encoding_dict):
        quant_cholesky_elements, feature_dc_index = encoding_dict["quant_cholesky_elements"], encoding_dict["feature_dc_index"]
        total_bits = 0
        initial_bits, codebook_bits = 0, 0
        for quantizer_index, layer in enumerate(self.features_dc_quantizer.quantizer.layers):
            codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits
        initial_bits += self.cholesky_quantizer.scale.numel()*torch.finfo(self.cholesky_quantizer.scale.dtype).bits
        initial_bits += self.cholesky_quantizer.beta.numel()*torch.finfo(self.cholesky_quantizer.beta.dtype).bits
        initial_bits += codebook_bits

        total_bits += initial_bits
        total_bits += self._xyz.numel()*16

        feature_dc_index = feature_dc_index.int().cpu().numpy()
        index_max = np.max(feature_dc_index)
        max_bit = np.ceil(np.log2(index_max)) #calculate max bit for feature_dc_index
        total_bits += feature_dc_index.size * max_bit #get_np_size(encoding_dict["feature_dc_index"]) * 8
        
        quant_cholesky_elements = quant_cholesky_elements.cpu().numpy()
        total_bits += quant_cholesky_elements.size * 6 #cholesky bits 

        position_bits = self._xyz.numel()*16
        cholesky_bits, feature_dc_bits = 0, 0
        cholesky_bits += self.cholesky_quantizer.scale.numel()*torch.finfo(self.cholesky_quantizer.scale.dtype).bits
        cholesky_bits += self.cholesky_quantizer.beta.numel()*torch.finfo(self.cholesky_quantizer.beta.dtype).bits
        cholesky_bits += quant_cholesky_elements.size * 6
        feature_dc_bits += codebook_bits
        feature_dc_bits += feature_dc_index.size * max_bit

        bpp = total_bits/self.H/self.W
        position_bpp = position_bits/self.H/self.W
        cholesky_bpp = cholesky_bits/self.H/self.W
        feature_dc_bpp = feature_dc_bits/self.H/self.W
        return {"bpp": bpp, "position_bpp": position_bpp, 
            "cholesky_bpp": cholesky_bpp, "feature_dc_bpp": feature_dc_bpp}

    def compress(self):
        means = torch.tanh(self.xyz_quantizer(self._xyz))
        quant_cholesky_elements, cholesky_elements = self.cholesky_quantizer.compress(self._cholesky)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        colors, feature_dc_index = self.features_dc_quantizer.compress(self.get_features)
        cholesky_compressed, cholesky_histogram_table, cholesky_unique = compress_matrix_flatten_categorical(quant_cholesky_elements.int().flatten().tolist())
        feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = compress_matrix_flatten_categorical(feature_dc_index.int().flatten().tolist())
        return {"xyz":self._xyz.half(), "feature_dc_index": feature_dc_index, "quant_cholesky_elements": quant_cholesky_elements, 
            "feature_dc_bitstream":[feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique], 
            "cholesky_bitstream":[cholesky_compressed, cholesky_histogram_table, cholesky_unique]}

    def decompress(self, encoding_dict):
        xyz = encoding_dict["xyz"]
        num_points, device = xyz.size(0), xyz.device
        feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = encoding_dict["feature_dc_bitstream"]
        cholesky_compressed, cholesky_histogram_table, cholesky_unique = encoding_dict["cholesky_bitstream"]
        feature_dc_index = decompress_matrix_flatten_categorical(feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique, num_points*2, (num_points, 2))
        quant_cholesky_elements = decompress_matrix_flatten_categorical(cholesky_compressed, cholesky_histogram_table, cholesky_unique, num_points*3, (num_points, 3))
        feature_dc_index = torch.from_numpy(feature_dc_index).to(device).int() #[800, 2]
        quant_cholesky_elements = torch.from_numpy(quant_cholesky_elements).to(device).float() #[800, 3]

        means = torch.tanh(xyz.float())
        cholesky_elements = self.cholesky_quantizer.decompress(quant_cholesky_elements)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        colors = self.features_dc_quantizer.decompress(feature_dc_index)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render":out_img}
   
    def analysis(self, encoding_dict):
        quant_cholesky_elements, feature_dc_index = encoding_dict["quant_cholesky_elements"], encoding_dict["feature_dc_index"]
        cholesky_compressed, cholesky_histogram_table, cholesky_unique = compress_matrix_flatten_categorical(quant_cholesky_elements.int().flatten().tolist())
        feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = compress_matrix_flatten_categorical(feature_dc_index.int().flatten().tolist())  
        cholesky_lookup = dict(zip(cholesky_unique, cholesky_histogram_table.astype(np.float64) / np.sum(cholesky_histogram_table).astype(np.float64)))
        feature_dc_lookup = dict(zip(feature_dc_unique, feature_dc_histogram_table.astype(np.float64) / np.sum(feature_dc_histogram_table).astype(np.float64)))

        total_bits = 0
        initial_bits, codebook_bits = 0, 0
        for quantizer_index, layer in enumerate(self.features_dc_quantizer.quantizer.layers):
            codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits
        initial_bits += self.cholesky_quantizer.scale.numel()*torch.finfo(self.cholesky_quantizer.scale.dtype).bits
        initial_bits += self.cholesky_quantizer.beta.numel()*torch.finfo(self.cholesky_quantizer.beta.dtype).bits
        initial_bits += get_np_size(cholesky_histogram_table) * 8
        initial_bits += get_np_size(cholesky_unique) * 8 
        initial_bits += get_np_size(feature_dc_histogram_table) * 8
        initial_bits += get_np_size(feature_dc_unique) * 8  
        initial_bits += codebook_bits

        total_bits += initial_bits
        total_bits += self._xyz.numel()*16
        total_bits += get_np_size(cholesky_compressed) * 8
        total_bits += get_np_size(feature_dc_compressed) * 8

        position_bits = self._xyz.numel()*16
        cholesky_bits, feature_dc_bits = 0, 0
        cholesky_bits += self.cholesky_quantizer.scale.numel()*torch.finfo(self.cholesky_quantizer.scale.dtype).bits
        cholesky_bits += self.cholesky_quantizer.beta.numel()*torch.finfo(self.cholesky_quantizer.beta.dtype).bits
        cholesky_bits += get_np_size(cholesky_histogram_table) * 8
        cholesky_bits += get_np_size(cholesky_unique) * 8   
        cholesky_bits += get_np_size(cholesky_compressed) * 8
        feature_dc_bits += codebook_bits
        feature_dc_bits += get_np_size(feature_dc_histogram_table) * 8
        feature_dc_bits += get_np_size(feature_dc_unique) * 8  
        feature_dc_bits += get_np_size(feature_dc_compressed) * 8

        bpp = total_bits/self.H/self.W
        position_bpp = position_bits/self.H/self.W
        cholesky_bpp = cholesky_bits/self.H/self.W
        feature_dc_bpp = feature_dc_bits/self.H/self.W
        return {"bpp": bpp, "position_bpp": position_bpp, 
            "cholesky_bpp": cholesky_bpp, "feature_dc_bpp": feature_dc_bpp,}
