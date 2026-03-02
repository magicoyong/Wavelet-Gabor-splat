from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum, rasterize_gabor_sum
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from quantize import *
from optimizer import Adan

## dwt and gabor training the same time
## 使用可微分 Haar 小波 + soft mask，让 threshold 参与梯度训练
class All_Cholesky(nn.Module):
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
        )
        self.device = kwargs["device"]

        # 可训练阈值：控制高低频划分边界
        self.init_threshold = nn.Parameter(torch.tensor(float(kwargs.get("threshold", 2.5))))
        self._threshold_raw = nn.Parameter(torch.tensor(math.log(math.exp(self.init_threshold) - 1.0)))
        self.threshold = kwargs.get("threshold", 2.5)

        # sigmoid 锐度：越大越接近硬阈值，可在训练过程中退火
        self.temperature = kwargs.get("temperature", 5.0)
        self.level = kwargs["level"]
        # wavelet 类型（目前仅实现 Haar 的可微版本）
        self.wavelet = kwargs.get("wavelet", "haar")

        # 所有 primitives 使用统一参数，不再拆分 high/low
        xyz = kwargs["xyz"]
        cholesky = kwargs["conic"]
        features = kwargs["features"]
        N = xyz.shape[0]

        self._mu = nn.Parameter(xyz.detach().clone())
        self._cholesky = nn.Parameter(cholesky.detach().clone())
        self._features_dc = nn.Parameter(features.detach().clone())
        self.register_buffer('opacity', torch.ones((N, 1)))

        self.num_gabor = kwargs["num_gabor"]
        # Gabor 参数为所有 N 个点分配（低频点的 gabor 贡献会被 soft mask 抑制到 ~0）
        self.gabor_freqs = nn.Parameter(torch.ones(N * self.num_gabor, 2) * 0.1)
        self.gabor_weights = nn.Parameter(torch.ones(N * self.num_gabor, 1) * 0.01)

        self.last_size = (self.H, self.W)
        self.quantize = kwargs["quantize"]  # T or F

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

    # ------------------------------------------------------------------ #
    #  可微分 Haar 小波：PyTorch 纯张量运算，支持对 mu 和 threshold 反传梯度
    # ------------------------------------------------------------------ #
    def _haar_decompose_one_level(self, x):
        """单层 Haar 小波分解。x: [M] -> cA: [ceil(M/2)], cD: [ceil(M/2)]"""
        n = x.shape[0]
        if n % 2 != 0:
            x = torch.cat([x, x[-1:]])  # 奇数长度时，复制最后一个元素填充
        cA = (x[0::2] + x[1::2]) / math.sqrt(2)
        cD = (x[0::2] - x[1::2]) / math.sqrt(2)
        return cA, cD

    def _haar_reconstruct_high(self, coeffs_detail, original_lengths):
        """
        从细节系数重建高频信号（将近似系数置 0）。
        coeffs_detail: list of tensors [cD_L, cD_{L-1}, ..., cD_1] (从最粗尺度到最细尺度)
        original_lengths: list of int，每层分解前的信号长度（从最粗到最细）
        """
        level = len(coeffs_detail)
        # 最粗尺度的近似系数置零
        a = torch.zeros_like(coeffs_detail[0])
        for i in range(level):
            d = coeffs_detail[i]
            # 上采样 + 合成
            recon = torch.zeros(len(a) * 2, device=a.device, dtype=a.dtype)
            recon[0::2] = (a + d) / math.sqrt(2)
            recon[1::2] = (a - d) / math.sqrt(2)
            a = recon[:original_lengths[i]]  # 裁剪到原始长度
        return a

    def _compute_hf_energy(self, mu):
        """
        可微分地计算每个 primitive 的高频能量。
        mu: [N, 2]  (tanh 后的坐标)
        返回: [N] 每个点的高频 L2 能量
        """
        N = mu.shape[0]
        high_freq_components = []
        for dim in range(mu.shape[1]):  # 对 x, y 两个维度分别做小波
            signal = mu[:, dim]  # [N]
            # 多层分解
            coeffs_detail = []   # [cD_L, ..., cD_1]
            original_lengths = []  # 每层重建目标长度
            a = signal
            for _ in range(self.level):
                orig_len = a.shape[0]
                a, d = self._haar_decompose_one_level(a)
                coeffs_detail.append(d)
                original_lengths.append(orig_len)
            # 逆序：从最粗到最细
            coeffs_detail.reverse()
            original_lengths.reverse()
            # 重建高频部分
            hf_signal = self._haar_reconstruct_high(coeffs_detail, original_lengths)
            high_freq_components.append(hf_signal[:N])

        high_freq = torch.stack(high_freq_components, dim=1)  # [N, 2]
        energy = torch.norm(high_freq, dim=1)  # [N]
        return energy

    def _compute_soft_mask(self, mu):
        """
        计算每个 primitive 的高频软掩码。
        返回 [N]：值接近 1 = 高频（使用 Gabor），值接近 0 = 低频（纯高斯）。
        梯度可回传到 mu 和 self.threshold。
        """
        energy = self._compute_hf_energy(mu)
        
        # softplus 保证 threshold > 0
        threshold = F.softplus(self._threshold_raw)
        mask = torch.sigmoid((energy - threshold) * self.temperature)
        self.threshold = threshold
        return mask

    def _init_data(self):
        self.cholesky_quantizer._init_data(self._cholesky)

    @property
    def get_mu(self):
        return torch.tanh(self._mu)

    @property
    def get_features(self):
        return self._features_dc

    @property
    def get_cholesky_elements(self):
        return self._cholesky + self.cholesky_bound

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
        mu = self.get_mu                        # [N, 2]
        cholesky = self.get_cholesky_elements    # [N, 3]
        features = self.get_features             # [N, 3]

        # 每次 forward 动态计算 soft mask（可微）
        soft_mask = self._compute_soft_mask(mu)  # [N]

        # 将 per-point mask 扩展到 gabor 参数维度: 每个点有 num_gabor 个 gabor 分量
        # gabor_mask[i*num_gabor : (i+1)*num_gabor] = soft_mask[i]
        gabor_mask = soft_mask.unsqueeze(1).expand(-1, self.num_gabor).reshape(-1, 1)  # [N*num_gabor, 1]

        # 低频点的 gabor_weight ≈ 0（纯高斯），高频点保留完整 gabor_weight
        masked_gabor_weights = self.gabor_weights * gabor_mask  # [N*num_gabor, 1]

        # 统一投影
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(
            mu, cholesky, self.H, self.W, self.tile_bounds)

        # 统一 Gabor 光栅化（gabor_weight ≈ 0 的点等价于普通高斯）
        out_img = rasterize_gabor_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            features, self.opacity,
            self.gabor_freqs[:, 0], self.gabor_freqs[:, 1], masked_gabor_weights.squeeze(-1),
            self.num_gabor,
            self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False)

        out_img = torch.clamp(out_img, 0, 1)  # [H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}
    
## training
    def train_iter(self, gt_image):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        # 正则项1：鼓励 threshold 不要太小（保持在初始值附近）
        
        reg_threshold = 0.05 * F.mse_loss(self.threshold, self.init_threshold.detach())
        
        # 正则项2：鼓励 mask 的熵最大化（即高低频比例不要太极端）
        soft_mask = self._compute_soft_mask(self.get_mu)
        mean_mask = soft_mask.mean()
        reg_entropy = -0.001 * (mean_mask * torch.log(mean_mask + 1e-8) 
                                + (1 - mean_mask) * torch.log(1 - mean_mask + 1e-8))
        total_loss = loss + reg_threshold + reg_entropy
        total_loss.backward()
        # loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

        self.scheduler.step()
        return loss, psnr

    def forward_quantize(self):
        N = self._mu.shape[0]
        l_vqm, m_bit = 0, 16*N*2
        means = torch.tanh(self.xyz_quantizer(self._mu))
        cholesky_elements, l_vqs, s_bit = self.cholesky_quantizer(self._cholesky)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        l_vqr, r_bit = 0, 0
        colors, l_vqc, c_bit = self.features_dc_quantizer(self.get_features)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, self.opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
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
        means = torch.tanh(self.xyz_quantizer(self._mu))
        quant_cholesky_elements, cholesky_elements = self.cholesky_quantizer.compress(self._cholesky)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        colors, feature_dc_index = self.features_dc_quantizer.compress(self.get_features)
        return {"xyz":self._mu.half(), "feature_dc_index": feature_dc_index, "quant_cholesky_elements": quant_cholesky_elements,}

    def decompress_wo_ec(self, encoding_dict):
        xyz, feature_dc_index, quant_cholesky_elements = encoding_dict["xyz"], encoding_dict["feature_dc_index"], encoding_dict["quant_cholesky_elements"]
        means = torch.tanh(xyz.float())
        cholesky_elements = self.cholesky_quantizer.decompress(quant_cholesky_elements)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        colors = self.features_dc_quantizer.decompress(feature_dc_index)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, self.opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
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
        total_bits += self._mu.numel()*16

        feature_dc_index = feature_dc_index.int().cpu().numpy()
        index_max = np.max(feature_dc_index)
        max_bit = np.ceil(np.log2(index_max)) #calculate max bit for feature_dc_index
        total_bits += feature_dc_index.size * max_bit #get_np_size(encoding_dict["feature_dc_index"]) * 8
        
        quant_cholesky_elements = quant_cholesky_elements.cpu().numpy()
        total_bits += quant_cholesky_elements.size * 6 #cholesky bits 

        position_bits = self._mu.numel()*16
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
        means = torch.tanh(self.xyz_quantizer(self._mu))
        quant_cholesky_elements, cholesky_elements = self.cholesky_quantizer.compress(self._cholesky)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        colors, feature_dc_index = self.features_dc_quantizer.compress(self.get_features)
        cholesky_compressed, cholesky_histogram_table, cholesky_unique = compress_matrix_flatten_categorical(quant_cholesky_elements.int().flatten().tolist())
        feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = compress_matrix_flatten_categorical(feature_dc_index.int().flatten().tolist())
        return {"xyz":self._mu.half(), "feature_dc_index": feature_dc_index, "quant_cholesky_elements": quant_cholesky_elements, 
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
                colors, self.opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
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
        total_bits += self._mu.numel()*16
        total_bits += get_np_size(cholesky_compressed) * 8
        total_bits += get_np_size(feature_dc_compressed) * 8

        position_bits = self._mu.numel()*16
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
