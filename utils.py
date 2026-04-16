import os
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
import torch
import numpy as np
from pytorch_wavelets import DWTForward

class LogWriter:
    def __init__(self, file_path, train=True):
        os.makedirs(file_path, exist_ok=True)
        self.file_path = os.path.join(file_path, "train.txt" if train else "test.txt")

    def write(self, text):
        # 打印到控制台
        print(text)
        # 追加到文件
        with open(self.file_path, 'a') as file:
            file.write(text + '\n')


def loss_fn(pred, target, loss_type='L2', lambda_value=0.7):
    target = target.detach()
    pred = pred.float()
    target  = target.float()
    if loss_type == 'L2':
        loss = F.mse_loss(pred, target)
    elif loss_type == 'L1':
        loss = F.l1_loss(pred, target)
    elif loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    elif loss_type == 'Fusion1':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion2':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion3':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * F.l1_loss(pred, target)
    elif loss_type == 'Fusion4':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion_hinerv':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value)  * (1 - ms_ssim(pred, target, data_range=1, size_average=True, win_size=5))
    return loss

def get_dwt_subbands(x: torch.Tensor) -> dict:
        """Get all DWT subbands using pytorch_wavelets package.
        
        Args:
            x: (N, C, H, W) input tensor
            
        Returns:
            Dictionary with keys: 'LL1', 'LH1', 'HL1', 'HH1', 'LL2', 'LH2', 'HL2', 'HH2'
        """
        device = x.device
        dtype = x.dtype
        
        # Initialize DWT for 2 levels using Haar (db1) wavelet
        # J=2 means it computes Level 1 and Level 2
        # mode='symmetric' is similar to reflect padding
        dwt = DWTForward(J=2, mode='symmetric', wave='db1').to(device)
        
        # Yl is the low-pass coefficients at the coarsest level (LL2)
        # Yh is a list of high-pass coefficients at each level (fine to coarse)
        # Yh[0] contains (LH1, HL1, HH1)
        # Yh[1] contains (LH2, HL2, HH2)
        Yl, Yh = dwt(x)
        
        LL2 = Yl
        
        # Level 1 high-pass
        LH1, HL1, HH1 = Yh[0][:,:,0,:,:], Yh[0][:,:,1,:,:], Yh[0][:,:,2,:,:]
        
        # Level 2 high-pass
        LH2, HL2, HH2 = Yh[1][:,:,0,:,:], Yh[1][:,:,1,:,:], Yh[1][:,:,2,:,:]
        
        # To get LL1, we can run 1-level DWT 
        dwt1 = DWTForward(J=1, mode='symmetric', wave='db1').to(device)
        
        # Level 1
        LL1, Yh1 = dwt1(x)
        LH1, HL1, HH1 = Yh1[0][:,:,0,:,:], Yh1[0][:,:,1,:,:], Yh1[0][:,:,2,:,:]
        
        # Level 2 (input is LL1)
        LL2, Yh2 = dwt1(LL1)
        LH2, HL2, HH2 = Yh2[0][:,:,0,:,:], Yh2[0][:,:,1,:,:], Yh2[0][:,:,2,:,:]
        
        return {
            'LL1': LL1, 'LH1': LH1, 'HL1': HL1, 'HH1': HH1,
            'LL2': LL2, 'LH2': LH2, 'HL2': HL2, 'HH2': HH2,
        }

def dwt_loss (image): #, lambda_value):
   # Ensure batch dimension
    pred_batched = image.unsqueeze(0) if image.dim() == 3 else image
            
    # Get all DWT subbands
    pred_bands = get_dwt_subbands(pred_batched)
            
    # Compute Charbonnier losses for all subbands
    total_dwt_loss = 0.0
    lambda_value = {
        "dwt_ll1_weight": 0.0, "dwt_lh1_weight": 0.5,
        "dwt_hl1_weight": 0.5, "dwt_hh1_weight": 1.0,
        "dwt_ll2_weight": 0.0, "dwt_lh2_weight": 0.0,
        "dwt_hl2_weight": 0.0, "dwt_hh2_weight": 0.0,
    }
            
    # Level 1 subbands (1/2 resolution)
    # if lambda_value["dwt_ll1_weight"] != 0.0:
    #     total_dwt_loss += lambda_value["dwt_ll1_weight"] * pred_bands['LL1'].abs().mean()
    # if lambda_value["dwt_lh1_weight"] != 0.0:
    #     total_dwt_loss += lambda_value["dwt_lh1_weight"] * pred_bands['LH1'].abs().mean()
    # if lambda_value["dwt_hl1_weight"] != 0.0:
    #     total_dwt_loss += lambda_value["dwt_hl1_weight"] * pred_bands['HL1'].abs().mean()
    # if lambda_value["dwt_hh1_weight"] != 0.0:
    #     total_dwt_loss += lambda_value["dwt_hh1_weight"] * pred_bands['HH1'].abs().mean()

    if lambda_value["dwt_ll1_weight"] != 0.0:
        total_dwt_loss += lambda_value["dwt_ll1_weight"] * pred_bands['LL1'].pow(2).mean()
    if lambda_value["dwt_lh1_weight"] != 0.0:
        total_dwt_loss += lambda_value["dwt_lh1_weight"] * pred_bands['LH1'].pow(2).mean()
    if lambda_value["dwt_hl1_weight"] != 0.0:
        total_dwt_loss += lambda_value["dwt_hl1_weight"] * pred_bands['HL1'].pow(2).mean()
    if lambda_value["dwt_hh1_weight"] != 0.0:
        total_dwt_loss += lambda_value["dwt_hh1_weight"] * pred_bands['HH1'].pow(2).mean()
            
    # Level 2 subbands (1/4 resolution)
    # if lambda_value["dwt_ll2_weight"] != 0.0:
    #     total_dwt_loss += lambda_value["dwt_ll2_weight"] * F.l1_loss(pred_bands['LL2'], gt_bands['LL2'])
    # if lambda_value["dwt_lh2_weight"] != 0.0:
    #     total_dwt_loss += lambda_value["dwt_lh2_weight"] * F.l1_loss(pred_bands['LH2'], gt_bands['LH2'])
    # if lambda_value["dwt_hl2_weight"]!= 0.0:
    #     total_dwt_loss += lambda_value["dwt_hl2_weight"] * F.l1_loss(pred_bands['HL2'], gt_bands['HL2'])
    # if lambda_value["dwt_hh2_weight"] != 0.0:
    #     total_dwt_loss += lambda_value["dwt_hh2_weight"] * F.l1_loss(pred_bands['HH2'], gt_bands['HH2'])
    return total_dwt_loss
        
def strip_lowerdiag(L):
    if L.shape[1] == 3:
        uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 0, 2]
        uncertainty[:, 3] = L[:, 1, 1]
        uncertainty[:, 4] = L[:, 1, 2]
        uncertainty[:, 5] = L[:, 2, 2]

    elif L.shape[1] == 2:
        uncertainty = torch.zeros((L.shape[0], 3), dtype=torch.float, device="cuda")
        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 1, 1]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_rotation_2d(r):
    '''
    Build rotation matrix in 2D.
    '''
    R = torch.zeros((r.size(0), 2, 2), device='cuda')
    R[:, 0, 0] = torch.cos(r)[:, 0]
    R[:, 0, 1] = -torch.sin(r)[:, 0]
    R[:, 1, 0] = torch.sin(r)[:, 0]
    R[:, 1, 1] = torch.cos(r)[:, 0]
    return R

def build_scaling_rotation_2d(s, r, device):
    L = torch.zeros((s.shape[0], 2, 2), dtype=torch.float, device='cuda')
    R = build_rotation_2d(r, device)
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L = R @ L
    return L
    
def build_covariance_from_scaling_rotation_2d(scaling, scaling_modifier, rotation, device):
    '''
    Build covariance metrix from rotation and scale matricies.
    '''
    L = build_scaling_rotation_2d(scaling_modifier * scaling, rotation, device)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def build_triangular(r):
    R = torch.zeros((r.size(0), 2, 2), device=r.device)
    R[:, 0, 0] = r[:, 0]
    R[:, 1, 0] = r[:, 1]
    R[:, 1, 1] = r[:, 2]
    return R

# def dwt_loss (pred, target): #, lambda_value):
#    # Ensure batch dimension
#     pred_batched = pred.unsqueeze(0) if pred.dim() == 3 else pred
#     gt_batched = target.unsqueeze(0) if target.dim() == 3 else target
            
#     # Get all DWT subbands
#     pred_bands = get_dwt_subbands(pred_batched)
#     gt_bands = get_dwt_subbands(gt_batched)
            
#     # Compute Charbonnier losses for all subbands
#     total_dwt_loss = 0.0
#     lambda_value = {
#         "dwt_ll1_weight": 1.0, "dwt_lh1_weight": 1.0,
#         "dwt_hl1_weight": 1.0, "dwt_hh1_weight": 0.0,
#         "dwt_ll2_weight": 0.0, "dwt_lh2_weight": 0.0,
#         "dwt_hl2_weight": 0.0, "dwt_hh2_weight": 0.0,
#     }
#     # # Level 1 subbands (1/2 resolution)
#     #     self.dwt_ll1_weight = 1.0
#     #     self.dwt_lh1_weight = 1.0
#     #     self.dwt_hl1_weight = 1.0
#     #     self.dwt_hh1_weight = 0.0
#     #     # Level 2 subbands (1/4 resolution)
#     #     self.dwt_ll2_weight = 0.0
#     #     self.dwt_lh2_weight = 0.0
#     #     self.dwt_hl2_weight = 0.0
#     #     self.dwt_hh2_weight = 0.0
            
#     # Level 1 subbands (1/2 resolution)
#     if lambda_value["dwt_ll1_weight"] != 0.0:
#         total_dwt_loss += lambda_value["dwt_ll1_weight"] * F.l1_loss(pred_bands['LL1'], gt_bands['LL1'])
#     if lambda_value["dwt_lh1_weight"] != 0.0:
#         total_dwt_loss += lambda_value["dwt_lh1_weight"] * F.l1_loss(pred_bands['LH1'], gt_bands['LH1'])
#     if lambda_value["dwt_hl1_weight"] != 0.0:
#         total_dwt_loss += lambda_value["dwt_hl1_weight"] * F.l1_loss(pred_bands['HL1'], gt_bands['HL1'])
#     if lambda_value["dwt_hh1_weight"] != 0.0:
#         total_dwt_loss += lambda_value["dwt_hh1_weight"] * F.l1_loss(pred_bands['HH1'], gt_bands['HH1'])
            
#     # Level 2 subbands (1/4 resolution)
#     if lambda_value["dwt_ll2_weight"] != 0.0:
#         total_dwt_loss += lambda_value["dwt_ll2_weight"] * F.l1_loss(pred_bands['LL2'], gt_bands['LL2'])
#     if lambda_value["dwt_lh2_weight"] != 0.0:
#         total_dwt_loss += lambda_value["dwt_lh2_weight"] * F.l1_loss(pred_bands['LH2'], gt_bands['LH2'])
#     if lambda_value["dwt_hl2_weight"]!= 0.0:
#         total_dwt_loss += lambda_value["dwt_hl2_weight"] * F.l1_loss(pred_bands['HL2'], gt_bands['HL2'])
#     if lambda_value["dwt_hh2_weight"] != 0.0:
#         total_dwt_loss += lambda_value["dwt_hh2_weight"] * F.l1_loss(pred_bands['HH2'], gt_bands['HH2'])
#     dwt_loss = total_dwt_loss
#     return dwt_loss
        
# def strip_lowerdiag(L):
#     if L.shape[1] == 3:
#         uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
#         uncertainty[:, 0] = L[:, 0, 0]
#         uncertainty[:, 1] = L[:, 0, 1]
#         uncertainty[:, 2] = L[:, 0, 2]
#         uncertainty[:, 3] = L[:, 1, 1]
#         uncertainty[:, 4] = L[:, 1, 2]
#         uncertainty[:, 5] = L[:, 2, 2]

#     elif L.shape[1] == 2:
#         uncertainty = torch.zeros((L.shape[0], 3), dtype=torch.float, device="cuda")
#         uncertainty[:, 0] = L[:, 0, 0]
#         uncertainty[:, 1] = L[:, 0, 1]
#         uncertainty[:, 2] = L[:, 1, 1]
#     return uncertainty