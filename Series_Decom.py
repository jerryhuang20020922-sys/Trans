import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

class Temporal_Decomposition_DWT_GPU(nn.Module):
    """
    多层DWT分解模块:分解出多层高频和低频分量,然后重构为原始长度
    输入: [B, T, N]
    输出: cA, cD [B, T, N] 低频和高频分量(通过IDWT重建为原始长度)
    
    优先使用 pytorch_wavelets,如果不可用则使用自定义实现
    """
    def __init__(self, wave='db4', J=2):  # 默认使用2层分解，更好地捕获多尺度特征
        super().__init__()
        self.wave = wave
        self.J = J

    def forward(self, x):
        """
        x: [B, T, N]
        return: cA, cD [B, T, N] 低频和高频分量(通过IDWT重建为原始长度)
        """
        B, T, N = x.shape

        # 转换为 [B, N, T] 格式以适应pywt
        x_trans = x.permute(0, 2, 1).cpu().numpy()  # [B, N, T]
        
        # 初始化结果数组
        xl = torch.zeros_like(x)
        xh = torch.zeros_like(x)
        
        # 对每个batch和node进行DWT分解
        for b in range(B):
            for n in range(N):
                coef = pywt.wavedec(x_trans[b, n, :], self.wave, level=self.J)
                coefl = [coef[0]] + [None] * (len(coef) - 1)
                coefh = [None] + coef[1:]
                xl[b, :, n] = torch.from_numpy(pywt.waverec(coefl, self.wave))[:T].to(x.device)
                xh[b, :, n] = torch.from_numpy(pywt.waverec(coefh, self.wave))[:T].to(x.device)
        
        return xl, xh


def disentangle_btn(x, wavelet='db1', level=1):
    """
    输入输出均为 [B, T, N]，直接用原来的 batch-wise DWT 方法，无循环。
    """
    # 转置到 [B, N, T]，最后一维是时间
    x = x.transpose(0, 2, 1)  # [B, N, T]

    # 做 DWT
    coef = pywt.wavedec(x, wavelet, level=level, axis=-1)

    # 构建低频分量
    coefl = [coef[0]] + [None]*(len(coef)-1)
    xl = pywt.waverec(coefl, wavelet, axis=-1)
    xl = xl[:, :, :x.shape[2]]  # 保持原长度

    # 构建高频分量
    coefh = [None] + coef[1:]
    xh = pywt.waverec(coefh, wavelet, axis=-1)
    xh = xh[:, :, :x.shape[2]]

    # 转置回 [B, T, N]
    xl = xl.transpose(0, 2, 1)
    xh = xh.transpose(0, 2, 1)

    return xl, xh


