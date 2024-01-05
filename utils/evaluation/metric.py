import numpy as np
import torch


def psnr(a, b, return_map=False):
    mse_map = torch.nn.functional.mse_loss(a, b, reduction="none")
    psnr_map = -10 * torch.log10(mse_map)
    if return_map:
        return psnr_map
    else:
        return psnr_map.mean()


def mse2psnr(a):
    return -10 * torch.log10(a)


def abs2psnr(a):
    return -10 * torch.log10(a.pow(2))


def psnr2mse(a):
    return 10 ** (-a / 10)


def correlation(a, b):
    x = torch.stack([a.flatten(), b.flatten()], dim=0)  # (2, N)
    corr = x.corrcoef()  # (2, 2)
    corr = corr[0, 1]  # only this one is meaningful
    return corr
