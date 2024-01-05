from abc import ABC, abstractmethod
import numpy as np
import torch
from torchvision.transforms import v2 as T


def get_crop_params(input_size, output_size, deterministic):
    """Get random crop parameters for a given image and output size.
    Args:
        img: numpy array hwc
        output_size (tuple): Expected output size of the crop.
    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    in_h, in_w = input_size
    out_h, out_w = output_size

    # i, j, h, w
    if deterministic:
        i, j = 0, 0
    else:
        i = np.random.randint(0, in_h - out_h + 1)
        j = np.random.randint(0, in_w - out_w + 1)
    return torch.tensor([i, j, out_h, out_w])


class Cropper(ABC):
    def __init__(self, output_size, deterministic=False):
        self.output_size = output_size
        self.deterministic = deterministic

    @abstractmethod
    def __call__(self, *args):
        raise NotImplementedError


class RandomCropperBatchSeparate(Cropper):
    """For an input tensor, assuming it's batched, and apply **DIFF** crop params
    to each item in the batch.
    """

    def __call__(self, imgs):
        # x: (B, C, H, W), (B, H, W)
        if imgs.ndim not in [3, 4]:
            raise ValueError("imgs.ndim must be one of [3, 4]")

        out_list = []
        crop_param_list = []
        for img in imgs:
            crop_param = get_crop_params(img.shape[-2:], self.output_size, self.deterministic)
            img = T.functional.crop(img, *crop_param)
            out_list.append(img)
            crop_param_list.append(crop_param)
        out_list = torch.stack(out_list)
        crop_param_list = torch.stack(crop_param_list)
        return {
            "out": out_list,  # (B, C, H, W) or (B, H, W)
            "crop_param": crop_param_list,  # (B, 4)
        }


class RandomCropperBatchSame(Cropper):
    """For a list of input tensors, assuming they're batched, and apply **SAME**
    crop params to all.
    """

    def __call__(self, *args):
        # use one set of crop params for all input
        crop_param = get_crop_params(args[0].shape[-2:], self.output_size, self.deterministic)
        out = [T.functional.crop(x, *crop_param) for x in args]
        return {
            "out": out,
            "crop_param": crop_param,
        }


class CropperFactory:
    def __init__(self, output_size, same_on_batch, deterministic=False):
        self.output_size = output_size
        if same_on_batch:
            self.cropper = RandomCropperBatchSame(output_size, deterministic)
        else:
            self.cropper = RandomCropperBatchSeparate(output_size, deterministic)

    def __call__(self, *args):
        return self.cropper(*args)
