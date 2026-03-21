import imageio
import numpy as np
from dataclasses import dataclass
from PIL import Image
from typing import List


@dataclass
class ImageNetMeanStd:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


def f32(img):
    img = img.astype(np.float32)
    img = img / 255.0
    return img


def u8(img):
    img = img * 255.0
    img = img.astype(np.uint8)
    return img


def image_read(p):
    img = np.array(Image.open(p))
    img = f32(img)
    return img


def metric_map_read(p, vrange: List[int]):
    """Read metric maps and convert to float.
    Note:
        - when read/write int32 to png, it acutally reads/writes uint16 but looks like int32.
        - uint16 has range [0, 65535]
    """
    m = np.array(Image.open(p))  # HW np.int32
    m = m.astype(np.float32)
    if vrange == [0, 1]:
        m = m / 65535
    elif vrange == [-1, 1]:
        m = m / 32767 - 1
    else:
        raise ValueError("Invalid range for metric map reading. Must be '[0,1]' or '[-1,1]'")
    return m  # HW np.float32


def metric_map_write(p, m, vrange: List[int]):
    """Convert float metric maps to integer and write to png.
    Note:
        - when read/write int32 to png, it acutally reads/writes uint16 but looks like int32.
        - uint16 has range [0, 65535]
    """
    if vrange == [0, 1]:
        m = m * 65535  # [0,1] -> [0, 65535]
    elif vrange == [-1, 1]:
        m = (m + 1) * 32767  # [-1,1] -> [0, 2] -> [0, 65534]
    else:
        raise ValueError("Invalid range for metric map writing. Must be '[0,1]' or '[-1,1]'")
    m = m.astype(np.int32)
    # set compression level 0 for even faster writing
    imageio.imwrite(p, m)
