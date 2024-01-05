import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont
from utils.io.images import u8


def jigsaw_to_image(x, grid_size):
    """
    :param x:           (B, N_patch_h * N_patch_w, patch_size_h, patch_size_w)
    :param grid_size:   a tuple: (N_patch_h, N_patch_w)
    :return:            (B, H, W)
    """
    batch_size, num_patches, jigsaw_h, jigsaw_w = x.size()
    assert num_patches == grid_size[0] * grid_size[1]
    x_image = x.view(batch_size, grid_size[0], grid_size[1], jigsaw_h, jigsaw_w)
    output_h = grid_size[0] * jigsaw_h
    output_w = grid_size[1] * jigsaw_w
    x_image = x_image.permute(0, 1, 3, 2, 4).contiguous()
    x_image = x_image.view(batch_size, output_h, output_w)
    return x_image


def de_norm_img(img, mean_std):
    """De-normalize images that are normalized by mean and std in ImageNet-style.
    :param img: (H, W, 3)
    :param mean_std: (6, )
    """
    mean, std = mean_std[:3], mean_std[3:]
    img = img * std[None, None]
    img = img + mean[None, None]
    return img


def gray2rgb(img, vrange, cmap="turbo"):
    """
    Args:
        img:    HW, numpy.float32
        vrange: (min, max), float
        cmap:   str
    """
    vmin, vmax = vrange
    norm_op = plt.Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap)

    img = norm_op(img)
    img = colormap(img)
    rgb_image = u8(img[:, :, :3])
    return rgb_image


def attn2rgb(attn_map, cmap="turbo"):
    """Visualise attention map in rgb.
    The attn_map is softmaxed so we need to use log to make it more visible.
    Args:
        attn_map:   HW, numpy.float32
        cmap:       str
    """
    eps = 1e-8  # to avoid log(0)
    attn_map = attn_map.clip(0, 1)
    attn_map = attn_map + eps  # (1e-8, 1 + 1e-8)
    attn_map = attn_map.clip(0, 1)  # (1e-8, 1)
    # invert softmax (exp'd) attn weights
    attn_map = np.log(attn_map)  # (np.log(eps), 0)
    attn_map = attn_map - np.log(eps)  # (0, -np.log(eps))

    # some norm_op and colormap
    norm_op = plt.Normalize(vmin=0, vmax=-np.log(eps))
    colormap = cm.get_cmap(cmap)
    attn_map = norm_op(attn_map)
    attn_map = colormap(attn_map)
    rgb_image = u8(attn_map[:, :, :3])
    return rgb_image


def img_add_text(
    img_rgb,
    text,
    text_position=(20, 20),
    text_colour=(255, 255, 255),
    font_size=50,
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
):
    img = Image.fromarray(img_rgb)
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(img)
    draw.text(text_position, text, text_colour, font=font)
    img = np.array(img)
    return img
