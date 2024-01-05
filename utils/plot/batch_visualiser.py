from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision.utils import make_grid
from PIL import Image
from utils.io.images import u8
from utils.misc.image import de_norm_img
from utils.check_config import check_reference_type


class BatchVisualiserBase(ABC):
    """Organise batch data and preview in a single image."""

    def __init__(self, cfg, img_mean_std, ref_type):
        self.cfg = cfg
        self.img_mean_std = img_mean_std.cpu().numpy()
        self.ref_type = ref_type
        self.metric_type = cfg.model.predict.metric.type
        self.metric_min = cfg.model.predict.metric.min
        self.metric_max = cfg.model.predict.metric.max

    @abstractmethod
    def vis(self, **kwargs):
        raise NotImplementedError


class BatchVisualiserRef(BatchVisualiserBase):
    @torch.no_grad()
    def vis(self, batch_input, batch_output, vis_id=0):
        ref_type = self.ref_type

        fig = plt.figure(figsize=(6, 6.5), layout="constrained")
        # plt.style.use("dark_background")
        # fig.set_facecolor("gray")

        # arrange subfigure
        N_ref_imgs = batch_input[f"reference/{ref_type}/imgs"].shape[1]
        h_ratios_subfig = [3, np.ceil(N_ref_imgs / 5)]
        fig_top, fig_bottom = fig.subfigures(2, 1, height_ratios=h_ratios_subfig)

        # arrange subplot for top fig
        inner = [
            ["score_map/gt", f"score_map/ref_{ref_type}"],
        ]

        mosaic_top = [
            ["query", inner],
        ]
        width_ratio_top = [3, 2]
        ax_dict_top = fig_top.subplot_mosaic(
            mosaic_top,
            empty_sentinel=None,
            width_ratios=width_ratio_top,
        )

        if "item_paths" in batch_input.keys():
            # 1. anonymize the path
            # 2. split to two rows
            # 3. add query path to title
            query_path = batch_input["item_paths"]["query/img"][vis_id]
            query_path = query_path.replace(str(Path("~").expanduser()), "~/")
            query_path = query_path.split("/")
            query_path.insert(len(query_path) // 2, "\n")
            query_path = Path(*query_path)
            fig_top.suptitle(f"{query_path}", fontsize=7)

        # arrange subplot for bottom fig
        mosaic_bottom = [[f"ref/{ref_type}/imgs"]]
        height_ratios = [1]
        ax_dict_bottom = fig_bottom.subplot_mosaic(
            mosaic_bottom,
            empty_sentinel=None,
            height_ratios=height_ratios,
        )

        # getting batch input output data
        img_dict = {"query": batch_input["query/img"][vis_id]}  # 3HW
        img_dict[f"ref/{ref_type}/imgs"] = make_grid(
            batch_input[f"reference/{ref_type}/imgs"][vis_id], nrow=5
        )  # 3HW
        img_dict["score_map/gt"] = batch_input[f"query/score_map"][vis_id]  # HW
        img_dict[f"score_map/ref_{ref_type}"] = batch_output[f"score_map_ref_{ref_type}"][
            vis_id
        ]  # HW

        # plot top fig
        for k, v in img_dict.items():
            if k not in ax_dict_top.keys():
                continue
            if k.startswith("score_map"):
                tmp_img = v.cpu().numpy()
                ax_dict_top[k].imshow(
                    tmp_img,
                    vmin=self.metric_min,
                    vmax=self.metric_max,
                    cmap="turbo",
                )
                title_txt = k.replace("score", self.metric_type)
                title_txt = title_txt.replace("_map", "").replace("/", " ").replace("_", " ")
                title_txt = title_txt + f"\n{tmp_img.mean():.3f}"
                if k == f"score_map/ref_{ref_type}":
                    if f"l1_diff_map_ref_{ref_type}" in batch_output.keys():
                        loss = batch_output[f"l1_diff_map_ref_{ref_type}"][vis_id].mean().item()
                        title_txt = title_txt + f" ∆{loss:.2f}"
                ax_dict_top[k].set_title(title_txt, fontsize=8)
            elif k.startswith("delta_map"):
                tmp_img = v.cpu().numpy()
                # ax_dict_top[k].imshow(tmp_img, vmin=-0.5, vmax=0.5, cmap="turbo")
                ax_dict_top[k].imshow(tmp_img, vmin=-0.5, vmax=0.5, cmap="bwr")
                ax_dict_top[k].set_title(k.replace("/", "\n"), fontsize=8)
            else:
                tmp_img = v.permute(1, 2, 0).cpu().numpy()  # HW3
                tmp_img = de_norm_img(tmp_img, self.img_mean_std)
                tmp_img = tmp_img.clip(0, 1)
                tmp_img = u8(tmp_img)
                ax_dict_top[k].imshow(tmp_img)
                ax_dict_top[k].set_title(k)
            ax_dict_top[k].set_xticks([])
            ax_dict_top[k].set_yticks([])

        # add a colorbar, turn off ticks
        ax_colorbar = fig_top.colorbar(
            plt.cm.ScalarMappable(cmap="turbo"), ax=ax_dict_top["query"], fraction=0.046, pad=0.04
        )
        ax_colorbar.ax.set_yticklabels([])

        # plot bottom fig
        for k, v in img_dict.items():
            if k not in ax_dict_bottom.keys():
                continue
            if k.startswith("attn_weights_map"):
                # tmp_img = v.permute(1, 2, 0)  # HW3
                tmp_img = v[0]  # HW
                tmp_img = tmp_img.clamp(0, 1)
                tmp_img = tmp_img.cpu().numpy()
                ax_dict_bottom[k].imshow(tmp_img, vmin=0, cmap="turbo")
                ax_dict_bottom[k].set_title(k)
            else:
                tmp_img = v.permute(1, 2, 0).cpu().numpy()  # HW3
                tmp_img = de_norm_img(tmp_img, self.img_mean_std)
                tmp_img = tmp_img.clip(0, 1)
                tmp_img = u8(tmp_img)
                ax_dict_bottom[k].imshow(tmp_img)
                ax_dict_bottom[k].set_title(k)

            ax_dict_bottom[k].set_xticks([])
            ax_dict_bottom[k].set_yticks([])

        # output
        fig.canvas.draw()
        out_img = Image.frombytes(
            "RGBA",
            fig.canvas.get_width_height(),
            fig.canvas.buffer_rgba(),
        ).convert("RGB")

        plt.close()
        out_img = wandb.Image(out_img, file_type="jpg")
        return out_img


class BatchVisualiserRefAttnMap(BatchVisualiserBase):
    def __init__(self, cfg, img_mean_std, ref_type, check_patch_mode):
        super().__init__(cfg, img_mean_std, ref_type)
        self.check_patch_mode = check_patch_mode

    @torch.no_grad()
    def vis(self, batch_input, batch_output, vis_id=0):
        fig = plt.figure(figsize=(6, 6.5), layout="constrained")
        # plt.style.use("dark_background")
        fig.set_facecolor("gray")

        has_reference_cross = "reference/cross/imgs" in batch_input.keys()
        has_attn_weights_cross = batch_output.get("attn_weights_map_ref_cross", None) is not None
        P = patch_size = 14

        # arrange subfigure
        h_ratios_subfig = [2, 1]

        fig_top, fig_bottom = fig.subfigures(2, 1, height_ratios=h_ratios_subfig)

        # arrange subplot for top fig
        inner = []
        if has_reference_cross:
            inner.append(["score_map/gt1", "score_map/ref_cross", "score_map/diff_cross"])
        inner = np.array(inner).T.tolist()  # vertical layout

        mosaic_top = [
            ["query", inner],
        ]
        width_ratio_top = [4, 1]

        ax_dict_top = fig_top.subplot_mosaic(
            mosaic_top,
            empty_sentinel=None,
            width_ratios=width_ratio_top,
        )

        # arrange subplot for bottom fig
        mosaic_bottom = []
        if has_reference_cross:
            mosaic_bottom.append(["ref/cross/imgs"])
        if has_attn_weights_cross:
            mosaic_bottom.append(["attn_weights_map_ref_cross"])

        height_ratios = [1] * len(mosaic_bottom)

        ax_dict_bottom = fig_bottom.subplot_mosaic(
            mosaic_bottom,
            empty_sentinel=None,
            height_ratios=height_ratios,
        )

        # getting batch input output data
        img_dict = {"query": batch_input["query/img"][vis_id]}  # 3HW
        if has_reference_cross:
            img_dict["ref/cross/imgs"] = make_grid(
                batch_input["reference/cross/imgs"][vis_id], nrow=5
            )  # 3HW
            img_dict["score_map/gt1"] = batch_input["query/score_map"][vis_id]
            img_dict["score_map/ref_cross"] = batch_output["score_map_ref_cross"][vis_id]
            if "l1_diff_map_ref_cross" in batch_output.keys():
                img_dict["score_map/diff_cross"] = batch_output["l1_diff_map_ref_cross"][vis_id]

        if has_attn_weights_cross:
            attn_weights_map_ref_cross = batch_output["attn_weights_map_ref_cross"]  # BHWNHW
            tmp_h, tmp_w = attn_weights_map_ref_cross.shape[1:3]
            if self.check_patch_mode == "centre":
                query_patch_cross = (tmp_h // 2, tmp_w // 2)
            elif self.check_patch_mode == "random":
                query_patch_cross = (np.random.randint(0, tmp_h), np.random.randint(0, tmp_w))
            else:
                raise ValueError(f"Unknown check_patch_mode: {self.check_patch_mode}")

            # NHW
            attn_weights_map_ref_cross = attn_weights_map_ref_cross[vis_id][query_patch_cross]
            img_dict["attn_weights_map_ref_cross"] = make_grid(
                attn_weights_map_ref_cross[:, None], nrow=5  # make grid expects N3HW
            )

        # plot top fig
        for k, v in img_dict.items():
            if k not in ax_dict_top.keys():
                continue
            if k.startswith("score_map"):
                tmp_img = v.cpu().numpy()
                ax_dict_top[k].imshow(tmp_img, vmin=0, vmax=1, cmap="turbo")
                ax_dict_top[k].set_title(k.replace("/", "\n"), fontsize=8)
            else:
                tmp_img = v.permute(1, 2, 0).cpu().numpy()  # HW3
                tmp_img = de_norm_img(tmp_img, self.img_mean_std)
                tmp_img = tmp_img.clip(0, 1)
                tmp_img = u8(tmp_img)
                ax_dict_top[k].imshow(tmp_img)
                ax_dict_top[k].set_title(k)

                if has_attn_weights_cross and k == "query":
                    # draw a rectangle patch
                    query_pixel = (query_patch_cross[0] * P, query_patch_cross[1] * P)
                    rect = Rectangle(
                        (query_pixel[1], query_pixel[0]),
                        P,
                        P,
                        linewidth=2,
                        edgecolor="magenta",
                        facecolor="none",
                    )
                    ax_dict_top[k].add_patch(rect)
            ax_dict_top[k].set_xticks([])
            ax_dict_top[k].set_yticks([])

        # plot bottom fig
        for k, v in img_dict.items():
            if k not in ax_dict_bottom.keys():
                continue
            if k.startswith("attn_weights_map"):
                num_stabler = 1e-8  # to avoid log(0)
                tmp_img = v[0]  # HW
                tmp_img = tmp_img.clamp(0, 1)
                tmp_img = tmp_img.cpu().numpy()
                # invert softmax (exp'd) attn weights
                tmp_img = np.log(tmp_img + num_stabler) - np.log(num_stabler)
                ax_dict_bottom[k].imshow(tmp_img, vmax=-np.log(num_stabler), cmap="turbo")
                ax_dict_bottom[k].set_title(k)
            else:
                tmp_img = v.permute(1, 2, 0).cpu().numpy()  # HW3
                tmp_img = de_norm_img(tmp_img, self.img_mean_std)
                tmp_img = tmp_img.clip(0, 1)
                tmp_img = u8(tmp_img)
                ax_dict_bottom[k].imshow(tmp_img)
                ax_dict_bottom[k].set_title(k)

            ax_dict_bottom[k].set_xticks([])
            ax_dict_bottom[k].set_yticks([])

        # output
        fig.canvas.draw()
        out_img = Image.frombytes(
            "RGBA",
            fig.canvas.get_width_height(),
            fig.canvas.buffer_rgba(),
        ).convert("RGB")

        plt.close()
        out_img = wandb.Image(out_img, file_type="jpg")
        return out_img


class BatchVisualiserRefFree(BatchVisualiserBase):
    @torch.no_grad()
    def vis(self, batch_input, batch_output, vis_id=0):
        fig = plt.figure(figsize=(6.5, 5), layout="constrained")
        # plt.style.use("dark_background")
        # fig.set_facecolor("gray")

        ref_type = self.ref_type
        inner = [["score_map/gt", f"score_map/ref_{ref_type}", "score_map/diff"]]
        inner = np.array(inner).T.tolist()

        mosaic = [
            ["query", inner],
        ]
        width_ratio = [5, 2]
        ax_dict = fig.subplot_mosaic(
            mosaic,
            empty_sentinel=None,
            width_ratios=width_ratio,
        )

        if "item_paths" in batch_input.keys():
            # 1. anonymize the path
            # 2. split to two rows
            # 3. add query path to title
            query_path = batch_input["item_paths"]["query/img"][vis_id]
            query_path = query_path.replace(str(Path("~").expanduser()), "~/")
            query_path = query_path.split("/")
            query_path.insert(len(query_path) // 2, "\n")
            query_path = Path(*query_path)
            fig.suptitle(f"{query_path}", fontsize=7)

        # getting batch input output data
        img_dict = {"query": batch_input["query/img"][vis_id]}  # 3HW
        img_dict["score_map/gt"] = batch_input[f"query/score_map"][vis_id]  # HW

        # plot top fig
        for k, v in img_dict.items():
            if k not in ax_dict.keys():
                continue
            if k.startswith("score_map"):
                tmp_img = v.cpu().numpy()
                ax_dict[k].imshow(
                    tmp_img,
                    vmin=self.metric_min,
                    vmax=self.metric_max,
                    cmap="turbo",
                )
                title_txt = k.replace("score", self.metric_type)
                title_txt = title_txt.replace("_map", "").replace("/", " ").replace("_", " ")
                title_txt = title_txt + f"\n{tmp_img.mean():.3f}"
                if k == f"score_map/ref_{ref_type}":
                    loss = batch_output[f"l1_diff_map_ref_{ref_type}"][vis_id].mean().item()
                    title_txt = title_txt + f" ∆{loss:.2f}"
                ax_dict[k].set_title(title_txt, fontsize=8)
            else:
                tmp_img = v.permute(1, 2, 0).cpu().numpy()  # HW3
                tmp_img = de_norm_img(tmp_img, self.img_mean_std)
                tmp_img = tmp_img.clip(0, 1)
                tmp_img = u8(tmp_img)
                ax_dict[k].imshow(tmp_img)
                ax_dict[k].set_title(k)
            ax_dict[k].set_xticks([])
            ax_dict[k].set_yticks([])

        # add a colorbar, turn off ticks
        ax_colorbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap="turbo"), ax=ax_dict["query"], fraction=0.046, pad=0.04
        )
        ax_colorbar.ax.set_yticklabels([])

        # output
        fig.canvas.draw()
        out_img = Image.frombytes(
            "RGBA",
            fig.canvas.get_width_height(),
            fig.canvas.buffer_rgba(),
        ).convert("RGB")

        plt.close()
        out_img = wandb.Image(out_img, file_type="jpg")
        return out_img


class BatchVisualiserFactory:
    def __init__(self, cfg, img_mean_std):
        self.cfg = cfg
        self.img_mean_std = img_mean_std
        self.ref_type = check_reference_type(self.cfg.model.do_reference_cross)

        if self.ref_type in ["cross"]:
            if self.cfg.model.need_attn_weights:
                self.visualiser = BatchVisualiserRefAttnMap(
                    cfg, self.img_mean_std, self.ref_type, check_patch_mode="centre"
                )
            else:
                self.visualiser = BatchVisualiserRef(cfg, self.img_mean_std, self.ref_type)
        else:
            raise NotImplementedError

    def __call__(self):
        return self.visualiser
