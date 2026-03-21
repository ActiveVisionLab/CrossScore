"""CrossScoreNet: the core neural network for CrossScore inference."""

import torch
from transformers import Dinov2Config, Dinov2Model
from omegaconf import OmegaConf

from crossscore.utils.io.images import ImageNetMeanStd
from crossscore.model.cross_reference import CrossReferenceNet
from crossscore.model.positional_encoding import MultiViewPosionalEmbeddings


class CrossScoreNet(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        img_norm_stat = ImageNetMeanStd()
        self.register_buffer(
            "img_mean_std", torch.tensor([*img_norm_stat.mean, *img_norm_stat.std])
        )

        # backbone, freeze
        self.dinov2_cfg = Dinov2Config.from_pretrained(self.cfg.model.backbone.from_pretrained)
        self.backbone = Dinov2Model.from_pretrained(self.cfg.model.backbone.from_pretrained)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # positional encoding layer
        self.pos_enc_fn = MultiViewPosionalEmbeddings(
            positional_encoding_h=self.cfg.model.pos_enc.multi_view.h,
            positional_encoding_w=self.cfg.model.pos_enc.multi_view.w,
            interpolate_mode=self.cfg.model.pos_enc.multi_view.interpolate_mode,
            req_grad=self.cfg.model.pos_enc.multi_view.req_grad,
            patch_size=self.cfg.model.patch_size,
            hidden_size=self.dinov2_cfg.hidden_size,
        )

        # cross reference predictor
        if self.cfg.model.do_reference_cross:
            self.ref_cross = CrossReferenceNet(cfg=self.cfg, dinov2_cfg=self.dinov2_cfg)

    def forward(
        self,
        query_img,
        ref_cross_imgs,
        need_attn_weights=False,
        need_attn_weights_head_id=0,
        norm_img=False,
    ):
        """
        :param query_img:       (B, 3, H, W)
        :param ref_cross_imgs:  (B, N_ref_cross, 3, H, W)
        :param norm_img:        bool, normalise an image with pixel value in [0, 1] with imagenet mean and std.
        """
        B = query_img.shape[0]
        H, W = query_img.shape[-2:]
        N_patch_h = H // self.cfg.model.patch_size
        N_patch_w = W // self.cfg.model.patch_size

        if norm_img:
            img_mean = self.img_mean_std[None, :3, None, None]
            img_std = self.img_mean_std[None, 3:, None, None]
            query_img = (query_img - img_mean) / img_std
            if ref_cross_imgs is not None:
                ref_cross_imgs = (ref_cross_imgs - img_mean[:, None]) / img_std[:, None]

        featmaps = self.get_featmaps(query_img, ref_cross_imgs)
        results = {}

        # processing (and predicting) for query
        featmaps["query"] = self.pos_enc_fn(featmaps["query"], N_view=1, img_h=H, img_w=W)

        if self.cfg.model.do_reference_cross:
            N_ref_cross = ref_cross_imgs.shape[1]

            # (B, N_ref_cross*num_patches, hidden_size)
            featmaps["ref_cross"] = self.pos_enc_fn(
                featmaps["ref_cross"],
                N_view=N_ref_cross,
                img_h=H,
                img_w=W,
            )

            # prediction
            dim_params = {
                "B": B,
                "N_patch_h": N_patch_h,
                "N_patch_w": N_patch_w,
                "N_ref": N_ref_cross,
            }
            results_ref_cross = self.ref_cross(
                featmaps["query"],
                featmaps["ref_cross"],
                None,
                dim_params,
                need_attn_weights,
                need_attn_weights_head_id,
            )
            results["score_map_ref_cross"] = results_ref_cross["score_map"]
            results["attn_weights_map_ref_cross"] = results_ref_cross["attn_weights_map_mha"]
        return results

    @torch.no_grad()
    def get_featmaps(self, query_img, ref_cross_imgs):
        """
        :param query_img:   (B, 3, H, W)
        :param ref_cross:   (B, N_ref_cross, 3, H, W)
        """
        B = query_img.shape[0]
        H, W = query_img.shape[-2:]
        N_patch_h = H // self.cfg.model.patch_size
        N_patch_w = W // self.cfg.model.patch_size
        N_query = 1
        N_ref_cross = 0 if ref_cross_imgs is None else ref_cross_imgs.shape[1]
        N_all_imgs = N_query + N_ref_cross

        # concat all images to go through backbone for once
        all_imgs = [query_img.view(B, 1, 3, H, W)]
        if ref_cross_imgs is not None:
            all_imgs.append(ref_cross_imgs)
        all_imgs = torch.cat(all_imgs, dim=1)
        all_imgs = all_imgs.view(B * N_all_imgs, 3, H, W)

        # bbo: backbone output
        bbo_all = self.backbone(all_imgs)
        featmap_all = bbo_all.last_hidden_state[:, 1:]
        featmap_all = featmap_all.view(B, N_all_imgs, N_patch_h * N_patch_w, -1)

        # query
        featmap_query = featmap_all[:, 0]  # (B, num_patches, hidden_size)
        N_patches = featmap_query.shape[1]
        hidden_size = featmap_query.shape[2]

        # cross ref
        if ref_cross_imgs is not None:
            featmap_ref_cross = featmap_all[:, -N_ref_cross:]
            featmap_ref_cross = featmap_ref_cross.reshape(B, N_ref_cross * N_patches, hidden_size)
        else:
            featmap_ref_cross = None

        featmaps = {
            "query": featmap_query,  # (B, num_patches, hidden_size)
            "ref_cross": featmap_ref_cross,  # (B, N_ref_cross*num_patches, hidden_size)
        }
        return featmaps


def load_model(ckpt_path: str, device: str = "cpu") -> CrossScoreNet:
    """Load a CrossScoreNet model from a Lightning or direct checkpoint.

    Args:
        ckpt_path: Path to the .ckpt file.
        device: Device to load the model on.

    Returns:
        CrossScoreNet model in eval mode.
    """
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Extract config from checkpoint (saved by Lightning's save_hyperparameters)
    if "hyper_parameters" in checkpoint:
        cfg = OmegaConf.create(checkpoint["hyper_parameters"])
    else:
        # Fallback: use default config
        from pathlib import Path

        config_dir = Path(__file__).parent.parent / "config"
        model_cfg = OmegaConf.load(config_dir / "model" / "model.yaml")
        cfg = OmegaConf.create({"model": model_cfg})

    model = CrossScoreNet(cfg)

    # Handle Lightning checkpoint format (keys prefixed with "model.")
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        # Strip "model." prefix from Lightning checkpoint keys
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model.to(device)
    return model
