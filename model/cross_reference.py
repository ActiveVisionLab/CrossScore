import torch
from .customised_transformer.transformer import (
    TransformerDecoderLayerCustomised,
    TransformerDecoderCustomised,
)
from .regression_layer import RegressionLayer
from utils.misc.image import jigsaw_to_image


class CrossReferenceNet(torch.nn.Module):
    def __init__(self, cfg, dinov2_cfg):
        super().__init__()
        self.cfg = cfg
        self.dinov2_cfg = dinov2_cfg

        # set up input projection
        self.input_proj = torch.nn.Identity()

        # set up output final activation function
        self.final_activation_fn = RegressionLayer(
            metric_type=self.cfg.model.predict.metric.type,
            metric_min=self.cfg.model.predict.metric.min,
            metric_max=self.cfg.model.predict.metric.max,
            pow_factor=self.cfg.model.predict.metric.power_factor,
        )

        # layers
        self.attn = TransformerDecoderCustomised(
            decoder_layer=TransformerDecoderLayerCustomised(
                d_model=self.dinov2_cfg.hidden_size,
                nhead=8,
                dim_feedforward=self.dinov2_cfg.hidden_size,
                dropout=0.0,
                batch_first=True,
                do_self_attn=self.cfg.model.decoder_do_self_attn,
                do_short_cut=self.cfg.model.decoder_do_short_cut,
            ),
            num_layers=2,
        )

        # set up head out dimension
        out_size = cfg.model.patch_size**2

        # head
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.dinov2_cfg.hidden_size, self.dinov2_cfg.hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.dinov2_cfg.hidden_size, out_size),
            self.final_activation_fn,
        )

    def forward(
        self,
        featmap_query,
        featmap_ref,
        memory_mask,
        dim_params,
        need_attn_weights,
        need_attn_weights_head_id,
    ):
        """
        :param featmap_query:   (B, num_patches, hidden_size)
        :param featmap_ref:     (B, N_ref * num_patches, hidden_size)
        :param memory_mask:     None
        :param dim_params:      dict
        """
        B = dim_params["B"]
        N_patch_h = dim_params["N_patch_h"]
        N_patch_w = dim_params["N_patch_w"]
        N_ref = dim_params["N_ref"]

        results = {}
        score_map, _, mha_weights = self.attn(
            tgt=featmap_query,
            memory=featmap_ref,
            memory_mask=memory_mask,
            need_weights=need_attn_weights,
            need_weights_head_id=need_attn_weights_head_id,
        )  # (B, num_patches_tgt, hidden_size), (B, num_patches_tgt, num_patches_mem)

        # return score map
        score_map = self.head(score_map)  # (B, num_patches, num_ssim_pixels)

        # reshape to image size
        P = self.cfg.model.patch_size
        score_map = score_map.view(B, -1, P, P)
        score_map = jigsaw_to_image(score_map, grid_size=(N_patch_h, N_patch_w))  # (B, H, W)
        results["score_map"] = score_map

        # return attn weights
        if need_attn_weights:
            mha_weights = mha_weights.view(B, N_patch_h, N_patch_w, N_ref, N_patch_h, N_patch_w)
        results["attn_weights_map_mha"] = mha_weights
        return results
