import torch


class MultiViewPosionalEmbeddings(torch.nn.Module):

    def __init__(
        self,
        positional_encoding_h,
        positional_encoding_w,
        interpolate_mode,
        req_grad,
        patch_size=14,
        hidden_size=384,
    ):
        """Apply positional encoding to input multi-view embeddings.
        Conceptually, posional encoding are in (pe_h, pe_w, C) shape.
        We can interpolate in 2D to adapt to different input image size, but
        no interpolation in the view dimension.

        Shorthand:
            P:  patch size
            C:  hidden size
            N:  number of
            pe: positional encoding
            mv: multi-view
            emb: embedding

        :param patch_size:  DINOv2 patch size   def 14
        :param hidden_size: DINOv2 hidden size  def 384 (dinov2_small)
        """
        super().__init__()
        self.P = patch_size
        self.C = hidden_size
        self.pe_h = positional_encoding_h
        self.pe_w = positional_encoding_w
        self.interpolate_mode = interpolate_mode
        self.PE = torch.nn.Parameter(
            torch.randn(1, self.pe_h, self.pe_w, self.C),
            req_grad,
        )

    def forward(self, mv_emb, N_view, img_h, img_w):
        """
        :param mv_emb: (B, N_patch, C), N_patch: N_view * emb_h * emb_w
        """
        B = mv_emb.shape[0]
        emb_h = img_h // self.P
        emb_w = img_w // self.P

        # a short cut no need to interpolate
        if emb_h == self.pe_h and emb_w == self.pe_w:
            # no need to interpolate
            mv_emb = mv_emb.view(B, N_view, emb_h, emb_w, self.C)
            mv_emb = mv_emb + self.PE
            mv_emb = mv_emb.view(B, N_view * emb_h * emb_w, self.C)
            return mv_emb

        # 1. interpolate position embedding
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        _PE = torch.nn.functional.interpolate(
            self.PE.permute(0, 3, 1, 2),
            scale_factor=(
                (emb_h + 1e-4) / self.pe_h,
                (emb_w + 1e-4) / self.pe_w,
            ),
            mode=self.interpolate_mode,
            align_corners=True,
        )  # (1, C, emb_h, emb_w)

        # 2. embed and reshape back
        mv_emb = mv_emb.view(B, N_view, emb_h, emb_w, self.C)
        mv_emb = mv_emb + _PE.permute(0, 2, 3, 1)[None]
        mv_emb = mv_emb.reshape(B, N_view * emb_h * emb_w, self.C)
        return mv_emb
