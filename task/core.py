"""Training-only Lightning module. Not part of the pip package.

Imports the CrossScoreNet model from the crossscore package and wraps it
in a LightningModule for training/validation/testing.
"""

from pathlib import Path
import torch
import lightning
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.utilities import rank_zero_only

# Model from the pip package
from crossscore.task.core import CrossScoreNet

# Training-only imports
from crossscore.utils.io.images import ImageNetMeanStd


class CrossScoreLightningModule(lightning.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.save_hyperparameters(OmegaConf.to_container(self.cfg, resolve=True))

        # init my network (from pip package)
        self.model = CrossScoreNet(cfg=self.cfg)

        # lazy imports for training-only deps
        from crossscore.utils.check_config import check_reference_type

        # init loss fn
        if self.cfg.model.loss.fn == "l1":
            self.loss_fn = torch.nn.L1Loss()
        else:
            raise NotImplementedError

        # logging related names
        self.ref_mode_names = []
        if self.cfg.model.do_reference_cross:
            self.ref_mode_names.append("ref_cross")

    def _core_step(self, batch, batch_idx, skip_loss=False):
        outputs = self.model(
            query_img=batch["query/img"],
            ref_cross_imgs=batch.get("reference/cross/imgs", None),
            need_attn_weights=self.cfg.model.need_attn_weights,
            need_attn_weights_head_id=self.cfg.model.need_attn_weights_head_id,
            norm_img=False,
        )

        if skip_loss:
            return outputs

        score_map = batch["query/score_map"]
        loss = []

        if self.cfg.model.do_reference_cross:
            score_map_cross = outputs["score_map_ref_cross"]
            l1_diff_map_cross = torch.abs(score_map_cross - score_map)
            if self.cfg.model.loss.fn == "l1":
                loss_cross = l1_diff_map_cross.mean()
            else:
                loss_cross = self.loss_fn(score_map_cross, score_map)
            outputs["loss_cross"] = loss_cross
            outputs["l1_diff_map_ref_cross"] = l1_diff_map_cross
            loss.append(loss_cross)

        loss = torch.stack(loss).sum()
        outputs["loss"] = loss
        return outputs

    def training_step(self, batch, batch_idx):
        return self._core_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._core_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._core_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        return self._core_step(batch, batch_idx, skip_loss=True)

    @rank_zero_only
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log("train/loss", outputs["loss"], prog_bar=True)

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.log("validation/loss", outputs["loss"], prog_bar=True)

    def configure_optimizers(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params=parameters,
            lr=self.cfg.trainer.optimizer.lr,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.cfg.trainer.lr_scheduler.step_size,
            gamma=self.cfg.trainer.lr_scheduler.gamma,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.cfg.trainer.lr_scheduler.step_interval,
                "frequency": 1,
            },
        }
