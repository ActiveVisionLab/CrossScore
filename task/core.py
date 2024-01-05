from pathlib import Path
import torch
import lightning
import wandb
from transformers import Dinov2Config, Dinov2Model
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.utilities import rank_zero_only
from utils.evaluation.metric import abs2psnr, correlation
from utils.evaluation.metric_logger import (
    MetricLoggerScalar,
    MetricLoggerHistogram,
    MetricLoggerCorrelation,
    MetricLoggerImg,
)
from utils.plot.batch_visualiser import BatchVisualiserFactory
from utils.io.images import ImageNetMeanStd
from utils.io.batch_writer import BatchWriter
from utils.io.score_summariser import (
    SummaryWriterPredictedOnline,
    SummaryWriterPredictedOnlineTestPrediction,
)
from model.cross_reference import CrossReferenceNet
from model.positional_encoding import MultiViewPosionalEmbeddings


class CrossScoreNet(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # used in 1. denormalising images for visualisation
        # and 2. normalising images for training when required
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
        need_attn_weights,
        need_attn_weights_head_id,
        norm_img,
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
            img_std = self.img_mean_std[None, :3, None, None]
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


class CrossScoreLightningModule(lightning.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # write config to wandb
        self.save_hyperparameters(OmegaConf.to_container(self.cfg, resolve=True))

        # init my network
        self.model = CrossScoreNet(cfg=self.cfg)

        # init visualiser
        self.visualiser = BatchVisualiserFactory(self.cfg, self.model.img_mean_std)()

        # init loss fn
        if self.cfg.model.loss.fn == "l1":
            self.loss_fn = torch.nn.L1Loss()
            self.to_psnr_fn = abs2psnr
        else:
            raise NotImplementedError

        # logging related names
        self.ref_mode_names = []
        if self.cfg.model.do_reference_cross:
            self.ref_mode_names.append("ref_cross")

    def on_fit_start(self):
        # reset logging cache
        if self.global_rank == 0:
            self._reset_logging_cache_train()
        self._reset_logging_cache_validation()

        self.frame_score_summariser = SummaryWriterPredictedOnline(
            metric_type=self.cfg.model.predict.metric.type,
            metric_min=self.cfg.model.predict.metric.min,
        )

    def on_test_start(self):
        Path(self.cfg.logger.test.out_dir, "vis").mkdir(parents=True, exist_ok=True)
        if self.cfg.logger.test.write.flag.batch:
            self.batch_writer = BatchWriter(self.cfg, "test", self.model.img_mean_std)
        else:
            self.batch_writer = None

        self.frame_score_summariser = SummaryWriterPredictedOnlineTestPrediction(
            metric_type=self.cfg.model.predict.metric.type,
            metric_min=self.cfg.model.predict.metric.min,
            dir_out=self.cfg.logger.test.out_dir,
        )

    def on_predict_start(self):
        Path(self.cfg.logger.predict.out_dir, "vis").mkdir(parents=True, exist_ok=True)
        if self.cfg.logger.predict.write.flag.batch:
            self.batch_writer = BatchWriter(self.cfg, "predict", self.model.img_mean_std)
        else:
            self.batch_writer = None

        self.frame_score_summariser = SummaryWriterPredictedOnlineTestPrediction(
            metric_type=self.cfg.model.predict.metric.type,
            metric_min=self.cfg.model.predict.metric.min,
            dir_out=self.cfg.logger.predict.out_dir,
        )

    def _reset_logging_cache_train(self):
        self.train_cache = {
            "loss": {
                k: MetricLoggerScalar(max_length=self.cfg.logger.cache_size.train.n_scalar)
                for k in ["final", "reg_self", "reg_cross"] + self.ref_mode_names
            },
            "correlation": {
                k: MetricLoggerCorrelation(max_length=self.cfg.logger.cache_size.train.n_scalar)
                for k in self.ref_mode_names
            },
            "map": {
                "score": {
                    k: MetricLoggerHistogram(max_length=self.cfg.logger.cache_size.train.n_scalar)
                    for k in self.ref_mode_names
                },
                "l1_diff": {
                    k: MetricLoggerHistogram(max_length=self.cfg.logger.cache_size.train.n_scalar)
                    for k in self.ref_mode_names
                },
                "delta": {
                    k: MetricLoggerHistogram(max_length=self.cfg.logger.cache_size.train.n_scalar)
                    for k in ["self", "cross"]
                },
            },
        }

    def _reset_logging_cache_validation(self):
        self.validation_cache = {
            "loss": {
                k: MetricLoggerScalar(max_length=None)
                for k in ["final", "reg_self", "reg_cross"] + self.ref_mode_names
            },
            "correlation": {
                k: MetricLoggerCorrelation(max_length=None) for k in self.ref_mode_names
            },
            "fig": {k: MetricLoggerImg(max_length=None) for k in ["batch"]},
        }

    def _core_step(self, batch, batch_idx, skip_loss=False):
        outputs = self.model(
            query_img=batch["query/img"],  # (B, C, H, W)
            ref_cross_imgs=batch.get("reference/cross/imgs", None),  # (B, N_ref_cross, C, H, W)
            need_attn_weights=self.cfg.model.need_attn_weights,
            need_attn_weights_head_id=self.cfg.model.need_attn_weights_head_id,
            norm_img=False,
        )

        if skip_loss:  # only used in predict_step
            return outputs

        score_map = batch["query/score_map"]  # (B, H, W)

        loss = []
        # cross reference model predicts
        if self.cfg.model.do_reference_cross:
            score_map_cross = outputs["score_map_ref_cross"]  # (B, H, W)
            l1_diff_map_cross = torch.abs(score_map_cross - score_map)  # (B, H, W)
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
        outputs = self._core_step(batch, batch_idx)
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self._core_step(batch, batch_idx)
        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self._core_step(batch, batch_idx)
        return outputs

    def predict_step(self, batch, batch_idx):
        outputs = self._core_step(batch, batch_idx, skip_loss=True)
        return outputs

    @rank_zero_only
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.train_cache["loss"]["final"].update(outputs["loss"])

        if self.cfg.model.do_reference_cross:
            self.train_cache["loss"]["ref_cross"].update(outputs["loss_cross"])
            self.train_cache["correlation"]["ref_cross"].update(
                outputs["score_map_ref_cross"], batch["query/score_map"]
            )
            self.train_cache["map"]["score"]["ref_cross"].update(outputs["score_map_ref_cross"])
            self.train_cache["map"]["l1_diff"]["ref_cross"].update(outputs["l1_diff_map_ref_cross"])

        # logger vis batch
        if self.global_step % self.cfg.logger.vis_imgs_every_n_train_steps == 0:
            fig = self.visualiser.vis(batch, outputs)
            self.logger.experiment.log({"train_batch": fig})

        # logger vis X batches statics
        if self.global_step % self.cfg.logger.vis_scalar_every_n_train_steps == 0:
            # log loss
            tmp_loss = self.train_cache["loss"]["final"].compute()
            self.log("train/loss", tmp_loss, prog_bar=True)

            if self.cfg.model.do_reference_cross:
                tmp_loss_cross = self.train_cache["loss"]["ref_cross"].compute()
                self.log("train/loss_cross", tmp_loss_cross)

            # log psnr
            if self.cfg.model.do_reference_cross:
                self.log("train/psnr_cross", self.to_psnr_fn(tmp_loss_cross))

            # log correlation
            if self.cfg.model.do_reference_cross:
                self.log(
                    "train/correlation_cross",
                    self.train_cache["correlation"]["ref_cross"].compute(),
                )

        # logger vis X batches histogram
        if self.global_step % self.cfg.logger.vis_histogram_every_n_train_steps == 0:
            if self.cfg.model.do_reference_cross:
                self.logger.experiment.log(
                    {
                        "train/score_histogram_cross": wandb.Histogram(
                            np_histogram=self.train_cache["map"]["score"]["ref_cross"].compute()
                        ),
                        "train/l1_diff_histogram_cross": wandb.Histogram(
                            np_histogram=self.train_cache["map"]["l1_diff"]["ref_cross"].compute()
                        ),
                    }
                )

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.validation_cache["loss"]["final"].update(outputs["loss"])

        if self.cfg.model.do_reference_cross:
            self.validation_cache["loss"]["ref_cross"].update(outputs["loss_cross"])
            self.validation_cache["correlation"]["ref_cross"].update(
                outputs["score_map_ref_cross"], batch["query/score_map"]
            )

        self.frame_score_summariser.update(batch_input=batch, batch_output=outputs)

        if batch_idx < self.cfg.logger.cache_size.validation.n_fig:
            fig = self.visualiser.vis(batch, outputs)
            self.validation_cache["fig"]["batch"].update(fig)

    def on_test_batch_end(self, outputs, batch, batch_idx):
        results = {"test/loss": outputs["loss"]}

        if self.cfg.model.do_reference_cross:
            corr = correlation(outputs["score_map_ref_cross"], batch["query/score_map"])
            psnr = self.to_psnr_fn(outputs["loss_cross"])
            results["test/loss_cross"] = outputs["loss_cross"]
            results["test/corr_cross"] = corr
            results["test/psnr_cross"] = psnr

        self.log_dict(
            results,
            on_step=self.cfg.logger.test.on_step,
            sync_dist=self.cfg.logger.test.sync_dist,
        )

        self.frame_score_summariser.update(batch_input=batch, batch_output=outputs)

        # write image to vis
        if (
            self.cfg.logger.test.write.config.vis_img_every_n_steps > 0
            and batch_idx % self.cfg.logger.test.write.config.vis_img_every_n_steps == 0
        ):
            fig = self.visualiser.vis(batch, outputs)
            fig.image.save(
                Path(
                    self.cfg.logger.test.out_dir,
                    "vis",
                    f"r{self.local_rank}_B{str(batch_idx).zfill(4)}_b{0}.png",
                )
            )

        if self.cfg.logger.test.write.flag.batch:
            self.batch_writer.write_out(
                batch_input=batch,
                batch_output=outputs,
                local_rank=self.local_rank,
                batch_idx=batch_idx,
            )

    def on_predict_batch_end(self, outputs, batch, batch_idx):
        self.frame_score_summariser.update(batch_input=batch, batch_output=outputs)

        # write image to vis
        if (
            self.cfg.logger.predict.write.config.vis_img_every_n_steps > 0
            and batch_idx % self.cfg.logger.predict.write.config.vis_img_every_n_steps == 0
        ):
            fig = self.visualiser.vis(batch, outputs)
            fig.image.save(
                Path(
                    self.cfg.logger.predict.out_dir,
                    "vis",
                    f"r{self.local_rank}_B{str(batch_idx).zfill(4)}_b{0}.png",
                )
            )

        if self.cfg.logger.predict.write.flag.batch:
            self.batch_writer.write_out(
                batch_input=batch,
                batch_output=outputs,
                local_rank=self.local_rank,
                batch_idx=batch_idx,
            )

    @rank_zero_only
    def on_train_epoch_end(self):
        self._reset_logging_cache_train()

    def on_validation_epoch_end(self):
        sync_dist = True
        self.log(
            "validation/loss",
            self.validation_cache["loss"]["final"].compute(),
            prog_bar=True,
            sync_dist=sync_dist,
        )
        self.logger.experiment.log(
            {"validation_batch": self.validation_cache["fig"]["batch"].compute()},
        )

        if self.cfg.model.do_reference_cross:
            self.log(
                "validation/loss_cross",
                self.validation_cache["loss"]["ref_cross"].compute(),
                sync_dist=sync_dist,
            )
            self.log(
                "validation/correlation_cross",
                self.validation_cache["correlation"]["ref_cross"].compute(),
                sync_dist=sync_dist,
            )
            self.log(
                "validation/psnr_cross",
                self.to_psnr_fn(self.validation_cache["loss"]["ref_cross"].compute()),
                sync_dist=sync_dist,
            )

        self._reset_logging_cache_validation()
        self.frame_score_summariser.reset()

    def on_test_epoch_end(self):
        self.frame_score_summariser.summarise()

    def on_predict_epoch_end(self):
        self.frame_score_summariser.summarise()

    def configure_optimizers(self):
        # how to use configure_optimizers:
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers

        # we freeze backbone and we only pass parameters that requires grad to optimizer:
        # https://discuss.pytorch.org/t/how-to-train-a-part-of-a-network/8923
        # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#convnet-as-fixed-feature-extractor
        # https://discuss.pytorch.org/t/for-freezing-certain-layers-why-do-i-need-a-two-step-process/175289/2
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

        results = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.cfg.trainer.lr_scheduler.step_interval,
                "frequency": 1,
            },
        }
        return results
