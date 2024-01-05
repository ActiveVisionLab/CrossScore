import sys
from pathlib import Path
from datetime import timedelta

sys.path.append(str(Path(__file__).parents[1]))

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
import lightning
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.utilities import rank_zero_only
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from core import CrossScoreLightningModule
from dataloading.data_manager import get_dataset
from dataloading.transformation.crop import CropperFactory
from utils.io.images import ImageNetMeanStd
from utils.check_config import ConfigChecker


@hydra.main(version_base="1.3", config_path="../config", config_name="default")
def train(cfg: DictConfig):
    lightning.seed_everything(cfg.lightning.seed, workers=True)
    NUM_GPUS = len(cfg.trainer.devices)
    ConfigChecker(cfg).check_train_val()

    # use hydra timestamp as log dir
    log_dir = Path(HydraConfig.get().runtime.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # init wandb logger  # trick 1 for ddp
    wandb_logger = WandbLogger(
        project=cfg.project.name,
        save_dir=log_dir,
        log_model=False,  # let lightning's checkpoint callback handle it
    )

    @rank_zero_only
    def set_wandb_exp_name():  # trick 2 for ddp
        wandb_exp_id = wandb_logger.experiment.id  # trick 2 for ddp
        if type(wandb_exp_id) == str:  # trick 2 for ddp
            if cfg.alias is None or cfg.alias == "":
                wandb_logger.experiment.name = wandb_exp_id
            else:
                wandb_logger.experiment.name = wandb_exp_id + "_" + cfg.alias

    # sync wandb run name with id, in lightning wandb's "run" is renamed to "experiment"
    set_wandb_exp_name()  # trick 2 for ddp

    # init dataset
    img_norm_stat = ImageNetMeanStd()
    transforms = {
        "query_crop": CropperFactory(
            output_size=(cfg.data.transforms.crop_size, cfg.data.transforms.crop_size),
            same_on_batch=True,
            deterministic=cfg.trainer.overfit_batches > 0,
        ),
        "reference_crop": CropperFactory(
            output_size=(cfg.data.transforms.crop_size, cfg.data.transforms.crop_size),
            same_on_batch=False,
            deterministic=cfg.trainer.overfit_batches > 0,
        ),
        "img": T.Normalize(
            mean=img_norm_stat.mean,
            std=img_norm_stat.std,
        ),
    }

    if cfg.this_main.resize_short_side > 0:
        transforms["resize"] = T.Resize(
            cfg.this_main.resize_short_side,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )

    dataset = {
        "train": get_dataset(cfg, transforms, "train"),
        "test": get_dataset(cfg, transforms, "test", return_item_paths=True),
    }

    dataloader_train = DataLoader(
        dataset["train"],
        batch_size=cfg.data.loader.train.batch_size,
        shuffle=cfg.data.loader.train.shuffle,
        num_workers=cfg.data.loader.train.num_workers,
        pin_memory=cfg.data.loader.train.pin_memory,
        persistent_workers=cfg.data.loader.train.persistent_workers,
        prefetch_factor=cfg.data.loader.train.prefetch_factor,
    )
    dataloader_test = DataLoader(
        dataset["test"],
        batch_size=cfg.data.loader.validation.batch_size,
        shuffle=cfg.data.loader.validation.shuffle,
        num_workers=cfg.data.loader.validation.num_workers,
        pin_memory=cfg.data.loader.validation.pin_memory,
        persistent_workers=cfg.data.loader.validation.persistent_workers,
        prefetch_factor=cfg.data.loader.validation.prefetch_factor,
    )

    # lightning model
    model = CrossScoreLightningModule(cfg)

    # DDP strategy
    if NUM_GPUS > 1:
        strategy = DDPStrategy(find_unused_parameters=False, static_graph=True)
        use_distributed_sampler = True
    else:
        strategy = "auto"
        use_distributed_sampler = False

    # checkpoint callback
    ckpt_dir = log_dir / "ckpt"
    if cfg.trainer.checkpointing.train_time_interval is not None:
        train_time_interval = timedelta(hours=float(cfg.trainer.checkpointing.train_time_interval))
    else:
        train_time_interval = None
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        every_n_epochs=cfg.trainer.checkpointing.every_n_epochs,
        every_n_train_steps=cfg.trainer.checkpointing.every_n_train_steps,
        train_time_interval=train_time_interval,
        save_last=cfg.trainer.checkpointing.save_last,
        save_top_k=cfg.trainer.checkpointing.save_top_k,
    )

    # learning rate monitor
    lr_monitor = LearningRateMonitor()

    if cfg.trainer.do_profiling:
        from lightning.pytorch.profilers import PyTorchProfiler
        from datetime import datetime

        profiler = PyTorchProfiler(
            dirpath=log_dir / "profiler",
            filename=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            schedule=torch.profiler.schedule(wait=10, warmup=2, active=10, repeat=1),
        )
    else:
        profiler = None

    # lightning trainer
    trainer = lightning.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        max_epochs=cfg.trainer.max_epochs,
        max_steps=cfg.trainer.max_steps,
        strategy=strategy,
        use_distributed_sampler=use_distributed_sampler,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        overfit_batches=cfg.trainer.overfit_batches,
        logger=wandb_logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=[checkpoint_callback, lr_monitor],
        profiler=profiler,
    )

    trainer.fit(
        model,
        dataloader_train,
        dataloader_test,
        ckpt_path=cfg.trainer.ckpt_path_to_load,
    )


if __name__ == "__main__":
    train()
