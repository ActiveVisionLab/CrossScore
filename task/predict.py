from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
import lightning
from lightning.pytorch.strategies import DDPStrategy
import hydra
from omegaconf import DictConfig, open_dict

from core import CrossScoreLightningModule
from dataloading.dataset.simple_reference import SimpleReference
from dataloading.transformation.crop import CropperFactory
from utils.io.images import ImageNetMeanStd


@hydra.main(version_base="1.3", config_path="../config", config_name="default_predict")
def predict(cfg: DictConfig):
    lightning.seed_everything(cfg.lightning.seed, workers=True)
    NUM_GPUS = len(cfg.trainer.devices)

    # double check batch size with user
    if (
        cfg.this_main.force_batch_size is False
        and cfg.data.loader.validation.batch_size > 8
        and cfg.this_main.crop_mode is None
    ):
        tmp = input(
            "Test full image resolution in a large batch size "
            f"{cfg.data.loader.validation.batch_size}. "
            "Press Enter to continue, or enter a new batch size: "
        )
        if tmp == "":
            pass
        elif tmp.isdigit():
            new_batch_size = int(tmp)
            with open_dict(cfg):
                cfg.data.loader.validation.batch_size = new_batch_size
                print(f"Set batch size to {new_batch_size}")
        else:
            raise ValueError("Invalid input")

    if cfg.trainer.ckpt_path_to_load is None:
        # no ckpt, use current timestamp as log dir
        now = datetime.now().strftime("%Y%m%d_%H%M%S.%f")
        log_dir = Path("log") / now
        test_dir_name = "predict_empty_ckpt"
    else:
        # has ckpt, use the log dir of the ckpt path
        log_dir = Path(cfg.trainer.ckpt_path_to_load).parents[1]
        test_dir_name = "predict"
    log_dir = log_dir / test_dir_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # add logdir to cfg so lightning model can write to it
    with open_dict(cfg):
        if cfg.logger.predict.out_dir is None:
            now = datetime.now().strftime("%Y%m%d_%H%M%S.%f")
            cfg.logger.predict.out_dir = f"{log_dir}/{now}"
        if cfg.alias != "":
            cfg.logger.predict.out_dir += f"_{cfg.alias}"

    # init dataset
    img_norm_stat = ImageNetMeanStd()
    transforms = {
        "img": T.Normalize(
            mean=img_norm_stat.mean,
            std=img_norm_stat.std,
        ),
    }

    if cfg.this_main.crop_mode == "dataset_default":
        transforms["query_crop"] = CropperFactory(
            output_size=(cfg.data.transforms.crop_size, cfg.data.transforms.crop_size),
            same_on_batch=True,
            deterministic=True,
        )
        transforms["reference_crop"] = CropperFactory(
            output_size=(cfg.data.transforms.crop_size, cfg.data.transforms.crop_size),
            same_on_batch=False,
            deterministic=True,
        )

    if cfg.this_main.resize_short_side > 0:
        transforms["resize"] = T.Resize(
            cfg.this_main.resize_short_side,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )

    dataset = {
        "predict": SimpleReference(
            query_dir=cfg.data.dataset.query_dir,
            reference_dir=cfg.data.dataset.reference_dir,
            transforms=transforms,
            neighbour_config=cfg.data.neighbour_config,
            return_item_paths=True,
            zero_reference=cfg.data.dataset.zero_reference,
        ),
    }

    dataloader_predict = DataLoader(
        dataset["predict"],
        batch_size=cfg.data.loader.validation.batch_size,
        shuffle=False,
        num_workers=cfg.data.loader.validation.num_workers,
        pin_memory=True,
        persistent_workers=False,
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

    # lightning trainer
    trainer = lightning.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        strategy=strategy,
        use_distributed_sampler=use_distributed_sampler,
        limit_test_batches=cfg.trainer.limit_test_batches,
        logger=False,
    )

    trainer.predict(
        model,
        dataloader_predict,
        ckpt_path=cfg.trainer.ckpt_path_to_load,
    )


if __name__ == "__main__":
    with torch.no_grad():
        predict()
