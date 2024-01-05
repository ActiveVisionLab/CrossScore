import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
import lightning
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import CSVLogger
import hydra
from omegaconf import DictConfig, open_dict

from core import CrossScoreLightningModule
from dataloading.data_manager import get_dataset
from dataloading.transformation.crop import CropperFactory
from utils.io.images import ImageNetMeanStd


@hydra.main(version_base="1.3", config_path="../config", config_name="default_test")
def test(cfg: DictConfig):
    lightning.seed_everything(cfg.lightning.seed, workers=True)
    NUM_GPUS = len(cfg.trainer.devices)

    if (
        cfg.this_main.force_batch_size is False
        and cfg.data.loader.validation.batch_size > 8
        and cfg.this_main.crop_mode in [None, "integer_patches"]
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
        import datetime

        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")
        log_dir = Path("log") / now
        test_dir_name = "test_empty_ckpt"
    else:
        # has ckpt, use the log dir of the ckpt path
        log_dir = Path(cfg.trainer.ckpt_path_to_load).parents[1]
        test_dir_name = "test"
    log_dir.mkdir(parents=True, exist_ok=True)

    # init csv logger
    csv_logger = CSVLogger(save_dir=log_dir, name=test_dir_name)

    # add logdir to cfg so lightning model can write to it
    with open_dict(cfg):
        if cfg.logger.test.out_dir is None:
            cfg.logger.test.out_dir = csv_logger.log_dir + f"_{cfg.alias}"  # the version_* dir

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
    elif cfg.this_main.crop_mode == "integer_patches":
        transforms["crop_integer_patches"] = "adaptive"

    if cfg.this_main.resize_short_side > 0:
        transforms["resize"] = T.Resize(
            cfg.this_main.resize_short_side,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )

    dataset = {
        "test": get_dataset(cfg, transforms, cfg.this_main.data_split, return_item_paths=True)
    }
    dataloader_test = DataLoader(
        dataset["test"],
        batch_size=cfg.data.loader.validation.batch_size,
        shuffle=cfg.data.loader.validation.shuffle,
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
        logger=csv_logger,
    )

    trainer.test(
        model,
        dataloader_test,
        ckpt_path=cfg.trainer.ckpt_path_to_load,
    )


if __name__ == "__main__":
    with torch.no_grad():
        test()
