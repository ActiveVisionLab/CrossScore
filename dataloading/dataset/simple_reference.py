import os, sys
from pathlib import Path
import torch
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parents[2]))
from dataloading.dataset.nvs_dataset import NvsDataset, NeighbourSelector, vis_batch


class SimpleReference(NvsDataset):
    def __init__(
        self,
        query_dir,
        reference_dir,
        transforms,
        neighbour_config,
        return_debug_info=False,
        return_item_paths=False,
        **kwargs,
    ):
        self.transforms = transforms
        self.neighbour_config = neighbour_config
        self.return_debug_info = return_debug_info
        self.return_item_paths = return_item_paths
        self.zero_reference = kwargs.get("zero_reference", False)

        self._detect_conflict_transforms()
        self.metric_config = self._build_empty_metric_config()

        self.all_paths = self.get_paths(query_dir, reference_dir)
        self.neighbour_selector = NeighbourSelector(self.all_paths, self.neighbour_config)

    def _build_empty_metric_config(self):
        cfg = {
            "type": None,
            "vrange": None,
            "load_dir": None,
        }
        cfg = OmegaConf.create(cfg)
        return cfg

    @staticmethod
    def get_paths(query_dir, reference_dir):
        """Define query and reference paths for ONE scene.
        This function is written in a way that mimics the
        NvsDataset.get_paths(), so that we can reuse most NvsDataset methods.

        :param scene_name:      str
        :param query_dir:       str, a dir that contains query images
        :param reference_dir:   str, a dir that contains reference images
        """

        query_dir = os.path.expanduser(query_dir)
        reference_dir = os.path.expanduser(reference_dir)
        query_paths = [os.path.join(query_dir, p) for p in sorted(os.listdir(query_dir))]
        reference_paths = [
            os.path.join(reference_dir, p) for p in sorted(os.listdir(reference_dir))
        ]

        fake_iter = -1
        query = {
            "images": {fake_iter: query_paths},
            "score_map": {fake_iter: ["empty_image"] * len(query_paths)},
            "N_iters": 1,
            "N_imgs_per_iter": len(query_paths),
        }
        reference = {
            "cross": {
                "images": {fake_iter: reference_paths},
                "N_iters": 1,
                "N_imgs_per_iter": len(reference_paths),
            }
        }

        # use query dir as scene name and anonymize the path
        scene_name = str(query_dir).replace(str(Path.home()), "~")
        results = {
            scene_name: {
                "gs_test": {
                    "query": query,
                    "reference": reference,
                },
            }
        }
        return results


if __name__ == "__main__":
    from lightning import seed_everything
    from torchvision.transforms import v2 as T
    from tqdm import tqdm
    from dataloading.transformation.crop import CropperFactory
    from utils.io.images import ImageNetMeanStd
    from omegaconf import OmegaConf

    seed_everything(1)
    cfg = {
        "data": {
            "dataset": {
                "query_dir": "datadir/processed_training_ready/gaussian/map-free-reloc/res_540/s00000/test/ours_1000/renders",
                "reference_dir": "datadir/processed_training_ready/gaussian/map-free-reloc/res_540/s00000/train/ours_1000/gt",
                "resolution": "res_540",
                "zero_reference": False,
            },
            "loader": {
                "batch_size": 8,
                "num_workers": 4,
                "shuffle": True,
                "pin_memory": True,
                "persistent_workers": False,
            },
            "transforms": {
                "crop_size": 392,
            },
            "neighbour_config": {
                "strategy": "random",
                "cross": 5,
                "deterministic": False,
            },
        },
        "this_main": {
            "skip_vis": False,
            "save_fig_dir": "./debug/dataset/simple_reference",
            "epochs": 1,
            "skip_batches": -1,
            "deterministic_crop": False,
            "return_debug_info": True,
            "return_item_paths": True,
            "resize_short_side": 518,  # -1 to disable
            # "crop_mode": "dataset_default",
            "crop_mode": None,
        },
        "model": {
            "patch_size": 14,
            "do_reference_cross": True,
            "predict": {
                "metric": {
                    "type": "ssim",
                    "min": 0,
                    "max": 1,
                },
            },
        },
    }
    cfg = OmegaConf.create(cfg)

    # overwrite cfg in some conditions
    if cfg.data.dataset.resolution == "res_540":
        cfg.data.transforms.crop_size = 518
    elif cfg.data.dataset.resolution == "res_400":
        cfg.data.transforms.crop_size = 392
    elif cfg.data.dataset.resolution == "res_200":
        cfg.data.transforms.crop_size = 196
    else:
        raise ValueError("Unknown resolution")

    if cfg.data.loader.num_workers == 0:
        cfg.data.loader.persistent_workers = False

    # for dataloader
    img_norm_stat = ImageNetMeanStd()
    transforms = {
        "img": T.Normalize(mean=img_norm_stat.mean, std=img_norm_stat.std),
    }

    if cfg.this_main.crop_mode == "dataset_default":
        transforms["query_crop"] = CropperFactory(
            output_size=(cfg.data.transforms.crop_size, cfg.data.transforms.crop_size),
            same_on_batch=True,
            deterministic=cfg.this_main.deterministic_crop,
        )
        transforms["reference_crop"] = CropperFactory(
            output_size=(cfg.data.transforms.crop_size, cfg.data.transforms.crop_size),
            same_on_batch=False,
            deterministic=cfg.this_main.deterministic_crop,
        )

    if cfg.this_main.resize_short_side > 0:
        transforms["resize"] = T.Resize(
            cfg.this_main.resize_short_side,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )

    dataset = SimpleReference(
        query_dir=cfg.data.dataset.query_dir,
        reference_dir=cfg.data.dataset.reference_dir,
        transforms=transforms,
        neighbour_config=cfg.data.neighbour_config,
        return_debug_info=cfg.this_main.return_debug_info,
        return_item_paths=cfg.this_main.return_item_paths,
        zero_reference=cfg.data.dataset.zero_reference,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.data.loader.batch_size,
        shuffle=cfg.data.loader.shuffle,
        num_workers=cfg.data.loader.num_workers,
        pin_memory=cfg.data.loader.pin_memory,
        persistent_workers=cfg.data.loader.persistent_workers,
        collate_fn=dataset.collate_fn_debug if cfg.this_main.return_debug_info else None,
    )

    # actual looping dataset
    for e in tqdm(range(cfg.this_main.epochs), desc="Epoch", dynamic_ncols=True):
        for b, batch in enumerate(tqdm(dataloader, desc="Batch", dynamic_ncols=True)):
            if cfg.this_main.skip_batches > 0 and b >= cfg.this_main.skip_batches:
                break

            if cfg.this_main.skip_vis:
                continue

            vis_batch(
                cfg,
                batch,
                metric_type=cfg.model.predict.metric.type,
                metric_min=cfg.model.predict.metric.min,
                metric_max=cfg.model.predict.metric.max,
                e=e,
                b=b,
            )
