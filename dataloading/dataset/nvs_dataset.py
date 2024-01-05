import os, sys, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parents[2]))
from utils.io.images import metric_map_read, image_read
from utils.neighbour.sampler import SamplerFactory
from utils.check_config import ConfigChecker


class NeighbourSelector:
    """Return paths for neighbouring images (and metric maps) for query and reference."""

    def __init__(self, paths, neighbour_config):
        self.paths = paths
        self.neighbour_config = neighbour_config

        self.idx_to_property_mapper = self._build_idx_to_property_mapper(self.paths)
        self.all_scene_names = np.array(sorted(self.paths.keys()))

        self.neighbour_sampler_ref_cross = None
        if self.neighbour_config["cross"] > 0:
            # since query is not in cross ref set, we can only do random sampling
            self.neighbour_sampler_ref_cross = SamplerFactory(
                strategy_name="random",
                N_sample=self.neighbour_config["cross"],
                include_query=False,
                deterministic=self.neighbour_config["deterministic"],
            )

    @staticmethod
    def _build_idx_to_property_mapper(paths):
        """Only consider query, since the dataloading idx is based on number of query images"""

        scene_name_list = sorted(paths.keys())
        gaussian_split_list = ["train", "test"]
        global_idx = 0

        idx_to_property_mapper = {}
        for sn in scene_name_list:
            for gs_split in gaussian_split_list:
                if f"gs_{gs_split}" not in paths[sn].keys():
                    continue
                n_iter = paths[sn][f"gs_{gs_split}"]["query"]["N_iters"]
                n_imgs_per_iter = paths[sn][f"gs_{gs_split}"]["query"]["N_imgs_per_iter"]
                n_imgs = n_iter * n_imgs_per_iter
                for idx in range(n_imgs):
                    idx_to_property_mapper[global_idx] = {
                        "scene_name": sn,
                        "gaussian_split": gs_split,
                        "iter_idx": idx // n_imgs_per_iter,
                        "img_idx": idx % n_imgs_per_iter,
                    }
                    global_idx += 1
        return idx_to_property_mapper

    def __len__(self):
        return len(self.idx_to_property_mapper)

    def __getitem__(self, idx):
        results = {
            "query/img": None,
            "query/score_map": None,
            "reference/cross/imgs": [],
        }

        scene_name, gaussian_split, iter_idx, img_idx = self.idx_to_property_mapper[idx].values()
        tmp_split_paths = self.paths[scene_name][f"gs_{gaussian_split}"]
        iter_name = list(tmp_split_paths["query"]["images"].keys())[iter_idx]

        results["query/img"] = tmp_split_paths["query"]["images"][iter_name][img_idx]
        results["query/score_map"] = tmp_split_paths["query"]["score_map"][iter_name][img_idx]

        # sampling reference images for cross set
        if self.neighbour_sampler_ref_cross is not None:
            ref_list_cross = tmp_split_paths["reference"]["cross"]["images"][iter_name]
            results["reference/cross/imgs"] = self.neighbour_sampler_ref_cross(
                query=None, ref_list=ref_list_cross
            )

        return results


class NvsDataset(Dataset):

    def __init__(
        self,
        dataset_path,
        resolution,
        data_split,
        transforms,
        neighbour_config,
        metric_type,
        metric_min,
        metric_max,
        return_debug_info=False,
        return_item_paths=False,
        **kwargs,
    ):
        """
        :param scene_path:  Gaussian Splatting output scene dir that contains point cloud, test, train etc.
        :param query_split: train or test
        :param transforms:  a dict of transforms for all, img, metric_map
        """
        self.transforms = transforms
        self.neighbour_config = neighbour_config
        self.return_debug_info = return_debug_info
        self.return_item_paths = return_item_paths
        self.zero_reference = kwargs.get("zero_reference", False)
        self.num_gaussians_iters = kwargs.get("num_gaussians_iters", -1)

        if data_split not in ["train", "test", "val", "val_small", "test_small"]:
            raise ValueError(f"Unknown data_split {data_split}")

        self._detect_conflict_transforms()
        self.metric_config = self._build_metric_config(metric_type, metric_min, metric_max)

        # read split json for scene names
        if resolution is None:
            resolution = os.listdir(dataset_path)[0]
        self.dataset_path = Path(dataset_path, resolution)
        with open(self.dataset_path / "split.json", "r") as f:
            scene_names = json.load(f)[data_split]
            scene_paths = [self.dataset_path / n for n in sorted(scene_names)]

            # We use same split for all processed methods (e.g. gaussian, nerfacto, etc.).
            # Some scenes may not be processed by some methods, so we need to filter out.
            scene_paths = [p for p in scene_paths if p.exists()]

        # Define query and ref sets. Get all paths points to images and metric maps.
        self.all_paths = self.get_paths(
            scene_paths, self.num_gaussians_iters, self.metric_config.load_dir
        )

        self.neighbour_selector = NeighbourSelector(
            self.all_paths,
            self.neighbour_config,
        )

    def __getitem__(self, idx):
        # neighouring logic
        item_paths = self.neighbour_selector[idx]

        # load content from related paths
        result = self.load_content(item_paths, self.zero_reference, self.metric_config)

        if "resize" in self.transforms:
            result = self.resize_all(result)

        if "crop_integer_patches" in self.transforms:
            result = self.adaptive_crop_integer_patches_all(result)

        if self.return_debug_info:
            result["debug"] = {
                "query/ori_img": result["query/img"],
                "query/ori_score_map": result["query/score_map"],
                "reference/cross/ori_imgs": result["reference/cross/imgs"],
            }

        if self.return_item_paths:
            result["item_paths"] = item_paths

        # apply transforms to query
        transformed_query = self.transform_query(
            result["query/img"],
            result["query/score_map"],
        )
        result["query/img"] = transformed_query["img"]
        result["query/score_map"] = transformed_query["score_map"]
        if self.return_debug_info:
            result["debug"]["query/crop_param"] = transformed_query["crop_param"]

        if self.neighbour_config["cross"] > 0:
            transformed_ref_cross = self.transform_reference(result["reference/cross/imgs"])
            result["reference/cross/imgs"] = transformed_ref_cross["imgs"]
            if self.return_debug_info:
                result["debug"]["reference/cross/crop_param"] = transformed_ref_cross["crop_param"]
        else:
            del result["reference/cross/imgs"]
        return result

    @staticmethod
    def collate_fn_debug(batch):
        """Only return the first item in the batch, because the original images
        before cropping are in different sizes.
        Using [None] to add a batch dimension at the front.
        """
        result = {
            "query/img": batch[0]["query/img"][None],
            "query/score_map": batch[0]["query/score_map"][None],
        }

        result["debug"] = {
            "query/ori_img": batch[0]["debug"]["query/ori_img"][None],
            "query/ori_score_map": batch[0]["debug"]["query/ori_score_map"][None],
            "query/crop_param": batch[0]["debug"]["query/crop_param"][None],
        }

        result["item_paths"] = batch[0]["item_paths"]

        if "reference/cross/imgs" in batch[0].keys():
            result["reference/cross/imgs"] = batch[0]["reference/cross/imgs"][None]
            result["debug"]["reference/cross/ori_imgs"] = batch[0]["debug"][
                "reference/cross/ori_imgs"
            ][None]
            result["debug"]["reference/cross/crop_param"] = batch[0]["debug"][
                "reference/cross/crop_param"
            ][None]

        return result

    def __len__(self):
        return len(self.neighbour_selector)

    def resize_all(self, results):
        results["query/img"] = self.transforms["resize"](results["query/img"])
        results["query/score_map"] = self.transforms["resize"](results["query/score_map"][None])[0]
        if "reference/cross/imgs" in results.keys():
            results["reference/cross/imgs"] = self.transforms["resize"](
                results["reference/cross/imgs"]
            )
        return results

    def adaptive_crop_integer_patches_all(self, results):
        """
        Adaptively crop all images to the closest integer patch size.
        This is needed for test_steps, where we need to compute loss on images in arbitrary sizes.
        """
        P = 14  # dinov2 patch size
        ori_h, ori_w = results["query/img"].shape[-2:]
        new_h = ori_h - ori_h % P
        new_w = ori_w - ori_w % P
        results["query/img"] = results["query/img"][:, :new_h, :new_w]
        results["query/score_map"] = results["query/score_map"][:new_h, :new_w]
        if len(results["reference/cross/imgs"]) > 0:
            results["reference/cross/imgs"] = results["reference/cross/imgs"][:, :, :new_h, :new_w]
        return results

    def transform_query(self, img, score_map):
        if self.transforms.get("query_crop", None) is not None:
            crop_results = self.transforms["query_crop"](img, score_map)
            img = crop_results["out"][0]
            score_map = crop_results["out"][1]
            crop_param = crop_results["crop_param"]
        else:
            crop_param = torch.tensor([0, 0, *img.shape[-2:]])  # (4)

        if self.transforms.get("img", None) is not None:
            img = self.transforms["img"](img)

        if self.transforms.get("metric_map", None) is not None:
            score_map = self.transforms["metric_map"](score_map[None, None])
            score_map = score_map[0, 0]

        return {
            "img": img,
            "score_map": score_map,
            "crop_param": crop_param,
        }

    def transform_reference(self, imgs):
        if self.transforms.get("reference_crop", None) is not None:
            crop_results = self.transforms["reference_crop"](imgs)
            imgs = crop_results["out"]
            crop_param = crop_results["crop_param"]
        else:
            crop_param = torch.stack(
                [torch.tensor([0, 0, *img.shape[-2:]]) for img in imgs]
            )  # (B, 4)

        if self.transforms.get("img", None) is not None:
            imgs = self.transforms["img"](imgs)
        return {
            "imgs": imgs,
            "crop_param": crop_param,
        }

    def _detect_conflict_transforms(self):
        if "resize" in self.transforms:
            crop_sizes = []
            if "query_crop" in self.transforms:
                crop_sizes.append(self.transforms["query_crop"].output_size)
            if "reference_crop" in self.transforms:
                crop_sizes.append(self.transforms["reference_crop"].output_size)

            if len(crop_sizes) > 0:
                max_crop_size = np.max(crop_sizes)
                min_resize_size = np.min(self.transforms["resize"].size)
                if min_resize_size < max_crop_size:
                    raise ValueError(
                        f"Required to resize image before crop, "
                        f"but min_resize_size {min_resize_size} "
                        f"< max_crop_size {max_crop_size}"
                    )

    def _build_metric_config(self, metric_type, metric_min, metric_max):
        """
        Convert predict_type to load_dir and vrange for metric map reading.
        Supported predict_types: ssim_0_1, ssim_-1_1, mse, mae
        """
        vrange = [metric_min, metric_max]

        if metric_type in ["ssim", "mae"]:
            load_dir = f"metric_map/{metric_type}"
        elif metric_type in ["mse"]:
            load_dir = "metric_map/mae"  # mse can be derived from mae
        else:
            raise ValueError(f"Invalid metric type {metric_type}")

        cfg = {
            "type": metric_type,
            "vrange": vrange,
            "load_dir": load_dir,
        }
        cfg = OmegaConf.create(cfg)
        return cfg

    @staticmethod
    def get_paths(scene_paths, num_gaussians_iters, metric_load_dir):
        """Get paths points to images and metric maps. Define query and refenence sets.
        Naming convention:
            Query:
                The (noisy) image we want to measure.
            Reference:
                Images for making predictions.
                For cross ref, we consider captured images that from ref splits.
            Example:
                When query_split is "train", we consider captured test images as cross ref.
                When query_split is "test", we consider captured training images as cross ref.
        """
        scene_name_list = sorted([scene_path.name for scene_path in scene_paths])
        all_paths = {
            scene_name: {
                "train": {
                    "renders": {},
                    "gt": {},
                    "score_map": {},
                },
                "test": {
                    "renders": {},
                    "gt": {},
                    "score_map": {},
                },
            }
            for scene_name in scene_name_list
        }

        for scene_path in scene_paths:
            scene_name = scene_path.name
            for gs_split in all_paths[scene_name].keys():
                dir_split = Path(scene_path, gs_split)
                dir_iter_list = sorted(os.listdir(dir_split), key=lambda x: int(x.split("_")[-1]))
                dir_iter_list = [Path(dir_split, d) for d in dir_iter_list]

                # This dataset contains images rendered from gaussian splatting checkpoints
                # at different iterations. Use this to use images from earlier checkpoints,
                # which have more artefacts.
                if num_gaussians_iters > 0:
                    dir_iter_list = dir_iter_list[:num_gaussians_iters]

                for dir_iter in dir_iter_list:
                    iter_num = int(dir_iter.name.split("_")[-1])
                    for img_type in all_paths[scene_name][gs_split].keys():
                        if img_type in ["renders", "gt"]:
                            img_dir = dir_iter / img_type
                        elif img_type == "score_map":
                            img_dir = dir_iter / metric_load_dir
                        else:
                            raise ValueError(f"Unknown img_type {img_type}")

                        if os.path.exists(img_dir):
                            img_names = sorted(os.listdir(img_dir))
                            paths = [img_dir / img_n for img_n in img_names]
                            paths = [str(p) for p in paths]
                        else:
                            # if no metric map available, use a placeholder "empty_image"
                            paths = ["empty_image"] * len(all_paths[scene_name][gs_split]["gt"])
                        all_paths[scene_name][gs_split][img_type][iter_num] = paths

                # all types of items should have the same item number as gt
                for img_type in all_paths[scene_name][gs_split].keys():
                    for iter_num in all_paths[scene_name][gs_split][img_type].keys():
                        N_imgs = len(all_paths[scene_name][gs_split][img_type][iter_num])
                        N_gt = len(all_paths[scene_name][gs_split]["gt"][iter_num])
                        if N_imgs != N_gt:
                            raise ValueError(
                                f"Number of items mismatch in "
                                f"{scene_name}/{gs_split}/{iter_num}/{img_type}"
                            )

        # assemble query and reference sets
        def get_cross_ref_split(query_split):
            all_splits = ["train", "test"]
            all_splits.remove(query_split)
            cross_ref_split = all_splits[0]
            return cross_ref_split

        results = {}
        for scene_name in scene_name_list:
            results[scene_name] = {}
            for gs_split in ["train", "test"]:
                cross_ref_split = get_cross_ref_split(gs_split)

                results[scene_name][f"gs_{gs_split}"] = {
                    "query": {
                        "images": all_paths[scene_name][gs_split]["renders"],
                        "score_map": all_paths[scene_name][gs_split]["score_map"],
                        "N_iters": len(all_paths[scene_name][gs_split]["renders"]),
                        "N_imgs_per_iter": len(
                            list(all_paths[scene_name][gs_split]["renders"].values())[0]
                        ),
                    },
                    "reference": {
                        "cross": {
                            "images": all_paths[scene_name][cross_ref_split]["gt"],
                            "N_iters": len(all_paths[scene_name][cross_ref_split]["gt"]),
                            "N_imgs_per_iter": len(
                                list(all_paths[scene_name][cross_ref_split]["gt"].values())[0]
                            ),
                        },
                    },
                }
        return results

    @staticmethod
    def load_content(item_paths, zero_reference, metric_config):
        results = {
            "query/img": None,
            "query/score_map": None,
            "reference/cross/imgs": [],
        }

        for k in item_paths.keys():
            if k == "query/img":
                results[k] = torch.tensor(image_read(item_paths[k])).permute(2, 0, 1)  # (3, H, W)
            elif k == "query/score_map":
                if metric_config.type == "ssim":
                    if item_paths[k] == "empty_image":
                        results[k] = torch.zeros_like(results["query/img"][0])  # (H, W)
                    else:
                        # (H, W), always read SSIM in range [-1, 1]
                        results[k] = torch.tensor(metric_map_read(item_paths[k], vrange=[-1, 1]))
                        if metric_config.vrange == [0, 1]:
                            results[k] = results[k].clamp(0, 1)
                elif metric_config.type in ["mse", "mae"]:
                    if item_paths[k] == "empty_image":
                        results[k] = torch.full_like(results["query/img"][0], torch.nan)  # (H, W)
                    else:
                        # (H, W), always read MAE in range [0, 1]
                        results[k] = torch.tensor(metric_map_read(item_paths[k], vrange=[0, 1]))
                        if metric_config.type == "mse":
                            results[k] = results[k].square()  # create mse from loaded mae
                elif metric_config.type is None:
                    # for SimpleReference, which doesn't need to load score maps
                    results[k] = torch.zeros_like(results["query/img"][0])
            elif k in ["reference/cross/imgs"]:
                if len(item_paths[k]) == 0:
                    continue  # if not loading this reference set, skip
                for path in item_paths[k]:
                    if path == "empty_image":
                        # NOTE: this assumes ref_img_size == query_img_size
                        tmp_img = torch.zeros_like(results["query/img"])  # (3, H, W)
                    else:
                        tmp_img = torch.tensor(image_read(path)).permute(2, 0, 1)  # (3, H, W)
                    results[k].append(tmp_img)  # (3, H, W)
                results[k] = torch.stack(results[k], dim=0)  # (N, 3, H, W)
                if zero_reference:
                    results[k] = torch.zeros_like(results[k])
            else:
                raise ValueError(f"Unknown key {k}")
        return results


def vis_batch(cfg, batch, metric_type, metric_min, metric_max, e, b):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    metric_vrange = [metric_min, metric_max]

    # mkdir to save figures
    save_fig_dir = Path(cfg.this_main.save_fig_dir).expanduser()
    save_fig_dir.mkdir(parents=True, exist_ok=True)

    # Vis batch[0] in two figures: one actual loaded and one with debug info
    # First figure with actual loaded data
    max_cols = max([3, cfg.data.neighbour_config.cross])
    _, axes = plt.subplots(2, max_cols, figsize=(15, 9))
    for ax in axes.flatten():
        ax.set_axis_off()

    # first row: query
    axes[0][0].imshow(batch["query/img"][0].permute(1, 2, 0).clip(0, 1))
    axes[0][0].set_title("query/img")
    axes[0][1].imshow(
        batch["query/score_map"][0],
        vmin=metric_vrange[0],
        vmax=metric_vrange[1],
        cmap="turbo",
    )
    axes[0][1].set_title(f"query/{metric_type}_map")
    for i in range(2):
        axes[0][i].set_axis_on()

    # second row: cross ref
    if "reference/cross/imgs" in batch.keys():
        for i in range(batch["reference/cross/imgs"].shape[1]):
            axes[1][i].imshow(batch["reference/cross/imgs"][0, i].permute(1, 2, 0).clip(0, 1))
            axes[1][i].set_title(f"reference/cross/imgs_{i}")
            axes[1][i].set_axis_on()

    plt.tight_layout()
    plt.savefig(save_fig_dir / f"e{e}b{b}.jpg")
    plt.close()

    # Second figure with full debug details in three rows
    if cfg.this_main.return_debug_info:
        _, axes = plt.subplots(2, max_cols, figsize=(20, 10))
        for ax in axes.flatten():
            ax.set_axis_off()

        # first row: query
        axes[0][0].imshow(batch["debug"]["query/ori_img"][0].permute(1, 2, 0).clip(0, 1))
        axes[0][0].set_title("query/ori_img")
        axes[0][1].imshow(
            batch["debug"][f"query/ori_score_map"][0],
            vmin=metric_vrange[0],
            vmax=metric_vrange[1],
            cmap="turbo",
        )
        axes[0][1].set_title(f"query/ori_{metric_type}_map")
        # crop box
        crop_param = batch["debug"]["query/crop_param"][0]
        for i in range(2):
            rect = Rectangle(
                (crop_param[1], crop_param[0]),
                crop_param[3],
                crop_param[2],
                linewidth=3,
                edgecolor="r",
                facecolor="none",
            )
            axes[0][i].add_patch(rect)
            axes[0][i].set_axis_on()

        # third row: cross ref
        if "reference/cross/imgs" in batch.keys():
            for i in range(batch["debug"]["reference/cross/ori_imgs"].shape[1]):
                axes[1][i].imshow(
                    batch["debug"]["reference/cross/ori_imgs"][0, i].permute(1, 2, 0).clip(0, 1)
                )
                axes[1][i].set_title(f"reference/cross/ori_imgs_{i}")
                # crop box
                crop_param = batch["debug"]["reference/cross/crop_param"][0][i]
                rect = Rectangle(
                    (crop_param[1], crop_param[0]),
                    crop_param[3],
                    crop_param[2],
                    linewidth=3,
                    edgecolor="r",
                    facecolor="none",
                )
                axes[1][i].add_patch(rect)
                axes[1][i].set_axis_on()

        plt.tight_layout()
        plt.savefig(save_fig_dir / f"e{e}b{b}_full.jpg")
        plt.close()


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
                "path": "datadir/processed_training_ready/gaussian/map-free-reloc",
                "resolution": "res_540",
                "num_gaussians_iters": 1,
                "zero_reference": False,
            },
            "loader": {
                "batch_size": 8,
                "num_workers": 4,
                "shuffle": True,
                "data_split": "train",
                "pin_memory": True,
                "persistent_workers": True,
            },
            "transforms": {
                "crop_size": 518,
            },
            "neighbour_config": {
                "strategy": "random",
                "cross": 5,
                "deterministic": False,
            },
        },
        "this_main": {
            "skip_vis": False,
            "save_fig_dir": "./debug/dataset/NvsData",
            "epochs": 3,
            "skip_batches": 5,
            "deterministic_crop": False,
            "return_debug_info": True,
            "return_item_paths": True,
            "resize_short_side": -1,  # -1 to disable
            "crop_mode": "dataset_default",
            # "crop_mode": None,
        },
        "model": {
            "patch_size": 14,
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

    # Overwrite cfg in some conditions
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

    # Check config
    ConfigChecker(cfg).check_dataset()

    # Init transforms for dataset
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

    # Init dataset and dataloader
    dataset = NvsDataset(
        dataset_path=cfg.data.dataset.path,
        resolution=cfg.data.dataset.resolution,
        data_split=cfg.data.loader.data_split,
        transforms=transforms,
        neighbour_config=cfg.data.neighbour_config,
        metric_type=cfg.model.predict.metric.type,
        metric_min=cfg.model.predict.metric.min,
        metric_max=cfg.model.predict.metric.max,
        return_debug_info=cfg.this_main.return_debug_info,
        return_item_paths=cfg.this_main.return_item_paths,
        num_gaussians_iters=cfg.data.dataset.num_gaussians_iters,
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

    # Actual looping dataset
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
