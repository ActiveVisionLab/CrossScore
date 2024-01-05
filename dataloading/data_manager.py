import torch
from omegaconf import OmegaConf, ListConfig
from dataloading.dataset.nvs_dataset import NvsDataset
from pprint import pprint


def get_dataset(cfg, transforms, data_split, return_item_paths=False):
    if isinstance(cfg.data.dataset.path, str):
        dataset_path_list = [cfg.data.dataset.path]
    elif isinstance(cfg.data.dataset.path, ListConfig):
        dataset_path_list = OmegaConf.to_object(cfg.data.dataset.path)
    else:
        raise ValueError("cfg.data.dataset.path should be a string or a ListConfig")

    N_dataset = len(dataset_path_list)
    print(f"Get {N_dataset} datasets for {data_split}:")
    pprint(f"cfg.data.dataset.path: {dataset_path_list}")
    print("==================================")

    dataset_list = []
    for i in range(N_dataset):
        dataset_list.append(
            NvsDataset(
                dataset_path=dataset_path_list[i],
                resolution=cfg.data.dataset.resolution,
                data_split=data_split,
                transforms=transforms,
                neighbour_config=cfg.data.neighbour_config,
                metric_type=cfg.model.predict.metric.type,
                metric_min=cfg.model.predict.metric.min,
                metric_max=cfg.model.predict.metric.max,
                return_item_paths=return_item_paths,
                num_gaussians_iters=cfg.data.dataset.num_gaussians_iters,
                zero_reference=cfg.data.dataset.zero_reference,
            )
        )
    if N_dataset == 1:
        dataset = dataset_list[0]
    else:
        dataset = torch.utils.data.ConcatDataset(dataset_list)
    return dataset
