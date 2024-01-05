from pathlib import Path
import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from utils.io.images import metric_map_read
from utils.evaluation.metric import mse2psnr


class ScoreReader(Dataset):
    def __init__(self, score_map_dir_list):
        # get all paths to read
        read_score_types = ["ssim", "mae"]
        self.read_paths_all = {k: [] for k in read_score_types}
        for score_read_type in read_score_types:
            for score_map_dir in score_map_dir_list:
                tmp_dir = os.path.join(score_map_dir, score_read_type)
                tmp_paths = [os.path.join(tmp_dir, n) for n in sorted(os.listdir(tmp_dir))]
                self.read_paths_all[score_read_type].extend(tmp_paths)

        # （N_frames, 2)
        self.read_paths_all = np.stack([self.read_paths_all[k] for k in read_score_types], axis=1)

    def __len__(self):
        return len(self.read_paths_all)

    def __getitem__(self, idx):
        path_ssim, path_mae = self.read_paths_all[idx]
        ssim_map = metric_map_read(path_ssim, vrange=[-1, 1])
        mae_map = metric_map_read(path_mae, vrange=[0, 1])
        mse_map = np.square(mae_map)

        score_ssim_n11 = ssim_map.mean()
        score_ssim_01 = ssim_map.clip(0, 1).mean()
        score_mae = mae_map.mean()
        score_mse = mse_map.mean()
        score_psnr = mse2psnr(torch.tensor([score_mse])).numpy()

        results = {
            "ssim_-1_1": score_ssim_n11,
            "ssim_0_1": score_ssim_01,
            "mae": score_mae,
            "mse": score_mse,
            "psnr": score_psnr,
            "path_ssim": path_ssim,
        }
        return results


class SummaryWriterGroundTruth:
    """
    Load the ground truth results from disk and summarise the results.
    """

    def __init__(self, dir_in, dir_out, num_workers, fast_debug, force):
        self.dir_in = Path(dir_in).expanduser()
        self.dir_out = Path(dir_out).expanduser()
        self.num_workers = num_workers
        self.fast_debug = fast_debug
        self.force = force

        self.dataset_type = self.dir_in.parent.name
        self.rendering_method = self.dir_in.parents[1].name
        self.csv_dir = self.dir_out / self.dataset_type
        self.csv_path = self.csv_dir / f"{self.rendering_method}.csv"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.columns = [
            "scene_name",
            "rendered_dir",
            "image_name",
            "gt_ssim_-1_1",
            "gt_ssim_0_1",
            "gt_mae",
            "gt_mse",
            "gt_psnr",
        ]

    def write_csv(self):
        write = self._check_write(self.csv_path, self.force)
        if write:
            rows = self._load_per_frame_score()
            df = DataFrame(data=rows, columns=self.columns)
            df.to_csv(self.csv_path, index=False, float_format="%.4f")

    def _check_write(self, csv_path, force):
        if csv_path.exists():
            if force:
                csv_path.unlink()
                print(f"Write to csv {csv_path} (OVERWRITE)")
                write = True
            else:
                print(f"Write to csv {csv_path} (SKIP)")
                write = False
        else:
            print(f"Write to csv {csv_path} (NORMAL)")
            write = True
        return write

    def _load_per_frame_score(self):
        # use glob to find all dir named "metric_map" in the scene dirs
        score_map_dir_list = sorted(glob(str(self.dir_in / "**/metric_map"), recursive=True))
        score_reader = ScoreReader(score_map_dir_list)
        score_loader = DataLoader(
            dataset=score_reader,
            batch_size=16,
            shuffle=False,
            num_workers=self.num_workers,
        )

        # process score maps to csv rows, each row contains the following columns
        rows = []
        for i, data in enumerate(tqdm(score_loader, desc=f"Loading gt scores", dynamic_ncols=True)):
            for j in range(len(data["path_ssim"])):
                path_ssim = data["path_ssim"][j]
                scene_name = path_ssim.split("/")[-6]
                rendered_dir = os.path.join(*path_ssim.split("/")[:-3])
                image_name = path_ssim.split("/")[-1]
                image_name = image_name.replace("frame_", "")
                tmp_row = [
                    scene_name,
                    rendered_dir,
                    image_name,
                    data["ssim_-1_1"][j].item(),
                    data["ssim_0_1"][j].item(),
                    data["mae"][j].item(),
                    data["mse"][j].item(),
                    data["psnr"][j].item(),
                ]
                rows.append(tmp_row)
            if self.fast_debug > 0 and i >= self.fast_debug:
                break
        return rows


class SummaryWriterPredictedOnline:
    """
    Used in at the fit/test/predict phase with Lightning Module.
    """

    def __init__(self, metric_type, metric_min):
        metric_type_str = self._get_metric_type_str(metric_type, metric_min)
        self.columns = [
            "scene_name",
            "rendered_dir",
            "image_name",
            f"pred_{metric_type_str}",
        ]
        self.reset()

    def _get_metric_type_str(self, metric_type, metric_min):
        if metric_type == "ssim":
            if metric_min == -1:
                metric_str = f"{metric_type}_-1_1"
            elif metric_min == 0:
                metric_str = f"{metric_type}_0_1"
        else:
            metric_str = f"{metric_type}"
        return metric_str

    def reset(self):
        self.rows = DataFrame(columns=self.columns)

    def update(self, batch_input, batch_output):
        """Store per frame scores to rows for each batch."""
        query_img_paths = batch_input["item_paths"]["query/img"]  # (B,)
        ref_types = [t for t in batch_output.keys() if t.startswith("score_map")]

        if len(ref_types) != 1:
            raise ValueError(f"Expect exactly one ref_type: self/cross, but got {ref_types}.")

        rows_batch = []
        for ref_type in ref_types:
            score_maps = batch_output[ref_type]  # (B, H, W)
            scores = score_maps.mean(dim=[-1, -2])  # (B,)

            scene_names = [p.split("/")[-5] for p in query_img_paths]

            rendered_dirs = [os.path.join(*p.split("/")[:-2]) for p in query_img_paths]
            image_names = [p.split("/")[-1] for p in query_img_paths]
            image_names = [n.replace("frame_", "") for n in image_names]

            for i in range(len(scene_names)):
                rows_batch.append(
                    [scene_names[i], rendered_dirs[i], image_names[i], scores[i].item()]
                )

        # concat rows to panda dataframe
        self.rows = pd.concat([self.rows, DataFrame(rows_batch, columns=self.columns)])

    def summarise(self):
        """Organise rows using dataset type and rendering method and sort them."""

        # get unique rendering methods
        rendering_method_list = self.rows["rendered_dir"].apply(lambda x: x.split("/")[-6]).unique()

        # get unique dataset types
        dataset_type_list = self.rows["rendered_dir"].apply(lambda x: x.split("/")[-5]).unique()

        # organise rows using dataset type and rendering method
        self.summary = {}
        for dataset_type in dataset_type_list:
            self.summary[dataset_type] = {}
            for rendering_method in rendering_method_list:
                # get rows with the same dataset type and rendering method
                tmp_rows = self.rows[
                    (self.rows["rendered_dir"].str.contains(rendering_method))
                    & (self.rows["rendered_dir"].str.contains(dataset_type))
                ]

                # sort rows by scene_name, rendered_dir, image_name
                tmp_rows = tmp_rows.sort_values(by=["scene_name", "rendered_dir", "image_name"])

                self.summary[dataset_type][rendering_method] = tmp_rows

    def __len__(self):
        return len(self.rows)

    def __repr__(self):
        return self.rows.__repr__()


class SummaryWriterPredictedOnlineTestPrediction(SummaryWriterPredictedOnline):
    """
    Used in at the end of validation/test/predict phase.
    Summarising with online predicted results to avoid reading from disks.
    """

    def __init__(self, metric_type, metric_min, dir_out):
        super().__init__(metric_type, metric_min)
        self.csv_dir = Path(dir_out).expanduser() / "score_summary"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.cache_csv_path = self.csv_dir / f"summarise_cache.csv"

    def summarise(self):
        super().summarise()

        # write to csv files
        for dataset_type, dataset_summary in self.summary.items():
            for rendering_method, rows in dataset_summary.items():
                tmp_csv_dir = self.csv_dir / dataset_type
                tmp_csv_dir.mkdir(parents=True, exist_ok=True)
                tmp_csv_path = tmp_csv_dir / f"{rendering_method}.csv"
                rows.to_csv(tmp_csv_path, index=False, float_format="%.4f")


class SummaryReader:
    @staticmethod
    def read_summary(summary_dir, dataset, method_list, scene_list, split_list, iter_list):
        summary_dir = Path(summary_dir).expanduser()
        summary_dir = summary_dir / dataset

        methods_available = [f.stem for f in summary_dir.iterdir() if f.is_file()]
        methods_to_read = []
        if method_list != [""]:
            for m in method_list:
                if m in methods_available:
                    methods_to_read.append(m)
                else:
                    raise ValueError(f"{m} is not available in {summary_dir}")
        else:
            methods_to_read = methods_available

        summary_files = [summary_dir / f"{m}.csv" for m in methods_to_read]

        # read csv files, create a new colume as 0th column for method_name
        summary = pd.concat(
            [pd.read_csv(f).assign(method_name=m) for f, m in zip(summary_files, methods_to_read)]
        )

        # filter with scene list using summary's scene_name column
        if scene_list != [""]:
            summary = summary[summary["scene_name"].isin(scene_list)]

        # filter split using rendered_dir column
        if split_list != [""]:
            new_s = []
            for split in split_list:
                tmp_s = summary[summary["rendered_dir"].str.split("/").str[-2] == split]
                new_s.append(tmp_s)
            summary = pd.concat(new_s)

        # filter iter using rendered_dir column, the last part of the path should be EXACTLY "ours_{iter}"
        if len(iter_list) > 0:
            new_s = []
            for i in iter_list:
                tmp_s = summary[summary["rendered_dir"].str.endswith(f"ours_{i}")]
                new_s.append(tmp_s)
            summary = pd.concat(new_s)

        # sort by scene_name, rendered_dir, image_name, method_name
        summary = summary.sort_values(["scene_name", "rendered_dir", "image_name", "method_name"])

        # reset index
        summary = summary.reset_index(drop=True)
        return summary

    @staticmethod
    def check_summary_gt_prediction_rows(summary_gt, summary_prediction):
        # these tow dataframes should have the same length
        # and the columns [rendered_dir, image_name] should be identical
        if len(summary_gt) != len(summary_prediction):
            raise ValueError("Summary GT and prediction have different length")

        if not summary_gt["rendered_dir"].equals(summary_prediction["rendered_dir"]):
            raise ValueError("Summary GT and prediction have different rendered_dir")

        if not summary_gt["image_name"].equals(summary_prediction["image_name"]):
            raise ValueError("Summary GT and prediction have different image_name")
